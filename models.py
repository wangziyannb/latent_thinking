"""Model wrapper reused from LatentMAS (HF path only).

Key features:
- Chat prompt rendering with tokenizer chat_template when available
- `generate_latent_batch`: latent-only autoregressive steps by feeding aligned last-layer hidden
  states back through `inputs_embeds`, accumulating `past_key_values`
- `generate_text_batch`: standard decoding from (optional) past_key_values

This file is adapted from https://github.com/Gen-Verse/LatentMAS (Apache-2.0).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _past_length(past_key_values) -> int:
    if past_key_values is None:
        return 0

    # New HF cache objects (DynamicCache / StaticCache / Cache)
    if hasattr(past_key_values, "get_seq_length"):
        # transformers.cache_utils.Cache implements this
        return int(past_key_values.get_seq_length())

    # Some versions expose .seqlen or similar
    if hasattr(past_key_values, "seqlen"):
        return int(past_key_values.seqlen)

    # Legacy tuple format: ((k,v), (k,v), ...)
    k = past_key_values[0][0]
    return int(k.shape[-2])

class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, *, latent_space_realign: bool = False):
        self.model_name = model_name
        self.device = device
        self.latent_space_realign = latent_space_realign
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)

        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True

        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device)

    # ---------- prompt helpers ----------
    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

        # fallback generic
        segments = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(self, messages: List[Dict], add_generation_prompt: bool = True):
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        return prompt_text, input_ids, attention_mask

    # ---------- latent alignment (from LatentMAS) ----------
    def _build_latent_realign_matrix(self, model, device) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)

        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")

        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)

        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)

        target_norm = input_weight.norm(dim=1).mean().detach()  # scalar
        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)
        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)
        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        if not self.latent_space_realign:
            return hidden
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)
        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    # ---------- generation ----------
    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple], torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)

        prompt_lengths = attention_mask.sum(dim=1).tolist()

        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        input_len = input_ids.shape[1]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )

        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)

        new_tokens = (outputs.sequences.shape[1] - input_len) * outputs.sequences.shape[0]
        # 更严格：每条样本实际生成长度（考虑提前 eos）
        per_sample_new = (outputs.sequences != self.tokenizer.pad_token_id).sum(dim=1) - input_len
        per_sample_new = torch.clamp(per_sample_new, min=0)

        return generations, outputs.past_key_values, per_sample_new

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        early_stop=False,
        early_stop_threshold=0.8,
        early_stop_probe_text="Judge whether it is true or false: It's time to output. My answer is:",
    ) -> Tuple:
        """Run latent-only steps and return updated past_key_values.

        Adapted from LatentMAS `generate_latent_batch`.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # [B, H]

        if early_stop:
            probe_ids = self.tokenizer(early_stop_probe_text, return_tensors="pt", add_special_tokens=False)[
                "input_ids"].to(
                self.device)
            if probe_ids.shape[0] == 1 and input_ids.shape[0] > 1:
                probe_ids = probe_ids.expand(input_ids.shape[0], -1)
            true_token_ids = self._true_token_candidates()

        for steps in range(latent_steps):
            latent_vec = self._apply_latent_realignment(last_hidden, self.model)
            latent_embed = latent_vec.unsqueeze(1)  # [B,1,H]

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )

            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            if early_stop:
                p_true = self._probe_p_true_no_pollute(past, probe_ids, true_token_ids)
                if float(p_true[0].item()) > early_stop_threshold:
                    return past, True, steps
        return past, False, steps

    def _true_token_candidates(self):
        # 只要能做到：从这些候选里找单 token 的 id
        cands = []
        for s in ["True", " True", "\nTrue", "\n True"]:
            ids = self.tokenizer(s, add_special_tokens=False).input_ids
            if len(ids) == 1:
                cands.append(ids[0])
        if len(cands) == 0:
            # fallback：用 "True" 的第一个 token（保证能算 next-token prob）
            ids = self.tokenizer("True", add_special_tokens=False).input_ids
            cands.append(ids[0])
        return cands

    @torch.no_grad()
    def _probe_p_true_no_pollute(self, past, probe_ids, true_token_ids):
        # 1) 用 legacy cache 做只读输入，避免 Cache object 原地更新风险
        probe_past = past
        if hasattr(past, "to_legacy_cache"):
            probe_past = past.to_legacy_cache()

        B = probe_ids.shape[0]
        past_len = _past_length(probe_past)
        attn = torch.ones((B, past_len + probe_ids.shape[1]),
                          dtype=torch.long, device=self.device)

        out = self.model(
            input_ids=probe_ids,
            attention_mask=attn,
            past_key_values=probe_past,
            use_cache=False,  # 2) 不产生/不更新 cache（更稳）
            return_dict=True,
        )

        logits = out.logits[:, -1, :].float()  # next token logits
        probs = F.softmax(logits, dim=-1)

        # 取 True 候选里的最大概率（更鲁棒）
        p_true = torch.stack([probs[:, tid] for tid in true_token_ids], dim=-1).max(dim=-1).values
        return p_true  # shape [B]