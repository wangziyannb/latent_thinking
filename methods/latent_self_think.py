from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from models import ModelWrapper
from utils import extract_gsm8k_answer, normalize_answer


@dataclass
class RunConfig:
    latent_steps: int = 40
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95


class LatentSelfThink:
    """Single-model latent thinking, reusing LatentMAS latent-step mechanism.

    Flow:
    1) Build a chat prompt for GSM8K.
    2) Run `generate_latent_batch(..., latent_steps=N)` to update KV cache without decoding.
    3) Re-feed a short decoding prompt (same question) with `past_key_values=past` and decode.

    Notes:
    - For fairness, decoding prompt is identical to the initial prompt here.
      You can optionally use a shorter decoding prompt (e.g. only 'Now answer:')
      as long as you keep it consistent across baselines.
    """

    def __init__(self, model: ModelWrapper, cfg: RunConfig):
        self.model = model
        self.cfg = cfg

    def _build_messages(self, question: str) -> List[Dict]:
        # Close to LatentMAS judger prompt for gsm8k.
        system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        user = (
            f"Target Question: {question}\n"
            "You must reason step-by-step to solve the provided Target Question. "
            "Now, reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @torch.no_grad()
    def run_one(self, item: Dict) -> Dict:
        question = item["question"]
        gold = item.get("gold")

        messages = self._build_messages(question)
        _, input_ids, attn = self.model.prepare_chat_input(messages, add_generation_prompt=True)

        past = None
        if self.cfg.latent_steps and self.cfg.latent_steps > 0:
            # 1) latent thinking (no decode)
            past = self.model.generate_latent_batch(
                input_ids=input_ids,
                attention_mask=attn,
                latent_steps=self.cfg.latent_steps,
                past_key_values=None,
            )

        # 2) decode final answer using the same prompt + past
        #    (You can replace with a shorter prompt if desired.)
        _, dec_ids, dec_attn = self.model.prepare_chat_input(messages, add_generation_prompt=True)
        texts, _, per_sample_new = self.model.generate_text_batch(
            input_ids=dec_ids,
            attention_mask=dec_attn,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            past_key_values=past,
        )

        decoded_tokens = int(per_sample_new[0])

        raw = texts[0]
        pred = normalize_answer(extract_gsm8k_answer(raw))
        gold_n = normalize_answer(str(gold)) if gold is not None else None
        ok = (pred == gold_n) if (pred and gold_n) else False

        return {
            "question": question,
            "gold": gold,
            "prediction": pred,
            "raw_prediction": raw,
            "correct": ok,
            "latent_steps": self.cfg.latent_steps,
            "decoded_tokens": decoded_tokens,
            "mode": "baseline" if (not self.cfg.latent_steps or self.cfg.latent_steps <= 0) else "latent",
        }
