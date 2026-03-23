from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from models import ModelWrapper
from utils import answers_match, extract_answer_for_task, normalize_prediction_for_task


@dataclass
class RunConfig:
    task: str = "gsm8k"
    latent_steps: int = 40
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95

    latent_early_stop: bool = False
    latent_early_stop_threshold: float = 0.8
    latent_early_stop_probe_text: str = "Judge whether it is true or false: It's time to output. My answer is:"

    latent_debug_decode: bool = False
    latent_debug_topk: int = 5
    latent_debug_temperature: float = 1.0

    # If True, ask the model to output only the final answer (no explanations).
    answer_only: bool = False

    # If True, always do a "prefill -> (optional latent steps) -> decode" loop.
    # This makes the baseline (latent_steps=0) match the latent setting's prompt reuse.
    loop_decode: bool = False
    decoding_new_message: bool = False


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

    def _build_messages(self, question: str, task: str) -> List[Dict]:
        system = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        if task == "math500":
            if self.cfg.answer_only:
                user = (
                    f"Math Problem: {question}\n"
                    "Return ONLY the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
                    "Do NOT include your reasoning steps."
                )
            else:
                user = (
                    f"Target Question: {question}\n"
                    "You must reason step-by-step to solve the provided Target Question. "
                    "Now, reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
                )
        else:
            if self.cfg.answer_only:
                user = (
                    f"Target Question: {question}\n"
                    "Return ONLY the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
                    "Do NOT include your reasoning steps."
                )
            else:
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
        task = item.get("task", self.cfg.task)
        question = item["question"]
        gold = item.get("gold")

        messages = self._build_messages(question, task)
        _, input_ids, attn = self.model.prepare_chat_input(messages, add_generation_prompt=True)

        past = None
        latent_steps_used = 0
        stop_p_true = False
        debug_steps = []

        if self.cfg.loop_decode or (self.cfg.latent_steps and self.cfg.latent_steps > 0):
            # 1) prefill + (optional) latent thinking (no decode)
            past, stop_p_true, latent_steps_used, debug_steps = self.model.generate_latent_batch(
                input_ids=input_ids,
                attention_mask=attn,
                latent_steps=max(int(self.cfg.latent_steps or 0), 0),
                past_key_values=None,
                early_stop=self.cfg.latent_early_stop,
                early_stop_threshold=self.cfg.latent_early_stop_threshold,
                early_stop_probe_text=self.cfg.latent_early_stop_probe_text,
                debug_decode=self.cfg.latent_debug_decode,
                debug_topk=self.cfg.latent_debug_topk,
                debug_temperature=self.cfg.latent_debug_temperature,
            )

        # 2) decode final answer using the same prompt + past
        #    (You can replace with a shorter prompt if desired.)
        if self.cfg.decoding_new_message:
            messages = [{"role": "user", "content": "\n"}]

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
        extracted = extract_answer_for_task(task, raw)
        pred = normalize_prediction_for_task(task, extracted)
        gold_n = normalize_prediction_for_task(task, str(gold)) if gold is not None else None
        ok = answers_match(task, pred, gold)

        result = {
            "task": task,
            "question": question,
            "gold": gold,
            "prediction": pred,
            "extracted_prediction": extracted,
            "raw_prediction": raw,
            "correct": ok,
            "latent_steps": self.cfg.latent_steps,
            "decoded_tokens": decoded_tokens,
            "mode": "baseline" if (not self.cfg.latent_steps or self.cfg.latent_steps <= 0) else "latent",
            "latent_steps_used": latent_steps_used,
            "early_stopped": (latent_steps_used < self.cfg.latent_steps),
            "early_stop_p_true": stop_p_true,
            "latent_debug_steps": debug_steps,
            "loop_decode": self.cfg.loop_decode,
        }
        if gold_n is not None:
            result["gold_normalized"] = gold_n

        for key in ("subject", "level", "unique_id"):
            if key in item:
                result[key] = item[key]
        return result
