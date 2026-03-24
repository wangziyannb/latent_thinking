from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from methods.phase_config import (
    PhaseOptions,
    build_messages,
    resolve_phase_config,
)
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

    answer_only: bool = False
    loop_decode: bool = False
    decoding_new_message: bool = False

    legacy_disable_thinking: bool = False
    strategy_label: str = ""

    latent_thinking_mode: str = "inherit"
    decode_thinking_mode: str = "inherit"
    latent_strip_think_tags: bool = False
    decode_strip_think_tags: bool = False
    close_latent_think_tag_before_decode: bool = False

    latent_prompt_preset: str = "shared"
    decode_prompt_preset: str = "shared"
    latent_system_prompt: str = ""
    latent_user_prompt: str = ""
    decode_system_prompt: str = ""
    decode_user_prompt: str = ""

    def latent_phase_options(self) -> PhaseOptions:
        return PhaseOptions(
            thinking_mode=self.latent_thinking_mode,
            strip_think_tags=self.latent_strip_think_tags,
            prompt_preset=self.latent_prompt_preset,
            system_prompt_override=self.latent_system_prompt,
            user_prompt_override=self.latent_user_prompt,
        )

    def decode_phase_options(self) -> PhaseOptions:
        return PhaseOptions(
            thinking_mode=self.decode_thinking_mode,
            strip_think_tags=self.decode_strip_think_tags,
            prompt_preset=self.decode_prompt_preset,
            system_prompt_override=self.decode_system_prompt,
            user_prompt_override=self.decode_user_prompt,
        )


class LatentSelfThink:
    """Single-model latent thinking with independent latent/decode controls."""

    def __init__(self, model: ModelWrapper, cfg: RunConfig):
        self.model = model
        self.cfg = cfg
        self._validate_static_config()

    def _validate_static_config(self) -> None:
        decode_prompt_changed = any(
            [
                self.cfg.decode_prompt_preset != "shared",
                bool(self.cfg.decode_system_prompt),
                bool(self.cfg.decode_user_prompt),
            ]
        )
        if self.cfg.decoding_new_message and decode_prompt_changed:
            raise ValueError(
                "--decoding_new_message is incompatible with decode prompt presets or overrides."
            )

    def _resolve_phase_configs(self, task: str, question: str):
        latent_phase = resolve_phase_config(
            phase_name="latent",
            options=self.cfg.latent_phase_options(),
            task=task,
            question=question,
            answer_only=self.cfg.answer_only,
            legacy_disable_thinking=self.cfg.legacy_disable_thinking,
        )
        decode_phase = resolve_phase_config(
            phase_name="decode",
            options=self.cfg.decode_phase_options(),
            task=task,
            question=question,
            answer_only=self.cfg.answer_only,
            legacy_disable_thinking=self.cfg.legacy_disable_thinking,
        )
        return latent_phase, decode_phase

    def _phase_result_payload(self, phase_cfg, messages):
        payload = phase_cfg.to_dict()
        payload["messages"] = messages
        return payload

    @torch.no_grad()
    def run_one(self, item: Dict) -> Dict:
        task = item.get("task", self.cfg.task)
        question = item["question"]
        gold = item.get("gold")

        latent_phase, decode_phase = self._resolve_phase_configs(task, question)
        latent_messages = build_messages(latent_phase)
        _, input_ids, attn = self.model.prepare_chat_input(
            latent_messages,
            add_generation_prompt=True,
            enable_thinking=latent_phase.enable_thinking,
            strip_think_tags=latent_phase.strip_think_tags,
        )

        past = None
        latent_steps_used = 0
        stop_p_true = False
        debug_steps = []

        if self.cfg.loop_decode or (self.cfg.latent_steps and self.cfg.latent_steps > 0):
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

        closed_latent_think_tag = False
        if (
            self.cfg.close_latent_think_tag_before_decode
            and past is not None
            and latent_phase.enable_thinking is not False
        ):
            past = self.model.append_text_to_past("</think>", past)
            closed_latent_think_tag = True

        decode_messages = build_messages(decode_phase)
        if self.cfg.decoding_new_message:
            decode_messages = [{"role": "user", "content": "\n"}]

        _, dec_ids, dec_attn = self.model.prepare_chat_input(
            decode_messages,
            add_generation_prompt=True,
            enable_thinking=decode_phase.enable_thinking,
            strip_think_tags=decode_phase.strip_think_tags,
        )

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
            "strategy_label": self.cfg.strategy_label,
            "closed_latent_think_tag_before_decode": closed_latent_think_tag,
            "phase_configs": {
                "latent": self._phase_result_payload(latent_phase, latent_messages),
                "decode": self._phase_result_payload(decode_phase, decode_messages),
            },
        }
        if gold_n is not None:
            result["gold_normalized"] = gold_n

        for key in ("subject", "level", "unique_id"):
            if key in item:
                result[key] = item[key]
        return result
