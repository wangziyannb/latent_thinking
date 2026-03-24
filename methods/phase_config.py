from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional


THINKING_MODES = ("inherit", "think", "no_think")
LATENT_PROMPT_PRESETS = ("shared", "thinker")
DECODE_PROMPT_PRESETS = ("shared", "actor")

_DEFAULT_SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

_TASK_LABELS = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "math-500": "MATH-500",
    "math_500": "MATH-500",
}


@dataclass(frozen=True)
class PhaseOptions:
    thinking_mode: str = "inherit"
    strip_think_tags: bool = False
    prompt_preset: str = "shared"
    system_prompt_override: str = ""
    user_prompt_override: str = ""


@dataclass(frozen=True)
class ResolvedPhaseConfig:
    phase_name: str
    thinking_mode: str
    enable_thinking: Optional[bool]
    strip_think_tags: bool
    prompt_preset: str
    system_prompt: str
    user_prompt: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "phase_name": self.phase_name,
            "thinking_mode": self.thinking_mode,
            "enable_thinking": self.enable_thinking,
            "strip_think_tags": self.strip_think_tags,
            "prompt_preset": self.prompt_preset,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
        }


def _canonical_task(task: str) -> str:
    key = (task or "gsm8k").strip().lower()
    if key not in _TASK_LABELS:
        raise ValueError(f"Unsupported task: {task!r}")
    if key in {"math-500", "math_500"}:
        return "math500"
    return key


def _task_label(task: str) -> str:
    return _TASK_LABELS[_canonical_task(task)]


def _replace_placeholders(template: str, values: Mapping[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _default_shared_user_prompt(task: str, *, answer_only: bool) -> str:
    task = _canonical_task(task)
    if task == "math500":
        if answer_only:
            return (
                "Math Problem: {question}\n"
                "Return ONLY the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
                "Do NOT include your reasoning steps."
            )
        return (
            "Target Question: {question}\n"
            "You must reason step-by-step to solve the provided Target Question. "
            "Now, reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
        )

    if answer_only:
        return (
            "Target Question: {question}\n"
            "Return ONLY the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
            "Do NOT include your reasoning steps."
        )
    return (
        "Target Question: {question}\n"
        "You must reason step-by-step to solve the provided Target Question. "
        "Now, reason step by step and output the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
    )


def _thinker_prompts() -> Dict[str, str]:
    return {
        "system": (
            "You are the Thinker agent. Privately reason about the problem and prepare a correct "
            "solution path for a separate Actor agent. Focus on decomposition, constraints, key "
            "equations, and intermediate results. Do not optimize for user-facing prose."
        ),
        "user": (
            "Target Problem ({task_label}): {question}\n"
            "Think carefully. First form a plan, then solve the necessary subproblems. Keep the "
            "reasoning internal; the Actor will present the final answer."
        ),
    }


def _actor_user_prompt(*, answer_only: bool) -> str:
    if answer_only:
        return (
            "Target Problem ({task_label}): {question}\n"
            "Return ONLY the final answer inside \\boxed{YOUR_FINAL_ANSWER}. "
            "Do NOT include your reasoning steps."
        )
    return (
        "Target Problem ({task_label}): {question}\n"
        "Produce the final solution and end with \\boxed{YOUR_FINAL_ANSWER}. If answer_only is "
        "enabled, return only the boxed final answer."
    )


def _actor_prompts(*, answer_only: bool) -> Dict[str, str]:
    return {
        "system": (
            "You are the Actor agent. Use the existing hidden reasoning state from the Thinker to "
            "produce the final answer for the user. Be concise, decisive, and do not restart a "
            "long scratchpad unless necessary."
        ),
        "user": _actor_user_prompt(answer_only=answer_only),
    }


def prompt_choices_for_phase(phase_name: str) -> List[str]:
    if phase_name == "latent":
        return list(LATENT_PROMPT_PRESETS)
    if phase_name == "decode":
        return list(DECODE_PROMPT_PRESETS)
    raise ValueError(f"Unsupported phase name: {phase_name!r}")


def resolve_thinking_mode(
    thinking_mode: str,
    *,
    legacy_disable_thinking: bool = False,
) -> tuple[str, Optional[bool]]:
    if thinking_mode not in THINKING_MODES:
        raise ValueError(
            f"Unsupported thinking mode: {thinking_mode!r}. Expected one of {THINKING_MODES}."
        )

    if thinking_mode == "think":
        return "think", True
    if thinking_mode == "no_think":
        return "no_think", False
    if legacy_disable_thinking:
        return "no_think", False
    return "inherit", None


def resolve_phase_config(
    *,
    phase_name: str,
    options: PhaseOptions,
    task: str,
    question: str,
    answer_only: bool,
    legacy_disable_thinking: bool = False,
) -> ResolvedPhaseConfig:
    prompt_choices = prompt_choices_for_phase(phase_name)
    if options.prompt_preset not in prompt_choices:
        raise ValueError(
            f"Unsupported {phase_name} prompt preset: {options.prompt_preset!r}. "
            f"Expected one of {tuple(prompt_choices)}."
        )

    resolved_mode, enable_thinking = resolve_thinking_mode(
        options.thinking_mode,
        legacy_disable_thinking=legacy_disable_thinking,
    )
    if options.strip_think_tags and resolved_mode != "no_think":
        raise ValueError(
            f"{phase_name}_strip_think_tags requires the resolved thinking mode to be 'no_think'."
        )

    if options.prompt_preset == "shared":
        system_prompt = _DEFAULT_SYSTEM_PROMPT
        user_prompt = _default_shared_user_prompt(task, answer_only=answer_only)
    elif options.prompt_preset == "thinker":
        preset = _thinker_prompts()
        system_prompt = preset["system"]
        user_prompt = preset["user"]
    else:
        preset = _actor_prompts(answer_only=answer_only)
        system_prompt = preset["system"]
        user_prompt = preset["user"]

    if options.system_prompt_override:
        system_prompt = options.system_prompt_override
    if options.user_prompt_override:
        user_prompt = options.user_prompt_override

    context = {
        "question": question,
        "task": _canonical_task(task),
        "task_label": _task_label(task),
    }
    rendered_system = _replace_placeholders(system_prompt, context)
    rendered_user = _replace_placeholders(user_prompt, context)

    return ResolvedPhaseConfig(
        phase_name=phase_name,
        thinking_mode=resolved_mode,
        enable_thinking=enable_thinking,
        strip_think_tags=options.strip_think_tags,
        prompt_preset=options.prompt_preset,
        system_prompt=rendered_system,
        user_prompt=rendered_user,
    )


def build_messages(phase_cfg: ResolvedPhaseConfig) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": phase_cfg.system_prompt},
        {"role": "user", "content": phase_cfg.user_prompt},
    ]
