from __future__ import annotations

from typing import Dict, Iterable, List

from datasets import load_dataset

from utils import extract_gold_from_gsm8k_solution


_TASK_ALIASES = {
    "gsm8k": "gsm8k",
    "math500": "math500",
    "math-500": "math500",
    "math_500": "math500",
}

_TASK_LABELS = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
}


def canonical_task_name(task: str) -> str:
    key = (task or "gsm8k").strip().lower()
    if key not in _TASK_ALIASES:
        raise ValueError(
            f"Unsupported task: {task!r}. Expected one of: "
            f"{', '.join(sorted(_TASK_ALIASES))}"
        )
    return _TASK_ALIASES[key]


def task_label(task: str) -> str:
    return _TASK_LABELS[canonical_task_name(task)]


def _load_task_dataset(task: str, split: str):
    task = canonical_task_name(task)
    if task == "gsm8k":
        return load_dataset("gsm8k", "main", split=split)
    if task == "math500":
        if split != "test":
            raise ValueError(
                "MATH-500 currently exposes only the 'test' split in "
                "HuggingFaceH4/MATH-500."
            )
        return load_dataset("HuggingFaceH4/MATH-500", split=split)
    raise AssertionError(f"Unhandled task: {task}")


def _convert_example(task: str, ex: Dict) -> Dict:
    task = canonical_task_name(task)
    if task == "gsm8k":
        q = ex["question"]
        sol = ex["answer"]
        gold = extract_gold_from_gsm8k_solution(sol)
        return {
            "task": task,
            "question": q,
            "solution": sol,
            "gold": gold,
        }

    if task == "math500":
        return {
            "task": task,
            "question": ex["problem"],
            "solution": ex.get("solution"),
            "gold": ex.get("answer"),
            "subject": ex.get("subject"),
            "level": ex.get("level"),
            "unique_id": ex.get("unique_id"),
        }

    raise AssertionError(f"Unhandled task: {task}")


def _take_examples(task: str, examples: Iterable[Dict], max_samples: int) -> List[Dict]:
    items: List[Dict] = []
    for ex in examples:
        items.append(_convert_example(task, ex))
        if max_samples != -1 and len(items) >= max_samples:
            break
    return items


def load_task(task: str = "gsm8k", split: str = "test", max_samples: int = -1) -> List[Dict]:
    ds = _load_task_dataset(task, split)
    return _take_examples(task, ds, max_samples)


def load_task_sharded(
    task: str = "gsm8k",
    split: str = "test",
    max_samples: int = -1,
    *,
    rank: int,
    world_size: int,
) -> List[Dict]:
    ds = _load_task_dataset(task, split)
    ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)
    return _take_examples(task, ds, max_samples)


def load_gsm8k(split: str = "test", max_samples: int = -1) -> List[Dict]:
    """Backward-compatible GSM8K loader."""
    return load_task("gsm8k", split=split, max_samples=max_samples)


def load_math500(split: str = "test", max_samples: int = -1) -> List[Dict]:
    return load_task("math500", split=split, max_samples=max_samples)
