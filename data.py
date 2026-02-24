from typing import Dict, List, Optional

from datasets import load_dataset

from utils import extract_gold_from_gsm8k_solution


def load_gsm8k(split: str = "test", max_samples: int = -1) -> List[Dict]:
    """Load GSM8K (main) and return a list of dicts with question/solution/gold."""
    ds = load_dataset("gsm8k", "main", split=split)
    items: List[Dict] = []
    for ex in ds:
        q = ex["question"]
        sol = ex["answer"]
        gold = extract_gold_from_gsm8k_solution(sol)
        items.append({"question": q, "solution": sol, "gold": gold})
        if max_samples != -1 and len(items) >= max_samples:
            break
    return items
