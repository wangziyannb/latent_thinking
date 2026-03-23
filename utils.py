import os
import random
import re
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def reserve_vram(device: torch.device, reserve_ratio: float = 0.0, reserve_mb: int = 0, safety_mb: int = 256):
    if reserve_ratio <= 0.0 and reserve_mb <= 0:
        return None
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except TypeError:
        # older torch: mem_get_info() doesn't accept device
        free_bytes, total_bytes = torch.cuda.mem_get_info()

    safety_bytes = int(safety_mb * 1024 * 1024)
    if free_bytes <= safety_bytes:
        return None

    if reserve_mb > 0:
        target_bytes = int(reserve_mb * 1024 * 1024)
    else:
        reserve_ratio = max(0.0, min(1.0, float(reserve_ratio)))
        target_bytes = int(free_bytes * reserve_ratio)

    # Clamp so we leave some free space
    target_bytes = min(target_bytes, max(0, free_bytes - safety_bytes))
    if target_bytes <= 0:
        return None

    # Allocate as uint8 to match bytes
    try:
        buf = torch.empty((target_bytes,), dtype=torch.uint8, device=device)
    except RuntimeError:
        # Fall back to a smaller allocation if we were too aggressive
        target_bytes = max(0, int((free_bytes - safety_bytes) * 0.5))
        if target_bytes <= 0:
            return None
        buf = torch.empty((target_bytes,), dtype=torch.uint8, device=device)

    mb = buf.numel() / (1024 * 1024)
    # try:
    #     free2, total2 = torch.cuda.mem_get_info(device)
    #     print(f"[reserve_vram] reserved ~{mb:.1f} MB on {device} (free now ~{free2/(1024*1024):.1f} MB / total {total2/(1024*1024):.1f} MB)")
    # except Exception:
    #     print(f"[reserve_vram] reserved ~{mb:.1f} MB on {device}")
    return buf

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract final numeric answer.

    - Prefer \boxed{...}
    - Otherwise take the last number in the text

    Copied from LatentMAS utils.py.
    """
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold_from_gsm8k_solution(solution: str) -> Optional[str]:
    """GSM8K gold is marked as '#### <number>' in the dataset."""
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", solution)
    return match.group(1) if match else None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()
