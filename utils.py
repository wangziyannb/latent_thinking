import os
import random
import re
from typing import List, Optional

import numpy as np
import torch

try:
    import sympy
    from sympy.parsing import sympy_parser
except ImportError:  # pragma: no cover - optional dependency
    sympy = None
    sympy_parser = None

try:
    from pylatexenc import latex2text
except ImportError:  # pragma: no cover - optional dependency
    latex2text = None


_BOX_COMMANDS = ("\\boxed", "\\fbox")
_LAST_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_TEXT_WRAPPER_RE = re.compile(r"^\\text\{(?P<text>.+)\}$", re.DOTALL)
_PLAIN_TEXT_ANSWER_RE = re.compile(r"^[A-Za-z][A-Za-z\s-]*$")


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


def _extract_braced_content(text: str, brace_start: int) -> Optional[str]:
    if brace_start >= len(text) or text[brace_start] != "{":
        return None

    depth = 0
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : idx]
    return None


def extract_boxed_text(text: str) -> Optional[str]:
    """Extract the last balanced \\boxed{...} or \\fbox{...} payload."""
    if not text:
        return None

    last_boxed = None
    for command in _BOX_COMMANDS:
        start = 0
        while True:
            idx = text.find(command, start)
            if idx == -1:
                break

            payload_start = idx + len(command)
            while payload_start < len(text) and text[payload_start].isspace():
                payload_start += 1

            if payload_start < len(text) and text[payload_start] == "{":
                content = _extract_braced_content(text, payload_start)
                if content is not None:
                    last_boxed = content.strip()

            start = idx + len(command)
    return last_boxed


def _strip_math_delimiters(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == "$" and text[-1] == "$":
        return text[1:-1].strip()
    return text


def _fix_sqrt(text: str) -> str:
    if "\\sqrt" not in text:
        return text

    pieces = text.split("\\sqrt")
    rebuilt = [pieces[0]]
    for piece in pieces[1:]:
        if piece and piece[0] != "{":
            rebuilt.append("\\sqrt{" + piece[0] + "}" + piece[1:])
        else:
            rebuilt.append("\\sqrt" + piece)
    return "".join(rebuilt)


def _fix_fracs(text: str) -> str:
    parts = text.split("\\frac")
    if len(parts) == 1:
        return text

    rebuilt = [parts[0]]
    for part in parts[1:]:
        rebuilt.append("\\frac")
        if not part:
            continue
        if part[0] == "{":
            rebuilt.append(part)
            continue

        if len(part) < 2:
            rebuilt.append(part)
            continue

        numerator = part[0]
        denominator = part[1]
        suffix = part[2:]
        if denominator == "{":
            rebuilt.append("{" + numerator + "}" + denominator + suffix)
        else:
            rebuilt.append("{" + numerator + "}{" + denominator + "}" + suffix)
    return "".join(rebuilt)


def _fix_simple_slash_fraction(text: str) -> str:
    if text.count("/") != 1:
        return text

    left, right = text.split("/")
    try:
        left_num = int(left)
        right_num = int(right)
    except ValueError:
        return text

    if text != f"{left_num}/{right_num}":
        return text
    return f"\\frac{{{left_num}}}{{{right_num}}}"


def _remove_right_units(text: str) -> str:
    if "\\text{ " not in text:
        return text

    prefix, _, _ = text.partition("\\text{ ")
    return prefix


def normalize_math_answer(answer: Optional[str]) -> Optional[str]:
    if answer is None:
        return None

    text = _strip_math_delimiters(answer.strip())
    match = _TEXT_WRAPPER_RE.match(text)
    if match is not None:
        text = match.group("text").strip()

    text = text.replace("\n", "")
    text = text.replace("\\!", "")
    text = text.replace("\\\\", "\\")
    text = text.replace("tfrac", "frac")
    text = text.replace("dfrac", "frac")
    text = text.replace("\\left", "")
    text = text.replace("\\right", "")
    text = text.replace("^{\\circ}", "")
    text = text.replace("^\\circ", "")
    text = text.replace("\\$", "")
    text = text.replace("\\%", "")
    text = text.replace("%", "")
    text = text.rstrip(".")
    text = _remove_right_units(text)
    text = text.replace(" .", " 0.")
    text = text.replace("{.", "{0.")

    if not text:
        return text

    if text[0] == ".":
        text = "0" + text

    if text.count("=") == 1:
        left, right = text.split("=")
        if len(left.strip()) <= 2:
            text = right

    text = _fix_sqrt(text)
    text = text.replace(" ", "")
    text = _fix_fracs(text)
    if text == "0.5":
        text = "\\frac{1}{2}"
    text = _fix_simple_slash_fraction(text)
    return text


def _split_top_level(text: str, sep: str) -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        elif ch == sep and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return [part.strip() for part in parts]


def _is_wrapped_pair(text: str, left: str, right: str) -> bool:
    return len(text) >= 2 and text[0] == left and text[-1] == right


def _sympy_parse(expr: str):
    if sympy_parser is None:
        raise ValueError("sympy parser unavailable")

    return sympy_parser.parse_expr(
        expr.replace("^", "**"),
        transformations=(
            sympy_parser.standard_transformations
            + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _latex_to_sympy_text(expr: str) -> str:
    expr = normalize_math_answer(expr) or ""
    if latex2text is not None:
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")
    return expr.strip()


def _math_tuple_match(pred: str, gold: str) -> bool:
    pred_parts = _split_top_level(pred[1:-1], ",")
    gold_parts = _split_top_level(gold[1:-1], ",")
    if len(pred_parts) != len(gold_parts):
        return False
    return all(math_answers_match(p, g) for p, g in zip(pred_parts, gold_parts))


def math_answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
    pred_norm = normalize_math_answer(pred)
    gold_norm = normalize_math_answer(gold)

    if not pred_norm or not gold_norm:
        return False

    if pred_norm == gold_norm:
        return True

    if (
        _PLAIN_TEXT_ANSWER_RE.fullmatch(pred_norm)
        and _PLAIN_TEXT_ANSWER_RE.fullmatch(gold_norm)
        and pred_norm.lower() == gold_norm.lower()
    ):
        return True

    if _is_wrapped_pair(pred_norm, "(", ")") and _is_wrapped_pair(gold_norm, "(", ")"):
        if _math_tuple_match(pred_norm, gold_norm):
            return True

    if sympy is None or sympy_parser is None:
        return False

    try:
        pred_expr = _sympy_parse(_latex_to_sympy_text(pred_norm))
        gold_expr = _sympy_parse(_latex_to_sympy_text(gold_norm))
        return bool(sympy.simplify(pred_expr - gold_expr) == 0)
    except Exception:
        return False


def extract_math_answer(text: str) -> Optional[str]:
    boxed = extract_boxed_text(text)
    if boxed:
        return boxed

    cleaned = _strip_math_delimiters(text.strip()).strip()
    if not cleaned:
        return None

    for line in reversed([line.strip() for line in cleaned.splitlines() if line.strip()]):
        lower = line.lower()
        if "final answer" in lower or "answer is" in lower:
            line = line.split(":")[-1].strip()
            return _strip_math_delimiters(line).rstrip(".").strip()
        if line.startswith("#"):
            continue
        return _strip_math_delimiters(line).rstrip(".").strip()
    return None

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract final numeric answer.

    - Prefer \boxed{...}
    - Otherwise take the last number in the text

    Copied from LatentMAS utils.py.
    """
    boxed = extract_boxed_text(text)
    if boxed:
        number = _LAST_NUMBER_RE.search(boxed)
        return number.group(0) if number else boxed.strip()

    numbers = _LAST_NUMBER_RE.findall(text)
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


def extract_answer_for_task(task: str, text: str) -> Optional[str]:
    key = (task or "gsm8k").strip().lower()
    if key == "gsm8k":
        return extract_gsm8k_answer(text)
    if key in {"math500", "math-500", "math_500"}:
        return extract_math_answer(text)
    raise ValueError(f"Unsupported task: {task!r}")


def normalize_prediction_for_task(task: str, answer: Optional[str]) -> Optional[str]:
    key = (task or "gsm8k").strip().lower()
    if key == "gsm8k":
        return normalize_answer(answer)
    if key in {"math500", "math-500", "math_500"}:
        return normalize_math_answer(answer)
    raise ValueError(f"Unsupported task: {task!r}")


def answers_match(task: str, pred: Optional[str], gold: Optional[str]) -> bool:
    key = (task or "gsm8k").strip().lower()
    if key == "gsm8k":
        return (
            normalize_answer(pred) == normalize_answer(str(gold))
            if pred and gold is not None
            else False
        )
    if key in {"math500", "math-500", "math_500"}:
        return math_answers_match(pred, str(gold) if gold is not None else None)
    raise ValueError(f"Unsupported task: {task!r}")
