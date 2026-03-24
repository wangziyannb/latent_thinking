import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data import canonical_task_name, load_task, task_label
from methods.latent_self_think import LatentSelfThink, RunConfig
from models import ModelWrapper
from utils import auto_device, set_seed


def _effective_batch_size(args) -> int:
    requested = max(int(args.batch_size or 1), 1)
    if args.latent_early_stop:
        return 1
    return requested


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--task", type=str, default="gsm8k")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--attn_implementation", type=str, default="auto", choices=("auto", "eager", "sdpa", "flash_attention_2"))
    p.add_argument("--disable_cudnn_sdp", action="store_true")

    p.add_argument("--latent_steps", type=int, default=40)
    p.add_argument("--latent_space_realign", action="store_true")

    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--disable_thinking", action="store_true", help="Legacy alias that maps both phases to no_think")
    p.add_argument("--latent_thinking_mode", type=str, default="inherit", choices=("inherit", "think", "no_think"))
    p.add_argument("--decode_thinking_mode", type=str, default="inherit", choices=("inherit", "think", "no_think"))
    p.add_argument("--latent_strip_think_tags", action="store_true")
    p.add_argument("--decode_strip_think_tags", action="store_true")
    p.add_argument("--close_latent_think_tag_before_decode", action="store_true")
    p.add_argument("--latent_prompt_preset", type=str, default="shared", choices=("shared", "thinker"))
    p.add_argument("--decode_prompt_preset", type=str, default="shared", choices=("shared", "actor"))
    p.add_argument("--latent_system_prompt", type=str, default="")
    p.add_argument("--latent_user_prompt", type=str, default="")
    p.add_argument("--decode_system_prompt", type=str, default="")
    p.add_argument("--decode_user_prompt", type=str, default="")
    p.add_argument("--strategy_label", type=str, default="")

    p.add_argument("--answer_only", action="store_true", help="Ask model to output only final answer")
    p.add_argument("--loop_decode", action="store_true", help="Always do prefill->(latent)->decode loop (fairer ablation)")

    p.add_argument("--latent_early_stop", action="store_true")
    p.add_argument("--latent_early_stop_threshold", type=float, default=0.8)
    p.add_argument("--latent_early_stop_probe_text", type=str, default="Judge whether it is true or false: Now I know how to solve this question. My answer is:")
    p.add_argument("--latent_debug_decode", action="store_true")
    p.add_argument("--decoding_new_message", action="store_true")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_jsonl", type=str, default="")
    p.add_argument("--log_dir", type=str, default="")
    return p


def build_run_config(args, task: str) -> RunConfig:
    return RunConfig(
        task=task,
        latent_steps=args.latent_steps,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        latent_early_stop=args.latent_early_stop,
        latent_early_stop_threshold=args.latent_early_stop_threshold,
        latent_early_stop_probe_text=args.latent_early_stop_probe_text,
        latent_debug_decode=args.latent_debug_decode,
        answer_only=args.answer_only,
        loop_decode=args.loop_decode,
        decoding_new_message=args.decoding_new_message,
        legacy_disable_thinking=args.disable_thinking,
        strategy_label=args.strategy_label,
        latent_thinking_mode=args.latent_thinking_mode,
        decode_thinking_mode=args.decode_thinking_mode,
        latent_strip_think_tags=args.latent_strip_think_tags,
        decode_strip_think_tags=args.decode_strip_think_tags,
        close_latent_think_tag_before_decode=args.close_latent_think_tag_before_decode,
        latent_prompt_preset=args.latent_prompt_preset,
        decode_prompt_preset=args.decode_prompt_preset,
        latent_system_prompt=args.latent_system_prompt,
        latent_user_prompt=args.latent_user_prompt,
        decode_system_prompt=args.decode_system_prompt,
        decode_user_prompt=args.decode_user_prompt,
    )


def main():
    args = build_parser().parse_args()
    task = canonical_task_name(args.task)

    set_seed(args.seed)
    device = auto_device(args.device)

    model = ModelWrapper(
        args.model_name,
        device,
        latent_space_realign=args.latent_space_realign,
        attn_implementation=args.attn_implementation,
        disable_cudnn_sdp=args.disable_cudnn_sdp,
    )
    runner = LatentSelfThink(model, build_run_config(args, task))

    items = load_task(task=task, split=args.split, max_samples=args.max_samples)
    effective_batch_size = _effective_batch_size(args)
    if args.latent_early_stop and args.batch_size > 1:
        print("latent_early_stop is enabled; forcing batch_size=1 for correctness.")

    preds = []
    correct = 0
    decoded_total = 0
    t0 = time.perf_counter()
    progress = tqdm(total=len(items), desc=task_label(task))
    for start in range(0, len(items), effective_batch_size):
        batch_items = items[start:start + effective_batch_size]
        outs = runner.run_batch(batch_items)
        preds.extend(outs)
        for out in outs:
            correct += 1 if out.get("correct") else 0
            decoded_total += int(out.get("decoded_tokens", 0))
        progress.update(len(batch_items))
    progress.close()
    elapsed = time.perf_counter() - t0

    acc = correct / len(preds) if preds else 0.0
    print(f"\nAccuracy: {acc:.4f} ({correct}/{len(preds)})")
    print(f"Total decoded tokens: {decoded_total}")
    print(f"Avg decoded tokens/sample: {decoded_total / len(preds):.2f}" if preds else "Avg decoded tokens/sample: 0.00")
    print(f"Total inference time (wall-clock): {elapsed:.3f}s")
    if elapsed > 0:
        print(f"Samples/sec: {len(preds) / elapsed:.3f}")
        print(f"Decoded tokens/sec: {decoded_total / elapsed:.3f}")

    if args.save_jsonl:
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for r in preds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved predictions to: {args.save_jsonl}")

    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(args.log_dir, f"{task}_{run_id}.json")
        log_data = {
            "timestamp": run_id,
            "task": task,
            "model_name": args.model_name,
            "split": args.split,
            "world_size": 1,
            "max_samples_per_rank": args.max_samples,
            "batch_size_requested": args.batch_size,
            "batch_size_effective": effective_batch_size,
            "attn_implementation": args.attn_implementation,
            "disable_cudnn_sdp": args.disable_cudnn_sdp,
            "latent_steps": args.latent_steps,
            "latent_space_realign": args.latent_space_realign,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "strategy_label": args.strategy_label,
            "disable_thinking": args.disable_thinking,
            "latent_thinking_mode": args.latent_thinking_mode,
            "decode_thinking_mode": args.decode_thinking_mode,
            "latent_strip_think_tags": args.latent_strip_think_tags,
            "decode_strip_think_tags": args.decode_strip_think_tags,
            "close_latent_think_tag_before_decode": args.close_latent_think_tag_before_decode,
            "latent_prompt_preset": args.latent_prompt_preset,
            "decode_prompt_preset": args.decode_prompt_preset,
            "latent_system_prompt": args.latent_system_prompt,
            "latent_user_prompt": args.latent_user_prompt,
            "decode_system_prompt": args.decode_system_prompt,
            "decode_user_prompt": args.decode_user_prompt,
            "latent_early_stop": args.latent_early_stop,
            "latent_early_stop_threshold": args.latent_early_stop_threshold,
            "latent_early_stop_probe_text": args.latent_early_stop_probe_text,
            "answer_only": args.answer_only,
            "loop_decode": args.loop_decode,
            "decoding_new_message": args.decoding_new_message,
            "predictions": {
                "merged_jsonl_path": args.save_jsonl,
            },
            "metrics": {
                "accuracy": acc,
                "correct_total": correct,
                "count_total": len(preds),
                "decoded_tokens_total": decoded_total,
                "avg_decoded_tokens_per_sample": decoded_total / len(preds) if preds else 0.0,
                "wall_clock_seconds": elapsed,
                "samples_per_sec": len(preds) / elapsed if elapsed > 0 else 0.0,
                "tokens_per_sec": decoded_total / elapsed if elapsed > 0 else 0.0,
            },
            "resolved_phase_configs_preview": preds[0].get("phase_configs") if preds else None,
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"Run log saved to: {log_path}")


if __name__ == "__main__":
    main()
