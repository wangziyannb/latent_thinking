import argparse
import datetime
import json
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from data import canonical_task_name, load_task_sharded, task_label
from methods.latent_self_think import LatentSelfThink, RunConfig
from models import ModelWrapper
from utils import reserve_vram, set_seed


def ddp_setup():
    if "RANK" not in os.environ:
        return False

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return True


def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--task", type=str, default="gsm8k")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=-1)

    p.add_argument("--reserve_vram_ratio", type=float, default=0, help="Reserve this fraction of currently free VRAM after model loads (0~1)")
    p.add_argument("--reserve_vram_mb", type=int, default=0, help="Reserve fixed VRAM in MB after model loads (overrides ratio if >0)")

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

    p.add_argument("--latent_steps", type=int, default=40)
    p.add_argument("--latent_space_realign", action="store_true")

    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--answer_only", action="store_true", help="Ask model to output only final answer")
    p.add_argument("--loop_decode", action="store_true", help="Always do prefill->(latent)->decode loop (fairer ablation)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_jsonl", type=str, default="")
    p.add_argument("--log_dir", type=str, default="logs")

    p.add_argument("--latent_early_stop", action="store_true")
    p.add_argument("--latent_early_stop_threshold", type=float, default=0.8)
    p.add_argument("--latent_early_stop_probe_text", type=str, default="Judge whether it is true or false: Now I know how to solve this question. My answer is:")

    p.add_argument("--latent_debug_decode", action="store_true")
    p.add_argument("--decoding_new_message", action="store_true")
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

    is_ddp = ddp_setup()
    rank, world_size = get_rank_world()

    set_seed(args.seed + rank)

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}") if torch.cuda.is_available() else torch.device("cpu")

    model = ModelWrapper(
        args.model_name,
        device,
        latent_space_realign=args.latent_space_realign,
    )
    _vram = reserve_vram(device, reserve_ratio=args.reserve_vram_ratio, reserve_mb=args.reserve_vram_mb)
    runner = LatentSelfThink(model, build_run_config(args, task))

    items = load_task_sharded(
        task=task,
        split=args.split,
        max_samples=args.max_samples,
        rank=rank,
        world_size=world_size,
    )

    iterator = tqdm(items, desc=f"{task_label(task)} rank {rank}/{world_size}", disable=(rank != 0))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    preds = []
    correct_local = 0
    decoded_local = 0
    for it in iterator:
        out = runner.run_one(it)
        preds.append(out)
        correct_local += 1 if out.get("correct") else 0
        decoded_local += int(out.get("decoded_tokens", 0))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_local = time.perf_counter() - t0

    correct_t = torch.tensor([correct_local], device=device, dtype=torch.long)
    count_t = torch.tensor([len(preds)], device=device, dtype=torch.long)
    decoded_t = torch.tensor([decoded_local], device=device, dtype=torch.long)
    elapsed_t = torch.tensor([elapsed_local], device=device, dtype=torch.float64)
    if is_ddp:
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(decoded_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed_t, op=dist.ReduceOp.MAX)

    decoded_total = int(decoded_t.item())
    correct_total = int(correct_t.item())
    count_total = int(count_t.item())
    acc = correct_total / count_total if count_total else 0.0
    elapsed_total = float(elapsed_t.item())

    tmp_path = ""
    if args.save_jsonl:
        tmp_path = f"{args.save_jsonl}.rank{rank}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in preds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    rank_stats = {
        "rank": rank,
        "num_samples": len(preds),
        "correct": correct_local,
        "decoded_tokens": decoded_local,
        "elapsed_s": elapsed_local,
        "tmp_pred_path": tmp_path,
    }

    if is_ddp:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, rank_stats)
        all_rank_stats = gathered
    else:
        all_rank_stats = [rank_stats]

    if is_ddp:
        dist.barrier()

    merged_jsonl_path = ""
    if rank == 0:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.save_jsonl:
            os.makedirs(args.log_dir, exist_ok=True)
            merged_jsonl_path = os.path.join(args.log_dir, f"{task}_{run_id}_predictions.jsonl")

            with open(merged_jsonl_path, "w", encoding="utf-8") as out_f:
                for r in range(world_size):
                    part = f"{args.save_jsonl}.rank{r}.tmp"
                    if os.path.exists(part):
                        with open(part, "r", encoding="utf-8") as in_f:
                            out_f.write(in_f.read())

            for r in range(world_size):
                part = f"{args.save_jsonl}.rank{r}.tmp"
                if os.path.exists(part):
                    os.remove(part)

            print(f"Saved predictions to: {merged_jsonl_path}")

        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"{task}_{run_id}.json")

        log_data = {
            "timestamp": run_id,
            "task": task,
            "model_name": args.model_name,
            "split": args.split,
            "world_size": world_size,
            "max_samples_per_rank": args.max_samples,
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
                "merged_jsonl_path": merged_jsonl_path,
            },
            "metrics": {
                "accuracy": acc,
                "correct_total": correct_total,
                "count_total": count_total,
                "decoded_tokens_total": decoded_total,
                "avg_decoded_tokens_per_sample": decoded_total / count_total if count_total else 0.0,
                "wall_clock_seconds": elapsed_total,
                "samples_per_sec": count_total / elapsed_total if elapsed_total > 0 else 0.0,
                "tokens_per_sec": decoded_total / elapsed_total if elapsed_total > 0 else 0.0,
            },
            "ranks": all_rank_stats,
            "resolved_phase_configs_preview": preds[0].get("phase_configs") if preds else None,
        }

        print(f"\nAccuracy: {acc:.4f} ({correct_total}/{count_total})")
        print(f"Total decoded tokens: {decoded_total}")
        print(f"Avg decoded tokens/sample: {decoded_total / count_total:.2f}" if count_total else "Avg decoded tokens/sample: 0.00")
        print(f"Total inference time (wall-clock): {elapsed_total:.3f}s")
        print(f"Samples/sec: {count_total / elapsed_total:.3f}" if elapsed_total > 0 else "Samples/sec: 0.000")
        print(f"Decoded tokens/sec: {decoded_total / elapsed_total:.3f}" if elapsed_total > 0 else "Decoded tokens/sec: 0.000")

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"Run log saved to: {log_path}")

    ddp_cleanup()


if __name__ == "__main__":
    main()
