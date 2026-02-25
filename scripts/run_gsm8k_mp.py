import argparse
import json
import os
import time
import datetime
from dataclasses import asdict
from typing import List

import torch
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset

from models import ModelWrapper
from methods.latent_self_think import LatentSelfThink, RunConfig
from utils import set_seed, extract_gold_from_gsm8k_solution


def ddp_setup():
    """
    torchrun 会注入这些环境变量：
      RANK, LOCAL_RANK, WORLD_SIZE
    """
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


def load_gsm8k_sharded(split: str, max_samples: int, rank: int, world_size: int):
    """
    用 datasets 的 shard 做确定性分片：
    - 每个 rank 只拿自己那份
    - max_samples 在分片后再截断，避免不同 rank 重叠
    """
    ds = load_dataset("gsm8k", "main", split=split)
    ds = ds.shard(num_shards=world_size, index=rank, contiguous=True)

    items = []
    for ex in ds:
        q = ex["question"]
        sol = ex["answer"]
        gold = extract_gold_from_gsm8k_solution(sol)
        items.append({"question": q, "solution": sol, "gold": gold})
        if max_samples != -1 and len(items) >= max_samples:
            break
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=-1)  # 注意：这是“每个 rank”的上限

    p.add_argument("--latent_steps", type=int, default=40)
    p.add_argument("--latent_space_realign", action="store_true")

    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_jsonl", type=str, default="")  # 最终合并后的 jsonl 路径（rank0 写）
    p.add_argument("--log_dir", type=str, default="logs")

    p.add_argument("--latent_early_stop", action="store_true")
    p.add_argument("--latent_early_stop_threshold", type=float, default=0.8)
    p.add_argument("--latent_early_stop_probe_text", type=str, default="Judge whether it is true or false: Now I know how to solve this question. My answer is:")

    args = p.parse_args()

    is_ddp = ddp_setup()
    rank, world_size = get_rank_world()

    # 不同 rank 用不同 seed，避免采样完全一致（可选）
    set_seed(args.seed + rank)

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}") if torch.cuda.is_available() else torch.device("cpu")

    model = ModelWrapper(args.model_name, device, latent_space_realign=args.latent_space_realign)
    runner = LatentSelfThink(
        model,
        RunConfig(
            latent_steps=args.latent_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            latent_early_stop=args.latent_early_stop,
            latent_early_stop_threshold= args.latent_early_stop_threshold,
            latent_early_stop_probe_text = args.latent_early_stop_probe_text,

    ),
    )

    items = load_gsm8k_sharded(args.split, args.max_samples, rank, world_size)

    # 只让 rank0 显示进度条更清爽
    iterator = tqdm(items, desc=f"GSM8K rank {rank}/{world_size}", disable=(rank != 0))

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
    t1 = time.perf_counter()
    elapsed_local = t1 - t0

    # --- 统计聚合：all_reduce 求和 ---
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

    # --- 保存：每个 rank 先落一个临时文件，rank0 合并 ---
    tmp_path = ""
    if args.save_jsonl:
        tmp_path = f"{args.save_jsonl}.rank{rank}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            for r in preds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 每个 rank 的统计（用于写入 log）
    rank_stats = {
        "rank": rank,
        "num_samples": len(preds),
        "correct": correct_local,
        "decoded_tokens": decoded_local,
        "elapsed_s": elapsed_local,
        "tmp_pred_path": tmp_path,  # 方便排查
    }

    # 收集所有 rank 的统计到 rank0
    if is_ddp:
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, rank_stats)
        all_rank_stats = gathered
    else:
        all_rank_stats = [rank_stats]

    # 等所有 rank 都把 tmp 写完
    if is_ddp:
        dist.barrier()

    merged_jsonl_path = ""
    if rank == 0:
        # 生成 run_id（和 log 共用）
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 1) rank0 合并预测文件
        if args.save_jsonl:
            # 创建 log_dir
            os.makedirs(args.log_dir, exist_ok=True)

            # 合并后的预测文件统一放到 log_dir
            merged_jsonl_path = os.path.join(
                args.log_dir,
                f"gsm8k_{run_id}_predictions.jsonl"
            )

            with open(merged_jsonl_path, "w", encoding="utf-8") as out_f:
                for r in range(world_size):
                    part = f"{args.save_jsonl}.rank{r}.tmp"
                    if os.path.exists(part):
                        with open(part, "r", encoding="utf-8") as in_f:
                            out_f.write(in_f.read())

            # 清理 tmp 文件
            for r in range(world_size):
                part = f"{args.save_jsonl}.rank{r}.tmp"
                if os.path.exists(part):
                    os.remove(part)

            print(f"Saved predictions to: {merged_jsonl_path}")

        # 2) rank0 写 log（包含 merged_jsonl_path + 每卡统计）
        os.makedirs(args.log_dir, exist_ok=True)
        log_path = os.path.join(args.log_dir, f"gsm8k_{run_id}.json")

        log_data = {
            "timestamp": run_id,
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

            "latent_early_stop": args.latent_early_stop,
            "latent_early_stop_threshold": args.latent_early_stop_threshold,
            "latent_early_stop_probe_text": args.latent_early_stop_probe_text,

            # 预测文件信息（合并后的）
            "predictions": {
                "merged_jsonl_path": merged_jsonl_path
            },

            # 全局指标（all_reduce 后）
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

            # 每张卡的统计（gather 后）
            "ranks": all_rank_stats,
        }

        # 终端仍然可以打印一份（可删）
        print(f"\nAccuracy: {acc:.4f} ({correct_total}/{count_total})")
        print(f"Total decoded tokens: {decoded_total}")
        print(f"Avg decoded tokens/sample: {decoded_total / count_total:.2f}")
        print(f"Total inference time (wall-clock): {elapsed_total:.3f}s")
        print(f"Samples/sec: {count_total / elapsed_total:.3f}")
        print(f"Decoded tokens/sec: {decoded_total / elapsed_total:.3f}")

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"Run log saved to: {log_path}")

    ddp_cleanup()


if __name__ == "__main__":
    main()