import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunSlot:
    slot_id: int
    gpu_group: str
    master_port: int


def load_matrix_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cli_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _build_cli_args(args: Dict) -> List[str]:
    cli_args: List[str] = []
    for key in sorted(args):
        value = args[key]
        if value is None or value is False:
            continue
        flag = _cli_flag(key)
        if value is True:
            cli_args.append(flag)
            continue
        cli_args.extend([flag, str(value)])
    return cli_args


def _resolve_script_path(script: str) -> Path:
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = REPO_ROOT / script_path
    return script_path.resolve()


def _launcher_prefix(
    kind: str,
    script_path: Path,
    nproc_per_node: Optional[int],
    *,
    master_port: Optional[int] = None,
) -> List[str]:
    if kind == "python":
        return [sys.executable, str(script_path)]
    if kind == "torchrun":
        if not nproc_per_node:
            raise ValueError("torchrun launcher requires nproc_per_node.")
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node",
            str(nproc_per_node),
        ]
        if master_port is not None:
            cmd.extend(["--master_port", str(master_port)])
        cmd.append(str(script_path))
        return cmd
    raise ValueError(f"Unsupported launcher kind: {kind!r}")


def _unique_run_dir(output_dir: Path, run_name: str) -> Path:
    base = output_dir / run_name
    if not base.exists():
        return base
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{run_name}_{suffix}"


def _default_paths(run_args: Dict, run_dir: Path) -> Dict:
    resolved = dict(run_args)
    if not resolved.get("log_dir"):
        resolved["log_dir"] = str(run_dir)
    if not resolved.get("save_jsonl"):
        resolved["save_jsonl"] = str(run_dir / "predictions.jsonl")
    return resolved


def parse_gpu_groups(raw: str) -> List[str]:
    groups: List[str] = []
    for group in (raw or "").split(";"):
        cleaned = group.strip()
        if cleaned:
            groups.append(cleaned)
    return groups


def select_runs(matrix_config: Dict, run_names: Optional[Sequence[str]] = None) -> List[Dict]:
    runs = list(matrix_config.get("runs", []))
    if not run_names:
        return runs

    wanted = {name.strip() for name in run_names if name.strip()}
    selected = [run for run in runs if run.get("name") in wanted]
    missing = sorted(wanted - {run.get("name") for run in selected})
    if missing:
        raise ValueError(f"Unknown run names requested: {missing}")
    return selected


def resolve_slots(
    matrix_config: Dict,
    *,
    gpu_groups_override: Optional[List[str]] = None,
    jobs_per_group_override: Optional[int] = None,
    base_master_port_override: Optional[int] = None,
) -> List[RunSlot]:
    scheduler_cfg = matrix_config.get("scheduler", {})
    gpu_groups = gpu_groups_override
    if gpu_groups is None:
        raw_groups = scheduler_cfg.get("gpu_groups", [])
        if isinstance(raw_groups, str):
            gpu_groups = parse_gpu_groups(raw_groups)
        else:
            gpu_groups = [str(group).strip() for group in raw_groups if str(group).strip()]

    jobs_per_group = jobs_per_group_override or int(scheduler_cfg.get("jobs_per_group", 1) or 1)
    base_master_port = base_master_port_override or int(scheduler_cfg.get("base_master_port", 29500) or 29500)

    if jobs_per_group <= 0:
        raise ValueError("jobs_per_group must be >= 1.")

    normalized_groups = gpu_groups or [""]
    slots: List[RunSlot] = []
    for group in normalized_groups:
        for _ in range(jobs_per_group):
            slots.append(
                RunSlot(
                    slot_id=len(slots),
                    gpu_group=group,
                    master_port=base_master_port + len(slots),
                )
            )
    return slots


def build_run_plan(
    matrix_config: Dict,
    run_entry: Dict,
    *,
    output_dir: Path,
    launcher_override: Optional[str] = None,
    nproc_per_node: Optional[int] = None,
    master_port: Optional[int] = None,
) -> Tuple[List[str], Dict, Path]:
    run_name = run_entry["name"]
    run_dir = _unique_run_dir(output_dir, run_name)

    merged_args = dict(matrix_config.get("base_args", {}))
    merged_args.update(run_entry.get("args", {}))
    merged_args = _default_paths(merged_args, run_dir)

    strategy_label = run_entry.get("strategy_label") or merged_args.get("strategy_label") or run_name
    merged_args["strategy_label"] = strategy_label

    launcher_cfg = matrix_config.get("launcher", {})
    launcher_kind = launcher_override or launcher_cfg.get("kind", "python")
    resolved_nproc = nproc_per_node or launcher_cfg.get("nproc_per_node")

    script_path = _resolve_script_path(matrix_config["script"])
    cmd = _launcher_prefix(
        launcher_kind,
        script_path,
        resolved_nproc,
        master_port=master_port,
    ) + _build_cli_args(merged_args)
    return cmd, merged_args, run_dir


def _discover_run_outputs(log_dir: Path) -> Tuple[str, str]:
    json_logs = sorted(
        [path for path in log_dir.glob("*.json") if path.name != "manifest.json"],
        key=lambda path: path.stat().st_mtime,
    )
    log_path = str(json_logs[-1]) if json_logs else ""

    predictions_path = ""
    if log_path:
        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        predictions_path = log_data.get("predictions", {}).get("merged_jsonl_path", "") or ""
    return log_path, predictions_path


def latent_step_display(log_data: Dict) -> str:
    latent_steps = int(log_data.get("latent_steps", 0) or 0)
    if latent_steps <= 0:
        return "/"
    if log_data.get("latent_early_stop"):
        return f"Early stop <={latent_steps}"
    return str(latent_steps)


def build_summary_row_from_log(log_data: Dict, *, run_status: str, log_path: str, predictions_path: str) -> Dict[str, object]:
    metrics = log_data.get("metrics", {})
    return {
        "model": log_data.get("model_name", ""),
        "strategy": log_data.get("strategy_label", ""),
        "task": log_data.get("task", ""),
        "latent_step": latent_step_display(log_data),
        "acc": metrics.get("accuracy"),
        "total_decoded_tokens": metrics.get("decoded_tokens_total"),
        "avg_decoded_per_sample": metrics.get("avg_decoded_tokens_per_sample"),
        "run_status": run_status,
        "log_path": log_path,
        "predictions_path": predictions_path,
    }


def build_summary_row_fallback(run_name: str, args: Dict, *, run_status: str, log_path: str = "", predictions_path: str = "") -> Dict[str, object]:
    latent_steps = int(args.get("latent_steps", 0) or 0)
    return {
        "model": args.get("model_name", ""),
        "strategy": args.get("strategy_label", run_name),
        "task": args.get("task", ""),
        "latent_step": "/" if latent_steps <= 0 else str(latent_steps),
        "acc": None,
        "total_decoded_tokens": None,
        "avg_decoded_per_sample": None,
        "run_status": run_status,
        "log_path": log_path,
        "predictions_path": predictions_path,
    }


def write_manifest(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_summary(output_dir: Path, rows: List[Dict[str, object]]) -> None:
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "model",
        "strategy",
        "task",
        "latent_step",
        "acc",
        "total_decoded_tokens",
        "avg_decoded_per_sample",
        "run_status",
        "log_path",
        "predictions_path",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _proc_env_for_slot(slot: RunSlot) -> Dict[str, str]:
    env = os.environ.copy()
    if slot.gpu_group:
        env["CUDA_VISIBLE_DEVICES"] = slot.gpu_group
    env["MASTER_PORT"] = str(slot.master_port)
    return env


def _finalize_job(job: Dict) -> Dict[str, object]:
    proc = job["proc"]
    return_code = proc.wait()

    log_dir = Path(job["resolved_args"]["log_dir"])
    log_path, predictions_path = _discover_run_outputs(log_dir)
    job["manifest"]["finished_at"] = datetime.now().isoformat()
    job["manifest"]["elapsed_seconds"] = time.time() - job["started_epoch"]
    job["manifest"]["exit_code"] = return_code
    job["manifest"]["run_status"] = "success" if return_code == 0 else "failed"
    job["manifest"]["log_path"] = log_path
    job["manifest"]["predictions_path"] = predictions_path
    write_manifest(job["run_dir"] / "manifest.json", job["manifest"])

    if log_path:
        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        return build_summary_row_from_log(
            log_data,
            run_status=job["manifest"]["run_status"],
            log_path=log_path,
            predictions_path=predictions_path,
        )

    return build_summary_row_fallback(
        job["run_entry"]["name"],
        job["resolved_args"],
        run_status=job["manifest"]["run_status"],
        log_path=log_path,
        predictions_path=predictions_path,
    )


def run_matrix(
    matrix_config: Dict,
    *,
    output_dir: Path,
    dry_run: bool = False,
    stop_on_error: bool = False,
    launcher_override: Optional[str] = None,
    nproc_per_node: Optional[int] = None,
    run_names: Optional[Sequence[str]] = None,
    gpu_groups_override: Optional[List[str]] = None,
    jobs_per_group_override: Optional[int] = None,
    base_master_port_override: Optional[int] = None,
) -> List[Dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    selected_runs = select_runs(matrix_config, run_names=run_names)
    slots = resolve_slots(
        matrix_config,
        gpu_groups_override=gpu_groups_override,
        jobs_per_group_override=jobs_per_group_override,
        base_master_port_override=base_master_port_override,
    )

    if dry_run:
        for idx, run_entry in enumerate(selected_runs):
            slot = slots[idx % len(slots)]
            cmd, resolved_args, run_dir = build_run_plan(
                matrix_config,
                run_entry,
                output_dir=output_dir,
                launcher_override=launcher_override,
                nproc_per_node=nproc_per_node,
                master_port=slot.master_port,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "name": run_entry["name"],
                "command": cmd,
                "args": resolved_args,
                "run_dir": str(run_dir),
                "slot_id": slot.slot_id,
                "gpu_group": slot.gpu_group,
                "master_port": slot.master_port,
                "started_at": datetime.now().isoformat(),
                "finished_at": datetime.now().isoformat(),
                "exit_code": None,
                "run_status": "dry_run",
            }
            write_manifest(run_dir / "manifest.json", manifest)
            rows.append(
                build_summary_row_fallback(
                    run_entry["name"],
                    resolved_args,
                    run_status="dry_run",
                )
            )
        write_summary(output_dir, rows)
        return rows

    free_slots = list(slots)
    pending_runs = list(selected_runs)
    active_jobs: List[Dict] = []
    stop_launching = False

    while pending_runs or active_jobs:
        while pending_runs and free_slots and not stop_launching:
            slot = free_slots.pop(0)
            run_entry = pending_runs.pop(0)
            cmd, resolved_args, run_dir = build_run_plan(
                matrix_config,
                run_entry,
                output_dir=output_dir,
                launcher_override=launcher_override,
                nproc_per_node=nproc_per_node,
                master_port=slot.master_port,
            )
            run_dir.mkdir(parents=True, exist_ok=True)

            manifest = {
                "name": run_entry["name"],
                "command": cmd,
                "args": resolved_args,
                "run_dir": str(run_dir),
                "slot_id": slot.slot_id,
                "gpu_group": slot.gpu_group,
                "master_port": slot.master_port,
                "started_at": datetime.now().isoformat(),
                "exit_code": None,
            }
            write_manifest(run_dir / "manifest.json", manifest)

            print(f"\n=== Running {run_entry['name']} on slot {slot.slot_id} (gpus={slot.gpu_group or 'all'}) ===")
            print(" ".join(cmd))
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=_proc_env_for_slot(slot),
            )
            active_jobs.append(
                {
                    "proc": proc,
                    "slot": slot,
                    "run_entry": run_entry,
                    "resolved_args": resolved_args,
                    "run_dir": run_dir,
                    "manifest": manifest,
                    "started_epoch": time.time(),
                }
            )

        if not active_jobs:
            break

        time.sleep(1.0)

        still_active: List[Dict] = []
        for job in active_jobs:
            if job["proc"].poll() is None:
                still_active.append(job)
                continue

            row = _finalize_job(job)
            rows.append(row)
            write_summary(output_dir, rows)

            free_slots.append(job["slot"])
            free_slots.sort(key=lambda slot: slot.slot_id)
            if job["manifest"]["exit_code"] != 0 and stop_on_error:
                stop_launching = True

        active_jobs = still_active

    write_summary(output_dir, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config_json", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--stop_on_error", action="store_true")
    p.add_argument("--launcher_override", type=str, default="", choices=("", "python", "torchrun"))
    p.add_argument("--nproc_per_node", type=int, default=0)
    p.add_argument("--run_names", type=str, default="", help="Comma-separated subset of run names to execute")
    p.add_argument("--gpu_groups", type=str, default="", help="Semicolon-separated GPU groups, e.g. '0,1,2,3;4,5,6,7'")
    p.add_argument("--jobs_per_group", type=int, default=0, help="How many concurrent jobs to stack on each GPU group")
    p.add_argument("--base_master_port", type=int, default=0, help="Base port used to assign unique torchrun master ports")
    return p


def main():
    args = build_parser().parse_args()
    config = load_matrix_config(args.config_json)
    run_matrix(
        config,
        output_dir=Path(args.output_dir),
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
        launcher_override=args.launcher_override or None,
        nproc_per_node=(args.nproc_per_node or None),
        run_names=[name for name in args.run_names.split(",") if name.strip()] or None,
        gpu_groups_override=parse_gpu_groups(args.gpu_groups) or None,
        jobs_per_group_override=(args.jobs_per_group or None),
        base_master_port_override=(args.base_master_port or None),
    )


if __name__ == "__main__":
    main()
