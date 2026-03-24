import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_ablation_matrix import (
    _discover_run_outputs,
    build_run_plan,
    build_summary_row_from_log,
    parse_gpu_groups,
    resolve_slots,
    select_runs,
    write_manifest,
)


class AblationRunnerTest(unittest.TestCase):
    def test_parse_gpu_groups(self):
        self.assertEqual(
            parse_gpu_groups("0,1,2,3;4,5,6,7"),
            ["0,1,2,3", "4,5,6,7"],
        )

    def test_resolve_slots_with_stacking(self):
        slots = resolve_slots(
            {"scheduler": {"gpu_groups": ["0,1,2,3", "4,5,6,7"], "jobs_per_group": 2, "base_master_port": 30000}}
        )
        self.assertEqual(len(slots), 4)
        self.assertEqual(slots[0].gpu_group, "0,1,2,3")
        self.assertEqual(slots[1].gpu_group, "0,1,2,3")
        self.assertEqual(slots[2].gpu_group, "4,5,6,7")
        self.assertEqual(slots[0].master_port, 30000)
        self.assertEqual(slots[3].master_port, 30003)

    def test_select_runs_subset(self):
        config = {
            "runs": [
                {"name": "a"},
                {"name": "b"},
                {"name": "c"},
            ]
        }
        selected = select_runs(config, run_names=["b", "c"])
        self.assertEqual([run["name"] for run in selected], ["b", "c"])

    def test_build_run_plan_merges_args_and_defaults_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            config = {
                "script": "scripts/run_gsm8k_mp.py",
                "launcher": {"kind": "torchrun", "nproc_per_node": 4},
                "base_args": {"model_name": "Qwen/Qwen3-4B", "task": "math500"},
            }
            run_entry = {
                "name": "think_think_40",
                "strategy_label": "Think + Think",
                "args": {"latent_steps": 40, "latent_thinking_mode": "think", "decode_thinking_mode": "think"},
            }
            cmd, args, run_dir = build_run_plan(config, run_entry, output_dir=output_dir, master_port=30123)

            self.assertIn("scripts/run_gsm8k_mp.py", " ".join(cmd))
            self.assertIn("30123", " ".join(cmd))
            self.assertEqual(args["strategy_label"], "Think + Think")
            self.assertEqual(args["task"], "math500")
            self.assertEqual(args["latent_steps"], 40)
            self.assertEqual(Path(args["log_dir"]), run_dir)
            self.assertEqual(Path(args["save_jsonl"]), run_dir / "predictions.jsonl")

    def test_manifest_and_summary_from_sample_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            log_path = run_dir / "math500_20260324_000000.json"
            log_payload = {
                "model_name": "Qwen/Qwen3-4B",
                "strategy_label": "Think + Think",
                "task": "math500",
                "latent_steps": 40,
                "latent_early_stop": True,
                "predictions": {"merged_jsonl_path": str(run_dir / "predictions.jsonl")},
                "metrics": {
                    "accuracy": 0.74,
                    "decoded_tokens_total": 1087172,
                    "avg_decoded_tokens_per_sample": 2174.344,
                },
            }
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_payload, f)

            manifest_path = run_dir / "manifest.json"
            write_manifest(manifest_path, {"command": ["python", "scripts/run_gsm8k_mp.py"]})

            discovered_log, predictions_path = _discover_run_outputs(run_dir)
            row = build_summary_row_from_log(
                log_payload,
                run_status="success",
                log_path=discovered_log,
                predictions_path=predictions_path,
            )

            self.assertEqual(discovered_log, str(log_path))
            self.assertEqual(predictions_path, str(run_dir / "predictions.jsonl"))
            self.assertEqual(row["strategy"], "Think + Think")
            self.assertEqual(row["latent_step"], "Early stop <=40")
            self.assertEqual(row["acc"], 0.74)


if __name__ == "__main__":
    unittest.main()
