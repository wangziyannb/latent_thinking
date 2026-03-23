# Latent Self-Think (Qwen) — reuse LatentMAS latent thinking in a single-model loop

This repo implements **single-model latent thinking** for Qwen/Qwen3-style causal LMs by reusing the core latent-step mechanism from **LatentMAS** (hidden-state -> aligned input-embedding -> forward with `inputs_embeds`, accumulating `past_key_values`).

We evaluate on **GSM8K** and **MATH-500**.

## Install

```bash
pip install -r requirements.txt
```

## Run GSM8K

```bash
python scripts/run_gsm8k.py \
  --model_name Qwen/Qwen3-4B \
  --task gsm8k \
  --device cuda \
  --split test \
  --max_samples 200 \
  --latent_steps 40 \
  --latent_space_realign \
  --max_new_tokens 256 \
  --temperature 0.6 \
  --top_p 0.95
```

## Run MATH-500

```bash
python scripts/run_gsm8k.py \
  --model_name Qwen/Qwen3-4B \
  --task math500 \
  --device cuda \
  --split test \
  --max_samples 200 \
  --latent_steps 40 \
  --latent_space_realign \
  --max_new_tokens 512 \
  --temperature 0.6 \
  --top_p 0.95
```

### Notes
- `--task` currently supports `gsm8k` and `math500` (`math-500` / `math_500` aliases also work).
- `MATH-500` currently exposes only the `test` split on Hugging Face.
- `latent_steps` controls how many latent-only forward steps are done **before** decoding.
- `--latent_space_realign` enables the linear realignment matrix (as in LatentMAS) to reduce distribution drift.

## Output
The scripts print per-example prediction metadata and a final accuracy summary.

For multi-GPU runs, `scripts/run_gsm8k_mp.py` now accepts the same `--task` flag and writes task-specific log filenames such as `math500_<timestamp>.json`.
