# Latent Self-Think (Qwen) — reuse LatentMAS latent thinking in a single-model loop

This repo implements **single-model latent thinking** for Qwen/Qwen3-style causal LMs by reusing the core latent-step mechanism from **LatentMAS** (hidden-state -> aligned input-embedding -> forward with `inputs_embeds`, accumulating `past_key_values`).

We evaluate on **GSM8K**.

## Install

```bash
pip install -r requirements.txt
```

## Run GSM8K

```bash
python scripts/run_gsm8k.py \
  --model_name Qwen/Qwen3-4B \
  --device cuda \
  --split test \
  --max_samples 200 \
  --latent_steps 40 \
  --latent_space_realign \
  --max_new_tokens 256 \
  --temperature 0.6 \
  --top_p 0.95
```

### Notes
- `latent_steps` controls how many latent-only forward steps are done **before** decoding.
- `--latent_space_realign` enables the linear realignment matrix (as in LatentMAS) to reduce distribution drift.

## Output
The script prints per-example prediction and a final accuracy summary.
