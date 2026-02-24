import argparse
import json
from tqdm import tqdm

from data import load_gsm8k
from models import ModelWrapper
from methods.latent_self_think import LatentSelfThink, RunConfig
from utils import auto_device, set_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=2000)

    p.add_argument("--latent_steps", type=int, default=40)
    p.add_argument("--latent_space_realign", action="store_true")

    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_jsonl", type=str, default="")

    args = p.parse_args()

    set_seed(args.seed)
    device = auto_device(args.device)

    model = ModelWrapper(args.model_name, device, latent_space_realign=args.latent_space_realign)
    runner = LatentSelfThink(
        model,
        RunConfig(
            latent_steps=args.latent_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        ),
    )

    items = load_gsm8k(split=args.split, max_samples=args.max_samples)

    preds = []
    correct = 0
    for it in tqdm(items, desc="GSM8K"):
        out = runner.run_one(it)
        preds.append(out)
        correct += 1 if out.get("correct") else 0

    acc = correct / len(preds) if preds else 0.0
    print(f"\nAccuracy: {acc:.4f} ({correct}/{len(preds)})")

    if args.save_jsonl:
        with open(args.save_jsonl, "w", encoding="utf-8") as f:
            for r in preds:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved predictions to: {args.save_jsonl}")


if __name__ == "__main__":
    main()
