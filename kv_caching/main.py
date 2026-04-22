

import os
import argparse
import torch

from trainer   import train
from inference import generate_text


def main():
    parser = argparse.ArgumentParser(description="GPT from scratch — TinyShakespeare")
    parser.add_argument("--iters",    type=int,  default=2000)
    parser.add_argument("--batch",    type=int,  default=32)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--prompt",   type=str,  default="HAMLET:")
    parser.add_argument("--tokens",   type=int,  default=500)
    parser.add_argument("--eval",     action="store_true",
                        help="Run evaluate.py automatically after training")
    parser.add_argument("--save_dir", type=str,  default=".",
                        help="Root dir for checkpoints/ and results/")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Train ─────────────────────────────────────────────────────────────
    model, dataset = train(
        max_iters  = args.iters,
        batch_size = args.batch,
        lr         = args.lr,
        device     = device,
        save_dir   = args.save_dir,
    )

    # ── Quick generation preview ───────────────────────────────────────────
    print(f"\n── Generated preview (temp=0.8) ───────────────────────────")
    text = generate_text(model, dataset, prompt=args.prompt,
                         max_new_tokens=args.tokens, temperature=0.8,
                         top_k=40, use_kv_cache=True)
    print(text)

    # ── Optional immediate eval ────────────────────────────────────────────
    if args.eval:
        print("\n── Running evaluation ──────────────────────────────────────")
        from evaluate import evaluate_single
        evaluate_single(iters=args.iters, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
