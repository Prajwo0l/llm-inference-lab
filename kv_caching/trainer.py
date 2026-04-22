"""
trainer.py
──────────
Training loop for the GPT model.

Features
────────
  * AdamW optimiser with weight decay
  * Cosine annealing LR scheduler
  * Gradient clipping
  * Loss curve logging to CSV  (results/loss_log_<iters>.csv)
  * Auto-saves checkpoint      (checkpoints/ckpt_<iters>.pt)

Usage
─────
    from trainer import train
    model, dataset = train(max_iters=2000)
    model, dataset = train(max_iters=10000)
"""

import os
import csv
import time
import torch

from config  import GPTConfig
from dataset import CharDataset
from model   import GPT


_DEFAULT_TEXT = (
    "To be or not to be, that is the question. "
    "Whether 'tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, "
    "or to take arms against a sea of troubles. " * 200
)


def train(
    text:       str  | None = None,
    max_iters:  int         = 2000,
    batch_size: int         = 32,
    lr:         float       = 3e-4,
    device:     str  | None = None,
    log_every:  int         = 100,
    save_dir:   str         = ".",
) -> tuple:
    """Train a character-level GPT and save checkpoint + loss log.

    Args:
        text       : Raw text. None → reads input.txt or uses default.
        max_iters  : Gradient update steps.
        batch_size : Batch size.
        lr         : Peak AdamW learning rate.
        device     : 'cpu', 'cuda', or None (auto).
        log_every  : Log loss every N steps.
        save_dir   : Root dir for checkpoints/ and results/ folders.

    Returns:
        (model, dataset)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on  : {device}")
    print(f"Iterations   : {max_iters}")
    print(f"Batch size   : {batch_size}")

    # ── Corpus ────────────────────────────────────────────────────────────
    if text is None:
        txt_path = os.path.join(save_dir, "input.txt")
        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8") as f:
                text = f.read()
            print(f"Corpus       : {txt_path}  ({len(text):,} chars)")
        else:
            text = _DEFAULT_TEXT
            print(f"Corpus       : built-in default  ({len(text):,} chars)")

    _cfg_block = GPTConfig().block_size
    dataset = CharDataset(text, block_size=_cfg_block)
    print(f"Vocab size   : {dataset.vocab_size}")

    # ── Model ─────────────────────────────────────────────────────────────
    config            = GPTConfig()
    config.vocab_size = dataset.vocab_size
    config.block_size = dataset.block_size
    model             = GPT(config).to(device)

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iters)

    # ── Output dirs ───────────────────────────────────────────────────────
    ckpt_dir    = os.path.join(save_dir, "checkpoints")
    results_dir = os.path.join(save_dir, "results")
    os.makedirs(ckpt_dir,    exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, f"loss_log_{max_iters}iters.csv")
    log_rows = []

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    model.train()
    t_start = time.time()
    best_loss = float("inf")

    for step in range(max_iters):
        ix = torch.randint(len(dataset), (batch_size,))
        x  = torch.stack([dataset[i][0] for i in ix]).to(device)
        y  = torch.stack([dataset[i][1] for i in ix]).to(device)

        _, loss, _ = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val

        if step % log_every == 0 or step == max_iters - 1:
            elapsed  = time.time() - t_start
            eta_secs = (elapsed / (step + 1)) * (max_iters - step - 1)
            log_rows.append({
                "step":    step,
                "loss":    round(loss_val, 6),
                "lr":      round(scheduler.get_last_lr()[0], 8),
                "elapsed": round(elapsed, 1),
            })
            print(
                f"  step {step:5d}/{max_iters}"
                f" | loss {loss_val:.4f}"
                f" | lr {scheduler.get_last_lr()[0]:.2e}"
                f" | elapsed {elapsed/60:.1f}m"
                f" | ETA {eta_secs/60:.1f}m"
            )

    total_time = time.time() - t_start
    print(f"{'─'*52}")
    print(f"Training done  : {total_time/60:.1f} min")
    print(f"Best loss seen : {best_loss:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_{max_iters}iters.pt")
    torch.save({
        "model_state":   model.state_dict(),
        "config":        config.__dict__,
        "vocab_size":    dataset.vocab_size,
        "stoi":          dataset.stoi,
        "itos":          dataset.itos,
        "max_iters":     max_iters,
        "best_loss":     best_loss,
        "total_time_s":  total_time,
    }, ckpt_path)
    print(f"Checkpoint     : {ckpt_path}")

    # ── Save loss log CSV ─────────────────────────────────────────────────
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "lr", "elapsed"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Loss log       : {log_path}")

    return model, dataset
