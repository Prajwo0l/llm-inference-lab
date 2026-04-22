"""
evaluate.py
───────────
Research evaluation script — GPT from scratch on TinyShakespeare.

Computes:
  1. Perplexity + BPC on train and val split
  2. KV cache vs naive benchmark (token-length sweep)
  3. Qualitative generation samples (temperature sweep)
  4. Loss curve plot
  5. Side-by-side comparison when both 2k and 10k checkpoints exist

Usage
─────
    python evaluate.py --iters 2000
    python evaluate.py --iters 5000
    python evaluate.py --iters 10000
    python evaluate.py --compare
"""

import os
import sys
import csv
import math
import time
import argparse
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from config  import GPTConfig
from dataset import CharDataset
from model   import GPT


# ── Plot theme ────────────────────────────────────────────────────────────────
BG    = "#0d1117"
PANEL = "#161b22"
GRID  = "#21262d"
TXT   = "#e6edf3"
MUTED = "#8b949e"
TEAL  = "#39d353"
RED   = "#ff7b72"
GOLD  = "#f0c040"
BLUE  = "#58a6ff"
PURP  = "#bc8cff"

THEME = {
    "figure.facecolor": BG,   "axes.facecolor":  PANEL,
    "axes.edgecolor":   GRID, "axes.labelcolor": MUTED,
    "axes.titlecolor":  TXT,  "xtick.color":     MUTED,
    "ytick.color":      MUTED,"grid.color":      GRID,
    "text.color":       TXT,  "font.family":     "monospace",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(ckpt_path, device="cpu"):
    ckpt   = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig()
    config.vocab_size = ckpt["vocab_size"]
    config.block_size = ckpt.get("config", {}).get("block_size", 128)
    model  = GPT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    class _DS:
        pass
    ds            = _DS()
    ds.vocab_size = ckpt["vocab_size"]
    ds.stoi       = ckpt["stoi"]
    ds.itos       = ckpt["itos"]
    ds.block_size = config.block_size

    def encode(s):
        return torch.tensor([ds.stoi.get(c, 0) for c in s], dtype=torch.long)
    def decode(t):
        return "".join(ds.itos.get(i.item() if hasattr(i, "item") else i, "?") for i in t)

    ds.encode = encode
    ds.decode = decode
    return model, ds, ckpt


def compute_perplexity(model, text, ds, device="cpu", block_size=128):
    model.eval()
    data         = torch.tensor([ds.stoi.get(c, 0) for c in text], dtype=torch.long)
    total_loss   = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(data) - block_size, block_size):
            x = data[i      : i + block_size].unsqueeze(0).to(device)
            y = data[i + 1  : i + block_size + 1].unsqueeze(0).to(device)
            _, loss, _ = model(x, targets=y)
            total_loss   += loss.item() * block_size
            total_tokens += block_size
    avg_loss   = total_loss / total_tokens
    return math.exp(avg_loss), avg_loss / math.log(2), avg_loss


def generate_samples(model, ds, device, prompts=None, temperatures=None):
    if prompts is None:
        prompts = ["HAMLET:", "To be, or not to be", "The king said", "O, what a"]
    if temperatures is None:
        temperatures = [0.6, 0.8, 1.0]
    samples = []
    for prompt in prompts:
        for temp in temperatures:
            ctx = ds.encode(prompt).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model.generate(ctx, max_new_tokens=300,
                                     temperature=temp, top_k=40,
                                     use_kv_cache=True)
            samples.append({
                "prompt":      prompt,
                "temperature": temp,
                "generated":   ds.decode(out[0].cpu()),
            })
    return samples


def load_loss_log(log_path):
    steps, losses = [], []
    if not os.path.exists(log_path):
        return steps, losses
    with open(log_path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return steps, losses


# ─────────────────────────────────────────────────────────────────────────────
# KV Cache benchmark
# ─────────────────────────────────────────────────────────────────────────────

def kv_cache_benchmark(model, ds, device, save_dir,
                       iters, runs=3,
                       sweep_lengths=None,
                       prompt="HAMLET:"):
    """
    Benchmark KV cache vs naive generation.

    Per-step timing shows O(T²) vs O(T) growth clearly.
    Sweep measures total time at multiple sequence lengths.

    Saves:
      results/kv_benchmark_<iters>iters.txt
      results/kv_benchmark_<iters>iters.png
    """
    if sweep_lengths is None:
        sweep_lengths = [50, 100, 150, 200, 250, 300, 400, 500]

    BLOCK   = 512   # must match model block_size
    MAX_NEW = 500   # tokens for per-step timing chart

    ctx = ds.encode(prompt).unsqueeze(0).to(device)
    assert ctx.shape[1] < BLOCK, \
        f"Prompt too long ({ctx.shape[1]} tokens) — must be < {BLOCK}"

    print(f"\n{'='*60}")
    print(f"  KV Cache Benchmark  —  {iters} iter checkpoint")
    print(f"  Prompt : \"{prompt}\"  ({ctx.shape[1]} tokens)")
    print(f"  Runs per sweep length : {runs}")
    print(f"{'='*60}")

    # ── Helper: timed full-sequence generation ────────────────────────────
    def timed_generate(use_cache, n_tokens):
        """Run n_tokens generation, return avg wall-clock seconds over `runs`."""
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model.generate(ctx.clone(), n_tokens,
                           temperature=1.0, top_k=None,
                           use_kv_cache=use_cache)
            times.append(time.perf_counter() - t0)
        return sum(times) / runs

    # ── Per-step timing (single run, step by step) ────────────────────────
    # Naive: crop ids to BLOCK each step (ids grows past block_size otherwise)
    # KV:    prefill once, then pass only the single new token each step
    print(f"\n  Collecting per-step timing ({MAX_NEW} tokens, single run)...")

    naive_ms = []
    kv_ms    = []

    model.eval()
    with torch.no_grad():

        # ── Naive ──────────────────────────────────────────────────────────
        ids = ctx.clone()
        for _ in range(MAX_NEW):
            # KEY FIX: always crop to the last BLOCK tokens before forwarding
            # Without this, ids grows beyond block_size and triggers the assert
            ids_in = ids[:, -BLOCK:]
            t0     = time.perf_counter()
            out, _, _ = model(ids_in, kv_caches=None)
            naive_ms.append((time.perf_counter() - t0) * 1000)
            next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_tok], dim=1)

        # ── KV cache ───────────────────────────────────────────────────────
        ids = ctx.clone()

        # Step 0: prefill — process the full prompt, build KV cache
        t0 = time.perf_counter()
        out, _, kv_caches = model(ids, kv_caches=None)
        kv_ms.append((time.perf_counter() - t0) * 1000)
        next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)

        # Steps 1..MAX_NEW-1: decode — one token at a time, no cropping needed
        for _ in range(MAX_NEW - 1):
            t0 = time.perf_counter()
            out, _, kv_caches = model(next_tok, kv_caches=kv_caches)
            kv_ms.append((time.perf_counter() - t0) * 1000)
            next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)

    # Both lists are exactly MAX_NEW long
    assert len(naive_ms) == MAX_NEW
    assert len(kv_ms)    == MAX_NEW

    # ── Sweep ─────────────────────────────────────────────────────────────
    print(f"\n  Running sweep ({runs} runs each)...")
    sweep_results = []
    sep = "  " + "─" * 65
    hdr = (f"  {'Tokens':>7}  {'Naive (s)':>10}  {'KV (s)':>8}"
           f"  {'Speedup':>9}  {'tok/s naive':>12}  {'tok/s kv':>10}")
    print(sep)
    print(hdr)
    print(sep)

    for n_tok in sweep_lengths:
        t_naive = timed_generate(use_cache=False, n_tokens=n_tok)
        t_kv    = timed_generate(use_cache=True,  n_tokens=n_tok)
        sp      = t_naive / t_kv
        tps_n   = n_tok / t_naive
        tps_k   = n_tok / t_kv
        sweep_results.append(dict(
            tokens=n_tok, naive_s=t_naive, kv_s=t_kv,
            speedup=sp, tps_naive=tps_n, tps_kv=tps_k,
        ))
        print(f"  {n_tok:>7}  {t_naive:>10.3f}  {t_kv:>8.3f}"
              f"  {sp:>8.2f}x  {tps_n:>12.1f}  {tps_k:>10.1f}")

    print(sep)
    best = sweep_results[-1]
    print(f"\n  At {best['tokens']} tokens:")
    print(f"    Naive   : {best['naive_s']:.3f} s  →  {best['tps_naive']:.1f} tok/s")
    print(f"    KV cache: {best['kv_s']:.3f} s  →  {best['tps_kv']:.1f} tok/s")
    print(f"    Speedup : {best['speedup']:.2f}x")
    print(f"    Reason  : naive recomputes all T tokens every step  O(T²)")
    print(f"              KV cache reads stored K,V, processes 1 token  O(T)")

    # ── Save numeric table ────────────────────────────────────────────────
    out_dir  = os.path.join(save_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"kv_benchmark_{iters}iters.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"KV Cache Benchmark — {iters} iterations\n")
        f.write(f"Model  : 1.82M params  block_size=128  n_layer=4  n_embd=192\n")
        f.write(f"Device : CPU (Intel i5-8300H)\n\n")
        f.write(f"{'Tokens':>7}  {'Naive (s)':>10}  {'KV (s)':>8}  "
                f"{'Speedup':>9}  {'tok/s naive':>12}  {'tok/s kv':>10}\n")
        f.write("─" * 65 + "\n")
        for r in sweep_results:
            f.write(f"{r['tokens']:>7}  {r['naive_s']:>10.3f}  {r['kv_s']:>8.3f}  "
                    f"{r['speedup']:>8.2f}x  {r['tps_naive']:>12.1f}  {r['tps_kv']:>10.1f}\n")
    print(f"\n  Table : {txt_path}")

    # ── Chart ─────────────────────────────────────────────────────────────
    plt.rcParams.update(THEME)
    step_axis = list(range(1, MAX_NEW + 1))
    cum_naive = np.cumsum(naive_ms)
    cum_kv    = np.cumsum(kv_ms)
    kern      = np.ones(10) / 10
    sm_naive  = np.convolve(naive_ms, kern, mode="same")
    sm_kv     = np.convolve(kv_ms,    kern, mode="same")
    sw_tok    = [r["tokens"]  for r in sweep_results]
    sw_sp     = [r["speedup"] for r in sweep_results]
    saved_pct = (1 - cum_kv[-1] / cum_naive[-1]) * 100

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"KV Cache vs Naive  ·  GPT from scratch ({iters} iters)  ·  CPU (i5-8300H)",
        fontsize=14, fontweight="bold", color=TXT, y=1.01,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.32)

    # 1 — per-step latency (full width)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(step_axis, naive_ms, color=RED,  lw=0.5, alpha=0.2)
    ax1.plot(step_axis, kv_ms,    color=TEAL, lw=0.5, alpha=0.2)
    ax1.plot(step_axis, sm_naive, color=RED,  lw=2.2,
             label="Naive — grows every step  O(T²)")
    ax1.plot(step_axis, sm_kv,    color=TEAL, lw=2.2,
             label="KV cache — stays flat     O(T)")
    ax1.fill_between(step_axis, sm_kv, sm_naive, alpha=0.08, color=BLUE)
    ax1.set_title("Latency per token step (ms)  —  naive grows, KV stays flat",
                  fontsize=11, pad=8)
    ax1.set_xlabel("Token step")
    ax1.set_ylabel("ms / token")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.28)

    # 2 — cumulative time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(step_axis, cum_naive, color=RED,  lw=2.2, label="Naive")
    ax2.plot(step_axis, cum_kv,    color=TEAL, lw=2.2, label="KV cache")
    ax2.fill_between(step_axis, cum_kv, cum_naive,
                     alpha=0.12, color=BLUE, label="Time saved")
    ax2.set_title("Cumulative time (ms)", fontsize=11, pad=8)
    ax2.set_xlabel("Tokens generated")
    ax2.set_ylabel("ms")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.28)
    ax2.text(0.97, 0.18, f"{saved_pct:.0f}%\ntime saved",
             transform=ax2.transAxes, ha="right", va="bottom",
             color=BLUE, fontsize=11, fontweight="bold")

    # 3 — speedup vs sequence length
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(sw_tok, sw_sp, color=GOLD, lw=2.2, marker="o",
             ms=7, label="Measured speedup")
    ax3.fill_between(sw_tok, 1, sw_sp, alpha=0.14, color=GOLD)
    ax3.axhline(1.0, color=MUTED, ls="--", lw=1.0)
    ax3.set_title("Speedup vs sequence length  —  grows with T",
                  fontsize=11, pad=8)
    ax3.set_xlabel("Tokens generated")
    ax3.set_ylabel("Speedup (x)")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.28)
    for x, y in zip(sw_tok, sw_sp):
        ax3.text(x, y + 0.03, f"{y:.2f}x", ha="center",
                 va="bottom", fontsize=8, color=GOLD)

    # 4 — throughput bar at max sweep length
    ax4 = fig.add_subplot(gs[2, 0])
    bars = ax4.bar(
        ["Naive", "KV cache"],
        [best["tps_naive"], best["tps_kv"]],
        color=[RED, TEAL], width=0.38, edgecolor=GRID, lw=1,
    )
    for bar, val in zip(bars, [best["tps_naive"], best["tps_kv"]]):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.1,
                 f"{val:.1f}\ntok/s",
                 ha="center", va="bottom",
                 fontweight="bold", fontsize=11, color=TXT)
    ax4.set_title(f"Throughput at {best['tokens']} tokens (tok/s)",
                  fontsize=11, pad=8)
    ax4.set_ylabel("tok/s")
    ax4.grid(axis="y", alpha=0.28)

    # 5 — stats panel
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    stats = (
        f"  Model        GPT from scratch\n"
        f"  Params       1.82 M\n"
        f"  Checkpoint   {iters} iterations\n"
        f"  Block size   {BLOCK} tokens\n"
        f"  n_layer      4\n"
        f"  n_embd       192\n"
        f"  Device       CPU  (i5-8300H)\n"
        f"  ────────────────────────────────\n"
        f"  Naive {best['tokens']}t   {best['naive_s']:.3f} s\n"
        f"  KV    {best['tokens']}t   {best['kv_s']:.3f} s\n"
        f"  Speedup      {best['speedup']:.2f}x\n"
        f"  Time saved   {saved_pct:.0f}%\n"
        f"  ────────────────────────────────\n"
        f"  Naive O(T²)  recomputes past tokens\n"
        f"  KV    O(T)   reads stored K,V tensors\n"
    )
    ax5.text(0.04, 0.95, stats,
             transform=ax5.transAxes, fontsize=9.5,
             verticalalignment="top", fontfamily="monospace",
             color=TXT,
             bbox=dict(boxstyle="round,pad=0.6",
                       facecolor=PANEL, edgecolor=GRID, linewidth=1.4))

    png_path = os.path.join(out_dir, f"kv_benchmark_{iters}iters.png")
    plt.savefig(png_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  Chart: {png_path}")
    print(f"{'='*60}")

    return sweep_results


# ─────────────────────────────────────────────────────────────────────────────
# Single-run evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_single(iters, save_dir="."):
    device    = "cpu"
    ckpt_path = os.path.join(save_dir, "checkpoints", f"ckpt_{iters}iters.pt")
    log_path  = os.path.join(save_dir, "results",     f"loss_log_{iters}iters.csv")
    out_dir   = os.path.join(save_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Evaluating: ckpt_{iters}iters.pt")
    print(f"{'='*60}")

    model, ds, ckpt = load_checkpoint(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params        : {n_params:.2f} M")
    print(f"  Vocab size    : {ds.vocab_size}")
    print(f"  Training time : {ckpt.get('total_time_s', 0)/60:.1f} min")
    print(f"  Best loss     : {ckpt.get('best_loss', 'N/A')}")

    # ── Corpus ────────────────────────────────────────────────────────────
    txt_path = os.path.join(save_dir, "input.txt")
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            full_text = f.read()
        val_text = full_text[int(0.9 * len(full_text)):]
        trn_text = full_text[:int(0.9 * len(full_text))]
        print(f"\n  Corpus        : {len(full_text):,} chars")
        print(f"  Val split     : {len(val_text):,} chars (last 10%)")
    else:
        val_text = trn_text = "To be or not to be. " * 100

    # ── Language model metrics ────────────────────────────────────────────
    print(f"\n  Computing perplexity...")
    trn_ppl, trn_bpc, trn_loss = compute_perplexity(model, trn_text, ds, device)
    val_ppl, val_bpc, val_loss = compute_perplexity(model, val_text, ds, device)
    overfit_gap = val_loss - trn_loss

    print(f"\n{'─'*60}")
    print(f"  {'Metric':<28} {'Train':>10}  {'Val':>10}")
    print(f"  {'-'*56}")
    print(f"  {'Cross-entropy loss':<28} {trn_loss:>10.4f}  {val_loss:>10.4f}")
    print(f"  {'Perplexity (PPL)':<28} {trn_ppl:>10.2f}  {val_ppl:>10.2f}")
    print(f"  {'Bits per char (BPC)':<28} {trn_bpc:>10.4f}  {val_bpc:>10.4f}")
    print(f"{'─'*60}")

    if   val_ppl < 5:   grade = "Excellent — strongly learned Shakespeare patterns"
    elif val_ppl < 15:  grade = "Good — coherent text with recognisable style"
    elif val_ppl < 40:  grade = "Developing — patterns forming, needs more training"
    elif val_ppl < 100: grade = "Early — loss still high"
    else:               grade = "Undertrained"
    print(f"\n  PPL {val_ppl:.2f}  →  {grade}")

    if   overfit_gap > 0.3: print(f"  Gap {overfit_gap:+.3f}  →  Overfitting present")
    elif overfit_gap > 0.1: print(f"  Gap {overfit_gap:+.3f}  →  Mild generalisation gap")
    else:                   print(f"  Gap {overfit_gap:+.3f}  →  Good fit (no overfitting)")

    # ── KV cache benchmark ────────────────────────────────────────────────
    sweep_results = kv_cache_benchmark(
        model, ds, device, save_dir, iters,
        runs=3,
        sweep_lengths=[50, 100, 150, 200, 250, 300, 400, 500],
        prompt="HAMLET:",
    )

    # ── Generation samples ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Generation samples")
    print(f"{'─'*60}")
    samples = generate_samples(model, ds, device,
                               prompts=["HAMLET:", "To be, or not to be"],
                               temperatures=[0.6, 0.8, 1.0])
    for s in samples:
        print(f"\n  Prompt: \"{s['prompt']}\"  temp={s['temperature']}")
        print(f"  {'─'*50}")
        print(f"  {s['generated'][:250].strip()}")

    sample_path = os.path.join(out_dir, f"samples_{iters}iters.txt")
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(f"GPT From Scratch — {iters} iterations\n")
        f.write(f"Val PPL: {val_ppl:.2f}  BPC: {val_bpc:.4f}\n")
        f.write("="*60 + "\n\n")
        for s in samples:
            f.write(f"Prompt: \"{s['prompt']}\"  temp={s['temperature']}\n")
            f.write(s["generated"] + "\n")
            f.write("-"*60 + "\n\n")
    print(f"\n  Samples : {sample_path}")

    # ── Loss curve ────────────────────────────────────────────────────────
    steps, losses = load_loss_log(log_path)
    if steps:
        plt.rcParams.update(THEME)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, losses, color=TEAL, lw=1.8, label="Training loss")
        ax.axhline(val_loss, color=RED,  lw=1.5, ls="--",
                   label=f"Val  {val_loss:.4f}")
        ax.axhline(trn_loss, color=GOLD, lw=1.0, ls=":",
                   label=f"Trn  {trn_loss:.4f}")
        ax.set_title(f"Loss curve — {iters} iters  |  Val PPL {val_ppl:.2f}",
                     fontsize=12, pad=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-entropy loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plot_path = os.path.join(out_dir, f"loss_curve_{iters}iters.png")
        plt.savefig(plot_path, dpi=130, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  Loss curve: {plot_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    best_kv = sweep_results[-1]
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {iters} iterations")
    print(f"{'='*60}")
    print(f"  Val Perplexity    : {val_ppl:.2f}")
    print(f"  Val BPC           : {val_bpc:.4f}")
    print(f"  Val loss          : {val_loss:.4f}")
    print(f"  Train loss        : {trn_loss:.4f}")
    print(f"  Overfit gap       : {overfit_gap:+.4f}")
    print(f"  KV speedup @500t  : {best_kv['speedup']:.2f}x")
    print(f"  Naive tok/s       : {best_kv['tps_naive']:.1f}")
    print(f"  KV tok/s          : {best_kv['tps_kv']:.1f}")
    print(f"{'='*60}\n")

    return {
        "iters":      iters,
        "val_ppl":    val_ppl,
        "val_bpc":    val_bpc,
        "val_loss":   val_loss,
        "trn_loss":   trn_loss,
        "gap":        overfit_gap,
        "kv_speedup": best_kv["speedup"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comparison across checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def compare(save_dir=".", iter_list=None):
    if iter_list is None:
        iter_list = [2000, 5000, 10000]

    results = {}
    for iters in iter_list:
        ckpt_path = os.path.join(save_dir, "checkpoints", f"ckpt_{iters}iters.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] ckpt_{iters}iters.pt not found")
            continue
        results[iters] = evaluate_single(iters, save_dir)

    if len(results) < 2:
        print("  Need at least 2 checkpoints to compare.")
        return

    keys_list = sorted(results.keys())
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {' vs '.join(str(k) for k in keys_list)} iterations")
    print(f"{'='*70}")
    col_w = 10
    header = f"  {'Metric':<25}" + "".join(f"  {k:>{col_w}}" for k in keys_list)
    print(header)
    print(f"  {'-'*65}")
    for label, key in [
        ("Val Perplexity",  "val_ppl"),
        ("Val BPC",         "val_bpc"),
        ("Val loss",        "val_loss"),
        ("Train loss",      "trn_loss"),
        ("Overfit gap",     "gap"),
        ("KV speedup @500t","kv_speedup"),
    ]:
        row = f"  {label:<25}"
        for k in keys_list:
            row += f"  {results[k][key]:>{col_w}.4f}"
        print(row)
    print(f"{'='*70}")

    # Combined loss curves
    plt.rcParams.update(THEME)
    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    colors = [TEAL, BLUE, PURP, GOLD]
    fig.suptitle("GPT From Scratch — Training Comparison", fontsize=13,
                 fontweight="bold", color=TXT)
    for ax, (iters, color) in zip(axes, zip(keys_list, colors)):
        log_path = os.path.join(save_dir, "results", f"loss_log_{iters}iters.csv")
        steps, losses = load_loss_log(log_path)
        if steps:
            ax.plot(steps, losses, color=color, lw=1.8, label="Loss")
        r = results[iters]
        ax.axhline(r["val_loss"], color=RED,  lw=1.5, ls="--",
                   label=f"Val  {r['val_loss']:.4f}")
        ax.axhline(r["trn_loss"], color=GOLD, lw=1.0, ls=":",
                   label=f"Trn  {r['trn_loss']:.4f}")
        ax.set_title(
            f"{iters} iters  |  PPL {r['val_ppl']:.2f}  |  KV {r['kv_speedup']:.2f}x",
            fontsize=11, pad=8,
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.28)

    out_path = os.path.join(save_dir, "results",
                            f"comparison_{'_'.join(str(k) for k in keys_list)}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Comparison plot: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",    type=int,  default=2000,
                        help="Which checkpoint to evaluate")
    parser.add_argument("--compare",  action="store_true",
                        help="Compare all available checkpoints")
    parser.add_argument("--save_dir", type=str,  default=".")
    args = parser.parse_args()

    if args.compare:
        compare(save_dir=args.save_dir)
    else:
        evaluate_single(iters=args.iters, save_dir=args.save_dir)
