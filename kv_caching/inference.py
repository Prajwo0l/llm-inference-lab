"""
inference.py
────────────
Text generation utilities — KV cache vs naive, with timing.

Usage
─────
    from inference import generate_text, benchmark

    # Generate text
    text = generate_text(model, dataset, prompt="To be", max_new_tokens=200)

    # Benchmark KV cache vs naive
    results = benchmark(model, dataset, prompt="To be", new_tokens=100)
"""

import time
import torch


def generate_text(
    model,
    dataset,
    prompt:         str   = "To be",
    max_new_tokens: int   = 500,
    temperature:    float = 0.8,
    top_k:          int   = 40,
    use_kv_cache:   bool  = True,
    device:         str | None = None,
) -> str:
    """Generate text from a prompt.

    Args:
        model         : Trained GPT instance.
        dataset       : CharDataset (provides encode/decode).
        prompt        : Starting text string.
        max_new_tokens: Number of tokens to generate.
        temperature   : Sampling temperature.
        top_k         : Top-k filtering (None = disabled).
        use_kv_cache  : Use KV cache for fast inference.
        device        : Device override, or None to match model.

    Returns:
        Generated text string (prompt + continuation).
    """
    if device is None:
        device = next(model.parameters()).device

    ctx = dataset.encode(prompt).unsqueeze(0).to(device)
    out = model.generate(ctx, max_new_tokens, temperature, top_k, use_kv_cache)
    return dataset.decode(out[0].cpu())


def benchmark(
    model,
    dataset,
    prompt:     str = "To be",
    new_tokens: int = 500,
    runs:       int = 3,
    device:     str | None = None,
) -> dict:
    """Compare generation speed: KV cache vs naive.

    Args:
        model      : Trained GPT instance.
        dataset    : CharDataset.
        prompt     : Starting text for all runs.
        new_tokens : Tokens to generate per run.
        runs       : Number of timed repetitions (averaged).
        device     : Device override, or None to match model.

    Returns:
        dict with keys: 'naive_avg_s', 'kv_avg_s', 'speedup'
    """
    if device is None:
        device = next(model.parameters()).device

    ctx = dataset.encode(prompt).unsqueeze(0).to(device)

    print(f"\n{'='*55}")
    print(f"  Benchmark | {runs} run(s) | {new_tokens} new tokens")
    print(f"  Prompt length : {ctx.shape[1]} tokens")
    print(f"{'='*55}")

    # ── Naive (no cache) ───────────────────────────────────────────────────
    naive_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.generate(ctx, new_tokens, temperature=1.0, use_kv_cache=False)
        naive_times.append(time.perf_counter() - t0)

    # ── KV cache ───────────────────────────────────────────────────────────
    kv_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.generate(ctx, new_tokens, temperature=1.0, use_kv_cache=True)
        kv_times.append(time.perf_counter() - t0)

    avg_naive = sum(naive_times) / runs
    avg_kv    = sum(kv_times)    / runs
    speedup   = avg_naive / avg_kv

    print(f"\n  No KV cache : {avg_naive:.3f} s avg")
    print(f"  KV cache    : {avg_kv:.3f} s avg")
    print(f"  Speedup     : {speedup:.2f}x")
    print(f"{'='*55}\n")

    return {"naive_avg_s": avg_naive, "kv_avg_s": avg_kv, "speedup": speedup}
