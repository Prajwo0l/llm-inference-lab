"""
model.py
────────
Full GPT model.

Architecture (top-down)
───────────────────────
  Token Embedding  (vocab_size → n_embd)
  Positional Embedding  (block_size → n_embd)
  Dropout
  N × Transformer Block  (with optional KV cache)
  LayerNorm
  LM Head  (n_embd → vocab_size)   ← weights tied to token embedding

Key features
────────────
  * Weight tying: lm_head and tok_embedding share parameters (saves ~10 M
    params for a 125 M model, from the GPT-2 paper).
  * KV cache: fast autoregressive inference with O(T) per step instead of O(T²).
  * Configurable via GPTConfig.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config             import GPTConfig
from layer_norm         import LayerNorm
from positional_encoding import PositionalEmbedding
from transformer_block  import Block


class GPT(nn.Module):
    """GPT language model built from scratch in pure PyTorch.

    Args:
        config (GPTConfig): All model hyperparameters.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = PositionalEmbedding(config.block_size, config.n_embd)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f    = LayerNorm(config.n_embd, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        # Weight initialisation
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print(f"GPT | {self.num_parameters() / 1e6:.2f}M parameters")

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        idx:       torch.Tensor,
        targets:   torch.Tensor | None = None,
        kv_caches: list | None         = None,
    ):
        """
        Args:
            idx       : (B, T) token indices.
            targets   : (B, T) target indices for cross-entropy loss, or None.
            kv_caches : List of per-layer cache dicts for fast inference,
                        or None for training.

        Returns:
            logits    : (B, T, vocab_size)
            loss      : scalar cross-entropy loss or None
            kv_caches : updated list of per-layer cache dicts
        """
        device = idx.device
        B, T   = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} > block_size {self.config.block_size}"

        # Offset positions when using KV cache (past tokens already processed)
        past_len = 0
        if kv_caches and kv_caches[0] is not None:
            past_len = kv_caches[0].get("cache_pos", 0)

        positions = torch.arange(past_len, past_len + T, dtype=torch.long, device=device)

        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))

        if kv_caches is None:
            kv_caches = [None] * self.config.n_layer

        # Extract cache_pos if present (decode mode); 0 means prefill
        cache_pos = None
        if kv_caches[0] is not None and "cache_pos" in kv_caches[0]:
            cache_pos = kv_caches[0]["cache_pos"] if kv_caches[0]["cache_pos"] > 0 else None

        new_caches = []
        for block, cache in zip(self.blocks, kv_caches):
            x, new_cache = block(x, kv_cache=cache, cache_pos=cache_pos)
            new_caches.append(new_cache)

        x      = self.ln_f(x)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, new_caches

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          int | None = None,
        use_kv_cache:   bool = True,
    ) -> torch.Tensor:
        """Autoregressive token generation.

        Args:
            idx           : (B, T) starting context.
            max_new_tokens: how many tokens to generate.
            temperature   : softmax temperature (lower = more deterministic).
            top_k         : nucleus/top-k filtering (None = no filtering).
            use_kv_cache  : True (fast, O(T)) or False (naive, O(T²)).

        Returns:
            idx: (B, T + max_new_tokens) complete sequence.
        """
        self.eval()
        if use_kv_cache:
            return self._generate_kv(idx, max_new_tokens, temperature, top_k)
        return self._generate_naive(idx, max_new_tokens, temperature, top_k)

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: int | None):
        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _generate_naive(self, idx, max_new_tokens, temperature, top_k):
        """Re-processes the full sequence at every step — O(T²) total."""
        for _ in range(max_new_tokens):
            idx_cond    = idx[:, -self.config.block_size:]
            logits, _, _ = self(idx_cond, kv_caches=None)
            next_tok    = self._sample(logits[:, -1, :], temperature, top_k)
            idx         = torch.cat([idx, next_tok], dim=1)
        return idx

    def _generate_kv(self, idx, max_new_tokens, temperature, top_k):
        """KV-cache generation with pre-allocated buffers — O(T) total.

        Optimisations
        ─────────────
          1. Buffers allocated ONCE before decode loop (no heap alloc per step)
          2. In-place writes via index assignment   (no torch.cat per step)
          3. Output tokens collected in pre-allocated tensor (no torch.cat)
          4. cache_pos integer tracks fill level    (no shape inspection)
        """
        B, T_prompt = idx.shape
        device      = idx.device
        max_total   = T_prompt + max_new_tokens

        # ── 1. Pre-allocate fixed KV buffers for every layer ───────────────────
        buf_len = min(max_total, self.config.block_size)
        kv_caches = [
            {
                "k":         torch.zeros(B, self.config.n_head,
                                        buf_len, self.config.n_embd // self.config.n_head,
                                        device=device),
                "v":         torch.zeros(B, self.config.n_head,
                                        buf_len, self.config.n_embd // self.config.n_head,
                                        device=device),
                "cache_pos": 0,          # tracks how many positions are filled
            }
            for _ in range(self.config.n_layer)
        ]

        # ── 2. Pre-allocate output token tensor ─────────────────────────────
        out_ids = torch.empty(B, max_total, dtype=torch.long, device=device)
        out_ids[:, :T_prompt] = idx

        # ── 3. Prefill: process full prompt, fill buffers positions 0..T_prompt-1
        logits, _, kv_caches = self(idx, kv_caches=kv_caches)
        # Advance cache_pos by prompt length
        for cache in kv_caches:
            cache["cache_pos"] = T_prompt

        next_tok = self._sample(logits[:, -1, :], temperature, top_k)
        out_ids[:, T_prompt] = next_tok.squeeze(-1)

        # ── 4. Decode loop: one token per step, zero allocation ──────────────
        for step in range(1, max_new_tokens):
            write_pos = T_prompt + step          # position being written this step
            if write_pos >= self.config.block_size:
                break                            # hit context limit

            logits, _, kv_caches = self(next_tok, kv_caches=kv_caches)

            # Advance cache_pos
            for cache in kv_caches:
                cache["cache_pos"] = write_pos

            next_tok = self._sample(logits[:, -1, :], temperature, top_k)
            out_ids[:, write_pos] = next_tok.squeeze(-1)

        filled = min(T_prompt + max_new_tokens, self.config.block_size)
        return out_ids[:, :filled]
