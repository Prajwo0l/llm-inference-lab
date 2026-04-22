
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Supports:
      * Standard forward pass (training, no cache)
      * Pre-allocated KV buffer pass (fast autoregressive inference)

    Args:
        config (GPTConfig): Model hyperparameters.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd must be divisible by n_head"

        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        # Fused Q, K, V projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal (lower-triangular) mask — registered as buffer (not a param)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(
        self,
        x:         torch.Tensor,
        kv_cache:  dict | None = None,
        cache_pos: int | None  = None,
    ):
        """
        Args:
            x         : Input tensor (B, T, C).
            kv_cache  : dict with pre-allocated buffers:
                          'k'  : (B, n_head, block_size, head_dim)  — fixed allocation
                          'v'  : (B, n_head, block_size, head_dim)
                        None during training or prefill building.
            cache_pos : int — next write position in the buffer (decode mode).
                        None during training / prefill.

        Returns:
            y         : Output tensor (B, T, C).
            kv_cache  : Same buffer dict (updated in-place, no copy).
        """
        B, T, C = x.shape

        # ── Project to Q, K, V ──────────────────────────────────────────────
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ── KV Cache: pre-allocated buffer path ─────────────────────────────
        if kv_cache is not None and cache_pos is not None:
            # Decode step: write one token in-place, no allocation, no copy
            kv_cache["k"][:, :, cache_pos : cache_pos + T, :] = k
            kv_cache["v"][:, :, cache_pos : cache_pos + T, :] = v
            filled = cache_pos + T          # how many positions are valid
            k = kv_cache["k"][:, :, :filled, :]   # view — no copy
            v = kv_cache["v"][:, :, :filled, :]
            T_k = filled
        elif kv_cache is not None:
            # Prefill: fill the buffer from position 0
            kv_cache["k"][:, :, :T, :] = k
            kv_cache["v"][:, :, :T, :] = v
            T_k = T
        else:
            # Training: no cache
            T_k = T
        # ────────────────────────────────────────────────────────────────────

        # ── Scaled dot-product attention ─────────────────────────────────────
        scale = 1.0 / math.sqrt(self.head_dim)
        att   = (q @ k.transpose(-2, -1)) * scale            # (B, n_head, T, T_k)

        att = att.masked_fill(
            self.causal_mask[:, :, T_k - T : T_k, :T_k] == 0,
            float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v                                          # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y, kv_cache
