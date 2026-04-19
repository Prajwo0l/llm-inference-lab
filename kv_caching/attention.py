import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig



class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Supports:
      * Standard forward pass (training, no cache)
      * KV cache forward pass (fast autoregressive inference)

    Args:
        config (GPTConfig): Model hyperparameters.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd must be divisible by n_head"

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.head_dim = config.n_embd // config.n_head

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

    def forward(self, x: torch.Tensor, kv_cache: dict | None = None):
        """
        Args:
            x        : Input tensor of shape (B, T, C).
            kv_cache : dict with keys 'k' and 'v' (each (B, n_head, T_past, head_dim))
                       or None for training / first prefill step.

        Returns:
            y        : Output tensor (B, T, C).
            kv_cache : Updated cache dict (same keys) or None if unused.
        """
        B, T, C = x.shape

        # ── Project to Q, K, V ──────────────────────────────────────────────
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape: (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ── KV Cache update ─────────────────────────────────────────────────
        if kv_cache is not None:
            if "k" in kv_cache:                              # past keys exist → append
                k = torch.cat([kv_cache["k"], k], dim=2)    # (B, n_head, T_past+T, head_dim)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache = {"k": k, "v": v}                     # save updated tensors
        # ────────────────────────────────────────────────────────────────────

        T_k = k.shape[2]   # total key length (past + current)

        # ── Scaled dot-product attention ─────────────────────────────────────
        scale = 1.0 / math.sqrt(self.head_dim)
        att   = (q @ k.transpose(-2, -1)) * scale            # (B, n_head, T, T_k)

        # Apply causal mask (handles cache offset: query sees only past + itself)
        att = att.masked_fill(
            self.causal_mask[:, :, T_k - T : T_k, :T_k] == 0,
            float("-inf")
        )
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted aggregation of values
        y = att @ v                                          # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)    # re-merge heads
        y = self.resid_dropout(self.c_proj(y))

        return y, kv_cache
