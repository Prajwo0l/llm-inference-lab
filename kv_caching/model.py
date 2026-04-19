import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from layer_norm import LayerNorm
from positional_encoding import PositionalEmbedding
from transformer_block import Block

class GPT(nn.Module):
    """GPT language model built from scratch in pure Pytorch"""
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.config=config

        self.tok_emb=nn.Embedding(config.vocab_size,config.n_embd)
        self.pos_emb=PositionalEmbedding(config.block_size,config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.ln_f = LayerNorm(config.n_embd,bias=config.bias)
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)


        #weight tying
        self.tok_emb.weight=self.lm_head.weight
        #weight initialization
        self.apply(self._init_weights)
        for name,param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(param,mean=0.0,std=0.02/math.sqrt(2 *config.n_layer))

        print(f"GPT | {self.num_paramters()/1e6:.2f}M paramters")


    def _init_weights(self,module:nn.Module):
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def num_paramters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    


    def forward(
            self,
            idx: torch.Tensor,
            targets :torch.Tensor | None=None,
            kv_caches : list| None = None,
    ):
        """
        Args : 
            IX : (B,T) token indices.
            targets : (B,T) target indices for cross-entropy loss 
            kv_caches :List or per-layer cache dicts for fast inference

        returns:
        logits :(B,T,vocab_size)
        loss: scalar cross-entropy loss or none
        kv_caches:updated list of per-layer cache dicts
        """
        device = idx.device
        B,T = idx.shape
        assert T <= self.config.block_size,\
            f"Seqeuence length {T} > block_size {self.config.block_size}"
        past_len=0
        if kv_caches and "k " in (kv_caches[0] or {}):
            past_len = kv_caches[0]['k'].shape[2]
        positions= torch.arange(past_len,past_len + T,dtype=torch.long,device=device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))

        if kv_caches is None:
            kv_caches = [None]* self.config.n_layer
        new_caches = []
        for block,cache in zip(self.blocks,kv_caches):
            x,new_cache = block(x,kv_cache=cache)
            new_caches.append(new_cache)
        x = self.ln_f(x)
        logits=self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1,logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits,loss,new_caches
    
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
        """KV-cache generation — O(T) total.

        Step 1 (Prefill) : process the full prompt, build cache.
        Step 2 (Decode)  : feed only the single newest token each step.
        """
        # Prefill
        logits, _, kv_caches = self(idx, kv_caches=None)
        next_tok = self._sample(logits[:, -1, :], temperature, top_k)
        idx = torch.cat([idx, next_tok], dim=1)

        # Decode loop
        for _ in range(max_new_tokens - 1):
            logits, _, kv_caches = self(next_tok, kv_caches=kv_caches)
            next_tok = self._sample(logits[:, -1, :], temperature, top_k)
            idx = torch.cat([idx, next_tok], dim=1)

        return idx
