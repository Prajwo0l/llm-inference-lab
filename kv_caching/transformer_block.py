import torch 
import torch.nn as nn

from config import GPTConfig
from layer_norm import LayerNorm
from attention import CausalSelfAttention
from feedforward import MLP


class Block(nn.Module):
    """Single transformer decoder block with KV cache support"""
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.ln_1=LayerNorm(config.n_embd,bias=config.bias)
        self.attn=CausalSelfAttention(config)
        self.ln_2=LayerNorm(config.n_embd,bias=config.bias)
        self.mlp=MLP(config)

    def forward(self,x:torch.Tensor,kv_cache:dict | None=None, cache_pos:int | None=None):
        """
        args : 
            x: (B,T,C)
            kv_cache:per-layer KV cache dict or None
            cache_pos: write position in the KV buffer (decode mode), or None
        Returns : 
        x : (B,T,C) updated hidden states
        kv_cache :updated cache dict or None
        """
        attn_out,kv_cache=self.attn(self.ln_1(x) , kv_cache=kv_cache, cache_pos=cache_pos)
        x= x + attn_out
        x = x +self.mlp(self.ln_2(x))
        return x , kv_cache