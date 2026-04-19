import torch 
import torch.nn as nn

from config import GPTConfig

class MLP(nn.Module):
    """ Two layer fed forward network with GELU Activation"""

    def __init__(self,config:GPTConfig):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,4*config.n_embd,bias=config.bias)
        self.gelu=nn.GELU()
        self.c_proj = nn.Linear(4* config.n_embd,config.n_embd,bias=config.bias)
        self.dropout =nn.Dropout(config.dropout)

    def forward(self,x:torch.Tensor)-> torch.Tensor:
        """
        Arhs:
            x: (B,T,C)
        Returns:
        (B,T,C)
        """
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))