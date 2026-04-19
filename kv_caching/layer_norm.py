import torch 
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    Layer Norm with Optional bias
    args:
    ndim:Number of features(last dimension of input)
    bias:If true,learns an additive bias 
    """
    def __init__(self,ndim:int,bias:bool=True):
        super().__init__()
        self.weight=nn.Parameter(torch.ones(ndim))
        self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self,x:torch.Tensor)-> torch.Tensor:
        return F.layer_norm(x,self.weight.shape,self.weight,self.bias,1e-5)
    
    