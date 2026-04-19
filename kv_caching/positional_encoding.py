import torch 
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    """Learned absolute postional embedding table"""

    def __init__(self,block_size:int,n_embd:int):
        super().__init__()
        self.embedding=nn.Embedding(block_size,n_embd)
    def forward(self,position:torch.Tensor)-> torch.Tensor:
        """
        Args:
            postion :1-D LongTensor of postion indices,shape(T,).
            when using a KV cache,pass offset positions
            eg torch.arange(past_len,past_len+T)

        Returns:
            Postional embeddings of shape(T,n_embd)
        """
        return self.embedding(position)