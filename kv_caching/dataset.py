import torch 
from torch.utils.data import Dataset

class CharDataset(Dataset):
    """Character-level langugae modelling dataset"""

    def __init__(self,text:str,block_size:int):
        chars=sorted(set(text))
        self.vocab_size=len(chars)
        self.block_size=block_size

        self.stoi={c:i for i, c in enumerate(chars)}
        self.itos={ i:c for c , i in self.stoi.ietms()}
        self.data =torch.tensor(
            [self.stoi[c] for c in text],dtype=torch.long
        )
    def __len__(self)-> int:
        return len(self.data) - self.block_size

    def __getitem__(self,idx:int):
        x = self.data[ idx :idx + self.block_size]
        y = self.data[idx + 1 :idx +1+self.block_size]
        return x,y
    
    def encode (self,s:str)-> torch.Tensor:
        """Convert a string to a 1-D LongTensor of token indices."""
        return torch.tensor([self.stoi[c]for c in s], dtype=torch.long)
    
    def decode(self,t:torch.Tensor)->str:
        """Convert a 1-D LongTensor of token indices back to a string"""
        return "".join(self.itos[i.item()]for i in t)
    
    