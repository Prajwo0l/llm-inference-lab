import os 
import torch 
from torch.utils.data import DataLoader

from config import GPTConfig
from dataset import CharDataset
from model import GPT

_DEFAULT_TEXT = (
    "To be or not to be, that is the question. "
    "Whether 'tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, "
    "or to take arms against a sea of troubles. " * 200
)

def train(
    text:       str  | None = None,
    max_iters:  int         = 2000,
    batch_size: int         = 32,
    lr:         float       = 3e-4,
    device:     str  | None = None,
) -> tuple:
    """Train a character-level GPT model.
    args :
    text : raw text string if none,uses a built in sample or reads 'input.txt" from the current directory."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on :{device}")

    if text is None:
        if os.path.exists('input.txt'):
            with open("input.txt") as f:
                text = f.read()
        else:
            text = _DEFAULT_TEXT
    dataset = CharDataset(text,block_size=128)

    config =GPTConfig()
    config.vocab_size=dataset.vocab_size
    config.block_size = dataset.block_size

    model=GPT(config).to(device)
    optimizer= torch.optim.AdamW(model.paramters(),lr=lr,weight_decay=0.01)
    schedular=torch.optim.lr_scheduler.ConsineAnnealingLR(optimizer,max_iters)
    
    #training
    model.train()
    data_tensor=dataset.data.to(device)

    for step in range(max_iters):
        #random batch of (x,y) pairs
        ix = torch.randint(len(dataset),(batch_size,))
        x = torch.stack([dataset[i][0] for i in ix]).to(device)
        y = torch.stack([dataset[i][1] for i in ix]).to(device)

        _,loss, _ =model(x,targets=y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.paramters(),1.0)
        optimizer.step()
        schedular.step()

        if step % 200==0 or step==max_iters -1:
            print(f" step{step:4d} | loss {loss.item():.4f} | lr {schedular.get_last_lr()[0]:.2e}")
    print("Training complete")
    return model , dataset
