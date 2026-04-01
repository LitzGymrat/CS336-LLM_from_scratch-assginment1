import torch
from torch import nn
import os
import typing

def save_checkpoint(model:nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    #不同对象的state_dict：存成一个大字典。方便查找
    checkpoint_state = {
        "model" : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "iteration" : iteration
    }

    torch.save(checkpoint_state,out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model:nn.Module, optimizer:torch.optim.Optimizer):
    state_checkpoint = torch.load(src)

    model.load_state_dict(state_checkpoint["model"])
    optimizer.load_state_dict(state_checkpoint["optimizer"])

    return state_checkpoint["iteration"]
    