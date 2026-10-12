import torch
import os
import typing
def save_checkpoint(model:torch.nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    #存储为字典
    checkpoint_data = {"model_state":model_state,"optimizer_state":optimizer_state,"iteration":iteration}
    
    torch.save(checkpoint_data,out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],model: torch.nn.Module,optimizer: torch.optim.Optimizer):
    checkpoint_data:dict =torch.load(src)
    model_state = checkpoint_data["model_state"]
    optimizer_state = checkpoint_data["optimizer_state"]
    iteration = checkpoint_data["iteration"]
    model.load_state_dict(model_state) #加载回去
    optimizer.load_state_dict(optimizer_state)
    #返回迭代次数，从而恢复学习率之类的。
    return iteration
