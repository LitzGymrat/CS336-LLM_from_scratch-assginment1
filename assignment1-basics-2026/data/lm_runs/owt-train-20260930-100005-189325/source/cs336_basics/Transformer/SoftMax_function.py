import torch
import torch.nn as nn
import math
import einops
from einops import reduce,einsum
def softmax(x:torch.Tensor,dim:int):
    #减去一个最大值，不影响softmax结果，防止指数爆炸导致数值溢出。
    
    shifted_x:torch.Tensor = x - x.max(dim = dim,keepdim = True).values #沿着某个维度会返回value + index,需要用.values读取对象属性  
    #指数化
    shifted_exp = torch.exp(shifted_x)
    return shifted_exp / shifted_exp.sum(dim = dim,keepdim = True)
