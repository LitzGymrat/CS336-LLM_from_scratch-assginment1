import torch
import math
from torch import nn
from einops import einsum
class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device = None,dtype = None):
        super().__init__()
        self.std = math.sqrt(2 / (in_features+out_features))
        #先建立一个空的张量                             #设备和类型
        empty_tensor = torch.empty(out_features,in_features,device=device,dtype=dtype)
        
        #设为nn.Parameter可训练参数
        self.weight= nn.Parameter(empty_tensor)

        #_：原地修改

        #_：原地修改
        nn.init.trunc_normal_(self.weight,mean = 0,std= self.std,a = -3*self.std,b = 3*self.std) 
    def forward(self,x:torch.Tensor)->torch.Tensor:
                        #支持任意前缀batch_like维度
                        #用省略号，前面维度留出空间（都是类batch处理）最后维度对齐即可
        return einsum(x,self.weight,"... d_in,d_out d_in -> ... d_out" )