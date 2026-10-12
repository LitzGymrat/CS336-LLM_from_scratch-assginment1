import torch
import torch.nn as nn
import math
import einops
class Linear(nn.Module):
    def __init__(self, d_in:int, d_out:int, device:torch.device|None = None, dtype:torch.dtype | None =None):

        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.device = device
        self.dtype = dtype
        std = math.sqrt(2/(d_out+d_in)) #先把标准差存下来
                                                        #设备和数据类型传给张量创建操作：而非张量填充操作
        weights = torch.nn.init.trunc_normal_(torch.empty(d_out,d_in,device = device,dtype = dtype),mean = 0,std = std,a=-3 * std,b=3*std)
        self.weights = nn.Parameter(weights)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        W = self.weights                                     #省略前面维度
        return einops.einsum(W,x,"d_out d_in , ... d_in -> ... d_out" )
