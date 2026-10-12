import torch
import torch.nn as nn
import math
import einops
from einops import reduce,einsum
class RMSNorm(nn.Module):
    def __init__(self, d_model:int, eps:float = 1e-5, device:torch.device|None = None, dtype:torch.dtype | None =None):

        super().__init__()
        self.d_in = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
                                                          #设备和数据类型传给张量创建操作：而非张量填充操作
        gamma = torch.nn.init.ones_(torch.empty(d_model,device = device,dtype = dtype))
        self.gamma = nn.Parameter(gamma)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        #normalize涉及平方，需要upcast到float32以防低精度下，平方溢出
        #例如，float16 最大只能表示 65504，而 300 的平方是 90000，存不下，结果就会变成 inf（无穷大）。这就叫“平方运算时溢出”
        in_dtype = x.dtype
        x = x.to(torch.float32)
        gamma = self.gamma
        rms = torch.sqrt(self.eps + reduce(x**2,"... d_model -> ... 1","mean"))   #使用reduce聚合维度，保留一个维度，以便广播操作
        result =  x * gamma / rms
                               #
        return result.to(in_dtype)
