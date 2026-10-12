import torch
import torch.nn as nn
import math
import einops
from einops import reduce,einsum
class SWIGLU(nn.Module):
    #d_ff应该可传入。如果没传入就算
    def __init__(self, d_model:int, d_ff:int|None = None,device:torch.device|None = None, dtype:torch.dtype | None =None,
                 gated:bool = True):

        super().__init__()
        self.d_in = d_model
        self.device = device
        self.dtype = dtype
        self.gated = gated
        if d_ff is None:
            target_d_ff = d_model * 8 / 3
            num_64 = round(target_d_ff / 64) #四舍五入：算出取整大约需要多少组64
            self.d_ff = num_64 * 64
        else:
            self.d_ff = d_ff
        std = math.sqrt(2/(self.d_ff + self.d_in))                                                 #设备和数据类型传给张量创建操作：而非张量填充操作
        w1_weights = torch.nn.init.trunc_normal_(torch.empty(self.d_ff,self.d_in,device = device,dtype = dtype),mean = 0,std = std,a=-3 * std,b=3*std)
        w2_weights = torch.nn.init.trunc_normal_(torch.empty(self.d_in,self.d_ff,device = device,dtype = dtype),mean = 0,std = std,a=-3 * std,b=3*std)
        if gated:
            w3_weights = torch.nn.init.trunc_normal_(torch.empty(self.d_ff,self.d_in,device = device,dtype = dtype),mean = 0,std = std,a=-3 * std,b=3*std)
        self.w1 = nn.Parameter(w1_weights)
        self.w2 = nn.Parameter(w2_weights)
        if gated:
            self.w3 = nn.Parameter(w3_weights)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        W1 = self.w1
        W2 = self.w2
        w1_x = einsum(W1,x,"d_ff d_in , ... d_in -> ... d_ff")
        SiLU_w1_x = w1_x * torch.sigmoid(w1_x)

                                #逐元素相乘门控
        if self.gated:
            w3_x = einsum(self.w3,x,"d_ff d_in , ... d_in -> ... d_ff")
            SiLU_w1_x = SiLU_w1_x * w3_x
        result = einsum(W2,SiLU_w1_x,"d_in d_ff , ... d_ff -> ... d_in")
                               #
        return result
