import torch
from torch import nn

from Linear_module.linear import Linear
class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dtype = None,device = None):
        #父类初始化
        super().__init__()
        if dtype is None:
            dtype = torch.float32

            #三个矩阵：升维+门控单元(套激活函数)+降维
            #使用矩阵乘法，相比偏置加法表达能力更强
            #RMSNorm使得偏置不太有意义？
                                    #d_out,d_in
        


        #或者复用前面的线性层，就可以直接self.gate_proj(x)，更清晰。
        #Linear:先传入in,再传入Out
        self.w1 = Linear(d_model,d_ff,dtype=dtype,device=device)#gate_proj
        self.w3 = Linear(d_model,d_ff,dtype=dtype,device=device)#up_proj
        self.w2 = Linear(d_ff,d_model,dtype=dtype,device=device)#down_proj

        

        #初始化
        

    # FFN(x) = SwiGLU(x,w1,w2,w3) = w3(SiLU(w1x) *(逐元素相乘) w2x)
    #
    def forward(self,x:torch.Tensor)->torch.Tensor: #一个矩阵：用空格隔开维度
        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate) #SiLU:x*sigmoid(x)逐元素相乘
        up = self.w3(x)

        gated_up = gate * up

        result = self.w2(gated_up)
        return result
