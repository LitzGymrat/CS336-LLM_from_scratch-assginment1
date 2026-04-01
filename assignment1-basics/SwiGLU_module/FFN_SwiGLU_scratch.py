import torch
from torch import nn
import math
from einops import einsum

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
        empty_tensor_gate = torch.empty(d_ff,d_model,dtype=dtype,device=device)
        empty_tensor_up = torch.empty(d_ff,d_model,dtype=dtype,device=device)
        empty_tensor_down = torch.empty(d_model,d_ff,dtype=dtype,device=device)


        #或者复用前面的线性层，就可以直接self.gate_proj(x)，更清晰。
        #self.gate_proj = Linear(d_ff,d_model,dtype=dtype,device=device)
        #self.up_proj = Linear(d_ff,d_model,dtype=dtype,device=device)
        #self.down_proj = Linear(d_ff,d_model,dtype=dtype,device=device)

        self.gate_proj = nn.Parameter(empty_tensor_gate)
        self.up_proj = nn.Parameter(empty_tensor_up)
        self.down_proj = nn.Parameter(empty_tensor_down)

        #初始化
        nn.init.kaiming_uniform_(self.gate_proj,a = math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj,a = math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj,a = math.sqrt(5))

    # FFN(x) = SwiGLU(x,w1,w2,w3) = w3(SiLU(w1x) *(逐元素相乘) w2x)
    #
    def forward(self,x:torch.Tensor)->torch.Tensor: #一个矩阵：用空格隔开维度
        gate = einsum(x,self.gate_proj,"... d_model,d_ff d_model -> ... d_ff")
        gate = gate * torch.sigmoid(gate) #SiLU:x*sigmoid(x)逐元素相乘
        up = einsum(x,self.up_proj,"... d_model,d_ff d_model -> ... d_ff")

        gated_up = gate * up

        result = einsum(gated_up,self.down_proj,"... d_ff,d_model d_ff -> ... d_model")
        return result
