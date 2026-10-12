import torch
import torch.nn as nn
import math
import einops
from einops import reduce,einsum
from cs336_basics.Transformer.RMSNorm_module import RMSNorm
from cs336_basics.Transformer.Swiglu_module import SWIGLU
from cs336_basics.Transformer.Multihead_self_attention_module import Multihead_self_attention

class Transformer_blk(nn.Module):
    #d_ff应该可传入。如果没传入就算
    def __init__(self, d_model:int, num_heads:int,d_ff:int,max_seq_len:int,theta:float,device:torch.device|None = None, dtype:torch.dtype | None =None,
                 norm_position:str = "pre", use_rope:bool = True, ffn_type:str = "swiglu"):

        super().__init__()
        self.d_in = d_model
        self.device = device
        self.dtype = dtype
        self.d_ff = d_ff
        self.num_heads = num_heads
        if norm_position not in ("pre", "post", "none"):
            raise ValueError(f"Unknown norm_position: {norm_position}")
        if ffn_type not in ("swiglu", "silu"):
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
        self.norm_position = norm_position
        #两个子层的rmsnorm各自参数独立，需要有两个
        self.rmsnorm_1 = nn.Identity() if norm_position == "none" else RMSNorm(d_model,device = device,dtype = dtype)
        self.rmsnorm_2 = nn.Identity() if norm_position == "none" else RMSNorm(d_model,device = device,dtype = dtype)
        self.swiglu = SWIGLU(d_model,d_ff,device = device,dtype = dtype,gated=ffn_type == "swiglu")
        self.msa = Multihead_self_attention(d_model,num_heads,theta,max_seq_len,device=device,dtype=dtype,use_rope=use_rope)


    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None = None) -> torch.Tensor:
        if self.norm_position == "post":
            y_1 = self.rmsnorm_1(x + self.msa(x,token_positions))
            return self.rmsnorm_2(y_1 + self.swiglu(y_1))
        y_1 = x + self.msa(self.rmsnorm_1(x),token_positions)
        y_2 = y_1 + self.swiglu(self.rmsnorm_2(y_1)) #第二个sublayer传入y_1
                               #
        return y_2
