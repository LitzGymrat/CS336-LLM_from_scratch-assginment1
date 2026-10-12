import torch
import torch.nn as nn
import math
import einops
from einops import reduce,einsum
from cs336_basics.Transformer.Linear_module import Linear
from cs336_basics.Transformer.Embedding_module import Embedding
from cs336_basics.Transformer.RMSNorm_module import RMSNorm
from cs336_basics.Transformer.Swiglu_module import SWIGLU
from cs336_basics.Transformer.Transformer_block import Transformer_blk
from jaxtyping import Int

class Transformer_lm(nn.Module):
    #d_ff应该可传入。如果没传入就算
    def __init__(self, vocab_size:int,d_model:int, num_heads:int,d_ff:int,num_layers:int,context_len:int,rope_theta:float,device:torch.device|None = None, dtype:torch.dtype | None =None,
                 norm_position:str = "pre", use_rope:bool = True, ffn_type:str = "swiglu"):

        super().__init__()
        self.d_in = d_model
        self.device = device
        self.dtype = dtype
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
                            #embedding大小由vocab_size决定（每个token都对应一个embedding
        self.embedding = Embedding(vocab_size,d_model,device = device,dtype=dtype)
                            #最后输出维度:(batch_size,...,seq_len,vocab_size)：每个token位置给每个词表位置都生成一个logit
        self.lm_head = Linear(d_model,vocab_size,device,dtype)
                                            #使用列表推导式:obj for _ in range(num)生成layers 个 blk构成的列表
        layers = [Transformer_blk(d_model,num_heads,d_ff,context_len,rope_theta,device,dtype,
                                  norm_position=norm_position,use_rope=use_rope,ffn_type=ffn_type)
                  for _ in range(num_layers)]
        self.layers = torch.nn.ModuleList(layers) #使用ModuleList传入模块列表
        #比起普通列表，会把列表内部模块注册为layers的子模块，查看参数时方便显示
        self.final_norm = nn.Identity() if norm_position == "none" else RMSNorm(d_model,device=device,dtype=dtype)

    def forward(self,in_indices: Int[torch.Tensor, " batch_size sequence_length"],token_positions:torch.Tensor|None = None) -> torch.Tensor:
        x = self.embedding.forward(in_indices)
        #一层层过
        for layer in self.layers:
            x = layer.forward(x,token_positions)
        x = self.final_norm.forward(x)

        logits = self.lm_head.forward(x)
                               #
        return logits
