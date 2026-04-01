import torch
from torch import nn
from RMSNorm_module.RMSNorm import RMSNorm
from SwiGLU_module.FFN_SwiGLU import SwiGLU


from multi_head_attn_module.multi_head_attention import Multi_Head_Attention
class Transformer_block(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,theta:int,
                 max_seq_len:int,
                 eps:float = 1e-5,#eps设置默认值
                 device = None,dtype = None):
        super().__init__()
        #不同地方的规范化函数需要建立不同的实例！
        self.max_seq_len = max_seq_len
        self.ln1 = RMSNorm(d_model,eps,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model,eps,device=device,dtype=dtype)
        self.ffn = SwiGLU(d_model,d_ff,dtype=dtype,device=device)
        self.attn = Multi_Head_Attention(d_model,num_heads,theta,max_seq_len,device,dtype)
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor = None):
        #一般的模型训练过程，token_positions就是0 -seqlen-1（当前序列长度）
        if token_positions is None:
            cur_seq_len = x.shape[-2]
            token_positions = torch.arange(0,cur_seq_len,device=x.device)
        prenormed_x = self.ln1(x)
        y_1 = x + self.attn(prenormed_x,token_positions)

        prenormed_y1 = self.ln2(y_1)

        result = y_1 + self.ffn(prenormed_y1)

        return result


