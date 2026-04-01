import torch
from torch import nn
from RMSNorm_module.RMSNorm import RMSNorm
from SwiGLU_module.FFN_SwiGLU import SwiGLU


from multi_head_attn_module.multi_head_attention_kv import Multi_Head_Attention_kv
class Transformer_block_kv(nn.Module):
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
        self.attn = Multi_Head_Attention_kv(d_model,num_heads,theta,max_seq_len,device,dtype)
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor = None,past_key_value:tuple[torch.Tensor,torch.Tensor] = None):
        #一般的模型训练过程，token_positions就是0 -seqlen-1（当前序列长度）
        #采用kvcache后情况会变化(decode阶段)
        if token_positions is None:
            cur_seq_len = x.shape[-2]
            token_positions = torch.arange(0,cur_seq_len,device=x.device)
        prenormed_x = self.ln1(x)
        y_1,present_key_value = self.attn(prenormed_x,token_positions,past_key_value)
        y_1  = x+y_1

        prenormed_y1 = self.ln2(y_1)

        result = y_1 + self.ffn(prenormed_y1)

        return result,present_key_value


