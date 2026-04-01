import torch
from einops import rearrange
from torch import nn
from scaled_dot_product_attention_module.s_dpa import scaled_dot_product_attention
from Linear_module.linear import Linear
from RoPE_module.RoPE import RoPE
class Multi_Head_Attention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,theta:int,max_seq_len:int,device = None,dtype = None):
         #父类初始化
        super().__init__()
        assert d_model % num_heads == 0
        #传入形状的时候要求是int，改为取整除法
        self.d_k = d_model // num_heads
        
        #传入x:(B,seq,d_model) ->(B,seq,d_model) 
        self.q_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        #传入x_head(... seq ,d_k)
        self.rope = RoPE(theta,self.d_k,max_seq_len,device=device)

        #还有最后融合的o_proj
        self.output_proj = Linear(d_model,d_model,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):
        #1.投影子空间得到Q,K,V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        #2.rarrange分头             #注意内存顺序：先分头，再分块
        Q = rearrange(Q,"... seq (num_heads d_k) -> ... seq num_heads d_k",d_k = self.d_k)
        K = rearrange(K,"... seq (num_heads d_k) -> ... seq num_heads d_k",d_k = self.d_k)
        V = rearrange(V,"... seq (num_heads d_k) -> ... seq num_heads d_k",d_k = self.d_k)

        #3.调整维度,方便batch_like运算；以及分头RoPE
        Q = rearrange(Q,"... seq num_heads d_k -> ... num_heads seq d_k")
        K = rearrange(K,"... seq num_heads d_k -> ... num_heads seq d_k")
        V = rearrange(V,"... seq num_heads d_k -> ... num_heads seq d_k")
        #4.每个头分别对q,k进行RoPE;接收返回值

        #Pytorch中nn.Module函数：不显式forward调用。直接类名调用即可
        Q = self.rope(Q,token_positions)
        K = self.rope(K,token_positions)

       
        #5.制作掩码下三角矩阵:第i个q只能看到前i个k
        seq = Q.shape[-2]
        #在forward中建立新张量：搬运到相同的设备中,指定数据类型
        mask  = torch.tril(torch.ones(seq,seq))
        mask = mask.to(device=x.device,dtype=torch.bool)
        #5.每个头进行缩放点积注意力
        attn_scores = scaled_dot_product_attention(Q,K,V,mask)
        #(...,num_heads,seq,d_k)
        #6.concat拼接(直接用einops调整)


        #7.维度调整回去                                                            #einops:括号顺序表示内存排列顺序：从外到内                         
        multi_head_attn = rearrange(attn_scores,"... num_heads seq d_k -> ... seq (num_heads d_k)")

        return self.output_proj(multi_head_attn)