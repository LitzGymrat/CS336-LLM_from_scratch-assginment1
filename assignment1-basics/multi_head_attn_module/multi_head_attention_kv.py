import torch
from einops import rearrange
from torch import nn
from scaled_dot_product_attention_module.s_dpa import scaled_dot_product_attention
from Linear_module.linear import Linear
from RoPE_module.RoPE import RoPE

class Multi_Head_Attention_kv(nn.Module):
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
                                                                #kv_cache：维护，传入之前的kv，和这一个(seq_len一般为1)拼起来
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor,past_key_value:tuple[torch.Tensor,torch.Tensor] = None):
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

        #先把新的Rope再接上去。
        Q = self.rope(Q,token_positions)
        K = self.rope(K,token_positions)

        if past_key_value is not None:
            past_key,past_value = past_key_value
            K = torch.concat([past_key,K],dim = -2) #沿着seq_len维度拼接
            V = torch.concat([past_value,V],dim = -2) 
        #把这轮的K,V打包准备返回
        cur_key_value = K,V

       
        #5.制作掩码下三角矩阵:第i个q只能看到前i个k
        L_q = Q.shape[-2]
        L_k = K.shape[-2]
        #在forward中建立新张量：搬运到相同的设备中,指定数据类型

        #decode阶段，Q = 1,如果K=101,加一个偏移量--->不需要掩码，实际上
        mask  = torch.tril(torch.ones(L_q,L_k),diagonal=L_k - L_q)
        mask = mask.to(device=x.device,dtype=torch.bool)
        #5.每个头进行缩放点积注意力
        attn_scores = scaled_dot_product_attention(Q,K,V,mask)
        #(...,num_heads,seq,d_k)
        #6.concat拼接(直接用einops调整)


        #7.维度调整回去                                                            #einops:括号顺序表示内存排列顺序：从外到内                         
        multi_head_attn = rearrange(attn_scores,"... num_heads seq d_k -> ... seq (num_heads d_k)")

        return self.output_proj(multi_head_attn),cur_key_value