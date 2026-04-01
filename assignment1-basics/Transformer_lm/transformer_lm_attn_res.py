import torch
from torch import nn
from RMSNorm_module.RMSNorm import RMSNorm
from embedding_module.embedding import Embedding
from transformer_blk_module.transformer_block import Transformer_block
from einops import einsum
from softmax_module.softmax import softmax
import math
from Linear_module.linear import Linear
from RMSNorm_module.RMSNorm import RMSNorm


class Transformer_lm(nn.Module):
    def __init__(self,vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,device = None,dtype = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model,device=device,dtype=dtype)
        
        
        self.layers = nn.ModuleList([Transformer_block(d_model,num_heads,d_ff,rope_theta,context_length,device=device,dtype=dtype)
                                      for i in range(num_layers)])
        self.ln_final= RMSNorm(d_model,device=device,dtype=dtype)
        #经过多个transformer_blk升降维交叉融合一顿操作 得到(batch_size,seq_len,d_Model)
        #lm_head(output_embedding)映射回词表，然后进softmax得到概率
        self.lm_head = Linear(d_model,vocab_size)
                    #"batch_size,seq_len"
        #给每一层建立一个可学习的Q      创新架构初始化：接近base
                                                #layers个，每个都是d_model维度
        self.layers_queries = nn.ParameterList(
            #list内部的，也要先用parameter包裹
            #类名一般首字母大写Parameter
            [nn.Parameter(torch.zeros(d_model,dtype=dtype,device=device)) for _ in range(num_layers)]
            )
        self.d_model = d_model
        self.rmsnorm = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        
    def forward(self,token_ids:torch.Tensor,return_hidden_states=False)->torch.Tensor:

        


        x = self.token_embeddings(token_ids)
        #第一层直接用Embedding输出

        #不用self.不绑在模型身上
        #局部变量
        hidden_states = [x]


        
        for i,block in enumerate(self.layers):
            #分为第一层和非第一层
            if i==0:
                #输入
                h_in = hidden_states[0]
            else:

                

                Q = self.layers_queries[i]
                #
                #把前面的隐藏层全部拼起来:(batch_size,seq_len,d_model) -->(layers,batch_size,seq_len,d_model)
                #没有归一化的原始数据作为V
                V = torch.stack(hidden_states)

                #对K做稳定处理，防止很大的dominate?
                K = self.rmsnorm(V)

                #计算点积注意力(后两个维度):得到每个batch每个seq中每个token对于各层残差的打分(layers,batch_size,seq_len)
                                    #1维不用写1
                dot_product = einsum(K,Q,"l b s d,d -> l b s")
                #缩放+softmax得到注意力分数
                scaled_dot_product = dot_product / math.sqrt(self.d_model)
                attn_scores = softmax(scaled_dot_product,i = 0) #沿着层的维度进行softmax
                #(layers,batch_size,seq_len)
                #加权求和
                

                h_in = einsum(V,attn_scores,"l b s d,l b s -> b s d")
            
            h_out = block(h_in)

            hidden_states.append(h_out)





            




        
        final_out = hidden_states[-1]
        final_out = self.ln_final(final_out)

        logits = self.lm_head(final_out)
        #forward最后不加softmax,保留原始对数得分，后续处理
        #logits = softmax(x,-1)#softmax指定维度
        "batch_size,seq_len,vocab_size"

        if return_hidden_states:
            return logits, hidden_states
        else:
            return logits
        
