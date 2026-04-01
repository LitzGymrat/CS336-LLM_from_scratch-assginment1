import torch
from einops import einsum
from softmax_module.softmax import softmax
import math
def scaled_dot_product_attention(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor = None)->torch.Tensor:
    #K:(batch,...,n,d_k) Q:(barch,...,m,d_k)
    d_k = K.shape[-1]
    dot_product  = einsum(Q,K,"... n d_k,... m d_k -> ... n m")
    scaled_dot_product =  dot_product / (math.sqrt(d_k))
    #进softmax之前加一下掩码:把不要的换成很小的数，进softmax概率->近似0
    if mask is not None:
        scaled_dot_product.masked_fill_(~mask,-1e9)
    attn_scores = softmax(scaled_dot_product,-1) #应该是一个Q针对不同K发生打分，沿着最后一个维度？
                                    #n个q,m个k-->对应m个v-->最后打分结果是n--d_v。每个q对应不同v
    
                                    #打分结果：注意力分数加权融合值的特征，进入下一步传输
    result = einsum(attn_scores,V,"... n m,... m d_v -> ... n d_v")

    return result


