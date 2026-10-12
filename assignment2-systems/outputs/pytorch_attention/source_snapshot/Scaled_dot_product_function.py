import torch
import torch.nn as nn
import math
import einops
from einops import reduce,einsum
from cs336_basics.Transformer.SoftMax_function import softmax
def scaled_dot_product_attention(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor|None = None):
    d_k = Q.shape[-1]
    #einops内部需要追踪维度，每个维度放进去的名字要唯一，不能出现两个seq_len
    scaled_dot_product = einsum(Q,K,"... queries_seq_len d_k , ... keys_seq_len d_k -> ... queries_seq_len keys_seq_len") / math.sqrt(d_k)
    #在softmax之前使用掩码：
    #mask为布尔值矩阵，if mask :存在歧义，用if mask is not None:判断是否传入
    #Boolean value of Tensor with more than one value is ambiguous
    if mask is not None:
        #mask = (~mask) * -math.inf #使用-无穷大表示掩码，传入softmax后softmax(-inf) = 0（取极限）
        #1 * -inf = -inf
        #0 * -inf = NaN ：0乘无穷大未定义，不应该用乘法
        #masked_fill(condition,value):condition为True的地方填入value:~进行取反——True的地方就是要填入-inf的地方
        #手写自己演绎：第n个query对第m（m>n）的k注意力分数应该为0！（m=n可以，可以关注自己）-->下三角矩阵
        #按条件填充的向量化操作，不需要通过乘法构造掩码
        
        scaled_dot_product = scaled_dot_product.masked_fill(~mask,-math.inf)
    attention_scores = softmax(scaled_dot_product,dim = -1)
    #输出根据注意力分数得到的V权重 :每个token位置都得到一组注意力分数 & V权重（但V此时仍然是高维向量）
    Output = einsum(attention_scores,V,"... queries_seq_len keys_seq_len, ... keys_seq_len d_v -> ... queries_seq_len d_v")
    return Output
