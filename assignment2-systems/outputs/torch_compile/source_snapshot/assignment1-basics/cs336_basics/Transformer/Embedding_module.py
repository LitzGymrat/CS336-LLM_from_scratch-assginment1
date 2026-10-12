import torch
import torch.nn as nn
import math
import einops
class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int, device:torch.device|None = None, dtype:torch.dtype | None =None):

        super().__init__()
        self.d_in = num_embeddings
        self.d_out = embedding_dim
        self.device = device
        self.dtype = dtype
       
                                                        #设备和数据类型传给张量创建操作：而非张量填充操作
        #embedding：不同于前面的权重做矩阵乘法，这里需要的是单个token-->映射到d_model维度的向量，直接查行即可，所以权重的第一维度是词表大小
        weights = torch.nn.init.trunc_normal_(torch.empty(num_embeddings,embedding_dim,device = device,dtype = dtype),mean = 0,std = 1,a=-3,b=3)
        self.weights = nn.Parameter(weights)

    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        W = self.weights                                     #省略前面维度
        return W[token_ids]
