import torch

from torch import nn



class Embedding(nn.Module):#词表大小       隐藏层(特征)维度
    def __init__(self,num_embeddings:int,embedding_dim:int,device = None,dtype =None):
        super().__init__()
                                        #d_in d_out
                                        #embedding层：跟Linear底层 Y = xW^T，W是d_out d_in不同
                                        #不需要发生矩阵乘法，直接按照直觉d_in,d_out，方便查表
        empty_tensor = torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype)

        self.weight = nn.Parameter(empty_tensor)

        nn.init.trunc_normal_(self.weight,mean=0,std=1,a=-3,b=3) #a:下界，b：上界
            #传入的都是张量来参与运算了
    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
                #token_ids :(b,seq)'

                #过程：seq中每个token转为0-1 one-hot向量，然后再乘一个embedding矩阵(相当于查表)
                #相当于取出矩阵对应index(token即为索引数值)行。例如词表大小为10000，embedding就是（10000，512）
                #每个token转换后最后长度 = d_model，即是特征维度，例如长度为10000->512
                 #切片：[vocab_size,d_model][b,seq]传出：(b , seq, d_model) seq维度每个元素即为一个token，每个token对应d_model特征维度
        
        #关键：shape维度的命名：描述的是空间结构，大小，而不表征内部的储存！
        #例如sequence_len维度，里面可以存token，也可能存的是字节，字符，等等
        #想清楚内部究竟是什么。
        return self.weight[token_ids]