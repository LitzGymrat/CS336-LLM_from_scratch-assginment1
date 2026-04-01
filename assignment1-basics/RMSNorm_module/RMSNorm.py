import torch
from torch import nn

class RMSNorm(nn.Module):
    #对每一个特征向量(维度为d_model)进行normalize
    def __init__(self,d_model:int,eps:float = 1e-5 ,device = None,dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        #初始化参数维度：想清楚作用的对象
        #gain作用于每个向量的每个隐藏层维度，如(512,)
        #初始化之前的dtype保护
        if dtype is None:
            dtype = torch.float32
            #torch.tensor(data)：将数据转为张量类型！
            #torch.ones/zeros/empty传入的才是维度！
        empty_tensor = torch.empty(d_model,device=device,dtype=dtype)
        
        self.weight = nn.Parameter(empty_tensor)

        #dtype保护
        

        #是用nn.init初始化
        nn.init.ones_(self.weight)
    def forward(self,x:torch.Tensor):




        #传入：batchsize,seq_len,d_model
        #计算前先转为f32，防止数值下溢
        intype = x.dtype
        x = x.to(torch.float32)
        #一步一步算
        x_sq = torch.square(x)
        mean_sq = x_sq.mean(dim = -1,keepdim=True) #keepdim把对应维度保留为1，方便后续广播
        #(B,T,1) 不是mean_sq / self.d_model，就是mean_sq。求和然后再除以数量
        rms = torch.sqrt((mean_sq) + self.eps)


        #(B,T,D) / (B,T,1)自动广播
        result = self.weight * x / rms

        return result.to(intype)
    