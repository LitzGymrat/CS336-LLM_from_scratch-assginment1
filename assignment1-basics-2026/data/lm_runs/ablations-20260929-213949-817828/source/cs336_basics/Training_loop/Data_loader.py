import torch
import numpy as np
from numpy.typing import NDArray
from torch.types import Device
def data_loader(token_ids:NDArray[np.int32],batch_size:int,seq_len:int,device:Device)->tuple[torch.Tensor]:
    #随机起始位置
    total_len = token_ids.size #NDArray.size返回总元素个数
    #需要sample batch_size个batch，1.批量取出 2.从0索引开始

    #差一错误！原式式 1<=i<=n-m，在python中由于是0-idx，从0开始，加上左闭右开,最后应该是0 <= i <= total_len-seq_len-1，写进去是[0,total_len-seq_len)
    sampled_starts = np.random.randint(0,total_len-seq_len,size = batch_size) #返回numpy数组。
        #torch.stack：沿着特定维度拼接多个张量                                      数据存储的类型不一定支持索引（embedding中要查表）统一转成torch.long
    sampled_data = torch.stack([torch.tensor(token_ids[i:i+seq_len],device = device,dtype = torch.long) for i in sampled_starts],dim=0)#搬运到相应设备
    sampled_next_token = torch.stack([torch.tensor(token_ids[i+1:i+seq_len+1],device = device,dtype = torch.long) for i in sampled_starts],dim=0)
    return sampled_data,sampled_next_token #（B，S）