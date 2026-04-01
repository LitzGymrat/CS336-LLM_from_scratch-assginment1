import torch

import numpy as np

    #传入:numpy数组(用省内存的uint16类型存储)，后续进张量运算再调回int64标准整数索引
def get_batch(ids:np.ndarray,b_size:int,s_len:int,device = None):

    #1.随机抽b_size个起点(需要有效,index应该在len(ids)-2 -->len(ids)-2-s_len之间)
    total_len = len(ids)
    #生成随机数字的时候是左闭右开的

    #而且切片右端点是不包含的！！！！！！！
    max_idx = total_len - s_len     #大小为b_size的一维张量
    start = torch.randint(0,max_idx,size = (b_size,),device=device)
    #最大是max_idx - 1 = total_len - s_len -1
    #切片最大时i+s_len+1-1 = total_len -1
                            #而且切片右端点是不包含的！！！！！！！
    #2.将b_size个起点对应的ids[start,start+s_len]的序列抽出来，用stack组装成一个input张量
    input = torch.stack([torch.from_numpy(ids[i:i+s_len].astype(np.int64))for i in start]) #将numpy ndarray的数据类型转换
    #3.对应下一位的targetA：ids[start+1,start+s_len+1]
    target = torch.stack([torch.from_numpy(ids[i+1:i+s_len+1].astype(np.int64)) for i in start]) 

    #4.搬移数据
    input.to(device=device)
    target.to(device=device)
    #4.打包成元组返回

    return input,target
