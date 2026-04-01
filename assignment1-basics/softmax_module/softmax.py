import torch
                        #对第i维度做softmax
def softmax(x:torch.Tensor,i:int)->torch.Tensor:
    #减去最大值，利于softmax数值稳定(不会有太大的放进去导致出现指数爆炸)

    #x[i]不是指第i维度！
    #需要计算梯度的forward阶段：慎用原地修改(需要根据计算图算梯度，不能把原来变量改了)
                        #返回的是元组。 使用keepdim便于广播

                        #不用取出来，直接对特定维度进行张量操作。 
    max_val = torch.max(x,dim=i,keepdim=True).values #是.values:属性不加括号。而非.values()从元组中取值而非调用函数
    #先调整数值，再运算
    shifted_x = x - max_val
    exp_x = torch.exp(shifted_x)
    sum_exp = torch.exp(shifted_x).sum(dim=i,keepdim=True)
    #减去最大值，利于softmax数值稳定(不会有太大的放进去导致出现指数爆炸)
    #前向传播过程：构建新的变量
    

    

    result = exp_x / sum_exp

    return result