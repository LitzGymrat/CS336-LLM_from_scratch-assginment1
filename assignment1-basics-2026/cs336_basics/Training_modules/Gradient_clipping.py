import torch
from collections.abc import Iterable
def gradient_clipping(parameters:Iterable[torch.nn.Parameter],max_l2_norm:float,eps:float = 1e-6):
    #parameters是生成器，内部通过yield逐个交出参数
    #遍历了一次就耗尽，所以要多次遍历的话需要自己创建一个list
    params = list(parameters) #给出已有参数的引用
    total_l2_norm_squared = 0
    #先算总体梯度的L2范数
    #不是针对每个参数！
    #统一缩放，保证缩放大小，方向不变）（如果每个参数缩放比例不同的话就会导致总体方向改变 ，比如3,4都被缩放到1；缩放到3/5,4/5
    for p in params: 
        if p.grad is None:#需要判断：实际中如果没发生反向传播或者冻结了参数，会出现Noned的情况
            continue
        else: #参与计算的是p.grad
            g = p.grad.data
            l2_norm_g = torch.linalg.norm(g) #l2范数：所有项平方加起来开根号
            total_l2_norm_squared += torch.square(l2_norm_g)#这算出的是平方和
    total_l2_norm = torch.sqrt(total_l2_norm_squared) #开根得到l2范数
    #分类处理截断
    if total_l2_norm < max_l2_norm:
        return
    else:
        down_scaler = max_l2_norm / (total_l2_norm + eps)#每一项都乘正数，总长度（l2范数）也乘正数-->如果加eps刚好，加了就是小一点
        #遍历参数缩放
        for p in params:
            if p.grad is None:#需要判断：实际中如果没发生反向传播或者冻结了参数，会出现None的情况
                continue
            else: #参与计算的是p.grad
                p.grad.data *= down_scaler #原地修改梯度
        return 