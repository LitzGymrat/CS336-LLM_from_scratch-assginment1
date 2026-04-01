import torch
def gradient_clipping(parameters:list[torch.Tensor],max_l2_norm:float,eps:float = 1e-6)->None:
    

    #需要计算所有参数构成的整体向量的L2范数，求出缩放因子。而不是单个参数
    #高维空间的一步梯度下降，每个参数对应一个维度的方向
    #如果每个维度不同缩放因子，最后整体的方向会改变
    #所以把参数都取出来，求出整体L2范数。算出统一的缩放因子

    #p.data:取出参数(在计算图内) 
    #使用p.grad.data取出底层数据，断开计算图。        #检查p.grad是否为None!而非.data。如果是None，没data。
    p_grads = [p.grad.data for p in parameters if p.grad is not None]

    if len(p_grads) == 0:
        return
    total_l2_norm_sq = 0
    

    for p_grad in p_grads: 
        #计算每个参数的L2范数（向量元素平方和的平方根）的平方，加起来
        total_l2_norm_sq += torch.sum(torch.square(p_grad))#使用.norm计算范数，整体计算，不需指定维度
    #整体的L2范数：sqrt(s1**2 + s2**2 +...)全部加起来开根
    global_l2_norm = torch.sqrt(total_l2_norm_sq)
    if global_l2_norm >= max_l2_norm:
        #缩放因子
        scaler = max_l2_norm / (eps + global_l2_norm)
        for p_grad in p_grads:
            #原地修改
            p_grad *= scaler
    else:
        pass

             

    
    
    