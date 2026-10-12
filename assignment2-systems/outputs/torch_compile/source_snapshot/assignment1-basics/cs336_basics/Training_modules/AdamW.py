import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),weight_decay=0.01,eps=1e-8):
        defaults = {"lr":lr,"betas":betas,"weight_decay":weight_decay,"eps":eps} #交给父类optimizer的默认参数，放到参数组中
        super().__init__(params,defaults)
    def step(self,closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            #一个参数组对应一组超参数（可能不同参数组对应不同权重对应不同超参数，相应的特殊超参数在传入时接受）
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]: #参数组中不只要优化的权重，把权重取出来

                #如果没有就跳过
                if p.grad is None:
                    continue

                state:dict = self.state[p] #取出p的状态
                
                #m = state.get("m",torch.zeros(p.shape))#创建新张量和原来参数的设备和类型相同：使用zeros_like
                #v = state.get("v",torch.zeros(p.shape))python会先计算函数实参，这样会每次先创建零张量
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                #给m,v赋值
                m = state["m"]
                v = state["v"]
                #取出/初始化一下一二阶动量（随着p迭代而变化，是stateful，存在state）：要和原参数加减，形状相同
                t = state.get("t",1) #
                grad = p.grad.data
                #根据当前迭代次数调整学习率
                beta_1 = betas[0]
                beta_2 = betas[1]

                                                    #除数内部有减法：要放括号
                lr_t = lr*(math.sqrt(1-beta_2**t) / (1-beta_1**t))#^:位运算符。指数运算使用**
                #weight_decay :使用-=原地修改权重，不然就创建新张量了
                p.data -= lr*weight_decay*p.data
                #更新两个动量
                m = beta_1 * m + (1-beta_1) * grad
                v = beta_2 * v + (1-beta_2) * grad**2
                p.data -= lr_t * (m / (torch.sqrt(v) + eps)) #张量开根号用torch.sqrt()
                #更新状态
                state["m"] = m
                state["v"] = v
                state["t"] = 1+t
                
        return loss
        