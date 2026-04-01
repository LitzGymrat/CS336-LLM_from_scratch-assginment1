import torch
import math
class AdamW(torch.optim.Optimizer):
            #传入待学习参数和超参数 元组：（）有序不可变。集合{}无序，不可用索引取值
    def __init__(self,params,lr = 1e-3,betas = (0.9,0.99), eps = 1e-5,weight_decay = 0.01):
        #拦截一下非法参数
        if lr <= 0:
            raise ValueError(f"invalid learning rate{lr}")
        #m ←β 1 m +(1 −β 1 ) g ..beta为1的话，梯度就不参与了，无意义。
        if not 0<= betas[0] < 1 or not 0<= betas[1] < 1: #两个都要写not
            raise ValueError(f"invalid beta at index 0 or 1{betas}")
        
        
        defaults = {"lr":lr,"betas":betas,"eps":eps,"weight_decay":weight_decay}
        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()

        #每个参数组分别处理
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
        
            #group是超参数，待学习参数的字典
            for p in group["params"]:
                #跳过无梯度者
                if p.grad is None:
                    continue
                #启动运算部分
                #运算中需要记录迭代的状态：使用state
                state = self.state[p]

                #动量是张量！需要严格初始化，和p.data的形状对应.同设备

                #状态为空，进行初始化
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]
                t = state["t"]

                #梯度
                grad = p.grad.data
                beta1 = betas[0]
                beta2 = betas[1]
                #一阶动量
                m = (1-beta1)*grad + beta1*m
                v = (1-beta2)*(grad**2) + beta2*v

                #调整学习率，修正偏差(一开始动量太小)
                corrected_lr = lr*(math.sqrt(1-beta2**t) / (1-beta1**t))

                #原地-=更新参数
                p.data -= corrected_lr*(m / (eps + torch.sqrt(v))) #张量使用torch运算
                #权重衰减
                p.data -= lr*weight_decay*p.data

                #更新状态迭代参数

                state["m"] = m
                state["v"] = v
                state["t"] = t+1
                


        return loss