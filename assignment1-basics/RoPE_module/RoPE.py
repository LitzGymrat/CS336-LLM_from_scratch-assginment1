import torch
from torch import nn
from einops import rearrange

class RoPE(nn.Module): #先分头，然后对d_k = d_model / num_head使用RoPE，然后QK^T注意力
    def __init__(self, theta:float, d_k:int,max_seq_len:int,device = None):
        super().__init__()
        #1.计算频率:只取偶数即可:左开右闭;转成浮点数后续参与运算
        #一对一对旋转
        k = torch.arange(0,d_k,2,device=device).float()
        inv_freq = 1.0 / (theta ** (k / d_k)) #1 / 1.0区别如何？

        #2.构建绝对位置编码，以便构建旋转角度矩阵
        positions = torch.arange(0,max_seq_len,1,device=device).float()

        #3.通过外积构建旋转角度矩阵(max_seq_len,d_k / 2)后续根据绝对位置索引直接查表即可
        angles = torch.outer(positions,inv_freq)

        #4.存储cos和sin（使用self.register_buffer(persistent=False)作为模型的一部分，会搬到GPU上运算)
        self.register_buffer('cos_table',torch.cos(angles),persistent=False)#不在state_dict中
        self.register_buffer('sin_table',torch.sin(angles),persistent=False)
    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        #1.希望把x(... seq,d_k)->(... seq d_k/2 2)，最后一个维度切两部分-->产生对子
        #每个对子旋转：索引0和1分别作用。
                    #einops表达式里面不能出现算术表达式，只出现纯字母
        x_pair = rearrange(x,"... seq (d two) -> ... seq d two",two = 2) #表明一下维度
        x_1 = x_pair[...,0]
        x_2 = x_pair[...,1] #取最后一个维度索引

        #取出需要的角度
        angle_sin = self.sin_table[token_positions]
        angle_cos = self.cos_table[token_positions]
        #旋转
        x_1_new = x_1*angle_cos - x_2*angle_sin
        x_2_new = x_2*angle_cos + x_1*angle_sin #公式留意
        #x_new ：（... d_k / 2 ） ->(...d_k / 2,2 )
        #拼回去：stack(构建新的维度拼接) vs cat
        #接住，不是原地修改
        x_pair_new = torch.stack([x_1_new,x_2_new],dim=-1)
        #再调回原来的形状
                    #einops表达式里面不能出现算术表达式，只出现纯字母
        x_new = rearrange(x_pair_new,"... seq d two -> ... seq (d two)",two = 2)

        return x_new