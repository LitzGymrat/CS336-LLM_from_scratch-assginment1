import torch
import torch.nn as nn
import math
import einops
from einops import reduce,einsum
class RoPE(nn.Module):
    def __init__(self, theta:float, d_k:int,max_seq_len: int, device:torch.device|None = None, dtype:torch.dtype | None =torch.float32):

        super().__init__()
        self.device = device
        self.dtype = dtype
        #旋转的角度表只和位置有关，可以复用
        #算出一个角度表：形状：(max_seq_len,d_k // 2)每个token位置的高维向量有d_k//2组
        angles_table = torch.zeros(max_seq_len,d_k//2,device = device,dtype = dtype)#创建张量时配置数据类型
        for token_pos in range(max_seq_len):
            #每个token位置--高维向量每个成对pair
            for pair_idx in range(d_k // 2):
                
                power_theta = 2*pair_idx / d_k
                omega = 1 / (theta**power_theta)
                Phi = omega * token_pos
                angles_table[token_pos][pair_idx] = Phi
        #再存为余弦，正弦
        cos_table = torch.cos(angles_table)
        sin_table = torch.sin(angles_table)
        #使用register_buffer登记两个表为这个模块需要保存的数据(统一管理，device也会一起移动)，但不是可学习参数
        self.register_buffer("cos_table",cos_table,persistent = False)#不登记到state_dict()：保存加载模型状态时使用的字典，不占用相关模型存档空间（RoPE的表可以重新算
        self.register_buffer("sin_table",sin_table,persistent = False)


    def forward(self,x:torch.Tensor,token_positions:torch.Tensor) -> torch.Tensor:
        # x : ... seq_len,d_model ; token_positions: ...,seq_len
        #完整的旋转大矩阵虽然也能算，但里面大部分是零。
        # 对于一个 d_k 维向量，普通稠密矩阵乘法需要约 \(O(d_k^2)\) 运算，
        # 而这里成对的逐元素计算只需要 \(O(d_k)\)。逐 pair 的 Python 循环也是 \(O(d_k)\)，
        # 张量写法主要消除了逐个执行 Python 操作的开销。
        #RoPE里面默认用了float32，可以处理一下数据类型问题
        in_dtype = x.dtype
        a = x[...,0::2]#从0开始每隔两个单位取一个（左闭右开，实际上中间隔了一个单位）
        b = x[...,1::2]#从1开始每隔两个单位取一个
        cos_values = self.cos_table[token_positions] #取值用方括号
        sin_values = self.sin_table[token_positions]
        a_rotated = a * cos_values - b * sin_values
        b_rotated = b * cos_values + a * sin_values                                 #3.复原组装：交错放入，然后展平，将pair和component位置调换--》调换维度实现交错--〉展平
        result = einops.rearrange([a_rotated,b_rotated],"component ... pair -> ... (pair component)") #1.由于分了组，最后一个维度是pair

        return result.to(in_dtype)                                           #2.中间可能还有batch_size,seq_len，但是只看第一个（列表传入的时候增加的维度）和最后一个维度Pair
