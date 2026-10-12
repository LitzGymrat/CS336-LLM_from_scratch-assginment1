import torch
import torch.nn as nn
import math
import einops
from einops import reduce,einsum,rearrange
                        #目标索引也是张量，有batch_size维度(batch_size,)每一个序列对应一个目标索引
def cross_entropy(o:torch.Tensor,target_indices:torch.Tensor)->torch.Tensor:
    #先进行公式推导，尽量抵消对数指数，以及LogSumExp保证数值稳定
    #输入的o是未经归一化的digits :(batch_size,...,seq_len,vocab_size)
    max_digit = o.max(dim=-1,keepdim=False).values#可以不保留维度运用广播机制？--(batch_size,seq)，最后维度一致
#不保留维度：
# o：         (batch, seq, vocab)
# max_digit： (batch, seq)
#无法相减：广播机制往左侧补成(1,batch, seq)无法对齐，需要(batch, seq,1)去相减-->(batch, seq, vocab)
    shifted_digits = o - rearrange(max_digit,"... -> ... 1") #做减法时补个1
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_digits),dim=-1,keepdim=False))
    # o[target_idx] :会从第一维去索引，而不是每一个batch的每个seq的词表中选出一个目标索引
    #某个位置应该取出：o[b, t, target_idx[b, t]]
    #1.统一indices和o维度大小:准备使用gather
    indices = rearrange(target_indices,"... -> ... 1") #(batch, seq,1);o:(batch, seq, vocab)

    target_logits = o.gather(dim=-1,index= indices)
    #输出形状和indices相同(batch, seq,1)
    #去掉临时添加的维度，和下面计算需要统一
    target_logits = rearrange(target_logits,"... 1 -> ...")
    loss = max_digit - target_logits + log_sum_exp #维度都为(batch_size,...,seq_len)
    #对着前面所有维度求平均
    avg_loss = torch.mean(loss)
    return avg_loss
