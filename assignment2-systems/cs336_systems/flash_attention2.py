import torch
import einops
from einops import rearrange,einsum
import triton

class FlashAttention2_with_tiling(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,is_causal = False):
        batch_size,n_queries,d = Q.shape
        n_keys = K.shape[1]
        device = Q.device
        dtype = Q.dtype
        #创建结果张量（因为分块计算是逐步填入的，先创建好）
        O = torch.empty_like(Q)
        #保存的logsumexp储存分母信息
        L = torch.empty((batch_size,n_queries),device = device,dtype = dtype)
        #确定分块大小
        Bq = 16
        Bk = 16
        ctx.Bq = Bq
        ctx.Bk = Bk
        #样本循环:每个样本的Q只针对那个样本的KV进行计算
        for b in range(batch_size):
            for q_start in range(0,n_queries,Bq): #起始点从0开始，每次增加Bq：从0到n_queries，终点不包含。只到n_queries-Bq
                q_block = Q[b,q_start:q_start+Bq,:] #取出相应块
                #创建用于在线计算的张量
                #第一维为Bq：针对每一行
                #累计对于各key的加权输出
                o_acc = torch.zeros((Bq,d),device = device,dtype = dtype)
                #记录分母
                l = torch.zeros((Bq,),device = device,dtype = dtype)
                #记录最大分数
                m = torch.full((Bq,),float("-inf"),device = device,dtype = dtype)
                #来取出k,v块

                for k_start in range(0,n_keys,Bk): #起始点从0开始，每次增加Bk
                    k_block = K[b,k_start:k_start+Bk,:] #取出相应块
                    v_block = V[b,k_start:k_start+Bk,:]
                    #计算q_block和k_block的注意力分数
                    s_block = einsum(q_block,k_block,"Bq d,Bk d -> Bq Bk") / (d ** 0.5) #缩放
                    #求出每个query对应各个K的注意力分数最大值（沿着列计算，dim = -1）,形状(Bq,)
                    #和原来比较得到更新最大值
                    m_new = torch.maximum(m,torch.amax(s_block,dim=-1))
                    #计算在线更新所需的值
                    #1.修正系数
                    alpha = torch.exp(m-m_new)
                    #2,本块的未归一化权重
                    #(Bq,Bk) - (Bq,1) --> (Bq,Bk)：利用广播机制，按行作用，作用到每列的每个元素
                    p_block = torch.exp(s_block-m_new[:,None])
                    #按行求和
                    p_block_sum  = torch.sum(p_block,dim = -1) #(Bq,)
                    #计算该块的v加权输出
                    o_block = einsum(p_block,v_block,"Bq Bk,Bk d -> Bq d") 
                    #更新累计值
                    l = l * alpha + p_block_sum
                    #o_acc为Bq,d;alpha默认为Bq，加个维度做广播机制
                    o_acc = o_acc * alpha[:,None] + o_block
                    m = m_new
                #退出K循环：针对这个query块的计算都完成，最后转换累计值为所需结果
                #算出这一块的结果
                #这一块归一化输出
                o_block = o_acc / l[:,None] #除以归一化分母
                #这一块累计分母：包含m和原来分母信息，方便通过注意力分数直接恢复概率
                l_block = m + torch.log(l)
                #写回最终结果
                O[b,q_start:q_start+Bq,:] = o_block
                L[b,q_start:q_start+Bq] = l_block
        #完成计算，保存反向所需数据
        ctx.save_for_backward(Q,K,V,O,L)
        return O
    def backward(ctx,grad_output):
        raise NotImplementedError


