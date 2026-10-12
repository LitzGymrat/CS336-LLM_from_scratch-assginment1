import triton
import torch
import triton.language as tl
import einops
from einops import reduce,einsum

@torch.compile
def flash_backward(Q, K, V, O, grad_output, L, is_causal):
    #1.恢复P,P = exp(S - L)
    #恢复S
    dO = grad_output
    d = Q.shape[-1]
    S = einsum(Q.float(),K.float(),"... Nq d,... Nk d -> ... Nq Nk") / (d ** 0.5)
    if is_causal:
            #创建新张量，移动设备s
            q_positions = torch.arange(0,Q.shape[-2],device=S.device) #(Bq,)
            k_positions = torch.arange(0,K.shape[-2],device=S.device) #(Bk,)
            #得到两者的位置，进行比较得到掩码矩阵
            allowed = q_positions[:,None] >= k_positions[None,:] #(Bq,1)和（1，Bk）广播比较-->(Bq,Bk)
            S = torch.where(allowed,S,S-(1e6))
    P = torch.exp(S - L[:,:,None]) #L :(B,Nq,)
    dV = einsum(P.to(dO.dtype),dO,"... Nq Nk,... Nq d -> ... Nk d")
    dP = einsum(dO.float(),V.float(),"... Nq d,... Nk d -> ... Nq Nk")
    D = torch.sum(O.float() * grad_output.float(),dim = -1,keepdim = True)
    dS = einsum(P,dP-D,"... Nq Nk,... Nq Nk -> ... Nq Nk")
    dQ = einsum(dS.to(K.dtype),K,"... Nq Nk,... Nk d-> ... Nq d") / (d ** 0.5)
    dK = einsum(dS.to(Q.dtype),Q,"... Nq Nk,... Nq d -> ... Nk d")/ (d ** 0.5)
    return dQ,dK,dV
@triton.jit
def flash_fwd_kernel(

    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal:tl.constexpr,
):
        # Program indices
        #一个program负责一个query块，和所有KV计算
        #启动的时候会用grid = (Tq,batch_size)启动kernel
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    q_start = query_tile_index * Q_TILE_SIZE #q起始索引位置
    #描述这个kernel需要读取/写入的块
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
        )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_start,0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
        )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(q_start,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
        )
    #读取当前program对应Q块
    q_block = tl.load(Q_block_ptr)
    #创建累积量
    o_acc = tl.zeros((Q_TILE_SIZE,D),dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,),dtype = tl.float32)
    m = tl.full((Q_TILE_SIZE,),float("-inf"),dtype=tl.float32)
    #进入key块循环：读取当前块的K和V
    for i in range(tl.cdiv(N_KEYS,K_TILE_SIZE)):
        k_block = tl.load(K_block_ptr,boundary_check=(0,1),padding_option = "zero")
        v_block = tl.load(V_block_ptr,boundary_check=(0,1),padding_option = "zero")
        #计算注意力分数
        s_block = tl.dot(q_block,tl.trans(k_block)) * scale
        if is_causal:
            q_positions = q_start + tl.arange(0,Q_TILE_SIZE) #(Bq,)
            k_positions = i*K_TILE_SIZE + tl.arange(0,K_TILE_SIZE) #(Bk,)
            #得到两者的位置，进行比较得到掩码矩阵
            allowed = q_positions[:,None] >= k_positions[None,:] #(Bq,1)和（1，Bk）广播比较-->(Bq,Bk)
            s_block = tl.where(allowed,s_block,s_block-(1e6))
        #算新最大值
        m_new = tl.maximum(m,tl.max(s_block,axis=1))
        #修正系数
        alpha = tl.exp(m - m_new)
        p_block = tl.exp(s_block-m_new[:,None])
                    #按行求和
        p_block_sum  = tl.sum(p_block,axis = -1) #(Bq,)
                    #计算该块的v加权输出
                    #dotY要求相同数据类型
                    #p_block由.dot返回，默认是FP32
                    #转一下数据类似
        p_block_v_type = p_block.to(v_block.dtype)
        
                    #更新累计值
        l = l * alpha + p_block_sum
                    #o_acc为Bq,d;alpha默认为Bq，加个维度做广播机制
        o_acc = o_acc * alpha[:,None]
        #累加使用acc :D = AB+C
        o_acc = tl.dot(p_block_v_type,v_block,acc = o_acc)
        m = m_new
        #移动K,V块指针
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE,0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE,0))
    #退出循环计算最终结果

    o_block = o_acc / l[:,None]
    o_block = o_block.to(O_ptr.type.element_ty)#累计的输出：采用外部张量类型
    l_block = m + tl.log(l)
    tl.store(O_block_ptr,o_block,boundary_check=(0,1))
    tl.store(L_block_ptr,l_block,boundary_check=(0,)) #0:整数；0,：包含一个元素的元组


class FlashAttention2_with_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V,is_causal=False):
        batch_size,n_queries,d = Q.shape
        n_keys = K.shape[1]
        device = Q.device
        dtype = Q.dtype
        #分配输出张量\
        O = torch.empty_like(Q)
        #保存的logsumexp储存分母信息
        L = torch.empty((batch_size,n_queries),device = device,dtype = torch.float32)
        Bq = 16
        Bk = 16
        ctx.Bq = Bq
        ctx.Bk = Bk
        #启动Tq * batch_size个program
        grid = (triton.cdiv(n_queries,Bq),batch_size)
        scale = 1 / (d**0.5)
        flash_fwd_kernel[grid](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        n_queries, n_keys,
        scale,
        D=d,
        Q_TILE_SIZE=Bq,
        K_TILE_SIZE=Bk,
        is_causal = is_causal,
)
        ctx.save_for_backward(Q,K,V,O,L)
        ctx.causal = is_causal
        #用Q,K重算注意力分数需要掩码。
        return O
    def backward(ctx,grad_output):
        Q,K,V,O,L = ctx.saved_tensors
        is_causal = ctx.causal
        dQ,dK,dV = flash_backward(Q,K,V,O,grad_output,L,is_causal)
        return dQ,dK,dV,None #和前向按照位置对应，Pytorch会沿着传播梯度
        raise NotImplementedError
