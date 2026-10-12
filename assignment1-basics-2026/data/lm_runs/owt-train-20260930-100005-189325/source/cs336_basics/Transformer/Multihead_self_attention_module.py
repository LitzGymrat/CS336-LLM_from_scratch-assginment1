import torch
import torch.nn as nn
import math
import einops
from jaxtyping import Int
from einops import reduce,einsum,rearrange
from cs336_basics.Transformer.Scaled_dot_product_function import scaled_dot_product_attention
from cs336_basics.Transformer.RoPE_module import RoPE
class Multihead_self_attention(nn.Module):
    #d_ff应该可传入。如果没传入就算
    def __init__(self, d_model:int, num_heads:int,theta:float,max_seq_len:int,device:torch.device|None = None, dtype:torch.dtype | None =None,
                 use_rope:bool = True):

        super().__init__()
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        std = math.sqrt(2/(d_model + d_model))                                                 #设备和数据类型传给张量创建操作：而非张量填充操作
        q_proj = torch.nn.init.trunc_normal_(torch.empty(d_model,d_model,device = device,dtype = dtype),mean = 0,std = std,a=-3 * std,b=3*std)
        k_proj = torch.nn.init.trunc_normal_(torch.empty(d_model,d_model,device = device,dtype = dtype),mean = 0,std = std,a=-3 * std,b=3*std)
        v_proj = torch.nn.init.trunc_normal_(torch.empty(d_model,d_model,device = device,dtype = dtype),mean = 0,std = std,a=-3 * std,b=3*std)
        o_proj = torch.nn.init.trunc_normal_(torch.empty(d_model,d_model,device = device,dtype = dtype),mean = 0,std = std,a=-3 * std,b=3*std)
        self.q_proj = nn.Parameter(q_proj)
        self.k_proj = nn.Parameter(k_proj)
        self.v_proj = nn.Parameter(v_proj)
        self.o_proj = nn.Parameter(o_proj)
        self.d_head = d_model // num_heads #单斜杠除法：结果为浮点数。使用双斜杠除法，如果两个数都是整数，结果为向下取整的整数
        self.num_heads = num_heads
        #self.rope = RoPE(theta,d_model,max_seq_len,device,dtype)
        #分头的rope:传入的维度应为d_head
        self.use_rope = use_rope
        if use_rope:
            self.rope = RoPE(theta,self.d_head,max_seq_len,device,dtype)
        #self.token_positions = token_positions ：不作为必填参数，
                            #位置：适合根据当前输入指定，同一个MSA对象可先后接受不同token位置序列进行运算
                            #可以不传入，使用x.shape[-2]生成默认位置
    def forward(self,x:torch.Tensor,token_positions:Int[torch.Tensor, "..."]|None = None) -> torch.Tensor:
        W_Q = self.q_proj
        W_K = self.k_proj
        W_V = self.v_proj
        W_O = self.o_proj

        
        #Q,K,V可以合并为一次矩阵乘法。
        #Q = einsum(W_Q,x,"(num_heads d_head) d_model,... seq_len d_model -> ... num_heads seq_len d_head")
        #K = einsum(W_K,x,"(num_heads d_head) d_model,... seq_len d_model -> ... num_heads seq_len d_head")
        #V = einsum(W_V,x,"(num_heads d_head) d_model,... seq_len d_model -> ... num_heads seq_len d_head")
        #einsum处理已经存在的维度，负责线性代数，不负责维度转换，所以无法用einsum一次性完成

        #einsum不出现像是 “(num_heads d_head)”这样的pattern表达，这种pattern用于shape rearrangement,不支持于einsum，改用q_dim这类
        Q = einsum(W_Q,x,"q_dim d_model,... seq_len d_model -> ... seq_len q_dim")
        K = einsum(W_K,x,"k_dim d_model,... seq_len d_model -> ... seq_len k_dim")
        V = einsum(W_V,x,"v_dim d_model,... seq_len d_model -> ... seq_len v_dim")
        #分为矩阵乘法和拆分成每个头调整维度两步
        Q = rearrange(Q," ... seq_len (num_heads d_head) ->  ... num_heads seq_len d_head",num_heads = self.num_heads)
        K = rearrange(K," ... seq_len (num_heads d_head) ->  ... num_heads seq_len d_head",num_heads = self.num_heads)
        V = rearrange(V," ... seq_len (num_heads d_head) ->  ... num_heads seq_len d_head",num_heads = self.num_heads)
        #Q和K需要携带位置信息：使用RoPE：注意力分数也应该受到相对位置影响。
        #V可以不用携带，Q，K已经带上
        #seq_len = Q[-2] :这样子取出来的是沿着第0维度的倒数第二个张量！需要使用.shape取出维度大小
        seq_len = Q.shape[-2] 
        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len,device = x.device) #默认从0开始生成默认位置
            Q = self.rope.forward(Q,token_positions)
            K = self.rope.forward(K,token_positions)
        #构造掩码矩阵
        
        ones = torch.ones(seq_len,seq_len,dtype = bool,device = x.device) #在forward里面构建配合输入的临时张量：一般配合输入的设备/精度：使用x.device
        mask = ~torch.triu(ones,diagonal = 1) #构造包含对角线的下三角掩码矩阵：先构造全1，然后讲上部分取出来1，然后整个取反
        #实施缩放点积注意力
        head_outputs = scaled_dot_product_attention(Q,K,V,mask)
        #调回按照token的排列--将不同头的结果组合回来
        merged  = rearrange(head_outputs,"... num_heads seq_len d_head -> ... seq_len (num_heads d_head)")

        #刚刚不同头是分开算的，这里最后一个o_proj再混合进行一次运算。
        result = einsum(W_O,merged," d_model_out d_model_in ,... seq_len d_model_in -> ... seq_len d_model_out") #留意轴的名称不重复！
                           #
        return result
