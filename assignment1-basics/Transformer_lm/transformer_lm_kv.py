import torch
from torch import nn
from RMSNorm_module.RMSNorm import RMSNorm
from embedding_module.embedding import Embedding


from Linear_module.linear import Linear
from transformer_blk_module.transformer_block_kv import Transformer_block_kv

class Transformer_lm_kv(nn.Module):
    def __init__(self,vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,device = None,dtype = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model,device=device,dtype=dtype)
        
        
        self.layers = nn.ModuleList([Transformer_block_kv(d_model,num_heads,d_ff,rope_theta,context_length,device=device,dtype=dtype)
                                      for i in range(num_layers)])
        self.ln_final= RMSNorm(d_model,device=device,dtype=dtype)
        #经过多个transformer_blk升降维交叉融合一顿操作 得到(batch_size,seq_len,d_Model)
        #lm_head(output_embedding)映射回词表，然后进softmax得到概率
        self.lm_head = Linear(d_model,vocab_size)
        self.device = device
        self.dtype= dtype
                    #"batch_size,seq_len"
    def forward(self,token_ids:torch.Tensor,past_key_value:list[tuple[torch.Tensor,torch.Tensor]] = None)->torch.Tensor:
        #记录这一次的kv

        present_key_value = []
        seq_len = token_ids.shape[-1] #token_ids：batch_size,seq_len。
        if past_key_value is None:
            #预填充阶段
            #存储每一层的K,V
            past_key_value = [None] * len(self.layers) #列表*长度
            #正常的位置索引
            token_positions = torch.arange(0,seq_len,dtype=torch.long,device=self.device)
        else:
            #decode阶段，只有一个Q
            #绝对位置索引，需要看下前面的K长度
            past_seq_len = past_key_value[0][0].shape[-2]
            token_positions = torch.tensor([past_seq_len],dtype=torch.long,device=self.device)
        
        x = self.token_embeddings(token_ids)

        


        for i,block in enumerate(self.layers): #传入对应层的KV
            x,cur_key_value = block(x,token_positions,past_key_value[i])
            present_key_value.append(cur_key_value)

        
        x = self.ln_final(x)

        logits = self.lm_head(x)
        #forward最后不加softmax,保留原始对数得分，后续处理
        #logits = softmax(x,-1)#softmax指定维度
        "batch_size,seq_len,vocab_size"

        return logits,present_key_value
        
