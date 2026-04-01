import torch
from torch import nn
from RMSNorm_module.RMSNorm import RMSNorm
from embedding_module.embedding import Embedding
from transformer_blk_module.transformer_block import Transformer_block

from Linear_module.linear import Linear

class Transformer_lm(nn.Module):
    def __init__(self,vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,device = None,dtype = None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model,device=device,dtype=dtype)
        
        
        self.layers = nn.ModuleList([Transformer_block(d_model,num_heads,d_ff,rope_theta,context_length,device=device,dtype=dtype)
                                      for i in range(num_layers)])
        self.ln_final= RMSNorm(d_model,device=device,dtype=dtype)
        #经过多个transformer_blk升降维交叉融合一顿操作 得到(batch_size,seq_len,d_Model)
        #lm_head(output_embedding)映射回词表，然后进softmax得到概率
        self.lm_head = Linear(d_model,vocab_size)
                    #"batch_size,seq_len"
    def forward(self,token_ids:torch.Tensor,return_hidden_states = False)->torch.Tensor:
        x = self.token_embeddings(token_ids)
        hidden_states = [x]

        for block in self.layers:
            x = block(x)
            hidden_states.append(x)
        
        x = self.ln_final(x)

        logits = self.lm_head(x)
        #forward最后不加softmax,保留原始对数得分，后续处理
        #logits = softmax(x,-1)#softmax指定维度
        "batch_size,seq_len,vocab_size"

        if return_hidden_states:
            return logits, hidden_states
        else:
            return logits
        
