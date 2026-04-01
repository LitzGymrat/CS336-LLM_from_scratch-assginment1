import torch        #(batch_size,seq_len,vocab_size)原始对数得分
def cross_entropy(logits:torch.Tensor,targets:torch.Tensor): #(batch_size,seq_len)

    #先求最大值，后续做logsumexp维护数值稳定
    max_logit = logits.max(dim = -1,keepdim=True).values#max返回的是
    #交叉熵实际就是softmax取负对数
    #但是为了数值稳定，展开计算。-->logsumexp 减去目标对数
    log_sum_exp =  max_logit + torch.log(torch.exp(logits - max_logit).sum(dim=-1,keepdim=True))

    #用gather取出目标的对数得分                 使用unsqueeze增加维度对齐（要求index维度对齐输出shape
    target_score = torch.gather(logits,dim=-1,index = targets.unsqueeze(-1))


    loss =  log_sum_exp - target_score

    return loss.mean() #返回scale loss

