from tokenizer.tokenizer import Tokenizer
from Transformer_lm.transformer_lm import Transformer_lm
from softmax_module.softmax import softmax

import torch

def generate(prompt:str,max_tokens:int,temperature:float,top_p:float)->str:
    vocab_path = "C:\\Users\\MilesZhu\\CS336-assignment1\\tinystory_train_vocab.pkl"
    special_tokens = ["<|endoftext|>"]

    merges_path = "C:\\Users\\MilesZhu\\CS336-assignment1\\tinystory_train_merges.pkl"
            #直接从类建立：不需要()
    tokenizer = Tokenizer.from_files(vocab_path,merges_path,special_tokens)
    
    vocab_size = 10000
    
    context_length=256
    d_model=512
    num_layers=4
    num_heads=16
    d_ff=1344
    rope_theta=10000
    device='cuda' if torch.cuda.is_available() is True else 'cpu'
    dtype=torch.float32
    #初始化模型
    model = Transformer_lm(vocab_size,context_length,d_model,num_layers,
                                    num_heads,d_ff,rope_theta,device = device,dtype=dtype)
    model.to(device)
    #加载权重
    state_dict = torch.load("path",map_location=device)

    model.load_state_dict(state_dict["model"])
    #开启评估模式，dropout等情况会调整
    model.eval()
    #编码
    input_ids = tokenizer.encode(prompt)  

    output = prompt
    

    next_token = ""
    #关掉梯度记录
    with torch.no_grad():
        while next_token != "<|endoftext|>" and len(input_ids) <= max_tokens:
            #得到输出词表logits


            #生成的时候，一般batch_size = 1

            #模型吃张量：将列表转为张量,[]增加一个batch_size维度？
            x = torch.tensor([input_ids],dtype=torch.long,device=device)
            logits = model(x)
            #得到概率
            #batch_size,seq_len,vocab_size
            prob = softmax(logits / temperature,dim = -1)
            #只需要取出末尾token对应概率
            next_token_prob = prob[:,-1,:]
            #batch_size=1,vocab_size
            #top_p采样


            #同时对索引(要用于取词)和概率进行排序
            sorted_prob,sorted_indices = torch.sort(next_token_prob,descending=True)


            #计算累计概率
            cum_prob = torch.cumsum(sorted_prob)

            
            #找到截断点，累积概率大于top_p
            trunc_idx = 0
            #数组里面的已经是累计概率了，不需要再累加
            for i,prob_cum in enumerate(cum_prob[-1]):
                
                if prob_cum > top_p:
                    trunc_idx = i
                    break
            #重新归一化
            top_p_prob = sorted_prob[0:trunc_idx+1]
            top_p_indices = sorted_indices[0:trunc_idx+1]

            top_p_sum = top_p_prob.sum()
            for p in top_p_prob:
                p /= top_p_sum

            #索引抽样:              #按照概率抽样：传入归一化后的概率
            sample_token_id = torch.multinomial(top_p_prob,1)
            #映射回去
            next_token_id = top_p_indices[sample_token_id]

            #取出tensor的数值，变回Python int
            next_token_id_val = next_token_id.item()
            #生成下个token
            next_token = tokenizer.decode(next_token_id_val)
            
            #接上去
            output += next_token
            #input_ids = input_ids.append(next_token_id)
            input_ids.append(next_token_id)#append原地修改
    return output





    

    