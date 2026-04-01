from tokenizer.tokenizer import Tokenizer
from Transformer_lm.transformer_lm import Transformer_lm
from softmax_module.softmax import softmax

import torch

def generate(prompt:str,max_tokens:int,temperature:float,top_p:float)->str:
    vocab_path = "C:\\Users\\MilesZhu\\CS336-assignment1\\tinystory_train_vocab.pkl"
    special_tokens = ["<|endoftext|>"]
    #指定权重路径
    ckpt_path = ""
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
    state_dict = torch.load(ckpt_path,map_location=device)

    model.load_state_dict(state_dict["model"])
    #开启评估模式，dropout等情况会调整
    model.eval()
    #编码
    input_ids = tokenizer.encode(prompt)  

    output = prompt
    #取出终止符的编码，方便比较
    end_bytes = "<|endoftext|>".encode()
    end_token_id = tokenizer.vocab_inv[end_bytes]
    
    
    #关掉梯度记录
    with torch.no_grad():
        
        while len(input_ids) <= max_tokens:
            #得到输出词表logits


            #生成的时候，一般batch_size = 1

            #模型吃张量：将列表转为张量,[]增加一个batch_size维度？
                    #截取最后context_len长度
            x = torch.tensor([input_ids[-context_length:]],dtype=torch.long,device=device)
            logits = model(x)
            #得到概率
            #batch_size,seq_len,vocab_size
            
            #只需要取出末尾token对应概率
            #先取出来，再做softmax
            next_token_logit = logits[:,-1,:]
            
            #特殊处理温度为0：直接输出最大者
            if temperature == 0:
                next_token_id = torch.argmax(next_token_logit,dim = -1)
            else:
                next_token_prob = softmax(next_token_logit / temperature,dim = -1)
                #batch_size=1,vocab_size
                #top_p采样

                #同时对索引(要用于取词)和概率进行排序
                sorted_prob,sorted_indices = torch.sort(next_token_prob,descending=True)


                #计算累计概率,维度指定，否则会展平成一维向量

                cum_prob = torch.cumsum(sorted_prob,dim=-1)

                
                token_to_remove = cum_prob > top_p
                #第一次超过top_P的词保留，所以要将波尔数组右移一位 .clone()深拷贝一份
                token_to_remove[:,1:] = token_to_remove[:,:-1].clone()#切片左闭右开

                #特殊处理第一个
                token_to_remove[:,0] = 0

                #魔法索引
                sorted_prob[token_to_remove] = 0.0

                

                #归一化,直接广播机制运算，避免低效for循环
                sorted_prob = sorted_prob / sorted_prob.sum(dim=-1,keepdim=True)

                #索引抽样:              #按照概率抽样：传入归一化后的概率
                sample_idx = torch.multinomial(sorted_prob,1)
                #传回的是(1,1)张量，不能直接取，使用torch.gather
                #映射回去                       #(1,vocab_size)
                next_token_id = torch.gather(sorted_indices,dim=-1,index=sample_idx)

            #取出tensor的数值，变回Python int
            next_token_id_val = next_token_id.item()
            
            
            
            
            #input_ids = input_ids.append(next_token_id)
            input_ids.append(next_token_id_val)#append原地修改
            if next_token_id_val == end_token_id:
                break

        #最后一次性decode，因为可能有多个token构成一个汉字等等
    output = tokenizer.decode(input_ids)
    return output





    

    