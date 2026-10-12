import torch
from cs336_basics.Transformer.Transformer_lm import Transformer_lm
from cs336_basics.Transformer.SoftMax_function import softmax
def decoding(in_tokens:list[int],lm:Transformer_lm,temp:float,top_p:float,max_out_len:int,device,end_of_text_token_idx:int)->list[int]:
    
        #python复制对象留意：默认两个引用赋值两个变量指向同一个列表,使用.copy创建新的对象
    input:list = in_tokens.copy()
    ori_len = len(in_tokens)
    #max_out_len:新生成token的数量，减去原来
    #&为按位与，python使用and
    lm.eval()
    while(len(input)-ori_len < max_out_len and input[-1] != end_of_text_token_idx):

                            #input用小写.tensor构造，指定整数类型
        # 前向关闭梯度记录，开eval模式
        
        with torch.no_grad():
            # 保留完整输出，但每次前向只看训练时的上下文长度。
            context = input[-lm.context_len:]
            logits = lm.forward(torch.tensor(context,device = device,dtype = torch.long))
        #(seq_len,vocab_size)，取最后一位(next-token prediction)?
        #(vocab_Size，)一维
        temp_scaled_probs = softmax(logits[-1]/temp,dim = -1)
        #top-p sampling
        sorted,ori_indices = torch.sort(temp_scaled_probs,descending = True) #先排序(返回values,indices)
        p_sum = 0
        i = 0
        #确定新的样本空间V(p)（截断低概率）
        v_p_values:list[torch.Tensor] = []
        v_p_indices :list[int] = []
        #分两个列表收集，然后zip组装成字典


        #浮点运算不一定能加到1，加一个遍历完所有元素的判断
        while(i < sorted.numel() and p_sum < top_p):
            p_sum += sorted[i]
            ori_index = ori_indices[i].item() #取回原来的索引 #索引取成Python整数,原本返回是torch.long张量
            v_p_indices.append(ori_index)
            v_p_values.append(temp_scaled_probs[ori_index]) #(vocab_Size，)再用整数索引-->0维度
            i += 1
        #新的样本空间概率缩放归一化 zip:两两配对构成元组
        #v_p = dict(zip(v_p_indices,v_p_values))
        probs = torch.stack(v_p_values) #直接转成张量方便后续计算
        probs /= probs.sum()
        #新的样本空间进行抽样：torch.multinomial
        #索引取成Python整数
        sampled_idx = torch.multinomial(probs,num_samples = 1).item()
        next_token_idx = v_p_indices[sampled_idx] 
        input.append(next_token_idx)
    new_tokens = input[ori_len:]
    return new_tokens


#问题1:既然只取最后一位，前面的还要算吗？（虽然训练的时候都是算的）里面的数据流动再复习一下
#问题2:这个top_p新的样本空间如果还是取最大的，那跟保留原来的样本空间有何区别？除非是在新的样本空间里面随机,这个要求里面并没有说最后是取最大的还是怎么样



###

# 如果你的 `list[torch.Tensor]` 里每个 Tensor 都想首尾拼起来，最终得到**一维 Tensor**，最直接是：

# ```python
# result = torch.cat(tensor_list, dim=0)
# ```

# 例如：

# ```python
# tensor_list = [
#     torch.tensor([1, 2]),
#     torch.tensor([3, 4]),
#     torch.tensor([5, 6])
# ]

# result = torch.cat(tensor_list, dim=0)
# print(result)
# # tensor([1, 2, 3, 4, 5, 6])
# ```

# 如果里面每个 Tensor 是**标量 Tensor**，比如：

# ```python
# tensor_list = [
#     torch.tensor(1),
#     torch.tensor(2),
#     torch.tensor(3)
# ]
# ```

# 因为 0 维 Tensor 不能直接 `cat`，这时候用：

# ```python
# result = torch.stack(tensor_list)
# # tensor([1, 2, 3])
# ```

# 所以简单记：

# - `list` 里是 `tensor([1,2])` 这种 → `torch.cat`
# - `list` 里是 `tensor(1)` 这种 → `torch.stack`
# - `torch.stack([tensor([1,2]), tensor([3,4])])` 会得到二维的 `[[1,2],[3,4]]`，不是你说的“一行”。

###
