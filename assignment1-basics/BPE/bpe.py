import regex as re
from collections import defaultdict

def train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]):
    #建立基础词表(无OOV)
    vocab = {i:bytes([i]) for i in range(256)}
    #整数映射到字节
    #加入特殊词元
    if special_tokens is not None:
        for st in special_tokens:
            new_id = len(vocab)
            vocab[new_id] = st.encode("utf-8") #值是编码后的


    #按照特殊词元切分文本
    with open(input_path,'r',encoding='utf-8') as f:
        text = f.read()
    if special_tokens:
        #转成正则表达式形式：
        # 1.外层括号表示保留 
        # 2.:.join表示用某个东西连接，传入给出str的可迭代对象re.escape(st) for st in special_tokens(和列表推导式不同)
        #3.|在正则中表示或 
        # 4.re.escape，给st套个完整，防止里面的魔法符号，影响

        pattern = "(" + "|".join(re.escape(st) for st in special_tokens) + ")"
        #切分成Chunks
        chunks = re.split(pattern,text) #返回列表
    else:
        chunks = [text] #也为列表
    #按照GPT2正则表达式切分文本
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #1.元组出现次数字典：defaultdict:给不存在的键一个默认值
    word_counts = defaultdict(int)

    #2.按照正则表达式切分
    for chunk in chunks:
        #跳过特殊词元和空词块
        if not chunk or (special_tokens and chunk in special_tokens):
            continue
        for match in re.finditer(PAT,chunk):
            word = match.group() #匹配到的match包含很多信息，取出字符串
            #编码
            utf8_bytes = word.encode("utf-8")
            #转成byte-level元组 如b"Miles",底层是[88,88,88,8,8]这类的。
            tuple_bytes = tuple(bytes([b]) for b in utf8_bytes)
            #再把每一个单字节分开来，组成元组，方便后面训练
            #字典的键不能是列表，必须是元组（一旦初始化就不可变），而列表是可变的
            #元组类似于(b'M',b'i',...)
            word_counts[tuple_bytes] += 1
        

    #3.BPE训练（合并）
    merges = []
            #词表大小上限
    while len(vocab) < vocab_size:
        #1.遍历word_counts，pairwise配对计数。如挨着的b'A'和b'B'
        pair_counts = defaultdict(int)
        for key in word_counts.keys():
            i = 0
            while i < len(key) - 1:
                tuple_pair = (key[i],key[i+1]) #tuple()接收一个可迭代对象组成元组，这里直接用括号就可以了
                pair_counts[tuple_pair] += word_counts[key] #加上频率
                i += 1


        #如果没有，跳出循环
        if not pair_counts:
            break
        #2.得到最高频率的，如果有一样，按照字典序取
        #找到最佳pair:
        #max函数，传入需要比较的对象；比较规则：比较元组，从左往右，先比pair_counts[p]数值，再按字典序比较



        best_pair = max(pair_counts.keys(),key= lambda p : (pair_counts[p],p))

        #3.将best_pair加入merge。合并，作为新的词元，加入vocab。
        merges.append(best_pair) 
        new_token = best_pair[0] + best_pair[1]
        new_id = len(vocab)
        vocab[new_id] = new_token

        #4.更新word_counts：将b'A'，b'B'都替换为b'AB'

        #用一个新的字典存过去（否则报错字典在被迭代的时候被修改大小）
        new_word_counts = defaultdict(int)

        for key,count in word_counts.items():
            i = 0
            #双指针：一个在旧词，一个在新词
            #构建一个空的新词列表，一点点翻译过去
            new_word = []
            while i < len(key):
                if i < len(key)-1 and key[i] == best_pair[0] and key[i+1] == best_pair[1]:
                    #传入  best_pair[0] +  best_pair[1]而非元组（把两个融合在一起，b'A'，b'B' ->b'AB'）
                    new_word.append(best_pair[0] + best_pair[1])
                    
                    i += 2 #跳两格
                else:
                    new_word.append(key[i])
                    i += 1
                        #新词列表转为不可变元组作为键
            new_word_counts[tuple(new_word)] = count
        #更新word_counts
        word_counts = new_word_counts




    return vocab,merges

#用例子做测试
if __name__ == "__main__":
    text_test = """low low low low low
lower lower widest widest widest
newest newest newest newest newest newest <|endoftext|>."""

    with open("test_bpe",'w',encoding='utf-8') as f:
        f.write(text_test)
    special_tokens = ["<|endoftext|>"]
    test_vocab,test_merges = train_bpe("test_bpe",263,special_tokens)
    for i,m in enumerate(test_merges):
        print(f"第{i+1}次合并：{m[0]} + {m[1]} --> {m[0]+m[1]}\n")

        #字典不可像列表等序列类型一样直接[:]切片（无序的哈希集合）
        #可以先把值取出来构建一个列表，再构建一个切片字典
    sliced_keys = list(test_vocab.keys())[256:]
    sliced_dict  = {k : test_vocab[k] for k in sliced_keys}
    print(sliced_dict)