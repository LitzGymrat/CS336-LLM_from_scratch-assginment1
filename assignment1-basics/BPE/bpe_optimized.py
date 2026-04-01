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

            #在循环中要使用的数据：避免内部频繁的类型转换，直接先转好，或者查表
            #tuple_bytes = tuple(bytes([b]) for b in utf8_bytes)
            tuple_bytes = tuple(vocab[b] for b in utf8_bytes)
            #再把每一个单字节分开来，组成元组，方便后面训练
            #字典的键不能是列表，必须是元组（一旦初始化就不可变），而列表是可变的
            #元组类似于(b'M',b'i',...)
            word_counts[tuple_bytes] += 1
        


        #初始化计数和倒排索引:只要一次
#1.遍历word_counts，pairwise配对计数。如挨着的b'A'和b'B'
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)#倒排索引：记录tuple出现在哪些word，是用集合set。
    for key in word_counts.keys():
        i = 0
        for i in range(len(key)-1):
            tuple_pair = (key[i],key[i+1]) #tuple()接收一个可迭代对象组成元组，这里直接用括号就可以了
            pair_counts[tuple_pair] += word_counts[key] #加上频率
            pair_to_words[tuple_pair].add(key)
            i += 1    
    #3.BPE训练（合并）
    merges = []
            #词表大小上限
    
    while len(vocab) < vocab_size:
        


        #如果没有，跳出循环
        if not pair_counts:
            break
        #2.得到最高频率的，如果有一样，按照字典序取
        #找到最佳pair:
        #max函数，传入需要比较的对象；比较规则：比较元组，从左往右，先比pair_counts[p]数值，再按字典序比较


        #需要频繁查找最大值，又要O(1)：优先级队列+哈希表？

        #best_pair = max(pair_counts.keys(),key= lambda p : (pair_counts[p],p))
                    #传入可迭代对象，元组比较：逐元素按顺序比较 items：元组集合？
        best_pair = max((count,pair) for pair,count in pair_counts.items())[1] #取出pair
        #3.将best_pair加入merge。合并，作为新的词元，加入vocab。
        merges.append(best_pair) 
        new_token = best_pair[0] + best_pair[1]
        new_id = len(vocab)
        vocab[new_id] = new_token

        #4.更新word_counts：将b'A'，b'B'都替换为b'AB'

        #只处理含有对应词对的老单词
        #更新paircount
        

        #需要在循环遍历容器（字典/集合）中增删：先转为list
        for old_word in list(pair_to_words[best_pair]):
            count = word_counts[old_word]
            i = 0
            for i in range(len(old_word)-1):
                old_pair = (old_word[i], old_word[i+1]) #键是元组
                pair_counts[old_pair] -= count
                #删完了就把集合删掉
                if pair_counts[old_pair] <= 0:
                    del pair_counts[old_pair]
                #更新倒排索引（处理旧词）
                #状态维护调整
                pair_to_words[old_pair].discard(old_word) #从集合中移除(discard：没有就不删，防止重复删除)
                if not pair_to_words[old_pair]:
                    del pair_to_words[old_pair]
                i += 1
            new_word = []
            i = 0 #每次循环前重置i
            while i < len(old_word):
                if i < len(old_word)-1 and old_word[i] == best_pair[0] and old_word[i+1] == best_pair[1]:
                    #传入  best_pair[0] +  best_pair[1]而非元组（把两个融合在一起，b'A'，b'B' ->b'AB'）
                    new_word.append(best_pair[0] + best_pair[1]) #融合才用加法
                    
                    i += 2 #跳两格
                else:
                    new_word.append(old_word[i])
                    i += 1
            
            #删去原来的
            new_word_tuple = tuple(new_word)
            del word_counts[old_word]
            #增加新的(现在不需要遍历整个字典，只遍历一拉出来的小部分，所以不需要创建新的)
            #列表不能作为键，转成元组
            word_counts[new_word_tuple] = count


            #更新paircount#更新pair_to_word倒排索引（新词）
            
            for i in range(len(new_word)-1):
                #构造pair:(a,b)
                                #直接括起来转为元组即可 tuple只接受一个参数
                new_pair_tuple = (new_word[i],new_word[i+1])
                pair_counts[new_pair_tuple] += count
                i += 1
                    #需要作为字典的键/集合的元素：必须是元组(可哈希)
                pair_to_words[new_pair_tuple].add(new_word_tuple)


            

        

        



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