import json
import regex
import os
from typing import BinaryIO
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool #进程池
from collections.abc import Iterable, Iterator

def initialize_vocab(special_tokens:list[str]) -> dict[int,bytes]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
        
        #处理特殊token

    for i,special_token in enumerate(special_tokens):
        vocab[i+256] = special_token.encode("utf-8")
    return vocab
#vocab：记录tokenID-->字符串                    #类型标注是用方括号描述
def pretokenize_and_count(raw_text:str,special_tokens:list[str]) -> dict[tuple[bytes,...],int]:
    #第一步：根据特殊token切分文段
    if special_tokens:
        escaped_tokens = []
        for token in special_tokens:
        #为了让特殊token在切分时完整切分，是用正则escape增加转义符
            escaped_tokens.append(regex.escape(token))
        #.join:连接多个字符串,形成按照特殊token切分的正则表达式
        delimiter = "|".join(escaped_tokens)
        #切分形成切分后的字符串列表
        sections = regex.split(delimiter,raw_text)
        #split：传入分隔符，得到按照分隔符分开的两侧文本（正则表达式PAT描述分隔符）
    else:
        #返回列表:[raw_text]
        sections = [raw_text]
    #第二步：进一步根据正则表达式切分--转bytes字典格式--计数

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    counts = {}
    for section in sections:
        #pieces = regex.split(PAT,section)
        #这里PAT描述的是需要的片段本身，如果用split反而会把他们当作分隔符去掉
        #findall返回所有匹配的字符串列表（不等于匹配对象）-遍历列表计数。但是这里我们节省内存，用finditer返回迭代器，逐个遍历匹配对象计数即可。
        matches = regex.finditer(PAT,section)
        for match in matches:
            #match是一个匹配对象，还包括匹配位置等等，用.group()取出其中的字符串
            piece = match.group()
            #字节串如b'low'，需要转为b'l',b'o',b'w'逐个字节。方便后续计数
            byte_parts = []
            encoded_piece = bytes(piece,"utf-8")
            for byte_value in encoded_piece:#将字符串转字节序列，是用bytes去转和.encode都可以
                single_byte = bytes([byte_value])
                byte_parts.append(single_byte) #需要多转一次，因为遍历bytes取出的是十进制整数值

            word = tuple(byte_parts)
            #查这个键是否存在
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
    return counts
def pretokenize_chunk(input_path:str,start:int,end:int,special_tokens:list[str]) -> dict[tuple[bytes,...],int]:
#并行化：针对每个分块设置一个pretokenize函数
    with open(input_path,"rb") as file:
        #找到开始位置（用二进制方式读取，因为前面找位置的时候是用二进制方式）
        file.seek(start)
        #找到对应字节片段
        chunk_bytes = file.read(end-start)
    chunk_text = chunk_bytes.decode("utf-8") #解码成文本
    chunk_counts = pretokenize_and_count(chunk_text,special_tokens)
    return chunk_counts 


#数字节对频次
def count_pair(counts:dict[tuple[bytes,...],int]) -> tuple[dict[tuple[bytes,bytes],int] , dict[tuple[bytes,bytes],set[tuple[bytes,...]]]]:#一个Word是一个tuple
    pair_counts = {}
    pair_to_word = {}
    for word,frequency in counts.items(): #.items取出字典的键和值
        for i in range(len(word) - 1):
            first = word[i]#拆出字节对
            second = word[i+1]
            #pair = tuple(first,second)不对。tuple接受可迭代对象转为元组
            pair = first,second 
            if pair not in pair_to_word:
                pair_to_word[pair] = set() #去重使用set：if word not in pair_to_wor_pair查找O（1）否则还要遍历列表

            if pair in pair_counts:
                pair_counts[pair] += frequency
            else:
                pair_counts[pair] = frequency

            if word not in pair_to_word[pair]:
                pair_to_word[pair].add(word)
            

    return pair_counts,pair_to_word

def merge_pair(word:tuple[bytes,...],pair:tuple[bytes,bytes]) -> tuple[bytes,...]:
    #构建一个新的word存储，边遍历边放进来
    #需要追加和修改：先用列表append。最后转元组（不可修改，如果要合并，更改只能创建新元组）
    new_word = []
    i = 0
    #while遍历
    while(i < len(word)):
        #分为遇到merged_token和没遇到两种情况
        if i+1 < len(word) and (word[i],word[i+1]) == pair:
            merged_token = word[i] + word[i+1]
            #用+拼接两个字节串
            new_word.append(merged_token)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word) #元组：不能修改，作为字典的键；list不能作为键

def train_bpe_loop(vocab:dict[int,bytes],vocab_size:int,counts:dict[tuple[bytes,...],int]) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    #训练循环：达到词表大小前
    merges = []
    #先算pair_counts
    pair_counts,pair_to_word = count_pair(counts)
    while(len(vocab) < vocab_size):
        
        #这是训练。训练之后，得到一个vocab词表和merge优先级+规则。用于实际的文本切分因为Merge是有次序之分的。
        #
        #先处理特殊 token，并做正则预分词。 特殊 token 直接对应自己的 ID，普通片段分别处理。
#把每个普通片段编码为 UTF-8，拆成单字节 token。
#按照训练时记录的优先级应用合并规则。 每次都操作合并后的当前序列，而且不跨越预分词边界。
#将最终 token 的字节串转换为 ID。

        
        if not pair_counts:
            break
        #取出最大的字节对频
        max_frequency = max(pair_counts.values())
        candidates = []
        for pair,frequency in pair_counts.items():#取出字典的键和值
            if frequency == max_frequency:
                #抽取出出现频次最多的作为候选
                candidates.append(pair)
        pair_to_merge = max(candidates)#选出字典序最大的
        merges.append(pair_to_merge)
        

        #注意，增量更新可能出现原来的for old_word in pair_to_word[pair_to_merge]:中集合在迭代中被修改
        #复制一个用于迭代的副本
        affected_words = pair_to_word[pair_to_merge].copy()
        for old_word in affected_words:
            merged_word = merge_pair(old_word,pair_to_merge)
            frequency = counts[old_word]
            #增量更新Counts
            del counts[old_word] #原地修改（字典可变
            counts[merged_word] = frequency
                    #word = merged_word :word是一个变量名，绑定了字典的旧键。修改word不会修改字典的键，只是重新绑定word
                    #需要新建一个字典重新让变量counts指向新的字典
                    # 。
            #增量更新pair_count和pair_to_word。将旧pair贡献移除，添加新pair贡献
            i = 0
            #移除旧片段贡献
            for i in range(len(old_word)-1):
                first = old_word[i]
                second = old_word[i+1]

                #pair为元组作为频次表的键，不是拼接。old_pair = first + second
                old_pair = first,second
                pair_counts[old_pair] -= frequency
                #如果pair数量为0了，移除索引关系
                #空pair_counts会触发终止。

                #如果一个word有多个aa位置，每个都remove会导致重复删除报错
                #在清零删除索引关系前使用discard。否则如果索引不存在也会报错。
                pair_to_word[old_pair].discard(old_word)

                if pair_counts[old_pair] == 0:
                    del pair_counts[old_pair]
                    del pair_to_word[old_pair]

                
            #添加新片段贡献
            for i in range(len(merged_word)-1):
                first = merged_word[i]
                second = merged_word[i+1]
                new_pair = first,second
                #新出现的情况需要额外处理，不能直接+=
                if new_pair not in pair_counts:
                    pair_counts[new_pair] = 0
                    pair_to_word[new_pair] = set()
                pair_counts[new_pair] += frequency
                pair_to_word[new_pair].add(merged_word)
        new_token = pair_to_merge[0] + pair_to_merge[1]
        new_idx = len(vocab)
        #字典：可变类型：修改对象本身，函数外也改变。
        vocab[new_idx] = new_token


    return vocab,merges


def train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
#初始化词表
    vocab = initialize_vocab(special_tokens)
    #读取训练文本
    with open(input_path,"r",encoding = "utf-8",newline = "") as file:
        raw_text = file.read()
   
        
        #预切分
    initial_counts = pretokenize_and_count(raw_text,special_tokens)

    vocab,merge = train_bpe_loop(vocab,vocab_size,initial_counts)
    return vocab,merge

#完整的训练流程接起来
def train_bpe_multiprocesses(input_path:str,vocab_size:int,special_tokens:list[str],num_processes:int=8,desired_num_chunks:int = 8) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
#初始化词表
    vocab = initialize_vocab(special_tokens)
    #读取训练文本
    #with open(input_path,"r",encoding = "utf-8",newline = "") as file:
        #raw_text = file.read()
    #将可以独立运行的进程并行化，优化efficiency：
    #根据特殊token找到切分边界：
    with open(input_path,"rb") as file:
        boundaries = find_chunk_boundaries(
            file,
            desired_num_chunks,
            b"<|endoftext|>"
        )

    #拆分成各个独立子任务
    tasks = []
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        task = (input_path,start,end,special_tokens)#每个独立子任务的参数
        tasks.append(task)
    with Pool(processes = num_processes) as pool:
        chunk_counts_list = pool.starmap(pretokenize_chunk,tasks)
        #构建进程池，将tasks(参数)分给这些进程，每个任务执行一次pretokenize_chunk
        #starmap等所有任务完成，把返回值组成一个列表返回。
    #汇总：
    initial_counts = {}
    for chunk_counts in chunk_counts_list:
        for word,frequency in chunk_counts.items():#.items
            if word in initial_counts:
                initial_counts[word] += frequency
            else:
                initial_counts[word] = frequency
    


    
    
        

    vocab,merge = train_bpe_loop(vocab,vocab_size,initial_counts)
    return vocab,merge

class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str] = None):
        self.vocab = vocab
        #查询字典在__init__构建一次即可
        
        self.merges = merges
        self.merge_to_rank = {merge:rank for rank,merge in enumerate(self.merges)}
        #if special_tokens:不管是不是None，属性需要赋值，这样后续也可以判断
        self.special_tokens = special_tokens
        #若特殊token尚未出现于词表需要加入
        #先判断，是None的话不能循环
        if self.special_tokens:
            for special_token in self.special_tokens:
                token_bytes = special_token.encode("utf-8")
                if token_bytes not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token.encode("utf-8")
        #加完特殊token再建立反向词表
        self.reversed_vocab = {token:token_id for token_id,token in self.vocab.items()}
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens=None):
        # 保存时用十六进制字符串表示 bytes，读取后恢复原来的类型。
        with open(vocab_filepath, "r", encoding="utf-8") as file:
            vocab_data = json.load(file)

        vocab = {}
        for token_id_text, token_hex in vocab_data.items():
            token_id = int(token_id_text)
            token_bytes = bytes.fromhex(token_hex)
            vocab[token_id] = token_bytes

        with open(merges_filepath, "r", encoding="utf-8") as file:
            merges_data = json.load(file)

        merges = []
        for first_hex, second_hex in merges_data:
            first = bytes.fromhex(first_hex)
            second = bytes.fromhex(second_hex)
            merges.append((first, second))

        return cls(vocab, merges, special_tokens)

    def merge_pair(self,word:tuple[bytes,...],pair:tuple[bytes,bytes]) -> tuple[bytes,...]:
        #构建一个新的word存储，边遍历边放进来
        #需要追加和修改：先用列表append。最后转元组（不可修改，如果要合并，更改只能创建新元组）
        new_word = []
        i = 0
        #while遍历
        while(i < len(word)):
            #分为遇到merged_token和没遇到两种情况
            if i+1 < len(word) and (word[i],word[i+1]) == pair:
                merged_token = word[i] + word[i+1]
                #用+拼接两个字节串
                new_word.append(merged_token)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word) #元组：不能修改，作为字典的键；list不能作为键

    def encode(self,raw_text:str) -> list[int]:
        #1.pretokenize
        if self.special_tokens:
                escaped_tokens = []
                #防止重叠特殊token：从长到短排列
                for token in sorted(self.special_tokens,key=len,reverse=True):
                #为了让特殊token在切分时完整切分，是用正则escape增加转义符
                    escaped_tokens.append(regex.escape(token))
                #.join:连接多个字符串,形成按照特殊token切分的正则表达式
                delimiter = "|".join(escaped_tokens)
                #切分形成切分后的字符串列表
                #encode阶段需要留下特殊token用于还原。（训练阶段可以不需要特殊token
                #两侧加括号，特殊token单独一组
                sections = regex.split("(" + delimiter + ")",raw_text)
                #split：传入分隔符，得到按照分隔符分开的两侧文本（正则表达式PAT描述分隔符）
        else:
                #返回列表:[raw_text]
            sections = [raw_text]
            #第二步：进一步根据正则表达式切分--转bytes字典格式--计数

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens_seq = []

        #merge_to_rank = {merge:rank for rank,merge in enumerate(self.merges)}
        for section in sections:
                #pieces = regex.split(PAT,section)
                #这里PAT描述的是需要的片段本身，如果用split反而会把他们当作分隔符去掉
                #findall返回所有匹配的字符串列表（不等于匹配对象）-遍历列表计数。但是这里我们节省内存，用finditer返回迭代器，逐个遍历匹配对象计数即可。
            
            #如果是特殊token直接添加，不需要再去逐个字节Merge
            if self.special_tokens and section in self.special_tokens:
                pre_tokens_seq.append(section.encode("utf-8"))#整个编码放进去（encode出来就是bytes而非一元组
                continue #跳过
            
            matches = regex.finditer(PAT,section)


            
            for match in matches:
                    #match是一个匹配对象，还包括匹配位置等等，用.group()取出其中的字符串
                piece = match.group()
                    #字节串如b'low'，需要转为b'l',b'o',b'w'逐个字节。方便后续计数
                byte_parts = []
                encoded_piece = bytes(piece,"utf-8")
                for byte_value in encoded_piece:#将字符串转字节序列，是用bytes去转和.encode都可以
                    single_byte = bytes([byte_value])
                    byte_parts.append(single_byte) #需要多转一次，因为遍历bytes取出的是十进制整数值

                pre_token = tuple(byte_parts)
                while True:
                    merge_candidates = []

                    #空间换时间：
                    #原来对所有m个Merge都去检查一次，时间复杂度O(mn)，其中n为len(pretoken)
                    # 但问题是这样子就很多浪费，因为很多merge中对应的字符根本就不在Pretoken内部
                    #理想状态应该是只针对Pretoken内部的字节对进行检查
                    #但是这就需要看每个字节对是否有对应的merge以及其中优先级
                    #为了解决这个问题，我们先构建一个merge_to_rank,建立每个Merge和优先级的映射
                    #然后每个字节对去查在不在merge中，如果在的话就进入候选，最后选出rank最小的进行合并
                    #
                    for i in range(len(pre_token) - 1):
                        byte_pair = pre_token[i],pre_token[i+1]
                        if byte_pair in self.merge_to_rank:
                            merge_candidates.append(byte_pair)
                    #if len(merge_candidates) != 0 和 if merge_candidates等价
                    if merge_candidates:                        #key:打分规则，使用匿名函数
                        lowest_rank_merge = min(merge_candidates,key = lambda x: self.merge_to_rank[x])
                        pre_token = self.merge_pair(pre_token,lowest_rank_merge)
                    else:
                        break



                #逐个合并
                #for pair_to_merge in self.merges:
                    #pre_token = self.merge_pair(pre_token,pair_to_merge)
                    #一个Pre_token是一个元组，我们最后需要的是逐个的元素，使用extend，将一个可迭代对象的元素逐个加入列表
                
                pre_tokens_seq.extend(pre_token)
            #字典推导式构建反向字典（移动到__init__构建）
        #reversed_vocab = {token:token_id for token_id,token in self.vocab.items()}
        #列表推导式（）
        encoded_text = [self.reversed_vocab[token] for token in pre_tokens_seq]
        return encoded_text
        #对于大文件，分片，控制每次保留的数据量。读一部分，处理一部分输出一部分
    
    
    
   
    
    def encode_iterable_marker(self, iterable: Iterable[str]) -> Iterator[int]:
        #Iterable[str]：可以逐个取出字符串
        #但不保证取出的每个可以独立安全编码
        buffer = "" #设置一个可以独立安全编码的缓冲区，留够了就encode一次。
        marker = "<|endoftext|>" #找到安全边界标记
        for chunk in iterable: 
            buffer = buffer + chunk
            #同一个chunk可能有多个marker，使用while True查找直到找不到。
            while True:
                split_position = buffer.find(marker) #寻找子串位置
                if split_position == -1:
                    break
                else:
                    end_position = split_position + len(marker)
                    document = buffer[:end_position] #
                    buffer = buffer[end_position:]
                ids = self.encode(document)
                for token_id in ids:
                    yield token_id#生成器
        for token_id in self.encode(buffer):
            yield token_id#最后一次buffer需要额外处理

    #注意ai过度设计+过度兜底
    #留意输入约定：保证输入迭代器返回的为多个可以安全输入的文档
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text) 
            #等价于
            # for token_id in self.encode(text)
            #   yield token_id
            #一行结束。！ 

    def decode(self,ids:list[int]) -> str:
        tokens = [self.vocab[token_id] for token_id in ids]
        combined_bytes = b"".join(tokens) #用空字节串连接起来
        #不是每个token单独解码，假如一个“黎”字编码为三个token，单独解码一个token会因为缺少后面字节报错
        #所有字节拼起来然后解码
        raw_text = combined_bytes.decode("utf-8",errors="replace")#用？替换非法字符
        return raw_text