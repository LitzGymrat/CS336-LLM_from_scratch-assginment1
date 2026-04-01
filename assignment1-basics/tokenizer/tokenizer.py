import regex as re
from collections import defaultdict
import pickle
from collections.abc import Iterable,Iterator
class Tokenizer:
    def __init__(self, vocab:dict[int,bytes], merges:list[tuple[bytes,bytes]], special_tokens:list[str]=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        #用于解码的反向映射字典
        self.vocab_inv = {v:k for k,v in vocab.items()}
        #优先级合并词表（必须按照训练时合并的顺序进行合并）
        self.sequntial_merges = {pair:i for i,pair in enumerate(merges)}

    @classmethod #通过类直接传入文件路径构造
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath,'rb') as f:
            loaded_vocab = pickle.load(f)
        with open(merges_filepath,'rb') as f:
            loaded_merges = pickle.load(f)

        return Tokenizer(loaded_vocab,loaded_merges,special_tokens)
    def encode(self,text:str)->list[int]:
        #先做正则&特殊词元切分
        if self.special_tokens:
        #转成正则表达式形式：
        # 1.外层括号表示保留 
        # 2.:.join表示用某个东西连接，传入给出str的可迭代对象re.escape(st) for st in special_tokens(和列表推导式不同)
        #3.|在正则中表示或 
        # 4.re.escape，给st套个完整，防止里面的魔法符号，影响

            #优先匹配长词：连续特殊词元截取整体
            sorted_special_tokens = sorted(self.special_tokens,key = len,reverse=True)
            pattern = "(" + "|".join(re.escape(st) for st in sorted_special_tokens) + ")"
        #切分成Chunks
            chunks = re.split(pattern,text) #返回列表
        else:
            chunks = [text] #也为列表
    #按照GPT2正则表达式切分文本
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
        ids = []
    #2.按照正则表达式切分
        for chunk in chunks:
        #跳过空词块
            if not chunk:
                continue
            #处理特殊词元
            if self.special_tokens and chunk in self.special_tokens:
                utf8_bytes = chunk.encode("utf-8")
                ids.append(self.vocab_inv[utf8_bytes])
                continue

            for match in re.finditer(PAT,chunk):
                word = match.group() #匹配到的match包含很多信息，取出字符串
            #编码
                utf8_bytes = word.encode("utf-8")
            #进行合并操作

                                       
                                        #1.迭代，索引bytes对象得到的都是整数(底层)，打印时渲染成ASCII字母。取出来转回字节
                                #GPT2词表打乱了顺序，直接还原。
                word_fragments = [bytes([b]) for b in utf8_bytes]
                while len(word_fragments) >= 2:
                    
                    pairs_to_merge = []
                    earliest_pair = None
                    for i in range(len(word_fragments)-1):
                        cur_pair = word_fragments[i],word_fragments[i+1]

                        #if cur_pair in self.sequntial_merges即可，默认检查字典的key，省去一步视图
                        if cur_pair in self.sequntial_merges:
                            if earliest_pair is None:
                                earliest_pair = cur_pair
                            else:
                                #第一轮遍历就直接维护最早者
                                if(self.sequntial_merges[cur_pair] < self.sequntial_merges[earliest_pair]):
                                    earliest_pair = cur_pair
                            pairs_to_merge.append(cur_pair)
                    if not pairs_to_merge:
                        break
                    #找到最早的对
                    #这里是：全部扫完，又跟一个min。相当于多扫了一遍，前面第一遍遍历就可以维护好！
                    #earliest_pair = min(pairs_to_merge,key= lambda p:self.sequntial_merges[p])
                    #如果没有了，就退出
                    
                    #进行合并
                    new_word_fragments = []
                    #跳步循环:只能用while:先重置i
                    i = 0

                    #while遍历到最后一个！！！
                    while i < len(word_fragments):
                        if i < len(word_fragments)-1 and word_fragments[i] == earliest_pair[0] and word_fragments[i+1] == earliest_pair[1]:
                        #传入  best_pair[0] +  best_pair[1]而非元组（把两个融合在一起，b'A'，b'B' ->b'AB'）
                            new_word_fragments.append(earliest_pair[0] + earliest_pair[1]) #融合才用加法
                        
                            i += 2 #跳两格
                        else:
                            new_word_fragments.append(word_fragments[i])
                            i += 1
                        #状态交接
                    word_fragments = new_word_fragments
                
                #合并完成，进行编码
                for fragment in word_fragments:
                    ids.append(self.vocab_inv[fragment]) #编码采用的是bytes->int vocab。(反向)
                
                

        return ids
    #迭代器
    def encode_iterable(self,iterable:Iterable[str])->Iterable[int]:
        for text_chunk in iterable:
            ids = self.encode(text_chunk)
            yield from ids
            #直接yield from
            #for token_id in ids:
                #yield token_id
        
    def decode(self,ids:list[int])->str:
        #b表示bytes
        #text = b""
        #for id in ids:
            #word_bytes = self.vocab[id]
            #先拼起来，再解码！不是一个个解码
            #text += word_bytes #替换无效的utf-8编码
                #先放在列表里
        bytes_list = [self.vocab[id] for id in ids]
                #直接接上去，避免频繁+=内存开销？
                #字符串/字节串是不可变类型，不能原地+=,必须申请新的内存

        text = b"".join(bytes_list)#一次性申请内存。
        return text.decode(encoding='utf-8',errors="replace")

