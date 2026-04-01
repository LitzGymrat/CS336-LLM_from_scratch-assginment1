from bpe_optimized import train_bpe
import pickle
import cProfile
import pstats
import time

def main():
    input_path = "C:\\Users\\MilesZhu\\CS336-assignment1\\data\\TinyStories-train.txt"
    special_tokens = ["<|endoftext|>"]

    vocab_size = 10000


    #cProfile:性能分析器:记录函数，调用栈时间
    #但是会跳过等待时间，用time记录总时间
    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    
    vocab,merges = train_bpe(input_path,vocab_size,special_tokens)
    
    profiler.disable()
    end_time = time.time()
    #pstats:转换，整理profiler数据(profile stats)
    stats = pstats.Stats(profiler).sort_stats('cumtime') #按照累计时间排序
    print("="*50)
    #打印前15耗时函数

    #找到最长的token
    print("="*50)
    print(f"总耗时:{end_time - start_time:.2f}") #:冒号+格式符
    stats.print_stats(15) 
    longest_token = max(vocab.values(),key = len)
    longest_word = longest_token.decode(encoding="utf-8",errors="replace")
    print(f"最长的token对应单词:{longest_word}\ntoken长度为{len(longest_token)}")

    #将得到的数据序列化存入硬盘
    with open("tinystory_train_vocab.pkl","wb") as f:
        pickle.dump(vocab,f)
    with open("tinystory_train_merges.pkl","wb") as f:
        pickle.dump(merges,f)

#执行
if __name__ == "__main__":
    main()