from tokenizer import Tokenizer
import pickle
import cProfile
import pstats
import time

def main():
    vocab_path = ""
    special_tokens = ["<|endoftext|>"]

    merges_path = ""

    tokenizer = Tokenizer().from_files(vocab_path,merges_path,special_tokens)

    #cProfile:性能分析器:记录函数，调用栈时间
    #但是会跳过等待时间，用time记录总时间
    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    
    with open("sampled_text_path","r",encoding="utf-8") as f:
        text = f.read()


    ids = tokenizer.encode(text)

    text_bytes = text.encode("utf-8")

    compression_ratio = len(text_bytes) / len(ids)

    print(f"压缩率(bytes / tokens)为{compression_ratio:.2f}\n")
    
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

    with open("tokenids_owt.pkl","wb") as f:
        pickle.dump(ids,f)

#执行
if __name__ == "__main__":
    main()