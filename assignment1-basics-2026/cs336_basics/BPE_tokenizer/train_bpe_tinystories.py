from cs336_basics.BPE_tokenizer.BPE_tokenizer import train_bpe_multiprocesses

#每次子进程会重新加载启动脚本。加一个是否为程序入口的判断，否则会再次启动训练。
if __name__ == "__main__":
    vocab, merges = train_bpe_multiprocesses(
        input_path="data/TinyStoriesV2-GPT4-valid.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_processes=4,
        desired_num_chunks=8,
    )

    print("训练完成", len(vocab), len(merges))