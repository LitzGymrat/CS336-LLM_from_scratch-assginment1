import numpy as np
from tokenizer.tokenizer import Tokenizer

# 加载词表
vocab_path = "../data/tinystory_train_vocab.pkl"
merges_path = "../data/tinystory_train_merges.pkl"
tokenizer = Tokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])

# 加载训练集和验证集的前 20 个 token
train_data = np.load("../data/tokenids_tinys_train.npy", mmap_mode='r')
valid_data = np.load("../data/tokenids_tinys_valid.npy", mmap_mode='r')

print("--- 训练集前 20 个词 ---")
print(tokenizer.decode(train_data[:20].tolist()))

print("\n--- 验证集前 20 个词 ---")
print(tokenizer.decode(valid_data[:20].tolist()))