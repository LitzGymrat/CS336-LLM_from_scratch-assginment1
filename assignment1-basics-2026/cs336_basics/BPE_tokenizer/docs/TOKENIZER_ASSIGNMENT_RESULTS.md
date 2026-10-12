# Tokenizer 实验答题汇总

2026-09-16。以下文字对应作业 tokenizer 部分的实验交付，每项一至两句。四份训练/验证数据的全量编码已完成，共 3,340,216,780 个 token；数量、耗时和 `.npy` 文件见 [实验报告](TOKENIZER_EXPERIMENTS_RESULTS.md) 及其产物清单。

## TinyStories 上的 BPE 训练（train_bpe_tinystories）

### (a) 训练时间、内存和最长 token

使用 8 个预处理工作进程训练词表大小为 10,000 的 TinyStories 分词器，共耗时 75.03 秒，采样得到的各进程 RSS 总和峰值为 3.79 GiB。最长的 token 为 `b' accomplishment'`、`b' disappointment'` 和 `b' responsibility'`，均为 15 字节；它们都是完整的英文单词，并保留了预分词时附带的前导空格，因此结果符合预期。

依据：[TinyStories 正式训练报告](TINYSTORIES_TRAINING_RESULTS.md)。RSS 是采样的进程组总和，共享页可能重复计入。

### (b) 性能分析（Profiling）

在 TinyStories 验证集上，性能分析显示：采用并行预分词后，主要瓶颈是训练中的合并循环，其中反复从所有字节对中选出下一组待合并项带来了较大开销。在完整训练集上，分阶段计时则显示准备与预分词占用了总共 75.03 秒中的 66.88 秒（约 89%），说明性能瓶颈会随语料规模变化。

依据：[验证集 cProfile 报告](BENCHMARK_MULTIPROCESSING_RESULTS.md)、[完整训练集阶段计时](TINYSTORIES_TRAINING_RESULTS.md)。没有把多个 worker 的累计时间相加当作墙钟时间；完整训练集的上述数字来自普通计时，不能称作完整训练集的 cProfile 结果。

## OpenWebText 上的 BPE 训练（train_bpe_expts_owt）

### (a) 最长 token

词表大小为 32,000 的 OpenWebText 分词器有两个并列最长的 token，均为 64 字节：连续 64 个连字符，以及将字符串 `ÃÂ` 重复 16 次得到的文本。这些内容可能来自网页中的重复分隔线和乱码，说明 BPE 会学习高频字节模式，即使它们不是普通单词。

依据：[OWT 正式训练报告](OWT_VALIDATION_RESULTS.md)。训练耗时 2,805.555 秒，采样进程组 RSS 峰值 8.365 GiB；词表和合并文件已保存。

### (b) 与 TinyStories 比较

TinyStories 分词器的最长 token 是普通英文单词，而 OpenWebText 分词器还学到了较长的格式符号序列和疑似乱码，这体现了后者语料更丰富、噪声更多的特点。在同一批 OpenWebText 样本上，OWT 分词器的压缩率为 4.22 字节/token，TinyStories 分词器为 3.11 字节/token；不过，两者的训练领域和词表大小均不同，不能将差异只归因于其中一个因素。

## 分词器实验（tokenizer_experiments）

### (a) 各自语料上的压缩率

从两个语料的验证集中各抽取 10 篇文档，词表大小为 10,000 的 TinyStories 分词器在 TinyStories 样本上的压缩率为 3.99 字节/token，词表大小为 32,000 的 OpenWebText 分词器在 OpenWebText 样本上的压缩率为 4.22 字节/token。

### (b) 用 TinyStories tokenizer 编码 OWT

对同一批 OpenWebText 样本改用 TinyStories 分词器后，压缩率从 4.22 降至 3.11 字节/token，token 数量增加了约 36.1%。TinyStories 分词器的词表更小、训练语料也不同，这两个因素都可能造成该差异。

### (c) 吞吐量与 The Pile 外推

在抽样的 OpenWebText 文档上，优化后的分词器单进程吞吐量约为 1.40 × 10^6 字节/秒，结果取自五次测量的中位数。假设文本特征相近且不计磁盘读写，按此速度编码 825 GB 文本约需 5.90 × 10^5 秒，即 6.8 天。

使用作业中的十进制 `825 GB`，没有把多进程全量处理或 tiktoken 原生实现的性能混入该结果。

### (d) uint16

两个词表的 token ID 都是小于 65,536 的非负整数，最大值分别为 9,999 和 31,999，因此 `uint16` 可以表示所有 ID，不会溢出。每个 ID 只占 2 字节，存储空间是 `int32` 的一半、`int64` 的四分之一。

依据：[抽样记录、全量编码与测试结果](TOKENIZER_EXPERIMENTS_RESULTS.md)。
