# Tokenizer 实验：压缩率、吞吐量和全量编码

日期：2026-09-16。使用用户当前 `BPE_tokenizer.py` 中的 `Tokenizer.encode`，以及分别在 TinyStories / OWT 训练集上学到的 10K / 32K 词表。

## OWT 最长 token

按字节数计算，有两个并列最长 token，长度均为 64 字节：

| ID | 内容 | Unicode 字符数 | UTF-8 字节数 |
|---|---|---:|---:|
| 25822 | `"ÃÂ" * 16` | 32 | 64 |
| 25836 | `"-" * 64` | 64 | 64 |

第二项是重复分隔线，第一项看起来是重复的乱码文本。BPE 根据字节对频次学习，并不要求一个 token 必须是自然语言单词。

## 抽样与统计方法

- 使用两个验证集，按 `<|endoftext|>` 分隔完整文档。
- 固定随机种子 42，使用蓄水池抽样（reservoir sampling），各等概率抽取 10 篇非空文档。
- TinyStories 验证集包含 27,630 篇非空文档，OWT 为 59,059 篇。
- 移除文档分隔标记，保留文档本身所有空白；仅排除全空白记录。
- 压缩率为样本原文的 UTF-8 总字节数除以总 token 数。
- OWT 的两种 tokenizer 对照使用完全相同的 10 篇文档。
- 三组抽样实验顺序执行，计时只覆盖预热后的 `encode` 调用；不包含加载、抽样、解码、参考实现检查和序列化。原始版本每组测一次；优化后沿用相同样本和词表，每组测五次取中位数。均未启用性能分析工具。

## (a) 与 (b)：压缩率

| 样本文档 | 分词器 | 原文总字节数 | Token 数 | 字节/token |
|---|---|---:|---:|---:|
| TinyStories 10 篇 | TinyStories 10K | 7,799 | 1,953 | 3.9933 |
| OWT 10 篇 | OWT 32K | 50,180 | 11,877 | 4.2250 |
| 同一批 OWT 10 篇 | TinyStories 10K | 50,180 | 16,159 | 3.1054 |

同一批 OWT 文档改用 TinyStories tokenizer，token 数增加 4,282 个，约 36.05%，每个 token 平均承载的字节数减少。这项比较同时涉及训练语料和词表大小的变化，不能单独归因为训练领域差异。

可用于作业的简短中文表述：

**(a)** 从两个语料的验证集中各抽取 10 篇文档，词表大小为 10,000 的 TinyStories 分词器在 TinyStories 样本上的压缩率为 3.99 字节/token，词表大小为 32,000 的 OpenWebText 分词器在 OpenWebText 样本上的压缩率为 4.22 字节/token。

**(b)** 对同一批 OpenWebText 样本改用 TinyStories 分词器后，压缩率从 4.22 降至 3.11 字节/token，token 数量增加了约 36.1%。TinyStories 分词器的词表更小、训练语料也不同，这两个因素都可能造成该差异。

## (c)：当前编码吞吐量

| 样本与分词器 | 编码耗时 | 字节/秒 | 825 GB 线性外推 |
|---|---:|---:|---:|
| TinyStories / TinyStories 10K | 0.00524 秒 | 1,488,252 | 约 6.42 天 |
| OWT / OWT 32K | 0.03591 秒 | 1,397,460 | 约 6.83 天 |
| OWT / TinyStories 10K | 0.03210 秒 | 1,563,015 | 约 6.11 天 |

采用十进制 `825 GB = 825 × 10^9 bytes`，公式为 `预计秒数 = 825 × 10^9 / 实测 bytes_per_second`。这些数字是优化后单进程编码的小样本外推，不包括磁盘读写，也不代表 The Pile 实际全量处理时间。

当前实现只检查片段中存在的相邻对，并通过预先构建的 `merge_to_rank` 选择最早的合并规则；反向词表也只初始化一次。OWT 原始版本耗时 24.182 秒、吞吐量 2,075 bytes/s；同一批样本优化后约快 673 倍，所有 ID 完全一致。

[优化后五次计时与校验记录](../../../data/tokenizer_experiments/20260916-124706-135165-rank-encoding/metrics.json)。另一次将相同样本重复成较长批次的测量得到约 1.55 MB/s，属于不同测量批次；本节保持使用上述五次整批文档计时的中位数。

**(c)** 在抽样的 OpenWebText 文档上，优化后的分词器单进程吞吐量约为 1.40 × 10^6 字节/秒，结果取自五次测量的中位数。假设文本特征相近且不计磁盘读写，按此速度编码 825 GB 文本约需 5.90 × 10^5 秒，即 6.8 天。

## 正确性与产物

30 次文档编码均满足 `decode(encode(text)) == text`，且输出 ID 与使用相同词表、按相邻字节对合并优先级编码的参考实现逐项一致。参考编码与解码检查位于计时区间之外。

- [完整测量记录](../../../data/tokenizer_experiments/20260916-083805-814679/metrics.json)
- [TinyStories 样本文档](../../../data/tokenizer_experiments/20260916-083805-814679/tinystories_sample.json)
- [OWT 样本文档](../../../data/tokenizer_experiments/20260916-083805-814679/owt_sample.json)
- [TinyStories 样本的 ID](../../../data/tokenizer_experiments/20260916-083805-814679/tinystories_with_tinystories_ids.json)
- [OWT 样本的 OWT tokenizer ID](../../../data/tokenizer_experiments/20260916-083805-814679/owt_with_owt_ids.json)
- [OWT 样本的 TinyStories tokenizer ID](../../../data/tokenizer_experiments/20260916-083805-814679/owt_with_tinystories_ids.json)

运行目录保留了用户实现和实验脚本快照，以及源数据、样本文档和训练产物的哈希。复现命令：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.tokenizer_experiments --seed 42 --documents 10
```

## (d)：全量编码

四份数据已于 2026-09-16 13:46:53（Asia/Shanghai）全部编码完成并通过校验。总耗时 2,787.67 秒（约 46 分 28 秒），共 3,340,216,780 个 token；四个一维 `uint16` 数组共占 6,680,434,072 字节（约 6.68 GB）。

| 数据 | 原文字节数 | Token 数 | `.npy` 文件字节数 | 处理耗时 |
|---|---:|---:|---:|---:|
| TinyStories 训练集 | 2,227,753,162 | 541,229,347 | 1,082,458,822 | 376.54 秒 |
| TinyStories 验证集 | 22,502,601 | 5,465,883 | 10,931,894 | 3.81 秒 |
| OWT 训练集 | 11,920,511,059 | 2,727,120,452 | 5,454,241,032 | 2,348.62 秒 |
| OWT 验证集 | 289,998,753 | 66,401,098 | 132,802,324 | 58.39 秒 |

以上全量处理时间包含进程池启动、读取、用户编码、独立参考检查、解码还原、写入、分片合并和哈希检查；与 (c) 单进程纯编码时间的计时范围不同。此处原文及输出均保留文档分隔标记，而 (a)(b) 抽样压缩率排除了这些标记。

输出目录为 `data/tokenized/20260916-user-bpe/`：

- [TinyStories 训练集](../../../data/tokenized/20260916-user-bpe/tinystories_train.npy)
- [TinyStories 验证集](../../../data/tokenized/20260916-user-bpe/tinystories_valid.npy)
- [OWT 训练集](../../../data/tokenized/20260916-user-bpe/owt_train.npy)
- [OWT 验证集](../../../data/tokenized/20260916-user-bpe/owt_valid.npy)
- [产物清单、词表来源与完整指标](../../../data/tokenized/20260916-user-bpe/manifest.json)
- [最终检查记录](../../../data/tokenized/20260916-user-bpe/verification.json)

使用用户自己的 `Tokenizer.encode`，8 个工作进程，目标约 4 MiB 的输入分块。只在 `<|endoftext|>` 起点分块，保留原始空白、文档顺序和特殊 token；每块输出均解码还原原文，并与使用同一词表和正则的 tiktoken 编码逐项核对。分块完成后按原顺序写入一维 NumPy 数组，核对保存后的 payload SHA-256，随后删除此次生成的临时分片。tiktoken 仅用于校验，保存的 ID 来自用户实现。

单块处理和最终合并均不需要把整份语料或整个 ID 数组读进内存。分块大小取决于目标大小和实际文档边界；沿用作业给定的文档分隔标记约定。

四份输入文件的 SHA-256 均与此前训练或抽样时的记录一致。每个数组重新通过 `np.load(..., mmap_mode="r", allow_pickle=False)` 打开检查，类型、长度和文件大小正确；训练集最大 ID 分别为 9,999 和 31,999。最终文件通过 payload SHA-256 比对，临时分片已删除，用户 `BPE_tokenizer.py` 在整个执行过程中没有变化。

**(d)** 两个词表的 token ID 都是小于 65,536 的非负整数，最大值分别为 9,999 和 31,999，因此 `uint16` 可以表示所有 ID，不会溢出。每个 ID 只占 2 字节，存储空间是 `int32` 的一半、`int64` 的四分之一。

从作业根目录运行：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.encode_datasets --output-dir data/tokenized/new-run --processes 8 --chunk-mib 4
```

后续训练按需读取：

```python
import numpy as np

tokens = np.load("data/tokenized/20260916-user-bpe/tinystories_train.npy", mmap_mode="r")
```

## 正式测试入口

`tests/adapters.py::get_tokenizer` 已连接到用户的 `BPE_tokenizer.Tokenizer`。2026-09-16 运行 `uv run --frozen pytest tests/test_train_bpe.py tests/test_tokenizer.py -q --tb=short`，结果为 **26 项通过、2 项跳过**；两项内存测试由官方测试在非 Linux 平台自动跳过。该结果没有证明任意字符串切块下的流式编码行为；本次数据处理使用明确的文档边界。
