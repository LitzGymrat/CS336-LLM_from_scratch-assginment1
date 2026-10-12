# 完整 TinyStories 训练结果

日期：2026-09-15。本次使用已验证的用户实现完成完整训练集上的 BPE 训练，保存了词表、按学习顺序排列的合并规则及源码快照。

## 配置与命令

- 输入：`data/TinyStoriesV2-GPT4-train.txt`，2,227,753,162 bytes，约 2.23 GB。
- 数据 SHA-256：`6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443`，与下载时校验值一致。
- 词表大小：10,000，包括 256 个基础字节和特殊 token `<|endoftext|>`。
- 并行预分词：8 个工作进程，8 个文件块；合并训练顺序执行。
- 环境：macOS 26.6.2 arm64，Python 3.12.14。
- 普通运行，未开启 cProfile；开启进程组内存采样。
- 用户训练源码 SHA-256：`aa3a7064a928c33e72d48f651be9d037cdcaaabe1d84fae895cc6b16d61d40ce`，与刚完成验证集测试的版本相同。

从 `assignment1-basics/` 启动：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --split train --processes 8 --chunks 8 --sample-memory
```

## 时间与内存

| 阶段 | 耗时 |
| --- | ---: |
| 准备、并行预分词与计数汇总 | 66.878 秒 |
| 合并训练 | 8.146 秒 |
| 总训练时间 | **75.030 秒，约 1 分 15 秒** |

计时不含 Python 启动、模块导入、结果保存、输入文件哈希和后续校验。准备阶段包含初始化词表、寻找分块边界、工作进程启动、读取各块、预分词、传输及汇总计数。合并计时包含一份供验证使用的计数字典浅拷贝。

训练期间主进程及其子进程 **RSS 总和的采样峰值为 4,068,163,584 bytes，约 3.789 GiB**。采样间隔 0.25 秒，共取得 287 次样本，最多观察到 10 个进程（主进程、8 个工作进程及资源追踪进程）。这是采样得到的近似峰值；共享页可能重复计入，短暂尖峰可能漏采。它不同于整个进程组独占的物理内存。

主进程单独的峰值 RSS 为 158,040,064 bytes；不能用这个数代替多进程总内存。

在完整训练集上，准备与预分词约占总时间的 89%。相比验证集，瓶颈比例发生了变化：预分词要处理全部文本，合并阶段处理的是按片段汇总后的频次字典。后续优化应以目标数据规模的测量为依据。

## 训练产物

- 不同预分词片段：59,933 种。
- 预分词片段总出现次数：536,592,168。
- 词表条目：10,000，ID 连续为 0–9,999。
- 特殊 token ID：256。
- 合并规则：9,743 条，按学习顺序保存。

最长 token 有三个，均为 15 bytes：

| ID | token |
| --- | --- |
| 7160 | `b' accomplishment'` |
| 9143 | `b' disappointment'` |
| 9379 | `b' responsibility'` |

这些都是完整英文单词加前导空格。空格来自预分词规则，训练后成为 token 内容的一部分，因此结果合理。

JSON 中用十六进制字符串保存原始字节：`vocab.json` 是 token ID 到字节内容的映射；`merges.json` 中每项是两个 token 的字节内容，列表顺序就是合并学习顺序。

## 校验

重新读取保存的 JSON 后，已核对基础 256 个单字节 token、特殊 token、连续编号、每条合并的两个来源 token 是否已存在，以及其拼接结果是否对应新增词表条目。普通 token 中没有 `<|` 标记。

已核对初始片段计数的条目数、出现次数总和及文件哈希，实际分块覆盖整个文件，全部内部边界均位于 `<|endoftext|>` 开头。训练源码和分块辅助函数与启动时的快照保持一致。

这些是产物结构和配置校验；本次没有第二次完整训练或独立完整训练集参考结果对照。此前已完成三个官方多进程训练测试，以及验证集上串行和并行的完整计数、词表、合并规则一致性检查。

## 文件

- [词表 vocab.json](../../../data/bpe_benchmarks/20260915-193145-068347-multiprocess-8p-8c-train-10000-baseline/vocab.json)
- [有序合并规则 merges.json](../../../data/bpe_benchmarks/20260915-193145-068347-multiprocess-8p-8c-train-10000-baseline/merges.json)
- [原始测量 metrics.json](../../../data/bpe_benchmarks/20260915-193145-068347-multiprocess-8p-8c-train-10000-baseline/metrics.json)
- [校验与最长 token validation.json](../../../data/bpe_benchmarks/20260915-193145-068347-multiprocess-8p-8c-train-10000-baseline/validation.json)
- [完整输出目录与源码快照](../../../data/bpe_benchmarks/20260915-193145-068347-multiprocess-8p-8c-train-10000-baseline/)
- [此前验证集性能分析](BENCHMARK_MULTIPROCESSING_RESULTS.md)
