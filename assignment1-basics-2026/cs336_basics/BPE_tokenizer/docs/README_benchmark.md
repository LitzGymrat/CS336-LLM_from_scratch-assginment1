# TinyStories 验证集基准

从 `assignment1-basics/` 目录运行当前实现：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_validation --label incremental
```

固定使用 `data/TinyStoriesV2-GPT4-valid.txt`、词表上限 10,000 和特殊 token `<|endoftext|>`。
脚本依次调用 `BPE_tokenizer.py` 中与 `train_bpe` 相同的步骤，分别记录读取、预分词、合并训练的时间。
基准脚本直接运行当前训练实现。`--label` 只用于标记结果版本，不改变算法；省略时使用 `current`。

需要分析函数耗时，等普通基准结束后单独运行：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_validation --label incremental --profile
```

`--profile` 使用 cProfile，会显著增加运行时间；不能把这个模式的耗时当作普通运行耗时。
每次运行写入独立的 `data/bpe_benchmarks/<时间>-<版本标签>-valid-10000-<模式>/` 目录：

- `metrics.json`：阶段时间、峰值内存、词表大小、最长 token、数据与源码 SHA-256。
- `vocab.json`：token ID 到字节内容，字节以十六进制字符串保存。
- `merges.json`：按学习顺序保存的合并规则，每个字节串同样用十六进制表示。
- `implementation.py`：本次运行的训练实现快照，便于之后比较。
- `training.prof`、`profile.txt`：仅 profile 模式生成的详细分析与文本报告。

计时不包含 Python 启动、模块导入和结果序列化。峰值内存取 macOS 的进程峰值 RSS，包含解释器和导入模块，单位转换为 MiB。
基准和 profile 的词表及合并规则应一致；比较优化版本时，也应固定数据、词表大小和特殊 token，并检查输出一致性。

这里用验证集作为调试语料，结果用于性能比较；完整 TinyStories 训练集的启动方法和结果见下文。

## 多进程入口计时与 profile

`benchmark_training.py` 直接调用实际训练入口。`--processes 0` 使用串行预分词，正数使用相应数量的工作进程；`--chunks` 控制目标分块数。下面各条命令依次执行：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --processes 0
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --processes 4 --chunks 8
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --processes 8 --chunks 8
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --processes 8 --chunks 8 --profile
```

准备阶段计时包含初始化、I/O、预分词、进程池启动及计数汇总。除词表、合并规则、计时和源码快照外，还保存规范化的 `initial_counts.json`，便于验证分块结果与串行一致。

并行 profile 分别采集主进程及每个分块任务，生成 `parent_profile.txt` 和 `workers_profile.txt`。工作进程汇总耗时包含并行重叠，不能当作总运行时间。记录的主进程峰值 RSS 也不能当作整个进程组的内存峰值。

[本次多进程性能与 profile 解读](BENCHMARK_MULTIPROCESSING_RESULTS.md)

## 完整 TinyStories 训练集

固定词表大小 10,000、特殊 token `<|endoftext|>`，使用 8 个工作进程、8 个文件块：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --split train --processes 8 --chunks 8 --sample-memory
```

`--split train` 选择完整训练集；省略时仍使用验证集。结果目录中的数据集标记也相应变为 `train`。

`--sample-memory` 每 0.25 秒记录主进程及其所有子进程的 RSS 总和，并报告采样峰值。它是近似的进程组内存指标，可能漏掉很短的尖峰，也可能重复计算共享页；它与仅统计主进程的 `parent_peak_rss_bytes` 不同。采样覆盖训练阶段，结果序列化和文件哈希在训练计时与采样结束后进行。

[已完成的完整 TinyStories 训练记录](TINYSTORIES_TRAINING_RESULTS.md)

## OpenWebText 32K 训练

在 `assignment1-basics/` 目录运行验证集试训：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --dataset owt --split valid --vocab-size 32000 --processes 8 --chunks 128 --sample-memory --skip-counts-export
```

正式训练使用相同命令并将 `--split valid` 改成 `--split train`。`--skip-counts-export` 关闭测量脚本额外保存的初始频次表副本及 JSON 导出，降低大语料的测量开销；仍记录初始片段种类和总频次。结果文件名包含数据集、分割、进程数、块数和词表大小。

单独比较预分词阶段时，依次执行：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_pretokenization --processes 8 --chunks 128
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_pretokenization --processes 128 --chunks 128
```

这个测量在实际多进程入口构建完整频次表后结束，记录全部计数的 SHA-256，便于核对并行配置的一致性。它不生成训练好的词表。计数哈希采用带长度字段的二进制记录，与旧的 `initial_counts.json` 哈希格式不同。

OWT 训练完成后，可以用 `verify_training_artifacts` 模块检查结果目录；它会读取下载校验记录，检查词表、合并顺序、分块边界和小样本编码解码，生成 `validation.json`。

[OWT 验证集结果、128 进程对比和正式训练日志](OWT_VALIDATION_RESULTS.md)
