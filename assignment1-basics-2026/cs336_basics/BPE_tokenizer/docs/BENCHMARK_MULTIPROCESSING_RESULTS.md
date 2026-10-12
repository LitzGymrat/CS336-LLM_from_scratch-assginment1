# TinyStories 验证集：多进程预分词与 cProfile

日期：2026-09-15。使用当前用户实现，未修改 `BPE_tokenizer.py`、分块辅助函数或启动脚本。

## 正确性检查

- 官方 `tests/test_train_bpe.py` 的三个测试均通过，实际目标是 `train_bpe_multiprocesses`，4 个进程、8 个目标块。
- 检查通过临时运行脚本将测试模块的调用入口指向多进程函数，未修改 `tests/adapters.py`；直接运行原有适配器仍测试串行入口。
- 三种配置各测三次，共九次普通运行。每次完整验证集的初始计数字典完全相同，13,111 种片段、5,419,001 次出现。
- 九次的 `vocab.json`、有序 `merges.json` 与此前 naive 基准逐字节一致：词表 10,000 项，合并规则 9,743 条。
- 单独的 profile 运行也与普通运行的计数、词表、合并规则完全一致。
- 实现 SHA-256：`aa3a7064a928c33e72d48f651be9d037cdcaaabe1d84fae895cc6b16d61d40ce`。
- 分块辅助函数 SHA-256：`f03de17baa3564ff2f6e17380904f092ffdd7eb8f21cf0816da38743509bcadf`。
- 初始计数字典规范化 JSON SHA-256：`811511626e0e0ee6fdf8ccb169364451b599b84a02dccb41effc09b5b4deb9c1`。

## 普通计时

机器为同一台 Apple M5 Mac，Python 3.12.14。输入完整 `TinyStoriesV2-GPT4-valid.txt`，22,502,601 bytes，词表上限 10,000，特殊 token 为 `<|endoftext|>`。输入 SHA-256 与此前一致：`6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d`。

每轮依次执行串行、4 进程、8 进程，重复三轮；没有同时运行其他训练，未启用 cProfile。两种并行配置实际均得到 8 块。

| 配置 | 准备与预分词中位数 | 合并训练中位数 | 总时间中位数 |
| --- | ---: | ---: | ---: |
| 串行预分词 + 增量合并 | 2.603 秒 | 1.909 秒 | 4.513 秒 |
| 4 进程、8 块 | 0.818 秒 | 1.938 秒 | 2.755 秒 |
| 8 进程、8 块 | 0.603 秒 | 1.982 秒 | 2.584 秒 |

每列分别取中位数，所以阶段中位数之和可能与总时间中位数略有差异。

| 配置 | 第一次总时间 | 第二次总时间 | 第三次总时间 |
| --- | ---: | ---: | ---: |
| 串行 | 4.513 秒 | 4.492 秒 | 4.541 秒 |
| 4 进程 | 2.755 秒 | 2.751 秒 | 2.773 秒 |
| 8 进程 | 2.603 秒 | 2.584 秒 | 2.561 秒 |

8 进程配置的总速度约为当前串行版本的 **1.75 倍**，耗时减少 **42.74%**。本次验证集上 8 进程比 4 进程稍快；合并仍顺序执行，因此增加预分词进程数不能让总时间按比例缩短。

计时调用的是实际 `train_bpe` / `train_bpe_multiprocesses`。准备阶段包含词表初始化、读取或定位边界、进程池启动、预分词、结果传输和计数汇总。合并阶段包含一份用于结果核对的浅拷贝。Python 启动、模块导入、结果保存、报告生成均在计时范围外。基准脚本只在测量进程内给合并函数包了一层轻量计时，并记录分块边界。

`parent_peak_rss_bytes` 仅表示主进程峰值内存，不能当作主进程和所有子进程的总内存。这里没有测量整个进程组的同步内存峰值。

## 怎样理解 profile

Profile 是程序运行过程的统计结果；profiling 指采集这些统计的过程。这里使用 Python 自带的 `cProfile`，观察函数调用次数和耗时。

- `ncalls`：函数调用次数。
- `tottime`：函数自身耗时，不包括它调用的其他函数。
- `cumtime`：函数及其内部调用的总耗时。

例如一个函数自身花 1 秒，调用别的函数花 2 秒，那么它的 `tottime` 是 1 秒，`cumtime` 是 3 秒。父函数的 `cumtime` 已经包含子函数耗时，不能把这些行全部相加。

官方说明：[Python profiler 文档](https://docs.python.org/3.12/library/profile.html)。分析工具本身有开销，所以用普通运行比较速度，用 profile 寻找开销的来源。

## 本次 profile 结果

单独对 **8 进程、8 块**运行一次。主进程和每个分块任务分别采集 cProfile；正常基准没有这些子任务包装和报告写入。

此次受分析工具影响的总耗时为 3.615 秒，其中准备阶段 1.394 秒、合并阶段 2.221 秒。不能用 3.615 秒替换普通基准的 2.584 秒中位数。

主进程的应用函数摘录：

| 函数 | ncalls | tottime | cumtime |
| --- | ---: | ---: | ---: |
| `train_bpe_loop` | 1 | 1.299 秒 | 2.220 秒 |
| `max`，仅训练循环中的调用 | 19,486 | 0.722 秒 | 0.722 秒 |
| `merge_pair` | 68,766 | 0.065 秒 | 0.096 秒 |
| `count_pair` | 1 | 0.019 秒 | 0.024 秒 |

训练 9,743 轮，每轮两次 `max`，对应 19,486 次调用。原始完整主进程报告的 `max` 还包括进程池等代码的额外调用；上表从调用者记录中只提取 `train_bpe_loop` 的贡献。

增量维护生效：`count_pair` 只调用一次，`merge_pair` 不再每轮遍历全部片段。普通基准中合并阶段约占总时间 76.7%。结合源码，下一处值得检查的是每轮全局扫描 `pair_counts`、筛选候选并取最大 pair 的开销；`max` 已明确占据约 0.722 秒，循环自身的 1.299 秒还包含候选遍历及其他 Python 操作，不能仅凭函数级报告把它全部归给某一行。

工作进程报告汇总了 8 个任务，实际观察到 8 个不同的工作进程 PID：

| 函数/操作 | 总调用次数 | 各任务 tottime 之和 | 各任务 cumtime 之和 |
| --- | ---: | ---: | ---: |
| `pretokenize_and_count` | 8 | 7.562 秒 | 9.953 秒 |
| 列表 `append` | 22,145,475 | 1.593 秒 | 1.593 秒 |

这些任务并行运行，单个任务受 profile 影响的时间为 1.184–1.277 秒。汇总的 9.953 秒是各任务时间相加，包含重叠时间，不是用户实际等待时间。逐字节构建和计数仍有大量工作，但此次普通基准的整体瓶颈已主要落在合并阶段。

主进程报告还包含管道读取、锁和等待等条目。等待工作进程的时间可能记在这些条目下，不能据此认定磁盘读取或锁本身是主要计算瓶颈。

## 复现与文件

从 `assignment1-basics/` 运行，每条命令单独执行：

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --processes 0
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --processes 4 --chunks 8
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --processes 8 --chunks 8
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_training --processes 8 --chunks 8 --profile
```

- [测量脚本](../benchmark_training.py)
- [九次普通计时与一致性检查](../../../data/bpe_benchmarks/20260915-192255-multiprocessing-comparison/comparison.json)
- [主进程 profile 报告](../../../data/bpe_benchmarks/20260915-192434-653948-multiprocess-8p-8c-valid-10000-profile/parent_profile.txt)
- [工作进程 profile 汇总](../../../data/bpe_benchmarks/20260915-192434-653948-multiprocess-8p-8c-valid-10000-profile/workers_profile.txt)
- [提取的关键函数统计](../../../data/bpe_benchmarks/20260915-192434-653948-multiprocess-8p-8c-valid-10000-profile/profile_summary.json)
- [profile 完整目录与源码快照](../../../data/bpe_benchmarks/20260915-192434-653948-multiprocess-8p-8c-valid-10000-profile/)

本报告记录完整验证集实验；随后完成的完整训练集实验见 [TinyStories 正式训练结果](TINYSTORIES_TRAINING_RESULTS.md)。
