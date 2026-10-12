# TinyStories 验证集：增量更新与 naive 版本对比

日期：2026-09-15。当前 `BPE_tokenizer.py` 使用一次性建立的 pair 频次表及 pair → word 集合索引，每轮只合并受影响片段，并移除旧片段贡献、添加新片段贡献。

## 验证

- 当前源文件的三个官方训练测试全部通过：`uv run --frozen pytest tests/test_train_bpe.py -q --tb=short --durations=3`，本次总耗时 0.87 秒。
- 完整验证集产出的 `vocab.json` 和有序 `merges.json` 与此前 naive 基准逐字节一致。
- 最终词表 10,000 项，合并规则 9,743 条；最长 token 仍为 `b' accomplishment'`，15 bytes，ID 7427。
- 本次未修改用户的训练实现；运行前后的源码及保存的快照 SHA-256 均为 `70044b764ff9e6e8bfc65e961f986c175a9837f97ba73e786cf05693d6213cba`。

## 测量配置

同一台 Apple M5 Mac、同一 Python 3.12.14 环境，使用完整 `TinyStoriesV2-GPT4-valid.txt`（22,502,601 bytes）、词表上限 10,000 和特殊 token `<|endoftext|>`。输入 SHA-256 与此前一致：`6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d`。

比较此前一次普通 naive 基准与本次一次普通增量基准，均未启用 cProfile，没有同时运行其他训练。计时不含 Python 启动、导入与结果保存。以下不是多次运行的统计均值；读取和预分词的小幅变化包含正常运行波动，不能归因于合并算法优化。

```sh
uv run --frozen python -u -m cs336_basics.BPE_tokenizer.benchmark_validation --label incremental
```

## 结果

| 阶段 | 此前 naive | 本次增量更新 | 加速倍数 |
| --- | ---: | ---: | ---: |
| 读取 | 0.043 秒 | 0.016 秒 | — |
| 预分词与计数 | 2.831 秒 | 2.615 秒 | — |
| 合并训练 | 50.106 秒 | 1.928 秒 | 25.99× |
| 总训练时间 | 52.980 秒 | 4.558 秒 | 11.62× |

总耗时减少 91.40%，合并训练耗时减少 96.15%。峰值进程 RSS 从 119.84 MiB 到 119.34 MiB，基本持平。

本次验证集运行中，预分词占总时间约 57.37%。减少无关片段的合并及重复计数已经显著改善速度；如果继续优化，预分词现在值得关注。完整训练集仍需另行测量，不能直接套用这里的时间或阶段占比。

## 原始结果

- [本次 metrics.json](../../../data/bpe_benchmarks/20260915-115954-328027-incremental-valid-10000-baseline/metrics.json)
- [本次对比数据与一致性检查](../../../data/bpe_benchmarks/20260915-115954-328027-incremental-valid-10000-baseline/comparison.json)
- [本次完整输出目录](../../../data/bpe_benchmarks/20260915-115954-328027-incremental-valid-10000-baseline/)
- [此前 naive 基准报告](BENCHMARK_VALIDATION_RESULTS.md)
