# CS336 Assignment 2：GPU 注意力与性能实验

本目录保存 2026 版 A2 的当前个人实现、官方测试、benchmark、profiling harness 和实验记录。官方起点：<https://github.com/stanford-cs336/assignment2-systems>，提交 `ca8bc81a59b70516f7ebb2da4808daade877c736`。

## 完成范围

完成基础计时/Nsight/混合精度/显存分析、激活 checkpointing 对照、普通注意力 benchmark、torch.compile 对照、FlashAttention PyTorch 参考、Triton 前向与 PyTorch 重算反向，以及 RTX 4090 上的选定性能实验。DDP、优化器分片、FSDP 和并行分析暂缓。Triton backward 未实现；没有完整模型接入自写 FlashAttention 的端到端性能结果。

## 环境与入口

Python 3.12 / PyTorch 2.11；本地依赖指向仓库中的 `../assignment1-basics-2026`。这个发布副本只改了依赖路径和显存脚本的源码归档路径，没有修改 kernel、模型、优化器或计时算法。原有 `assignment1-basics/` 的 KV-cache/AttnRes 实现保持原样。

```bash
uv sync --frozen --python 3.12
uv run pytest tests/test_attention.py
uv run python -m cs336_systems.benchmark --help
uv run python -m cs336_systems.benchmark_flash_forward --help
```

Triton 由匹配的 Linux PyTorch 环境依赖提供；CUDA 实验需要 Linux/NVIDIA，Mac 仅用于 PyTorch 参考与报告整理。RTX 4090 的实验不能替代题目要求的 B200 性能结果。

## 报告

| 主题 | 报告 |
|---|---|
| 基础计时与预热 | [benchmark_report.md](docs/benchmark_report.md) |
| Nsight / NVTX 与 kernel 分析 | [nsight_report.md](docs/nsight_report.md) |
| 混合精度 | [mixed_precision_report.md](docs/mixed_precision_report.md) |
| 模型与 block 显存 | [memory_profiling_report.md](docs/memory_profiling_report.md)、[block_memory_report.md](docs/block_memory_report.md) |
| 激活 checkpointing 分组 | [gradient_checkpointing_report.md](docs/gradient_checkpointing_report.md) |
| 普通注意力 | [pytorch_attention_report.md](docs/pytorch_attention_report.md) |
| torch.compile | [torch_compile_report.md](docs/torch_compile_report.md) |
| FlashAttention 短序列前向/反向 | [flash_attention_benchmark_report.md](docs/flash_attention_benchmark_report.md) |
| FlashAttention 完整尺寸前向扫描 | [flash_forward_sweep_report.md](docs/flash_forward_sweep_report.md) |

较大的 profile 和快照在根目录 README 所链接的 Release 中；解压后可恢复完整 outputs 目录。现有报告记录了部分课程配置因显存/预算采用替代规模的事实，未宣称完成整个作业的正式实验矩阵。
