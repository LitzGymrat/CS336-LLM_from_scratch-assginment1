# 单头 PyTorch attention 实验

正式实验已完成，20 组配置的实测表、图和 OOM 显存核算见 [测量报告](pytorch_attention_report.md)。本页保留脚本用法与推导依据。

依据本地 `cs336_assignment2_systems.pdf` 第 16 页的 `pytorch_attention`。
默认固定 batch=8、FP32、不加 causal mask，直接调用你 A1 的 `scaled_dot_product_attention`，不含 Q/K/V 投影、多头拼接或输出投影。

## 运行

在服务器的 assignment2-systems 目录运行，默认覆盖题目全部 20 组：

```bash
uv run python -m cs336_systems.benchmark_attention
```

默认每组预热 5 次，再测量 100 次前向和 100 次反向。需要检查某一组时，例如：

```bash
uv run python -m cs336_systems.benchmark_attention --d_models 16 --seq_lens 4096 --memory_profile --output_prefix outputs/pytorch_attention/d16_s4096
```

`--memory_profile` 另跑一个前向和反向，保存现有 memory_viz harness 的 pickle；这一步不参与耗时统计。全配置默认不记录调用栈，避免 profiler 开销影响计时。

## 测量边界

每一轮都先清掉 Q/K/V 的梯度，执行一个前向，再执行对应的反向。每次调用前后都同步 CUDA。前向保持 autograd 开启；反向用提前生成的、形状为 `(8, s, d)` 的输出梯度，不包含 loss 或优化器。每轮创建新计算图，反向后释放输出，不使用 `retain_graph=True`。

- `forward_mean_ms` / `backward_mean_ms`：两个阶段独立计时的平均值。
- `*_std_ms`：100 次耗时的总体标准差。
- `baseline_allocated_mib`：清空梯度后、前向开始前的活跃显存，含 Q/K/V 和外部输出梯度。
- `before_backward_allocated_mib`：前向完成、反向开始前的活跃显存，取各次测量最大值。
- `forward_retained_extra_mib`：上述值减去基线，只看前向新留下的输出和中间值，便于与张量公式对照。
- `forward_peak_allocated_mib` / `backward_peak_allocated_mib`：各阶段的峰值活跃显存，取所有测量的最大值，均包含基线。
- `status=oom`：该配置失败，`phase` 和 `error` 记录失败阶段及错误；保留此前采集到的原始耗时。

CSV/LaTeX/Typst 复用 `reporting.save_table()`；同名前缀的 JSON 保留每一次耗时。CPU 入口只用于小规模检查，CUDA 显存列留空，不作为 GPU 实验结果。

PyTorch 的 [memory_allocated](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.memory_allocated.html) 统计张量占用的活跃显存，不等于缓存池保留量或 nvidia-smi 的进程总显存；[max_memory_allocated](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_allocated.html) 给出测量范围内的峰值；[synchronize](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.synchronize.html) 等待该设备的工作完成。

## 从形状推导计算量

设 batch 为 B、序列长度为 s、单头维度为 d：

1. `Q, K, V` 的形状都是 `(B, s, d)`。
2. `Q @ K.transpose(-1, -2)` 得到 `(B, s, s)`，矩阵乘法 FLOPs 为 `2 B s² d`。
3. 缩放、softmax 保持 `(B, s, s)`。
4. 注意力概率矩阵乘 V 得到 `(B, s, d)`，矩阵乘法 FLOPs 为 `2 B s² d`。

因此前向主要矩阵乘法 FLOPs 是 `4 B s² d`。反向中，每个矩阵乘法分别计算两个输入的梯度，共四个矩阵乘法，所以是 `8 B s² d`。这些式子忽略 softmax、缩放等逐元素与归约运算；时间测量则包含它们。单头 attention 没有模型权重，梯度是对 Q/K/V 求的。

## 从当前实现推导显存

FP32 每个元素 4 字节。先分别数主要张量：

- Q/K/V：`3 B s d` 个 FP32 元素；脚本还预先持有一个输出梯度 `B s d`。
- 输出：`B s d` 个 FP32 元素。
- 一张注意力矩阵：`B s²` 个 FP32 元素。

对当前手写 softmax，用小输入的 `saved_tensors_hooks` 核对：Q/K/V 以原存储的引用保存；新保存的主要存储是指数矩阵和注意力概率矩阵，各 `B s²` 个 FP32 元素。指数矩阵虽然被两个反向节点引用，但只占一份存储，不能重复计数。此外保存了 `B s` 个 FP32 行和，以及 `B s` 个 int64 最大值索引。

于是，反向额外保存的张量存储估算为：

`M_saved_extra = 8 B s² + 12 B s` 字节。

包含脚本基线及输出的、反向开始前活跃显存估算为：

`M_before_backward ≈ 20 B s d + 8 B s² + 12 B s` 字节。

实际基线还可能含库工作空间等分配，因此比较时也看前向增量：`M_retained_extra ≈ 4 B s d + 8 B s² + 12 B s` 字节。此次 4090 短检查中，B=8、s=256、d=16 的基线为 16.750 MiB，反向前为 20.8984375 MiB，差值 4.1484375 MiB 正好与这个增量公式一致；这些是 3 次测量的检查结果，不是题目要求的正式 100 次实验。

这里只核算张量存储；实际 allocator 对齐、库工作空间等以实测为准。保存张量的清单依赖当前 softmax 实现，换成 torch.softmax、compile 或 FlashAttention 后需要重新核对。

固定 B=8，单张 FP32 注意力矩阵随 s 变化：

| s | 一张矩阵 | 两张主要保存矩阵 |
| --- | ---: | ---: |
| 256 | 2 MiB | 4 MiB |
| 1024 | 32 MiB | 64 MiB |
| 4096 | 512 MiB | 1 GiB |
| 8192 | 2 GiB | 4 GiB |
| 16384 | 8 GiB | 16 GiB |

这还没有算执行峰值：当前 softmax 在返回前可能同时持有输入分数、减最大值后的分数、指数矩阵和概率矩阵，共约四张 `(B,s,s)` 矩阵。s=16384 时，仅这四张矩阵就约 32 GiB，所以即使估算的“前向结束后保存量”约为 16 GiB，前向执行中也可能先 OOM。这个估算用于解释风险，最小 OOM 配置需要实际实验确定。

例如 B=8、s=16384、d=16：反向开始前的主要张量估算约为 16.041 GiB；当前 softmax 的前向临时峰值仅二次项就约 32 GiB。长度翻倍时，主要保存矩阵的显存变为四倍。FlashAttention 的方向是分块计算并结合反向重算，避免完整的 s×s 中间矩阵长期保存到显存。

正式实验的最小失败配置为 B=8、s=16384、d=16；诊断确认前向的 torch.exp 再申请 8 GiB 时失败。完整实测和逐项核算见测量报告；以上理论完成态估算与实际执行峰值仍分别列出。
