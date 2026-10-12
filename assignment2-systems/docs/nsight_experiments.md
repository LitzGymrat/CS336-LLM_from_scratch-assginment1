# Nsight 实验、记录与分析执行方案

2026-10-08，依据 [作业 §2.1.4 nsys_profile](../cs336_assignment2_systems.pdf) 和本次选定方案。状态：2026-10-08 已执行，15 份正式采集及 9 次容量探测完成；分析见 [nsight_report.md](nsight_report.md)。medium/1024 三种模式实际 OOM，按用户选择保留 batch=4 的显存限制。

## 一、实验目标和固定模式

每个可运行的“模型×上下文长度”配置，分别采集 **forward、forward_backward、train**。采用三个独立进程，直观比较各模式，不再从 train 报告替代独立 forward_backward 采集。

| 模式 | 实际执行 | 分析目标 |
| --- | --- | --- |
| forward | 仅模型前向 | (a) 前向总时间和旧 Python 结果对照；(b) 前向主要 kernel 及调用数；(c) 非矩阵乘法 kernel；(e) 注意力中的 softmax 与矩阵乘法 |
| forward_backward | 前向、loss、反向 | (b) 累计时间最多的 kernel 相比 forward 是否变化 |
| train | 清空梯度、前向、loss、反向、AdamW 更新 | (d) 各阶段及完整 step 的矩阵乘法/其他 kernel 时间占比，与 forward 比较 |

(b) 的题目核心是找 **Total Time（累计 GPU 时间）最多**的 kernel，再记录它单次前向的调用数。调用次数最多、Avg 和 Max 可以补充观察，不能把它们代替核心指标。[NVIDIA 统计字段说明](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)。

## 二、模型、长度与运行条件

| 模型 | d_model | d_ff | 层数 | 头数 | 候选长度 |
| --- | ---: | ---: | ---: | ---: | --- |
| small | 768 | 3072 | 12 | 12 | 256 / 512 / 1024 |
| medium | 1024 | 4096 | 24 | 16 | 256 / 512 / 1024 |

固定词表 10,000、batch=4、FP32、SwiGLU、pre-norm、RoPE、theta=10,000、lr=0.001。正式采集 warmup=5、steps=10；容量探测 warmup=1、steps=2。所有 GPU 实验串行运行，单独进程结束后再开始下一项。

当前 forward 不运行 backward，但 autograd 开启；沿用已有 Python 前向实验设置并在报告说明。当前随机初始化未显式设置 seed，记录该事实，不声称旧实验与新实验使用完全相同的权重。若改变 autograd/精度/batch，则视为新实验条件并重新探测容量。

这是一组候选配置；实际可运行的最大长度由各模式的 OOM 探测确认。每个模式都保留最长成功档及下一档 OOM，不把预估写成实测。预计 small 的边界是 1024→2048，medium 是 512→1024，但必须执行确认。预估表见 [memory_estimates.csv](../outputs/nsight_plan/memory_estimates.csv)。

## 三、执行顺序

### 1. 补齐注意力的两个标签并检查标注

执行时先完成机械标注，保留模型计算逻辑：

- 外层 `attention`：包住完整 Multihead_self_attention.forward 的计算，也就是 QKV 投影、RoPE、QKᵀ、mask、softmax、PV、输出投影；输入 RMSNorm 和残差加法不在范围里。
- 内层 `attention_softmax`：只包住 scaled_dot_product_attention 中现有 softmax 调用，包含其 max、减法、exp、sum、除法。
- 两个范围沿用 CUDA/标注开关控制，不新增阶段同步。

只需要这两个标签。矩阵乘法从 attention 范围关联的完整 kernel 名称中识别和累加；本轮不额外要求给 QKᵀ、PV 分别贴标签。FLOPs 的范围也与完整 MSA 对齐，包括四个投影和两次注意力矩阵乘法。

标注后做小规模输出/测试核对，确保标签没有改变数学计算；先采一份 small/512/forward，验证 measurement 只包含正式 10 步，forward→attention→attention_softmax 的关联可见。本轮已在测速脚本中临时包住原有 attention 和 softmax 调用，A1 数学计算未修改；CUDA 小例子的输出和梯度逐位核对一致。

### 2. 各模式独立探测容量直到 OOM

上一轮 small、medium 的三种模式在长度 512 都已成功，有原始 CSV 可追溯。因此从 1024 继续探测，不重跑旧 Python 基线。

- small：每个模式先测 1024，成功后测 2048；若 2048 仍成功，继续 4096、8192……，直到首次 OOM。
- medium：每个模式从 1024 开始；若成功，继续 2048、4096……，直到首次 OOM。
- 一种模式 OOM 后停止该模式继续增大，记录最长成功档和下一档失败，然后进行下一个模式。
- 若从 1024 已 OOM，最长成功档引用已有 512 成功记录。非 OOM 的程序错误先定位，不能记成显存上限。

探测脚本复用 benchmark.main；一个进程只有该配置的模型。记录 PyTorch allocator 的 peak_allocated/peak_reserved、退出码和完整 stderr。两项峰值不相加，也不代表驱动和 Nsight 的全部显存。[PyTorch 峰值定义](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.max_memory_allocated.html)。

### 3. 采集成功配置

优先完成 small/512 三种模式作为一组，确认报告有效，再按 small/256、small/最大成功长度、medium/256、medium/512 的顺序采集；若 medium/更长长度成功，就加入对应配置。每个可运行配置的三种模式分别保留报告；模式之间的容量上限可能不同。

候选 2×3×3 对应 18 个采集任务。若预估吻合，medium/1024 的三个任务标为 OOM，其他五个配置预计产生 15 份正式报告。实际是否成功均由运行确认。若上限比预估更长，将最大成功长度追加到表中；保持三个符合题目条件的代表长度并保留已有数据。

正式 nsys 采集若也发生 OOM，记录该模式 profiler 下的限制，并使用前一档可运行长度；不自动改变 batch/精度。每次进程的 stdout/stderr 保存到日志，避免 OOM 后只剩口头记忆。

### 4. 导出、核对并分析

每份成功报告导出 kernel summary、NVTX GPU projection 和必要的逐事件明细。确认 measurement/step 的次数与 10 次正式测量相符；按标签关联 kernel，父子范围的同一个 kernel 不重复加到同一分母中。已有 small/512 的旧 trace 仍保留，新注意力标注的 trace 单独保存。

## 四、每次记录哪些数据

| 记录表 | 保存内容 |
| --- | --- |
| experiment_manifest.csv | 模型/长度/mode/超参数/精度/autograd/预热/步数、状态、退出码、报告/日志路径、张量与缓存峰值、旧 Python 对照路径 |
| capacity_results.csv | 每模型每 mode 的最长成功长度、首次 OOM 长度、成功峰值、成功/失败日志、profiler 下最大成功长度 |
| kernel_observations.csv | 完整 kernel 名称、分类依据、累计时间、调用次数、每步调用数、Avg/Max、所选范围的时间占比、来源报告 |
| stage_observations.csv | forward/loss/backward/optimizer/total 各范围的 GPU span、kernel 累计时间、矩阵乘法/其他 kernel 时间及占比 |
| attention_observations.csv | 完整 MSA 的矩阵乘法时间、内部 softmax 全部 kernel 时间、同范围 FLOPs、时间比与 FLOPs 比 |

观察模板已建在 outputs/nsight_plan。空白字段就是待测数据，不填 0 冒充测量。每份新报告同时导出 benchmark 的计时 CSV/LaTeX/Typst；这些 profiler 内的 Python 时间另存，不能替代 (a) 的历史对照。

运行时附上本次源码 revision/文件 SHA256、GPU 名称及容量、Python/PyTorch/CUDA/Nsight 版本和关键精度设置。记录归因时保留完整 kernel 模板名，不只保留 vectorized_elementwise 这种公共前缀。

## 五、(a)–(e) 的分析口径

### (a) 前向时间与已有 Python 测量

旧 Python 结果只在相同配置下比较：

| 模型 | 长度 | 已有 Python 前向 mean ± std（ms） | 原始文件 |
| --- | ---: | ---: | --- |
| small | 512 | 25.85 ± 0.06 | [small_forward.csv](../outputs/benchmark/small_forward.csv) |
| medium | 512 | 80.62 ± 0.23 | [medium_forward.csv](../outputs/benchmark/medium_forward.csv) |

比较一次 forward 的 GPU 完整执行跨度和旧 Python 经过时间，另列 kernel 累计 GPU 时间，说明它们的口径不同。GPU projection 由第一个关联 GPU 操作开始到最后一个结束；内部 CPU NVTX 宽度不等于 GPU 完成时间。十步总时间需要转成每步量再比较。256/1024 无相同长度的旧计时，标明无历史对照，不跨长度混比，也不新增 Python 基线重测任务。[NVIDIA NVTX GPU projection 说明](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html#nvtx-gpu-proj-sum-nvtx-gpu-projection-summary)。

### (b) 前向与前向+反向的主要 kernel

在 forward 和 forward_backward 的正式范围分别按 Total Time 降序排序。记录榜首的完整名称、累计时间和 Instances，并确认它在一次前向内的调用次数。若采用十步累计次数除以 10，先确认各步调用模式一致。最多调用次数、平均/最大单次耗时另外保留，避免与累计最耗时混淆。forward_backward 的主比较范围包含该 mode 实际执行的 loss；可额外按阶段区分其贡献。

### (c) 非矩阵乘法 kernel 的归因

列出 forward 的主要非矩阵乘法 kernel，按完整模板名和 NVTX 关联判断运算。vectorized_elementwise 是通用逐元素实现；SwiGLU 的乘法/sigmoid、注意力缩放、RoPE、RMSNorm、softmax 等都可能产生逐元素 kernel，不能只凭这个公共名字归到 FFN。现有小模型报告里既有 MulFunctor，也有 exp_kernel_cuda；补标签后用实际事件关联核对归属。[PyTorch CUDA 逐元素实现](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/CUDALoops.cuh)。

### (d) 阶段和整体的矩阵乘法占比

对 train 的 forward、loss、backward、optimizer、total 分别计算：

```text
矩阵乘法占比 = 所选范围内 matmul kernel 的累计 GPU 时间 / 同范围内所有 kernel 的累计 GPU 时间
其他 kernel 占比 = 1 - 矩阵乘法占比
```

以相同配置的 forward 为对照。GPU 空闲、CPU 调度、同步等待、内存复制不混入这项 kernel 时间占比；需要时单列。不同阶段与 total 分别统计，不能把各阶段的百分比直接相加。逐元素、归约、mask、index 等可以另做分类，保留判定依据。

### (e) softmax 与注意力矩阵乘法

选一次前向中的所有 MSA 层，统计 attention_softmax 的全部关联 kernel 时间，并汇总 attention 外层的矩阵乘法 kernel 时间。两项都用同一次前向、相同层数和 batch，比较时间比；按同范围统计 FLOPs，再比较 FLOPs 比。完整 MSA 矩阵乘法 FLOPs 可复用既有 8Bsd²+4Bs²d 公式；softmax 明确其运算计数约定。不要把整个 attention 范围的 wall time 当作矩阵乘法时间。

## 六、输出的报告、表和图

执行完成后生成 docs/nsight_report.md，保留每个配置的观测值，并按 (a)–(e) 用中文写简短结论。

- 容量表：模型×模式的最后成功长度与首个 OOM。
- 前向时间对照表/图：长度 512 的旧 Python 与 Nsight 时间，不同计时口径分别列出。
- 主要 kernel 表：forward 与 forward_backward 的累计耗时排名、每步调用数和非矩阵乘法归因。
- 运算占比图：train 的各阶段及总体，相同配置 forward 的对照。
- 注意力图：softmax 与 MSA 矩阵乘法的 GPU 时间比和 FLOPs 比，两种量纲分开画。

图表通过已有 reporting 工具和 CSV 自动生成；每个图注明模型、长度、mode、精度、范围和单位。全部结论链接回真实 trace 和原始 CSV，不将预估、OOM 或未运行任务当作性能结果。

## 七、命令清单

在服务器 /root/autodl-tmp/cs336/assignment2-systems 中运行。下列为本轮执行方案的命令清单。实际成功、OOM 与日志以 experiment_manifest.csv 和 capacity_probe_manifest.csv 为准；避免直接重跑覆盖已完成的报告。

```sh
mkdir -p outputs/nsight_run/profiles outputs/nsight_run/memory_probes outputs/nsight_run/logs outputs/nsight_run/analysis
```

### 容量探测：三种模式分别到 OOM

下面是按当前预估准备的起始探测命令。若下一档成功，执行者会把 --context_len 翻倍并同步更改输出前缀，追加记录，直到实际 OOM。正式执行时会保存 stdout/stderr 日志。

**small / forward**

```sh
uv run python measure_gpu_memory.py --device cuda --mode forward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 1024 --warmup 1 --steps 2 --output_prefix outputs/nsight_run/memory_probes/small_s1024_forward
```

1024 成功后执行：

```sh
uv run python measure_gpu_memory.py --device cuda --mode forward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 2048 --warmup 1 --steps 2 --output_prefix outputs/nsight_run/memory_probes/small_s2048_forward
```

**small / forward_backward**

```sh
uv run python measure_gpu_memory.py --device cuda --mode forward_backward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 1024 --warmup 1 --steps 2 --output_prefix outputs/nsight_run/memory_probes/small_s1024_forward_backward
```

1024 成功后执行：

```sh
uv run python measure_gpu_memory.py --device cuda --mode forward_backward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 2048 --warmup 1 --steps 2 --output_prefix outputs/nsight_run/memory_probes/small_s2048_forward_backward
```

**small / train**

```sh
uv run python measure_gpu_memory.py --device cuda --mode train --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 1024 --warmup 1 --steps 2 --output_prefix outputs/nsight_run/memory_probes/small_s1024_train
```

1024 成功后执行：

```sh
uv run python measure_gpu_memory.py --device cuda --mode train --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 2048 --warmup 1 --steps 2 --output_prefix outputs/nsight_run/memory_probes/small_s2048_train
```

**medium / forward**

已有长度 512 成功；从 1024 开始：

```sh
uv run python measure_gpu_memory.py --device cuda --mode forward --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 1024 --warmup 1 --steps 2 --output_prefix outputs/nsight_run/memory_probes/medium_s1024_forward
```

**medium / forward_backward**

已有长度 512 成功；从 1024 开始：

```sh
uv run python measure_gpu_memory.py --device cuda --mode forward_backward --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 1024 --warmup 1 --steps 2 --output_prefix outputs/nsight_run/memory_probes/medium_s1024_forward_backward
```

**medium / train**

已有长度 512 成功；从 1024 开始：

```sh
uv run python measure_gpu_memory.py --device cuda --mode train --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 1024 --warmup 1 --steps 2 --output_prefix outputs/nsight_run/memory_probes/medium_s1024_train
```

### 正式采集：每个配置三条

容量成功且注意力标签就绪后执行。medium/1024 的每个模式都先经过容量验证。命令使用现有 measure_gpu_memory.py 入口，它仍调用同一个 benchmark.main，只额外记录整个进程的 PyTorch 显存峰值。

#### C1：small / 长度 256

**forward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s256_forward -- python measure_gpu_memory.py --device cuda --mode forward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 256 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s256_forward_timing
```

**forward_backward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s256_forward_backward -- python measure_gpu_memory.py --device cuda --mode forward_backward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 256 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s256_forward_backward_timing
```

**train**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s256_train -- python measure_gpu_memory.py --device cuda --mode train --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 256 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s256_train_timing
```

#### C2：small / 长度 512

**forward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s512_forward -- python measure_gpu_memory.py --device cuda --mode forward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 512 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s512_forward_timing
```

**forward_backward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s512_forward_backward -- python measure_gpu_memory.py --device cuda --mode forward_backward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 512 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s512_forward_backward_timing
```

**train**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s512_train -- python measure_gpu_memory.py --device cuda --mode train --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 512 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s512_train_timing
```

#### C3：small / 长度 1024

**forward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s1024_forward -- python measure_gpu_memory.py --device cuda --mode forward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 1024 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s1024_forward_timing
```

**forward_backward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s1024_forward_backward -- python measure_gpu_memory.py --device cuda --mode forward_backward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 1024 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s1024_forward_backward_timing
```

**train**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s1024_train -- python measure_gpu_memory.py --device cuda --mode train --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 1024 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s1024_train_timing
```

#### C4：medium / 长度 256

**forward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/medium_s256_forward -- python measure_gpu_memory.py --device cuda --mode forward --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 256 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/medium_s256_forward_timing
```

**forward_backward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/medium_s256_forward_backward -- python measure_gpu_memory.py --device cuda --mode forward_backward --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 256 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/medium_s256_forward_backward_timing
```

**train**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/medium_s256_train -- python measure_gpu_memory.py --device cuda --mode train --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 256 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/medium_s256_train_timing
```

#### C5：medium / 长度 512

**forward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/medium_s512_forward -- python measure_gpu_memory.py --device cuda --mode forward --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 512 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/medium_s512_forward_timing
```

**forward_backward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/medium_s512_forward_backward -- python measure_gpu_memory.py --device cuda --mode forward_backward --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 512 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/medium_s512_forward_backward_timing
```

**train**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/medium_s512_train -- python measure_gpu_memory.py --device cuda --mode train --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 512 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/medium_s512_train_timing
```

#### C6：medium / 长度 1024（先确认容量）

**forward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/medium_s1024_forward -- python measure_gpu_memory.py --device cuda --mode forward --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 1024 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/medium_s1024_forward_timing
```

**forward_backward**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/medium_s1024_forward_backward -- python measure_gpu_memory.py --device cuda --mode forward_backward --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 1024 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/medium_s1024_forward_backward_timing
```

**train**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/medium_s1024_train -- python measure_gpu_memory.py --device cuda --mode train --vocab_size 10000 --d_model 1024 --d_ff 4096 --num_layers 24 --num_heads 16 --batch_size 4 --context_len 1024 --nvtx --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/medium_s1024_train_timing
```

### 导出统计示例

```sh
nsys stats --report cuda_gpu_kern_sum,nvtx_gpu_proj_sum outputs/nsight_run/profiles/small_s512_forward.nsys-rep
```

各模式同样导出。总表用于初看和排序；细分阶段/注意力范围时依赖事件与 NVTX 的关联，不能只从整份总表推测模块归属。原始 trace 保留，CSV/图表/中文分析集中到 outputs/nsight_run/analysis 与 docs/nsight_report.md。

执行索引：[18 个候选采集任务](../outputs/nsight_plan/experiment_manifest.csv)、[9 个起始容量探测任务](../outputs/nsight_plan/capacity_probe_manifest.csv)。索引状态均为 not_run。
