# Nsight Systems 逐步实现清单

对应作业 §2.1.4 `nsys_profile`，PDF 第 4–6 页。2026-10-08 最新检查：普通计时、四种 mode、NVTX 开关、measurement/step 和计算阶段标签已接入。注意力标签及 15 份 trace 已完成 CUDA 核对；结果见 [nsight_report.md](nsight_report.md)。不同配置的命令、容量探测和显存预估见 [nsight_experiments.md](nsight_experiments.md)。

作业规定两种模型规模、三个大于 128 的二次幂序列长度，最大长度达到显存可容纳的上限。本文的标签名、实现顺序和第一次运行配置是学习建议。先让 small、序列长度 512 的一份报告能正确采集和阅读，再扩展实验组合。

## 最终要得到什么

一份 `.nsys-rep` 文件，能在 Nsight Systems 中查看时间线和 CUDA kernel 统计。我们要能从报告中辨认正式测量、单次 step、前向、loss、反向、优化器，以及注意力中的两个矩阵乘法和 softmax。

数据流：现有 benchmark 执行计算 → NVTX 标记代码区间 → `nsys profile` 采集区间内的 CUDA 活动 → 在报告中查看时间与调用次数 → 回答 (a)–(e)。

本文依据：[作业原文](../cs336_assignment2_systems.pdf)、[PyTorch 2.11 NVTX API](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.nvtx.range.html)、[NVIDIA 采集说明](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[NVIDIA 统计说明](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)。采集命令留给你在服务器启动；本文没有启动新实验。

## 1. 增加标注开关并划定正式采集区间

**目的：**开启 profiling 时，让工具识别正式测量；普通测速和 Mac 上的 CPU 调试继续使用原来的计算流程。

先在 `parse_args()` 增加一个布尔命令行开关 `--nvtx`，采用 `store_true`：省略时为 False，传入时为 True。通过 `main()` 把 `args.nvtx` 传给 `benchmark()` 的 `use_nvtx` 参数。

`--mode` 决定做前向、前向加反向或完整训练；`--nvtx` 决定是否贴标签。NVTX 标签本身不启动 Nsight，也不负责统计平均耗时。

需要认识两个工具：

- `torch.cuda.nvtx.range("名字")`：进入区间时开始标记，离开时结束标记。
- `contextlib.nullcontext()`：提供相同的 `with` 结构，但不做额外操作。

**API 小例子：**假设 `x`、`weight` 已是 CUDA 张量，给一次普通矩阵乘法贴标签可以写：

```python
with torch.cuda.nvtx.range("demo_matmul"):
    result = torch.matmul(x, weight)
```

标签覆盖缩进内的代码；矩阵乘法本身没有变化。`with` 的离开并不自动等待 GPU 完成。

为了让是否标注由开关控制，先根据 `use_nvtx` 选择 NVTX context 或 `nullcontext()`，然后进入选好的 context 执行原有工作。这个分支对应现有的 CPU/普通测速与 CUDA 采集两种实际用途，不需要再添加自动切换设备的逻辑。

**在你当前文件中的位置：**找到 `benchmark()` 中的 `#正式计时`，为调用测量函数的部分加一个外层标签，名字统一为 `measurement`。区间从 warm-up 循环结束后开始，到测量函数返回时结束；构造 `record`、打印和 `save_table()` 都放在区间外。当前测量函数每步结尾已有 CUDA 同步，因此函数返回时，该步 GPU 工作已完成。

**边界小例子：**`warmup=5, steps=10` 时执行 15 步，但 `measurement` 只包住后 10 步。标签应覆盖整个正式循环；不要为每一轮重新开启一个同名的采集外层，因为这里的采集命令只捕获第一个匹配区间。

完成检查：

- [x] `--help` 显示 `--nvtx`。
- [x] 默认关闭标注；开关从 parser 经 `main()` 进入 `benchmark()`。
- [x] `measurement` 内只有正式测量，预热和结果导出在外。
- [ ] 在 Mac 上不传 `--nvtx` 的原有 CPU 调试命令仍能运行。

**本节已在代码接入；下一步通过服务器采集验证。**

## 2. 给单步和计算阶段贴标签

**目的：**从整段采集中找到一次前向或一次完整训练，区分每个阶段关联的 CUDA kernel。

在 `measure_call()` 的每次循环里，加一个 `step` 区间，覆盖 `run_step()` 以及已有的步末 CUDA 同步。步骤开始前的同步保留原位置。这样 `step` 的结束对应 GPU 完成这一步工作，方便观察完整经过时间。

在 `run_step()` 内分别给现有操作加标签：

| 标签 | 包住的操作 |
| --- | --- |
| `forward` | 模型调用 |
| `loss` | 交叉熵计算 |
| `backward` | `loss.backward()` |
| `optimizer` | `optimizer.step()` |

同一个阶段名称用于所有包含该阶段的 mode。`forward` 标签只包模型调用；当前 `train_stages` 保存的 `forward_with_loss` 数字包含 loss，两者范围不同。

开关需要沿调用链继续传递：`benchmark()` → `measure_call()` → `run_step()`。每层明确接收并传递 `use_nvtx`，不要在函数内部另行读取命令行。warm-up 仍调用原来的 `run_step()`；它在外层 `measurement` 之外，所以不会进入本轮采集。

完整训练采集使用 `train`，观察只在整步边界同步的执行流程。内部阶段标签包住原来的调用即可；**加标签不要求在每个阶段新增同步。** `step` 末尾的同步保证外层采集结束前 GPU 工作完成。

**时间线小例子：**CPU 的 `forward` 标签结束时，GPU 可能还在计算前向。因此要用该标签关联的 CUDA kernel 看 GPU 运算时间，不能直接把内部 NVTX 标签的宽度当作 GPU 完成时间。GPU kernel 累计耗时、CPU 经过时间是两个统计口径。

完成检查：

- [x] 一次 `forward` step 有模型前向标签。
- [x] 一次 `train` step 能辨认 `forward`、`loss`、`backward`、`optimizer`。
- [x] 开关真正传到每个使用标签的函数。
- [x] 标签没有改变原来的计算顺序或新增阶段同步。

## 3. 在服务器采集第一份报告

**目的：**先验证标签、预热过滤和 CUDA kernel 记录都生效。本地脚本现已接受 `--nvtx`；把当前改动同步到服务器后运行。

进入服务器的 `/root/autodl-tmp/cs336/assignment2-systems`。先检查 `nsys --version` 和 benchmark 的 `--help`，确认运行的是带新开关的代码，再创建输出目录：

```sh
mkdir -p outputs/nsight_run/profiles
```

**第一份前向报告，一条命令：**

```sh
NSYS_NVTX_PROFILER_REGISTER_ONLY=0 uv run nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=nvtx --nvtx-capture=measurement --capture-range-end=stop --output=outputs/nsight_run/profiles/small_s512_forward -- python -m cs336_systems.benchmark --device cuda --mode forward --nvtx --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 512 --warmup 5 --steps 10 --output_prefix outputs/nsight_run/profiles/small_s512_forward_timing
```

这条命令的两层职责：`nsys profile` 负责采集；`--` 后仍是你的 Python benchmark 和它的原有模型参数。

| 采集设置 | 此处用途 |
| --- | --- |
| `--trace=cuda,nvtx` | 记录 CUDA 活动及手工标签 |
| `--sample=none --cpuctxsw=none` | 这轮聚焦 CUDA，关闭 CPU 采样和调度切换记录 |
| `--capture-range=nvtx --nvtx-capture=measurement` | 进入 `measurement` 时开始采集，排除预热 |
| `--capture-range-end=stop` | 区间结束后停止采集，程序继续完成结果导出 |
| `--output=...` | Nsight 报告文件名前缀 |
| benchmark 的 `--output_prefix=...` | 你原来的计时表文件名前缀 |

命令开头的环境变量只作用于这次运行。PyTorch 的这个接口使用普通 NVTX 字符串；NVIDIA 文档要求开启普通字符串匹配，才能用它触发采集，否则工具默认只考虑注册字符串。这是采集接口的匹配要求。[依据](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)

预期得到 `outputs/nsight_run/profiles/small_s512_forward.nsys-rep`，以及独立命名的计时 CSV/LaTeX/Typst。报告名已存在时换一个名字，保留之前的采集。

完成检查：

- [x] 生成 `.nsys-rep`，其中实际有 CUDA kernel 活动。
- [x] `measurement` 里有 10 个 `step`，没有前 5 次预热。
- [x] 模型创建、随机输入生成和表格导出没有混进采集。
- [x] 开着 profiler 的计时结果单独保存，未覆盖上一轮普通 benchmark。

## 4. 打开报告，先学会读三类数据

**目的：**确认前向时间和 kernel 统计各自在统计什么。

把 `.nsys-rep` 下载到 Mac，用 Nsight Systems 桌面应用打开。时间线上先找到 NVTX 的 `measurement` 和其中一次 `step`，再查看 CPU CUDA API 调用与 GPU kernel 行的关联。

先记录这三个量：

| 数据 | 含义 | 对应小问 |
| --- | --- | --- |
| 一个完整前向 step 的经过时间 | 从提交工作到同步完成，包含 CPU 调度和等待 | (a) |
| 各 kernel 的 `Total Time` | 该 kernel 所有执行实例的 GPU 耗时之和 | (b)、(c) |
| 各 kernel 的 `Instances` | 所选范围中的执行次数 | (b) |

在 `Stats System View` 的 `CUDA GPU Kernel Summary` 中查看统计。核对当前统计范围：整个报告包含 10 步，题目却问单次前向的调用次数；先选择一个完整 `step` 看统计，或核实每步调用一致后，将整段次数除以 10，明确这是每步平均值。

**统计小例子：**捕获 10 次前向，某 kernel 的 `Instances=120`，在每步调用一致的前提下，每次前向为 12 次；不能直接回答每次 120 次。kernel 表的 `Time (%)` 是相对于所列 GPU kernel 累计耗时的占比，不是整个程序经过时间的占比。[统计口径](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)

也可以先在服务器查看全采集范围的统计，帮助确认报告有内容：

```sh
nsys stats --report cuda_gpu_kern_sum,nvtx_sum outputs/nsight_run/profiles/small_s512_forward.nsys-rep
```

`nvtx_sum` 展示 CPU 标注区间的统计。内部阶段未逐段同步，读它时仍要遵守前面区分的时间口径。

(a) 先与上一轮同配置、仅模型前向的 small 基线 `25.85 ± 0.06 ms` 对照。Nsight 会引入记录开销，数值不必完全相同；判断时确认比较的是同样的模型、精度、输入形状和前向范围。

完成检查：

- [x] 能找到一次完整前向和它关联的 GPU kernel。
- [x] 记录前向时间、累计耗时最多的 kernel、单次前向调用次数。
- [x] 区分单次调用平均耗时、累计耗时、完整 step 经过时间。
- [x] 对非矩阵乘法 kernel，根据关联的代码操作判断用途，保留实际名称。

## 5. 采集完整训练，再细分注意力

**目的：**为 (b) 的前向加反向比较、(d) 的完整训练占比比较和 (e) 的注意力内部比较提供范围明确的数据。

按当前确定方案，分别采集 forward、forward_backward、train 三种 mode。forward_backward 用于 (b) 的独立对照，train 用于 (d) 的完整步骤。配置、命令、记录和分析统一见 [nsight_experiments.md](nsight_experiments.md)。保持模型、batch、精度、长度、预热和测量次数相同。

完整训练中按 `forward`、`backward`、`optimizer` 标签辨认 kernel，比较前向加反向的主要 kernel，以及加入 AdamW 后矩阵乘法和其他运算的累计 GPU 时间占比。计算占比时，分母统一为所选计算范围的 GPU kernel 时间，不混入 CUDA API 的 CPU 耗时。

当前 `forward` mode 保留自动求导，记录时写明这是仅做前向、autograd 开启的配置。(d) 将前向对照称为 inference；若另测关闭 autograd 的推理前向，需要单独记录这一条件，不与 (a) 原有前向基线混为同一配置。

按当前方案，只需补两个范围：

| 标签 | 当前范围 |
| --- | --- |
| attention | 完整 MSA 计算，包括 QKV/输出投影、QKᵀ、softmax、PV；不包含输入 RMSNorm 与残差加法 |
| attention_softmax | scaled_dot_product_attention 内现有 softmax 调用，包括 max、减法、exp、sum、除法 |

矩阵乘法在 attention 范围关联的 kernel 表中识别，不再额外要求给两次矩阵乘法贴标签。比较时间和 FLOPs 时采用相同的完整 MSA 范围与层数。标注遵守现有 CUDA/开关控制，不新增阶段同步；模型数学计算保留原样。

**小例子：**attention_softmax 关联多个 GPU kernel，必须累计整个 softmax 的 max/exp/sum/除法等操作；不能只找一个名称含 softmax 的 kernel。完整 MSA 的矩阵乘法还包括四个投影，不只 QKᵀ 与 PV。

完成检查：

- [x] forward、forward_backward、train 分别有报告。
- [x] 完整训练中的 loss 和优化器都进入采集。
- [ ] 能区分注意力分数矩阵乘法、softmax 和 V 加权求和。
- [x] softmax 的所有关联 kernel 都纳入统计。

## 6. 扩展正式配置并保存结果

**目的：**把工具使用扩展成作业要求的模型规模与上下文长度实验。

两种规模优先考虑已经测过的 small、medium。先考察 256、512，再逐级尝试更长的二次幂；第三个长度及最大长度要根据实际显存可容纳情况确定，不把 1024 或 2048 默认当作一定能跑。若当前配置不足以覆盖三个长度，再据实际结果确定可行的模型和实验设置，并记录任何设置变化。

为每个正式配置保留模型参数、batch、长度、精度、mode、预热次数、测量步数、Nsight 版本、`.nsys-rep` 路径以及 (a)–(e) 的观察数据。OOM 只记录实际发生的配置；预估显存不足不记为已运行失败。

每份 profile 的结果记录应包括：前向每步经过时间、主要 kernel 的完整名称与调用次数、非矩阵乘法 kernel 的用途、前向与训练中的矩阵乘法时间占比，以及注意力内部 softmax/矩阵乘法的 GPU 时间和 FLOPs 对比。先保存观测值，再整理题目要求的简短中文回答。

完成检查：

- [ ] 覆盖两种模型规模和三个符合要求的上下文长度，记录可行上限。
- [x] 所有比较能追溯到明确的配置、统计范围和报告文件。
- [x] (a)–(e) 都有实际观测数据，profiling 结果与普通测速结果分开保存。

当前状态：15 份正式采集、9 次容量探测已完成，注意力标签已接入并核对；见 [nsight_report.md](nsight_report.md)。medium 在 batch=4 下仅有 256/512 两档成功，按用户选择保留这一限制。
