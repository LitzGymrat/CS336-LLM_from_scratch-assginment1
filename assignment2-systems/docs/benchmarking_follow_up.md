# Benchmarking Script：逐步实现清单

本清单对应作业 §2.1.3 `benchmarking_script`，PDF 第 3–4 页。它是辅助学习的拆解，函数划分、参数名字和调试配置是建议；计时模式、warm-up、同步与实验次数来自题目。

2026-10-06 检查：`cs336_systems/benchmark.py` 已有 `torch`、`argparse`、`timeit` 三个导入，以及你刚添加的 `benchmark(...)` 函数声明；函数体仍待填写。下面的检查项是待完成内容，不表示已经实现或测试通过。保留你的 `benchmark()`，把它作为实验主体。

## 先明确最后要得到什么

最终，一条命令描述一次实验。脚本创建模型和随机输入，先运行若干次，再测量若干次，输出每次耗时及平均值、标准差。

例如，**完成相应参数解析后**，在 `assignment2-systems/` 下运行：

```sh
uv run python -m cs336_systems.benchmark --device cpu --mode forward --warmup 2 --steps 3
```

含义：使用 CPU，执行 2 次不计入结果的前向，再测量 3 次前向。正式 CUDA 实验时改设备和模型规模；CPU 小模型运行只用于检查程序流程。

数据流：

```text
命令行 → args → 模型和随机 batch → 一次选定的 step
                                       ↓
                            warm-up → 正式计时 → 耗时列表 → 表格
```

各步都建立在前一步已经能运行的基础上。当前先完成第 1 步。

## 1. 读参数：先让命令行能改变变量

**目的：**以后切换模型规模、计时模式和实现版本时，使用命令行改变配置。

**当前写什么：**写一个 `parse_args()`，负责建立 parser、声明参数并返回解析结果。在 `main()` 中接收结果，暂时只打印它。文件末尾用 `if __name__ == "__main__":` 调用 `main()`。后续 `main()` 再把这些值传给你现有的 `benchmark()`；模型、数据和计时循环写在 `benchmark()` 里面。

你当前的 `benchmark()` 只有声明，Python 要求函数体至少有一条语句。先写 `pass` 作为占位即可，否则连参数读取的检查也会被语法错误阻止；实现实验主体时再替换这个占位。

这里的 `main()` 是组织脚本流程的普通函数，不需要继承任何类。`parse_args()` 返回的对象通过 `args.d_model` 等属性取值。

先只用下面三个参数熟悉接口，再补表格中的其余参数：

```python
parser = argparse.ArgumentParser()
parser.add_argument("--d-model", type=int, default=32)
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward")
args = parser.parse_args()
```

这个片段演示 argparse 的用法，不是完整的 benchmark。把它放进 `parse_args()` 时，还需要把 `args` 返回给调用方。

例如命令中写 `--d-model 64`，随后 `args.d_model` 就是整数 `64`；不写该选项时则是 `32`。`type` 负责把输入字符串转成所需类型，`default` 是省略该选项时的值，`choices` 限定允许的选项。参数名中的 `-` 会转换成属性名中的 `_`。

建议这版采用下面的参数。默认值是便宜的调试配置，**不是正式实验配置**。

| 类别 | 命令行参数 | 类型/选项 | 调试默认值 | 下一步在哪里使用 |
| --- | --- | --- | --- | --- |
| 模型 | `--vocab-size` | int | 64 | 模型、随机 token ID 的上界 |
| 模型 | `--d-model` | int | 32 | 模型构造 |
| 模型 | `--d-ff` | int | 128 | 模型构造 |
| 模型 | `--num-layers` | int | 1 | 模型构造 |
| 模型 | `--num-heads` | int | 4 | 模型构造 |
| 模型 | `--context-len` | int | 8 | 模型 RoPE 表的容量、随机 batch 的序列长度 |
| 模型 | `--rope-theta` | float | 10000.0 | 模型构造 |
| 数据 | `--batch-size` | int | 2 | 随机 batch 的形状 |
| 运行 | `--device` | cpu / cuda | cpu | 模型和数据所在设备 |
| 测量 | `--mode` | forward / forward_backward / train | forward | 一次 step 执行哪些操作 |
| 测量 | `--warmup` | int | 2 | 不收集耗时的循环次数 |
| 测量 | `--steps` | int | 3 | 收集耗时的循环次数 |
| 优化器 | `--lr` | float | 0.001 | 创建 AdamW |

优化器的其他超参数先复用你 A1 的默认值。为了当前这道速度实验，不需要接入学习率调度、验证集、checkpoint 或文本数据集。

**完成标准：**

- [ ] `--help` 显示声明的参数。
- [ ] 不传选项时，打印的配置与默认值一致。
- [ ] 传 `--d-model 64 --warmup 0` 后，对应值确实改变。
- [ ] 这时还没有创建模型，没有开始计时。

运行检查：

```sh
uv run python -m cs336_systems.benchmark --help
uv run python -m cs336_systems.benchmark --d-model 64 --warmup 0
```

参考：[Python argparse 入门](https://docs.python.org/3.12/howto/argparse.html)、[参数名转换规则](https://docs.python.org/3.12/library/argparse.html#dest)。

## 2. 创建模型和 batch：接上你已完成的 A1

**目的：**得到可以真正送入模型的数据，先验证一次前向能跑通。

**输入：**第 1 步得到的 `args`。

**需要得到：**模型、输入 ID 张量、目标 ID 张量，三者都在指定设备上。

你自己的模型入口是：

```text
cs336_basics.Transformer.Transformer_lm.Transformer_lm
```

它接受 `vocab_size`、`d_model`、`num_heads`、`d_ff`、`num_layers`、`context_len`、`rope_theta`，以及 `device`、`dtype` 等可选项。逐个使用同名的 `args` 属性传入；保持现有关键字参数写法即可。

先用 FP32。数据是 token ID，所以它是整数张量，不随模型浮点精度改成浮点数。

随机数据的最小例子：

```python
example = torch.randint(0, 10, (2, 4))
```

这产生一个 `(2, 4)` 的整数张量，每个元素在 `0` 到 `9` 之间。你的实际输入应为 `(batch_size, context_len)`，上界换成 `vocab_size`。目标张量也可以独立随机生成同样的形状；这个实验只需要触发计算，不用评估语言预测质量。

贯穿后续的小配置：`B=2, S=8, V=64, d=32, L=1, H=4`。

| 量 | 预期形状 | 含义 |
| --- | --- | --- |
| 输入 ID | `(2, 8)` | 两条序列，每条 8 个 token |
| 目标 ID | `(2, 8)` | 每个位置有一个目标类别 |
| 模型输出 logits | `(2, 8, 64)` | 每个位置对 64 个类别的预测分数 |
| 交叉熵 | `()` | 所有位置平均后的标量 |

这就是复用 A1：随机 ID → 你的 LM → 你的 cross entropy。

**完成标准：**

- [ ] 一次前向返回 `(B, S, V)` 的 logits。
- [ ] 输入 ID 在词表范围内，模型和输入设备一致。
- [ ] 用目标张量计算 loss，得到有限的标量。
- [ ] 模型和数据只在准备阶段创建，不放进计时区间。

你的 loss 入口是 `cs336_basics.Training_modules.Crossentropyloss.cross_entropy(o, target_indices)`，已支持这里的形状。

参考：[torch.randint](https://docs.pytorch.org/docs/2.11/generated/torch.randint.html)。

## 3. 定义一次 step：先跑操作，再谈耗时

**目的：**明确重复测量的到底是哪段工作。

建议写一个 `run_step()`。传入模型、输入、目标、模式和 optimizer；它执行一遍指定工作。这个函数先负责计算，不负责循环、打印、保存结果。

| 模式 | 一次 step 的操作 | 参数会不会更新 |
| --- | --- | --- |
| `forward` | 模型前向 | 不会 |
| `forward_backward` | 清空梯度 → 前向 → loss → backward | 不会 |
| `train` | 清空梯度 → 前向 → loss → backward → optimizer.step | 会 |

把它看成相同流程的三个终点：只做到前向；做到反向；做到更新。

你的优化器入口是 `cs336_basics.Training_modules.AdamW.AdamW`。创建时传入 `model.parameters()`，复用 A1 的调用方式。optimizer 在循环前创建一次，内部状态贯穿后续 step。

**为什么要清空梯度？**PyTorch 的反向默认把梯度累加进 `.grad`。这里各次 step 独立测量，不做梯度累积，因此每次反向前清空。

**小例子：**同一个随机 batch 跑两次 `train`，两次都有新的前向和反向，但第二次继续使用同一个 optimizer 的动量状态。不要每轮重新创建模型或 optimizer。

第一版统一在训练设置、开启 autograd 的情况下测量三种模式；`forward` 测的是训练所用的前向。以后如果测关闭 autograd 的推理前向，要另标明条件。只保留当次必要的 logits/loss，不把它们累积进列表；后面列表中保存的是耗时数字。

**完成标准：**

- [ ] 三种模式分别能运行一次。
- [ ] 反向后，可训练参数出现 `.grad`。
- [ ] `forward_backward` 不调用更新；`train` 调用更新。
- [ ] 重复运行 `train` 时复用同一个 optimizer。

## 4. Warm-up：重复相同工作，但不记录

**目的：**减少首次执行时初始化、库加载、缓存建立等因素对稳定运行耗时的干扰。

输入是第 3 步的执行过程和 `warmup` 次数。调用它 `w` 次即可。warm-up 和正式测量必须使用相同模式。

**小例子：**`warmup=2, steps=3` 时，总共执行 5 次：前两次不进入耗时列表，后三次进入。因此最终列表长度应是 3。

`train` 模式的 warm-up 也正常更新参数、初始化优化器状态。这里衡量执行速度，不是在为 accuracy 做数据训练。

**完成标准：**

- [ ] `warmup=0` 时不执行 warm-up 循环。
- [ ] warm-up 的耗时不进入正式结果。
- [ ] 结束后沿用已经创建的模型、batch 和 optimizer。

## 5. 正式计时：每次得到一个耗时数字

**目的：**把“执行一次”转成“执行一次用了多少毫秒”。

先熟悉计时器，下面只演示 CPU 计时：

```python
start = timeit.default_timer()
sum(range(100000))
end = timeit.default_timer()
elapsed_ms = (end - start) * 1000
```

`default_timer()` 返回以秒计的时钟读数。相减得到耗时，再乘 1000 得到毫秒。

对 CUDA，一次测量的次序应为：

1. 起始计时前同步，确保之前提交的工作已经完成。
2. 读开始时间。
3. 执行当前模式的一次 step。
4. 调用 `torch.cuda.synchronize()`，等待这次 GPU 工作完成。
5. 读结束时间，相减并加入耗时列表。

**为什么结束计时放在同步之后？**模型调用返回时，GPU 可能还没算完。这个结束同步让时间覆盖实际完成工作所需的等待。

题目要求每个 step 后同步，包括 warm-up。CPU 调试时不调用 CUDA 同步；根据已指定的 `device` 区分这两种路径，是当前两种实际运行环境的需要，不做静默切换设备。

例如三个模拟耗时 `[10.0, 12.0, 11.0]`，表示三次 step 分别用了多少毫秒；列表不存模型输出。这三个数字是说明用的，不是真实测量。

**完成标准：**

- [ ] 正式执行 `n` 次，耗时列表恰好有 `n` 个数字。
- [ ] 每个 CUDA step 结束后都同步，结束时钟在同步之后读取。
- [ ] 初始化、生成随机数据、打印、导出表格都在计时区间外。
- [ ] `forward_backward` 的结果标为前向加反向总耗时，不误称纯反向耗时；`train` 的总耗时也包括 loss 和清空梯度。

参考：[timeit.default_timer](https://docs.python.org/3.12/library/timeit.html#timeit.default_timer)、[PyTorch CUDA 异步执行与计时](https://docs.pytorch.org/docs/2.11/notes/cuda.html#asynchronous-execution)。

## 6. 汇总结果：把一次实验变成一行记录

**目的：**供后续比较模型规模、精度与不同实现。

输入是耗时列表和本次配置。先打印每次耗时，再算均值、标准差。可以使用 Python `statistics.mean` 和 `statistics.stdev`；`stdev` 是样本标准差，要求至少两次测量，本清单默认 3 次、正式要求 10 次。参考：[Python statistics](https://docs.python.org/3.12/library/statistics.html#statistics.stdev)。

沿用刚才的模拟数字 `[10.0, 12.0, 11.0]`：均值 `11.0 ms`，样本标准差 `1.0 ms`。同时保留原始数字，才能看首次测量是否特别慢。

结果记录至少带上模型超参数、batch/序列长度、设备、精度、模式、warm-up 次数、测量次数、mean_ms、std_ms。输出时不要只留下耗时，导致以后无法确认实验条件。

已有 `cs336_systems/reporting.py` 的 `save_table(records, output_prefix)` 可直接使用。先把结果组织为一个字典，放入列表，再交给它导出 CSV/LaTeX/Typst；它不承担计时或统计工作。

**完成标准：**

- [ ] 能说明输出的时间对应什么操作、什么单位。
- [ ] 能从一行记录还原实验配置。
- [ ] mean/std 根据正式测量计算，未混入 warm-up。

### 为作业 (b) 补充各阶段耗时

完成上述三种模式的总耗时后，(b) 还要回答前向、反向和优化器各阶段耗时。不能直接把三种模式的总耗时都当成独立阶段耗时，也不要用两次独立实验的总时间作差冒充精确测量。

在训练 step 内分别围绕模型调用、`loss.backward()`、`optimizer.step()` 设置计时边界，并在各阶段边界同步 GPU。计算 loss 和清空梯度不要混进这三个单独阶段；说明它们仍在整步总耗时中。这样分别得到三个阶段的耗时列表，再各自汇总。

阶段同步会改变执行组织，因此这些时间描述的是明确同步边界下的测量。它们不是忽略 CPU 开销的纯 kernel 时间，也不保证三个阶段相加严格等于另一轮整步测量。

## 7. 把变体接到实际行为上

命令行选项只是输入，必须有使用该选项的代码才会改变实验。

**已经有的变体：**你的 LM 支持 `ffn_type="swiglu"/"silu"`、`norm_position` 和 `use_rope`。例如添加 `--ffn-type` 后，把 `args.ffn_type` 传给模型构造。第一版并不要求比较这些变体；它们说明现有模型如何接入选项。

**后续模块再增加：**进入混合精度实验时，添加 `--precision fp32/bf16` 并接入相应 autocast 行为；进入编译时，添加 compile 开关；写好 FlashAttention 后，再加注意力实现选择。现在先完成 FP32 基线。

**小例子：**`--ffn-type silu` → `args.ffn_type` 为 `"silu"` → 传给模型 → 模型创建 SiLU FFN。只读到 `"silu"` 而没传进去，模型行为不会改变。

## 8. 最后按题目运行实验

上面的调试默认值与这里的正式配置分开。题目 Table 1：

| 模型 | d_model | d_ff | num_layers | num_heads |
| --- | --- | --- | --- | --- |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

按作业 §2.1.2 使用词表 10000、batch size 4；除另有说明外 context length 512。这里 `d_ff` 直接采用表格值，不再套用 A1 的估算公式。

- [ ] (a)：脚本支持三种模式、配置、随机 batch、warm-up、计时和每步 CUDA 同步。
- [ ] (b)：各规模采用 5 次 warm-up、10 次正式测量，报告平均值和标准差；记录硬件与测量条件。
- [ ] (c)：分别重新启动进程，比较 0、1、2、5 次 warm-up。不要在已经充分预热的同一进程内把 `warmup` 改成 0，就当成没有预热的实验。

模型因显存不足不能运行时，记录对应配置和 OOM；不要换小了模型却沿用原规模标签。正式 CUDA 性能结果在 NVIDIA GPU 上测量；Mac 上的小 CPU 配置用于先打通脚本。

## 建议保留的文件结构

保留你现有的 `benchmark()`，在一个文件内按下面的职责组织即可：

| 函数 | 职责 |
| --- | --- |
| `parse_args()` | 读取命令行配置 |
| `run_step()` | 执行一次选定模式的计算 |
| `benchmark()` | 创建模型和数据、组织 warm-up/计时、汇总结果 |
| `main()` | 接收命令行配置，传给 `benchmark()`；第 1 步暂时只打印 |

调用关系是 `main → parse_args → benchmark → run_step`。`parse_args()` 返回配置后，`main()` 才调用 `benchmark()`；`benchmark()` 在循环中多次调用 `run_step()`。各函数放在同一个文件里即可。

`benchmark()` 的参数还需要加入 `batch_size`、`mode`、`warmup`、`steps`、`lr`，因为它要使用这些实验设置。先完成第 1 步，运行默认参数、覆盖参数和 `--help` 三种检查；确认之后再接第 2 步。
