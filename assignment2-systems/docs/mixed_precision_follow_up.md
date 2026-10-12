# 混合精度：类型观察与 benchmark 接入

对应作业 §2.1.5，PDF 第 8–9 页。2026-10-08 已完成代码接入、RTX 4090 小规模验证，以及 (c) 的 small、medium 性能对照，结果见 [分阶段对照报告](mixed_precision_report.md)。其他模型规模尚未测量。

## (a) ToyModel：跑一次完整训练，查看数据类型

文件：`cs336_systems/toy_mixed_precision.py`。模型结构沿用题目；`forward()` 中两个字段只记录中间输出的 dtype，不保存中间张量。

在远端 `assignment2-systems` 目录运行：

```bash
uv run python -m cs336_systems.toy_mixed_precision
```

默认 CUDA、FP16 autocast，随机 FP32 输入和整数类别标签，先预热 5 次，再测量 10 次完整训练 step。计时包含清空梯度、前向、交叉熵、反向、优化器，以及 FP16 的梯度缩放处理；打印放在计时循环外。

FP16 使用 PyTorch `GradScaler`，BF16 关闭 scaler。检查 BF16 类型只需：

```bash
uv run python -m cs336_systems.toy_mixed_precision --precision bf16
```

RTX 4090 / PyTorch 2.11.0+cu128 实测：

| 观察对象 | FP16 autocast | BF16 autocast |
| --- | --- | --- |
| 模型参数，在 autocast 内检查 | FP32 | FP32 |
| 原始随机输入 | FP32 | FP32 |
| `fc1` 输出 | FP16 | BF16 |
| `LayerNorm` 输出 | FP32 | FP32 |
| 预测 logits | FP16 | BF16 |
| 交叉熵 loss | FP32 | FP32 |
| 参数的 `.grad` | FP32 | FP32 |

这里的参数梯度指 FP32 参数上最终保存的 `.grad`，不表示反向传播的每个中间运算都是 FP32。CPU 与 CUDA 的 autocast 规则不同，作业这问以 CUDA 结果为准。

原始打印保存在 `outputs/mixed_precision_smoke/cuda/toy_fp16.log` 和 `toy_bf16.log`。

## (b) LayerNorm：先看统计量，再看根号和除法

LayerNorm 计算：

\[
\mu=\frac1d\sum_i x_i,\qquad
\sigma^2=\frac1d\sum_i(x_i-\mu)^2,\qquad
y_i=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
\]

敏感点包括均值与方差的求和、中心化减法以及平方。FP16 的平方可能溢出，例如 `300² = 90000` 超过其最大有限值 65504；即使没有溢出，均值误差也会改变所有中心化结果，随后方差误差通过归一化分母影响所有输出。

BF16 的指数位与 FP32 一样多，范围问题大幅减轻，但其显式小数位只有 7 位，少于 FP16 的 10 位；统计量的舍入问题仍然存在。[格式定义](https://docs.pytorch.org/docs/2.11/tensor_attributes.html#torch-dtype)

一个不涉及溢出的具体例子：输入 `[1, 1.0078125]`，这两个数都能被 BF16 精确保存。按公式逐项计算并让中间统计量也存成 BF16，与先把同一输入提升到 FP32 相比：

| 中间量 | BF16 保存统计量 | FP32 保存统计量 |
| --- | --- | --- |
| 均值 | 1 | 1.00390625 |
| `x - 均值` | `[0, 0.0078125]` | `[-0.00390625, 0.00390625]` |
| 方差 | 0.000030517578125 | 0.0000152587890625 |

BF16 存不下这里的精确均值，舍入后连方差都变成两倍。这是手动逐项、低精度保存统计量的示范，不是说 PyTorch 的 `nn.LayerNorm` 实际使用这种不稳定计算；CUDA autocast 会把它按 FP32 执行。

因此，换成 BF16 后仍有理由让归一化的统计计算保留 FP32。高精度计算减少后续新增的舍入误差，不能恢复输入降精度时已经丢失的信息。

公式与精度策略参考：[PyTorch LayerNorm](https://docs.pytorch.org/docs/2.11/generated/torch.nn.LayerNorm.html)、[CUDA autocast 运算规则](https://docs.pytorch.org/docs/2.11/amp.html#cuda-op-specific-behavior)。

## (c) benchmark：选择 FP32 或 BF16 autocast

`cs336_systems/benchmark.py` 增加 `--precision fp32/bf16`，默认保持 FP32。`precision_context()` 在 BF16 下返回 `torch.autocast(...)`，FP32 下返回 `nullcontext()`。

数据流是：FP32 模型参数 → autocast 内前向与 loss → 离开 context → backward → optimizer。参数始终保持 FP32，warm-up 和正式测量使用同一种精度。你自己的 RMSNorm 已在计算统计量前显式转为 FP32，无需另外修改 A1。

四种模式均已接入，保留原来的统计口径：

| mode | 记录内容 |
| --- | --- |
| `forward` | 前向总耗时，保留原有 autograd 行为 |
| `forward_backward` | 清空梯度、前向、loss、反向的总耗时 |
| `train` | 完整训练 step 总耗时 |
| `train_stages` | 分别记录 `forward_with_loss`、`backward`、`optimizer` |

独立的反向时间读取 `train_stages` 的 `backward` 行；`forward_backward` 的总耗时不能直接称作反向耗时。阶段测量沿用同步边界与优化器预热，因此也包含优化器状态所需的显存。

新增 `--seed`，默认 0，固定初始参数与随机数据。CSV 增加 `parameter_dtype`、`autocast`、`tf32_matmul` 和 `seed`，把参数存储精度与 autocast 精度分开记录。已有 NVTX 开关可继续使用。

一条 small 的 BF16 前向测量命令：

```bash
uv run python -m cs336_systems.benchmark --device cuda --precision bf16 --mode forward --vocab_size 10000 --d_model 768 --d_ff 3072 --num_layers 12 --num_heads 12 --batch_size 4 --context_len 512 --warmup 5 --steps 10 --output_prefix outputs/mixed_precision/small_bf16_forward
```

FP32 对照改为 `--precision fp32`，并更换输出前缀。测量前向加反向改为 `--mode forward_backward`；需要单独阶段时间改为 `--mode train_stages`。输出前缀同时注明规模、精度和 mode，避免覆盖结果。

作业 Table 1 的配置如下；各规模精度对照保持 batch=4、序列长度=512、词表=10000、warm-up=5、测量=10、seed=0：

| 规模 | d_model | d_ff | num_layers | num_heads |
| --- | ---: | ---: | ---: | ---: |
| small | 768 | 3072 | 12 | 12 |
| medium | 1024 | 4096 | 24 | 16 |
| large | 1280 | 5120 | 36 | 20 |
| xl | 2560 | 10240 | 32 | 32 |
| 10B | 4608 | 12288 | 50 | 36 |

超过 4090 显存的配置记录 OOM；不自动缩小模型、batch 或序列长度。比较加速比时只配对同一配置下都成功的 FP32/BF16 结果。BF16 autocast 仍保留 FP32 参数与梯度，不能把整个模型的显存直接按减半估计。

验证记录：`outputs/mixed_precision_smoke/cuda/validation.json`。FP32 的前向、loss、梯度和参数更新与接入前逐元素相同；CUDA 下两种精度的四种 mode 均通过，小模型 logits 类型、loss 类型、反向 context 边界、有限梯度和 FP32 优化器状态均已检查。这些小规模检查不作为各模型规模的正式性能结果。
