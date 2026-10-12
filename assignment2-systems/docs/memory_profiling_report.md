# 显存分析：small

2026-10-08，RTX 4090，PyTorch 2.11.0+cu128。按确认范围整理已有结果，没有重复启动实验。

模型：d_model=768、d_ff=3072、12 层、12 头、词表=10000、batch=4、SwiGLU、PreNorm、RoPE。预热 1 步，随后记录 2 步；每个模式独立进程。推理为 model.eval() + torch.no_grad()，完整训练包含清空梯度、前向、loss、反向和 AdamW 更新。BF16 使用 autocast；参数、参数梯度和动量仍为 FP32。

原题指定 xl 和长度 128/2048；本轮按选定范围使用 small 和 128/1024。结果不能作为 xl 的测量。

## 峰值与快照

主表使用 torch.cuda.max_memory_allocated()，单位 GiB；包含常驻张量。另存 peak_reserved_gib，表示 allocator 从 CUDA 保留的池，不等于当前活跃张量。CSV 同时保留两种口径。快照只追踪 PyTorch allocator，不能代表 nvidia-smi 中全部进程显存。[PyTorch 文档](https://docs.pytorch.org/docs/2.11/torch_cuda_memory.html)。

| 序列长度 | 精度 | 推理峰值 GiB | 训练峰值 GiB | 推理快照 | 训练快照 |
| ---: | --- | ---: | ---: | --- | --- |
| 128 | FP32 | 0.518 | 2.191 | [打开文件](../outputs/memory_profiling/approved_small/small_b4_s128_fp32_forward.pickle) | [打开文件](../outputs/memory_profiling/approved_small/small_b4_s128_fp32_train.pickle) |
| 128 | BF16 | 0.759 | 2.187 | [打开文件](../outputs/memory_profiling/approved_small/small_b4_s128_bf16_forward.pickle) | [打开文件](../outputs/memory_profiling/approved_small/small_b4_s128_bf16_train.pickle) |
| 1024 | FP32 | 1.302 | 10.904 | [打开文件](../outputs/memory_profiling/approved_small/small_b4_s1024_fp32_forward.pickle) | [打开文件](../outputs/memory_profiling/approved_small/small_b4_s1024_fp32_train.pickle) |
| 1024 | BF16 | 1.324 | 8.527 | [打开文件](../outputs/memory_profiling/approved_small/small_b4_s1024_bf16_forward.pickle) | [打开文件](../outputs/memory_profiling/approved_small/small_b4_s1024_bf16_train.pickle) |

打开 [memory_viz](https://pytorch.org/memory_viz)，把表中的 .pickle 拖进去，选择 Active Memory Timeline。建议先对照长度 1024 的 FP32 推理和训练快照，再对照两种精度的训练快照。

## 结果理解

长度 1024 时，BF16 训练峰值下降 21.8%；长度 128 时几乎没有变化。训练的常驻参数、梯度、AdamW 动量没有减半；长序列的激活占用更大，降低部分激活精度更容易体现收益。

推理峰值在本轮反而增加：FP32 参数仍在，autocast 会建立低精度权重缓存。分配栈还显示 BF16 的 attention softmax 中，exp 和除法仍产生 192 MiB 的 FP32 大张量。这些现象解释了为什么开启 BF16 不保证整步峰值减半；本轮没有另跑关闭权重缓存的消融来量化它的独立贡献。[autocast 权重缓存与算子精度](https://docs.pytorch.org/docs/2.11/amp.html)。

## (a)–(c) 怎样记录

(a) 从网页截取同一长度、精度下的推理与训练 Active Memory Timeline，共两张，描述可辨认的阶段。推理通常反复产生并释放每层临时张量；训练前向累积需要保存的张量，反向释放它们并产生梯度，优化器继续持有参数和两份动量并产生更新临时张量。网页中的峰不能单凭外形精确标为某个阶段，应结合调用顺序和分配栈判断。训练已预热，所以初始状态中已有动量和上一步梯度。

(b) 使用表中的 FP32 列，两个长度各列出推理和训练峰值。(c) 与 BF16 列比较，写 2–3 句。本次没有在浏览器中查看或生成截图，截图待你打开网页后保存。

## (d) 算的是哪个张量

目标是算残差主干中一份 token 表示的大小：比如进入或离开一个 TransformerBlock 的 x，形状是 (batch_size, seq_len, d_model)。每个 token 有 d_model 个数；FP32 每个数占 4 字节。

大小（MiB）= batch_size × seq_len × d_model × 4 / 1024²。这是一个张量，不是整个 block 保存的所有中间张量，也不是形状 (B,H,S,S) 的注意力分数。

本轮 small：4×128×768×4/1024² = **1.5 MiB**；长度 1024 时为 **12 MiB**。若按原题 xl（d_model=2560、batch=4），长度 128/2048 时分别为 **5/80 MiB**；按 Table 1 默认长度 512 则为 **20 MiB**，计算时要注明采用的长度。

## (e) 网页上找最大的分配

把 forward 快照的 Detail 调到 10%，悬停最大的色块，读大小和调用栈，说明哪行运算产生它。长度 1024 的 FP32 快照中，最大的新增分配为 **192 MiB**，来自注意力分数、masked_fill 和 softmax 中间结果。其形状是 (4,12,1024,1024)：4×12×1024²×4/1024² = 192 MiB。具体位置为 Scaled_dot_product_function.py:10/22 和 SoftMax_function.py:9/11/12；BF16 中仍有 192 MiB 的 exp 和归一化结果。snapshot_audit.json 列出了对应栈来源。

模型参数在开启记录前已经创建，因此这部分常驻分配没有原始分配栈；上述来源判断使用正式步骤中新增分配的栈。

## (f) 另一个测量目标

这一问要用 Nsight Systems 的显存追踪与 PyTorch NVTX 标签，查看单个 TransformerBlock，不只是读 memory_viz 的总峰值。先确定该 block 前向结束后为反向保留下来的张量总量，列出贡献最大的 5 个操作及占这份总量的百分比；累计分配过的所有临时张量不等于保存量。

再观察该 block 反向时，保存张量被释放、参数梯度被分配造成的显存净变化，并与参数数量×4 字节比较。还要区分过程中短暂产生的激活梯度与最终保留的参数梯度。交付 Nsight 截图和 1–2 段解释。

已完成你运行的正式 small/FP32、batch=4、长度 1024 单 block 采集的离线分析。新增保存存储 **734.094 MiB**，其中 softmax 的 exp 和归一化结果各 **192 MiB**，合计占 **52.31%**。最终梯度 **48.006 MiB**，等于参数梯度 **36.006 MiB** 加输入梯度 **12 MiB**，与实际张量大小一致。

`block_backward` 标签确实存在，位于报告内 **4.03368–4.12600 秒**，持续 **92.32 ms**。完整操作占比表、前向/反向显存重绘图、答题解释和证据见 [单 block 显存报告](block_memory_report.md)；采集方法见 [实验入口与口径](block_memory_experiment.md)。本轮没有重复启动 GPU 实验。

注意：(d) 的 residual stream 指残差主干上的表示；(f) 的 residuals 指为反向保存的张量，范围不同。

## 复核与复现

8 份快照均通过本机/远端 SHA256 核对；事件数未达到历史缓冲上限，含分配、释放和用户代码栈，未发现 OOM。requested tensor bytes 与 allocator 的 block bytes 口径不同，不能强求事件求和峰值等于 max_memory_allocated。snapshot_audit.json 保存事件数、申请字节峰值、最大新增分配来源和快照 SHA256。

原始数据和实验源码位于 outputs/memory_profiling/small/，每项已记录的进程返回码保存在 snapshot_audit.json。

仅重新整理结果：在 assignment2-systems 运行 `.venv/bin/python report_memory.py`。需要重新采集时，在 GPU 服务器运行 `.venv/bin/python run_memory_experiments.py --model small`，当前默认范围已改为四组、八份快照。不要把开启显存记录后的计时当作普通测速结果。

原批量调度器在最后一个子进程完成后被停止，最后一项 BF16/1024/train 没有在批量 execution.json 留下返回码；其正式两步日志、CSV、显存指标和快照均已完整生成并核对。该返回码保留为 null，没有补造退出码。
