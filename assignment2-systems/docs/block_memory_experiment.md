# memory_profiling (f)：一个 block 的显存变化

状态：你已完成正式 small 采集；原始文件已下载并核对，离线分析完成。结果见 [单 block 显存报告](block_memory_report.md)。

## 本轮只采集一组

沿用前面的 small 配置：FP32，batch=4，序列长度 1024，d_model=768，d_ff=3072，12 个注意力头，PreNorm、SwiGLU、RoPE。

实例化自己的 `Transformer_blk`，单独运行其中一个 block。输入是形状 `(4,1024,768)` 的随机 FP32 张量，并且需要梯度；这对应正常 Transformer 中来自上一层的输入。使用预先创建的同形状张量作为后续层传回的输出梯度。

预热 1 次，清空参数梯度和输入梯度，然后正式记录一次前向、一次反向。这里没有 embedding、LM head、交叉熵和优化器，因此读到的变化可以明确归属于这个 block。它是单 block 的实验，不是整个模型的显存峰值实验；本轮也不作为原题 xl 的结果。

## 启动和文件

在 GPU 服务器的 `/root/autodl-tmp/cs336/assignment2-systems` 中运行：

```bash
uv run python run_block_memory_profile.py
```

默认输出 `outputs/block_memory/small_b4_s1024_fp32` 前缀：

- `.nsys-rep`：用 Nsight Systems 打开。
- `.sqlite`：保留原始事件，便于后续按字节计算和核对。
- `.json`：配置、实际源码哈希、参数数量、实际参数梯度及输入梯度字节数。
- `.command.json`：真实采集命令、环境设置、harness 哈希与返回码。
- `.log`：采集日志。

默认不覆盖已有结果；重跑时指定新的 `--output_prefix`。

## 标注和采集口径

`torch.autograd.profiler.emit_nvtx(record_shapes=True)` 自动记录每个算子的名字、输入形状和前向/反向对应的序号。文档把反向的序号称为 `stashed seq`；本机对应的远端 PyTorch 2.11 实际标签中，两边都写作 `seq = N`，反向标签同时包含 `Backward`。根据所在阶段、算子名和相同序号配对，不要求字面上出现 `stashed seq`。[PyTorch 文档](https://docs.pytorch.org/docs/2.11/autograd.html#torch.autograd.profiler.emit_nvtx)

另外标出 `warmup`、`block_measurement`、`block_forward`、`block_backward`，以及 `before_forward`、`after_forward`、`after_backward` 三个边界。前向和反向区间的末尾同步 GPU。

Nsight 开启 `--cuda-memory-usage=true`，从进程启动采集，包括初始化和预热；分析时只选 `block_measurement`。这样可以在正式前向开始时看见已有的参数、输入和上游梯度基线。[Nsight 文档](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-gpu-memory-allocation-graph)

这一个子进程设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`。默认 caching allocator 释放张量时会把显存留在缓存池里，Nsight 的 CUDA malloc/free 曲线不能直接代表活跃张量的变化。关闭缓存使这里的分配、释放更易观察，也改变了执行开销，因此本实验不用于速度比较。不调用 `memory_allocated()` 等 allocator 指标来替代 Nsight 读数。[PyTorch 显存管理](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management)

## 看图和记录

打开 `.nsys-rep`，找到 NVTX 的 `block_measurement`，把 CUDA Memory Usage 和 NVTX 算子行放在一起。

1. 记录三个边界的显存量：`M0=before_forward`，`M1=after_forward`，`M2=after_backward`。前向峰值含临时张量，不能当作保存量。
2. 查找前向申请、前向结束仍然保留、反向才释放的分配，按最内层的申请算子归属。多个 view 若共享一个地址，只算同一块存储一次；已经常驻的参数不算新增激活。
3. 汇总保存量，列出贡献最大的 5 个操作。每项占比的分母都是同一个 block 的保存总量；不要用所有历史 malloc 的累计大小作分母，也不要在嵌套 NVTX 区间重复计数。
4. 根据反向的释放和新增分配，核对产生的梯度量。最终保留的参数梯度、传回上一层的输入梯度，与途中短暂存在的中间梯度需要分开。

在本入口中，输出张量在两个阶段边界都保留，输出梯度在正式前向前已经创建。若 CUDA 内存基线在预热后保持稳定，则新增保存量为：

```text
saved_bytes = M1 - M0 - output_bytes
gradient_bytes = M2 - M1 + saved_bytes
               = M2 - M0 - output_bytes
```

这里的 `gradient_bytes` 包含参数梯度和输入梯度。不要把 `M2-M1` 的净下降直接称为“释放了多少激活”。实际读数还需检查是否有库工作区等额外分配；不把上述关系作为未经核对的测量结果。

FP32 small block 的理论参数数量为 `4*d_model² + 3*d_model*d_ff + 2*d_model = 9,438,720`，参数梯度预期 **36.005859 MiB**。输入梯度为 `4*1024*768*4` 字节，即 **12 MiB**，两者合计 **48.005859 MiB**。这两个数也会从实际 `.grad` 张量中记录到 JSON，供与时间线推算对照；它们不是前向保存量。

## 最终交付

保存单 block 前向/反向的 NVTX 和 CUDA Memory Usage 截图，以及前五个贡献操作的证据。结果表包含：操作、保存字节数、占比；再记录保存总量、推算梯度量、参数梯度与输入梯度理论值。写 1–2 段解释。

正式结果：前向新增保存存储 **734.094 MiB**，最终参数与输入梯度合计 **48.006 MiB**。操作贡献、反向区间、原始记录和离线复现命令见 [单 block 显存报告](block_memory_report.md)。表格与重绘图已保存；对话中你提供的 Nsight 界面截图没有另存为本地文件。

## 小规模入口验证

2026-10-08，RTX 4090、PyTorch 2.11.0+cu128、Nsight Systems 2025.1.1。使用 d_model=32、d_ff=128、4 头、batch=1、长度 8 验证实际采集入口，原始文件位于 `outputs/setup_smoke/block_memory_20261008.*`。

三个边界和两个正式阶段各出现一次；174 个前向序号与 174 个反向序号对应。报告包含 390 次 kernel 调用（含预热）和 644 条 device 显存分配/释放事件。时间线推算的最终梯度为 66,816 字节，与实际参数梯度 65,792 字节加输入梯度 1,024 字节相符。

这证明入口、标签和读数核对方法可用，不是正式 small 配置的实验结果。
