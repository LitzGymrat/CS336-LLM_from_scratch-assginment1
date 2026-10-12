"""校验本次 8 组结果，复用 reporting 导出对照表、图和中文报告。"""

import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cs336_systems.reporting import plot_bars, save_table


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/torch_compile"


def load_record(name, samples, mean_key, std_key, sample_key, population=False):
    records = json.loads((OUTPUT / f"{name}.json").read_text())
    assert len(records) == 1
    record = records[0]
    assert len(record[sample_key]) == samples
    assert abs(statistics.mean(record[sample_key]) - record[mean_key]) < 1e-9
    stdev = statistics.pstdev if population else statistics.stdev
    assert abs(stdev(record[sample_key]) - record[std_key]) < 1e-9
    assert record["seed"] == 0 and record["warmup"] == 5
    assert not record["tf32_matmul"]
    if record["implementation"] == "compiled":
        assert record["compiled_graphs"] > 0
        assert record["graph_breaks"] == 0
    return record


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def main():
    manifest = json.loads((OUTPUT / "execution.json").read_text())
    assert len(manifest["runs"]) == 9  # 正确性检查 + 8 组正式测量。
    assert all(run["exit_code"] == 0 for run in manifest["runs"])
    snapshot = OUTPUT / "source_snapshot"
    for name, expected_hash in manifest["source_sha256"].items():
        assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == expected_hash
    verification = json.loads((OUTPUT / "verification.json").read_text())
    assert len(verification) == 2 and all(result["status"] == "passed" for result in verification)

    attention = {}
    first_calls = []
    for implementation in ("eager", "compiled"):
        name = f"attention_{implementation}"
        record = load_record(name, 100, "forward_mean_ms", "forward_std_ms", "forward_samples_ms", population=True)
        assert record["status"] == "ok" and record["phase"] == "complete"
        assert (record["batch_size"], record["seq_len"], record["d_model"]) == (8, 8192, 128)
        assert len(record["backward_samples_ms"]) == 100
        assert abs(statistics.mean(record["backward_samples_ms"]) - record["backward_mean_ms"]) < 1e-9
        assert abs(statistics.pstdev(record["backward_samples_ms"]) - record["backward_std_ms"]) < 1e-9
        attention[implementation] = record
        first_calls.append({"case": "注意力前向+反向", "implementation": implementation,
                            "first_execution_s": record["first_call_ms"] / 1000})

    timing_rows = []
    summaries = []
    for stage, label in (("forward", "注意力前向"), ("backward", "注意力反向")):
        for implementation, record in attention.items():
            timing_rows.append({"case": label, "implementation": implementation,
                                "mean_ms": record[f"{stage}_mean_ms"], "std_ms": record[f"{stage}_std_ms"]})
        summaries.append({"case": label,
                          "eager_mean_ms": attention["eager"][f"{stage}_mean_ms"],
                          "eager_std_ms": attention["eager"][f"{stage}_std_ms"],
                          "compiled_mean_ms": attention["compiled"][f"{stage}_mean_ms"],
                          "compiled_std_ms": attention["compiled"][f"{stage}_std_ms"]})

    model = {}
    for mode, label in (("forward", "模型前向"), ("forward_backward", "模型前向+反向"), ("train", "模型完整训练")):
        model[mode] = {}
        for implementation in ("eager", "compiled"):
            record = load_record(f"medium_{mode}_{implementation}", 10, "mean_ms", "std_ms", "samples_ms")
            assert (record["batch_size"], record["seq_len"], record["d_model"], record["d_ff"],
                    record["num_layers"], record["num_heads"], record["vocab_size"]) == (4, 512, 1024, 4096, 24, 16, 10000)
            assert record["mode"] == mode and record["precision"] == "torch.float32"
            assert not record["autocast"] and not record["inference"] and not record["memory_profile"]
            model[mode][implementation] = record
            timing_rows.append({"case": label, "implementation": implementation,
                                "mean_ms": record["mean_ms"], "std_ms": record["std_ms"]})
            first_calls.append({"case": label, "implementation": implementation,
                                "first_execution_s": record["first_call_ms"] / 1000})
        summaries.append({"case": label,
                          "eager_mean_ms": model[mode]["eager"]["mean_ms"],
                          "eager_std_ms": model[mode]["eager"]["std_ms"],
                          "compiled_mean_ms": model[mode]["compiled"]["mean_ms"],
                          "compiled_std_ms": model[mode]["compiled"]["std_ms"]})
    for row in summaries:
        row["speedup"] = row["eager_mean_ms"] / row["compiled_mean_ms"]

    memory_rows = []
    for field, label in (("before_backward_allocated_mib", "注意力反向开始前"),
                         ("forward_peak_allocated_mib", "注意力前向峰值"),
                         ("backward_peak_allocated_mib", "注意力反向峰值")):
        memory_rows.append({"case": label, "eager_gib": attention["eager"][field] / 1024,
                            "compiled_gib": attention["compiled"][field] / 1024})
    for mode, label in (("forward", "模型前向峰值"), ("forward_backward", "模型前反向峰值"), ("train", "模型训练峰值")):
        memory_rows.append({"case": label, "eager_gib": model[mode]["eager"]["peak_allocated_mib"] / 1024,
                            "compiled_gib": model[mode]["compiled"]["peak_allocated_mib"] / 1024})
    for row in memory_rows:
        row["reduction_percent"] = 100 * (1 - row["compiled_gib"] / row["eager_gib"])

    table = save_table(summaries, OUTPUT / "comparison")
    save_table(memory_rows, OUTPUT / "memory_comparison")
    save_table(first_calls, OUTPUT / "first_execution")
    plt.rcParams["font.family"] = ["Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["pdf.fonttype"] = 42
    timing_table = save_table(timing_rows, OUTPUT / "timing_plot_data")
    plot_bars(timing_table, "case", "mean_ms", "implementation", OUTPUT / "timing_comparison",
              xlabel="测量范围", ylabel="同步墙钟耗时（ms）", title="FP32：eager 与 torch.compile", error="std_ms")
    memory_plot_rows = []
    memory_labels = {"注意力反向开始前": "注意力\n反向开始前", "注意力前向峰值": "注意力\n前向峰值",
                     "注意力反向峰值": "注意力\n反向峰值", "模型前向峰值": "模型\n前向峰值",
                     "模型前反向峰值": "模型\n前反向峰值", "模型训练峰值": "模型\n训练峰值"}
    for row in memory_rows:
        for implementation in ("eager", "compiled"):
            memory_plot_rows.append({"case": memory_labels[row["case"]], "implementation": implementation,
                                     "allocated_gib": row[f"{implementation}_gib"]})
    memory_table = save_table(memory_plot_rows, OUTPUT / "memory_plot_data")
    plot_bars(memory_table, "case", "allocated_gib", "implementation", OUTPUT / "memory_comparison_plot",
              xlabel="测量范围", ylabel="活跃分配显存（GiB）", title="同配置下的显存对比")

    timing_md = markdown_table(["测量范围", "eager（ms）", "compiled（ms）", "加速比"], [
        [row["case"], f"{row['eager_mean_ms']:.3f} ± {row['eager_std_ms']:.3f}",
         f"{row['compiled_mean_ms']:.3f} ± {row['compiled_std_ms']:.3f}", f"{row['speedup']:.2f}×"] for row in summaries])
    memory_md = markdown_table(["测量范围", "eager（GiB）", "compiled（GiB）", "显存下降"], [
        [row["case"], f"{row['eager_gib']:.3f}", f"{row['compiled_gib']:.3f}",
         "≈0%" if abs(row['reduction_percent']) < 0.05 else f"{row['reduction_percent']:.1f}%"] for row in memory_rows])
    first_md = markdown_table(["测量范围", "实现", "首次执行（s）"], [
        [row["case"], row["implementation"], f"{row['first_execution_s']:.3f}"] for row in first_calls])
    output = attention["eager"]
    retained_eager = attention["eager"]["before_backward_allocated_mib"] / 1024
    retained_compiled = attention["compiled"]["before_backward_allocated_mib"] / 1024
    report = f"""# Torch Compile 对照实验

## 范围与测量方法

本次按选定方案完成 8 组对照：单头注意力 eager/compiled 各一次；medium Transformer 的前向、前向+反向、完整训练步骤，各测 eager/compiled。原题 (a) 要求覆盖前一题全部注意力配置，本次只选了 d=128、s=8192 这一代表性配置，不能据此声称完成整个配置扫描。

- 环境：{output['gpu']}；PyTorch {output['torch_version']}；CUDA {output['cuda_version']}。
- 注意力：B=8、s=8192、d=128，FP32，无 causal mask；前向和反向各 100 个样本。反向使用预先生成的外部输出梯度，每轮清除 Q/K/V 梯度，重新构建计算图。
- Transformer：medium，B=4、s=512、d_model=1024、d_ff=4096、24 层、16 头、词表 10000；pre-norm、SwiGLU、RoPE(θ=10000)，自写 AdamW(lr=0.001)。每组 10 个样本。前向仍开启 autograd，与之前的基线一致。
- 所有组 seed=0，FP32，TF32 关闭。eager 与 compiled 都重新测量，采用相同初始化与输入。各组先独立执行一次，再预热 5 次，最后计时；完整训练两侧执行相同次数的参数更新。
- 编译范围：`torch.compile(attention, backend="inductor", mode="default")` 或 `torch.compile(model, ...)`；loss 和 optimizer 不编译。每次计时调用前后同步 CUDA，测 CPU/GPU 同步墙钟时间；未开启 Nsight、NVTX 或显存历史采集。
- 表中 ± 为标准差：注意力使用总体标准差，Transformer 沿用脚本中的样本标准差。加速比=eager 平均耗时/compiled 平均耗时。

首次调用会触发延迟编译，所以首次执行与正式样本分开记录；默认 Inductor 缓存可能被正确性检查或其他进程复用，首次执行数值不是纯编译耗时，也不是清空缓存后的冷启动编译耗时。参见 [PyTorch compile 文档](https://docs.pytorch.org/docs/2.11/generated/torch.compile.html) 与 [官方教程的首次执行说明](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)。

## 正确性核对

在两组正式形状上，以同一模型/输入比较 eager 和 compiled。注意力输出最大绝对误差 {verification[0]['output_max_abs_error']:.3e}，Q/K/V 梯度最大误差 {max(verification[0]['gradient_max_abs_errors']):.3e}；medium logits 最大误差 {verification[1]['logits_max_abs_error']:.3e}，loss 误差 {verification[1]['loss_abs_error']:.3e}，全部 {verification[1]['parameter_tensors_checked']} 个参数张量的梯度最大误差 {verification[1]['gradient_max_abs_error']:.3e}。输出采用 rtol=1e-4、atol=1e-5，梯度采用 rtol=1e-3、atol=1e-6，检查全部通过。正式 compiled 结果均记录到编译图，graph break 为 0。

## 稳态耗时

{timing_md}

![耗时对比](../outputs/torch_compile/timing_comparison.png)

注意力反向单独计时，不含前向；模型“前向+反向”包括清梯度、模型前向、交叉熵和反向；“完整训练”进一步包含 AdamW 更新。两个模型模式的差值不能当作单独计时的优化器耗时。

## 显存

{memory_md}

![显存对比](../outputs/torch_compile/memory_comparison_plot.png)

数值来自 PyTorch CUDA allocator 的活跃分配量（allocated），包括当时活跃的参数、激活、梯度及已建立的优化器状态；它不是 reserved，也不是整张卡的 nvidia-smi 用量。峰值在预热后重置，因此不包含首次编译的临时峰值。注意力“反向开始前”是前向完成、计算图仍保留时的分配量。旧脚本中手写 softmax 的保存张量估算只适用于 eager，compiled 的缓存策略以实测为准。

## 首次执行

{first_md}

上述首次耗时包括该模式的一次实际执行。注意力包括前向和反向，Transformer 各模式与正式测量一致。

## 如何解释

`torch.compile` 优化的是已有计算的执行方式：可以减少 Python 调度与 kernel 启动开销，并融合相邻操作，减少中间张量写回与再次读取。它没有改变这里的注意力公式或两次矩阵乘法的理论 FLOPs。我们手写的 softmax、RMSNorm、RoPE 和 SwiGLU 含有许多逐元素操作与归约，因此存在融合空间；本次没有采集 kernel trace，不能仅凭测速断定某一具体 kernel 或 FlashAttention 是收益来源。这个机制解释参考 [PyTorch 官方性能教程](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)，具体收益由上表实测支持。

注意力前向加速 {summaries[0]['speedup']:.2f}×，反向加速 {summaries[1]['speedup']:.2f}×；medium 前向、前反向、完整训练分别加速 {summaries[2]['speedup']:.2f}×、{summaries[3]['speedup']:.2f}×、{summaries[4]['speedup']:.2f}×。完整训练仍包含未编译的 loss 和 AdamW，因此模型部分的加速不等于每个训练环节都按相同比例加速。

这组注意力在反向开始前的活跃显存从 {retained_eager:.3f} GiB 变为 {retained_compiled:.3f} GiB，几乎没有变化；下降明显的是执行过程的峰值。因此，这里不能说编译消除了前向结束后保留的大量激活。峰值下降与融合操作、减少临时张量物化的机制吻合，这是基于结果的解释，不是 kernel trace 的直接证据。编译也未解决显式注意力矩阵随序列长度二次增长的问题，后面的 FlashAttention 正是针对这一问题进一步改变计算与访存组织。

### 金字塔图与前向结束时的保存量

之前 memory_viz 的金字塔图表示整个模型在每个时刻的总活跃显存。前向逐层保存反向需要的激活，形成总体上升趋势；反向逐层释放它们，形成下降趋势，同时还会生成梯度。最高点通常在前向尾部或反向开头附近，但不能仅凭外形断定它恰好发生在前向返回的瞬间。

前向期间的显存可以近似理解为“常驻张量 + 已经保留的中间张量 + 当前运算的临时张量”。因此，某一瞬间的执行峰值与前向结束后保留的显存，是两个不同的量。

以当前手写 softmax 为例，依次计算“分数 x → 减最大值后的 shifted_x → 指数 shifted_exp → 概率 probabilities”。计算概率时，四份大矩阵可能同时存在；注意力返回后，前两份可以释放，当前 eager 实现主要保留指数和概率矩阵供反向使用。B=8、s=8192、FP32 时，每份大矩阵约 2 GiB，因此执行时四份矩阵约 8 GiB，结束后两份约 4 GiB，其余输入、输出和库分配解释了实测数值与整数 GiB 的差别。

编译器可以融合相邻运算，使部分中间结果不必写成完整张量再读回来。这样即使结束时保留量接近，执行峰值也能下降。[PyTorch 官方融合说明](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html#fuse-operations)。这次只确认了 compiled 的总量变化，没有逐块核对其保存清单，所以不能由“总量相同”进一步断言两种实现保存了完全相同的张量。

### FlashAttention 也会减少前向结束后的保存量

“反向需要某个中间值”不等于“必须从前向一直保存它”：也可以等反向用到时重算，这与前面学习的 activation checkpointing 思路相通。FlashAttention 分块计算并避免在显存中保留完整的注意力分数和概率矩阵，反向再按块恢复所需值；减少显存读写还可以抵消重算增加的运算，使其运行更快。[FlashAttention 原论文 §3.1](https://arxiv.org/html/2205.14135v2#S3.SS1)。

| 实现 | 前向结束后为反向保留的主要张量 | 反向如何得到注意力概率 |
| --- | --- | --- |
| 当前手写 attention | 引用 Q/K/V，保留输出、完整指数和概率矩阵及少量行统计量 | 使用已保存的完整中间矩阵 |
| FlashAttention-2 | 引用 Q/K/V，保留输出 O 和每行一个 logsumexp 统计量 L | 根据输入和 L 按块重算，用完块内中间值后释放 |

FlashAttention-2 的 L 只有每行一个数；输入与输出大小为 O(Bsd)，额外归一化统计量为 O(Bs)，避免为反向保存 O(Bs²) 的完整大矩阵。[FlashAttention-2 原论文 §3.1](https://arxiv.org/html/2307.08691v1#S3.SS1)。因此它既减少长期保留量，也降低执行峰值；这里是原理对比，本轮尚未测量 FlashAttention。

当前注意力实验在前向前清除了 Q/K/V 的梯度，它们的梯度要等反向才生成；脚本预先创建的输出梯度用于模拟上游梯度，两种实现采用相同输入。因此，“反向前显存没变化”的准确表述是“本次两种实现的前向结束保留总量几乎相同”，而不是“前向已经保存了所有要算出的梯度”。

## 可用于作业的回答

(a) 在单头注意力 B=8、d=128、s=8192、FP32 的代表性配置上，torch.compile 将前向从 {summaries[0]['eager_mean_ms']:.3f} ms 降至 {summaries[0]['compiled_mean_ms']:.3f} ms（{summaries[0]['speedup']:.2f}×），反向从 {summaries[1]['eager_mean_ms']:.3f} ms 降至 {summaries[1]['compiled_mean_ms']:.3f} ms（{summaries[1]['speedup']:.2f}×）。首次编译与执行不计入这些稳态均值；本次只测试选定配置。

(b) 在 medium、B=4、s=512、FP32 上，仅编译 Transformer 模型后，前向、前向+反向和完整训练步骤分别获得 {summaries[2]['speedup']:.2f}×、{summaries[3]['speedup']:.2f}×、{summaries[4]['speedup']:.2f}× 加速。各组采用相同输入、初始化和计时边界，loss 与 AdamW 未编译。

## 结果与复现

- 原始样本与完整条件：`outputs/torch_compile/attention_*.json`、`medium_*.json`。
- 对照表：`comparison.csv/.tex/.typ`、`memory_comparison.csv/.tex/.typ`。
- 正确性：`verification.json`；命令、退出码和源码哈希：`execution.json`。
- 实际执行源码：`source_snapshot/`；已有本地 benchmark 修改前备份：`source_before/`。
- 串行运行：服务器 A2 目录内执行 `uv run python -u run_compile_experiments.py`；本地重新生成报告：`uv run python report_compile.py`。重新运行会覆盖这些同名实验结果，应先保留已有目录。
"""
    (ROOT / "docs/torch_compile_report.md").write_text(report, encoding="utf-8")
    print(table.to_string(index=False))
    print("报告：docs/torch_compile_report.md")


if __name__ == "__main__":
    main()
