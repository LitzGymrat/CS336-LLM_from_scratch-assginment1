"""核对本轮单头 attention 原始结果，生成表格、图和中文报告；不运行 GPU 测量。"""

import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cs336_systems.reporting import plot_lines, save_table


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "outputs/pytorch_attention"
FIGURES = ROOT / "docs/figures"


def main():
    records = json.loads((RAW / "attention.json").read_text())
    execution = json.loads((RAW / "execution.json").read_text())
    diagnostic = json.loads((RAW / "oom_d16_s16384.diagnostic.json").read_text())
    for name, digest in execution["sha256"].items():
        assert hashlib.sha256((RAW / name).read_bytes()).hexdigest() == digest, name
    assert execution["formal_exit_code"] == execution["diagnostic_exit_code"] == 0
    expected = {(d, s) for d in (16, 32, 64, 128) for s in (256, 1024, 4096, 8192, 16384)}
    assert len(records) == 20
    assert {(row["d_model"], row["seq_len"]) for row in records} == expected
    for row in records:
        for name, value in {"batch_size": 8, "device": "cuda", "dtype": "torch.float32",
                            "steps": 100, "warmup": 5, "causal": False, "seed": 0}.items():
            assert row[name] == value, (row["d_model"], row["seq_len"], name)
        assert row["gpu"] == execution["gpu"]
        if row["status"] == "ok":
            for stage in ("forward", "backward"):
                times = row[f"{stage}_samples_ms"]
                assert len(times) == 100 and all(math.isfinite(t) and t > 0 for t in times)
                assert math.isclose(statistics.mean(times), row[f"{stage}_mean_ms"], abs_tol=1e-9)
                assert math.isclose(statistics.pstdev(times), row[f"{stage}_std_ms"], abs_tol=1e-9)
            assert math.isclose(row["forward_retained_extra_mib"],
                                row["forward_retained_extra_estimate_mib"], abs_tol=0.01)
            assert row["forward_peak_allocated_mib"] >= row["before_backward_allocated_mib"]
            assert row["backward_peak_allocated_mib"] >= row["before_backward_allocated_mib"]
        else:
            assert row["status"] == "oom" and row["error"]
    failures = [row for row in records if row["status"] == "oom"]
    successful = [row for row in records if row["status"] == "ok"]
    smallest = min(failures, key=lambda row: (row["seq_len"], row["d_model"]))
    assert (smallest["d_model"], smallest["seq_len"]) == (diagnostic["d_model"], diagnostic["seq_len"])
    assert diagnostic["status"] == "oom" and diagnostic["phase"] == "forward"
    assert "shifted_exp = torch.exp(shifted_x)" in diagnostic["traceback"]

    # 原始 CSV 保留全部条件；这份较窄的表用于报告。
    summary = []
    markdown_rows = []
    for row in records:
        item = {"d_model": row["d_model"], "seq_len": row["seq_len"], "status": row["status"]}
        for stage in ("forward", "backward"):
            item[f"{stage}_ms"] = (f"{row[f'{stage}_mean_ms']:.3f} ± {row[f'{stage}_std_ms']:.3f}"
                                    if row["status"] == "ok" else "未测量")
        for source, target in (("before_backward_allocated_mib", "before_backward_gib"),
                               ("forward_peak_allocated_mib", "forward_peak_gib"),
                               ("backward_peak_allocated_mib", "backward_peak_gib")):
            item[target] = row[source] / 1024 if row[source] is not None else None
        summary.append(item)
        cells = [str(item["d_model"]), str(item["seq_len"]), item["forward_ms"], item["backward_ms"]]
        cells.extend(f"{item[name]:.3f}" if item[name] is not None else "—"
                     for name in ("before_backward_gib", "forward_peak_gib", "backward_peak_gib"))
        cells.append("成功" if row["status"] == "ok" else f"OOM（{row['phase']}）")
        markdown_rows.append("| " + " | ".join(cells) + " |")
    save_table(summary, RAW / "summary")

    plt.rcParams.update({"font.family": ["Arial Unicode MS", "DejaVu Sans"],
                         "axes.unicode_minus": False, "pdf.fonttype": 42})
    data = pd.DataFrame(successful)
    data["维度"] = data.d_model.map(lambda value: f"d={value}")
    for stage, label in (("forward", "前向"), ("backward", "反向")):
        plot_lines(data, "seq_len", f"{stage}_mean_ms", "维度", FIGURES / f"attention_{stage}",
                   xlabel="序列长度", ylabel="耗时（ms）", title=f"单头 attention {label}：batch=8，FP32",
                   error=f"{stage}_std_ms", log_x=True, log_y=True)
    memory_rows = []
    for row in successful:
        if row["d_model"] == smallest["d_model"]:
            for column, label in (("forward_retained_extra_mib", "前向结束新增保留量"),
                                  ("forward_peak_allocated_mib", "前向峰值"),
                                  ("backward_peak_allocated_mib", "反向峰值")):
                memory_rows.append({"seq_len": row["seq_len"], "mib": row[column], "范围": label})
    plot_lines(pd.DataFrame(memory_rows), "seq_len", "mib", "范围", FIGURES / "attention_memory",
               xlabel="序列长度", ylabel="活跃显存（MiB）", title="单头 attention 显存：d=16，batch=8，FP32",
               log_x=True, log_y=True)

    indexed = {(row["d_model"], row["seq_len"]): row for row in successful}
    short, long = indexed[16, 4096], indexed[16, 8192]
    forward_ratio = long["forward_mean_ms"] / short["forward_mean_ms"]
    backward_ratio = long["backward_mean_ms"] / short["backward_mean_ms"]
    memory_ratio = long["forward_retained_extra_mib"] / short["forward_retained_extra_mib"]
    wide = indexed[128, 8192]
    dimension_forward_ratio = wide["forward_mean_ms"] / long["forward_mean_ms"]
    dimension_backward_ratio = wide["backward_mean_ms"] / long["backward_mean_ms"]
    matrix_gib = smallest["attention_matrix_mib"] / 1024
    b, s, d = smallest["batch_size"], smallest["seq_len"], smallest["d_model"]
    qkv_mib = 3 * b * s * d * 4 / 1024**2
    stream_mib = b * s * d * 4 / 1024**2
    row_mib = b * s * 12 / 1024**2
    memory_matches = max(abs(row["forward_retained_extra_mib"] - row["forward_retained_extra_estimate_mib"])
                         for row in successful)
    report = f"""# PyTorch 单头 attention 测量报告

本轮使用自己的 A1 attention，在 {execution['gpu']} 上完成题目全部 20 组配置：**{len(successful)} 组成功、{len(failures)} 组 OOM**。四种维度都能运行到长度 8192；长度 16384 全部在预热时 OOM。另对最小失败配置进行一次独立诊断，确认失败发生在前向 softmax 的 `torch.exp`，尚未进入反向。

## 配置与口径

batch=8；d∈{{16,32,64,128}}；s∈{{256,1024,4096,8192,16384}}；FP32；不加 causal mask；无 Q/K/V 投影和多头拼接。PyTorch {execution['torch_version']}，CUDA {execution['cuda_version']}，TF32 matmul={records[0]['tf32_matmul']}。每组预热 5 次，成功配置分别采集 100 次前向和反向，每次调用前后同步 CUDA。输入和外部输出梯度提前生成，前向保留 autograd；每轮创建新计算图并清空 Q/K/V 梯度，不含 loss 和优化器。

时间包含 Python 调用与同步等待，是平均值 ± **总体标准差**，不是纯 kernel 时间或置信区间。显存使用 PyTorch allocator 的活跃分配量，峰值包含基线；反向前显存是前向完成时的活跃量，不能当作执行峰值。[memory_allocated](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.memory_allocated.html)。

## 完整结果

| d | s | 前向 ms | 反向 ms | 反向前 GiB | 前向峰值 GiB | 反向峰值 GiB | 状态 |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
{chr(10).join(markdown_rows)}

OOM 发生于预热，所以没有填入虚假的 100 次测量均值或完成前向后的显存。边界仅限本轮离散配置：最长成功长度 8192，最短失败长度 16384，没有测定二者之间的精确临界值。

![前向耗时](figures/attention_forward.png)

![反向耗时](figures/attention_backward.png)

![保存量与峰值](figures/attention_memory.png)

显存图中“前向结束新增保留量”减去了 Q/K/V、外部输出梯度与其他常驻分配的基线；两条峰值线保留基线。因此图中比较的是三种明确不同的口径，不应把前向新增保留量解释为总显存。

## 最小 OOM 配置：从张量形状开始计算

采用 **B={b}、s={s}、d={d}、FP32**，单位 GiB=2³⁰ 字节，MiB=2²⁰ 字节。

1. Q/K/V 各为 `(B,s,d)`：三份共 `3×{b}×{s}×{d}×4` = **{qkv_mib:.1f} MiB**。外部输出梯度和输出各 **{stream_mib:.1f} MiB**。
2. 分数 `QKᵀ`、减最大值后的分数、指数矩阵和概率矩阵都为 `(B,s,s)`。**每一张**大小 `4×{b}×{s}²` = **{matrix_gib:.1f} GiB**。
3. 当前手写 softmax 为反向额外保存指数矩阵、概率矩阵，合计 **{2 * matrix_gib:.1f} GiB**；行和与最大值索引合计 `12Bs` = **{row_mib:.1f} MiB**。Q/K/V 以原存储的引用保存，不重复计分配。
4. 因而，假如前向能够完成，脚本反向前的主要张量总量约为 `20Bsd + 8Bs² + 12Bs` = **{smallest['before_backward_estimate_mib'] / 1024:.3f} GiB**。这个数还没有包含库工作空间和 allocator 对齐。
5. 实际前向在创建指数矩阵时已经失败：当时输入分数和减最大值后的分数仍活跃，占约 **{2 * matrix_gib:.1f} GiB**；再申请一张 **{matrix_gib:.1f} GiB** 指数矩阵，将使这三份矩阵达到 **{3 * matrix_gib:.1f} GiB**，加上输入和其他分配后无法放入本卡。诊断记录的已分配峰值为 **{diagnostic['peak_allocated_bytes'] / 1024**3:.3f} GiB**，错误要求再申请 8 GiB，报告剩余仅 7.00 GiB。这是前向执行峰值造成的失败，16 GiB 的保存量估算并不能保证前向能够完成。

若继续执行到 softmax 除法，当前局部变量还可能同时持有第四张概率矩阵，二次项达到约 **{4 * matrix_gib:.1f} GiB**；本轮实际没有执行到这个位置。诊断的 [.pickle](../outputs/pytorch_attention/oom_d16_s16384.pickle) 可在 [memory_viz](https://pytorch.org/memory_viz) 打开，错误与调用栈保存在 [diagnostic.json](../outputs/pytorch_attention/oom_d16_s16384.diagnostic.json)。

## 结果解释与作业回答

反向需要保存的主要张量是两张 `(B,s,s)` 矩阵，因此固定 batch 时按 **O(s²)** 增长。d=16 时，长度从 4096 翻倍到 8192，前向新增保留量从 {short['forward_retained_extra_mib']:.3f} MiB 增至 {long['forward_retained_extra_mib']:.3f} MiB，约 **{memory_ratio:.2f} 倍**；前向与反向耗时分别增为 **{forward_ratio:.2f}、{backward_ratio:.2f} 倍**。全部成功配置的前向新增保留量与逐项公式 `4Bsd + 8Bs² + 12Bs` 的最大差值为 {memory_matches:.6f} MiB，支持这里的保存量核算。

在长度 8192 下，维度从 16 增至 128，主要矩阵乘法 FLOPs 增加 8 倍，但前向与反向耗时仅增为 **{dimension_forward_ratio:.2f}、{dimension_backward_ratio:.2f} 倍**。这说明本轮耗时对序列长度更敏感；各维度处理的注意力矩阵大小相同，对这些 `(B,s,s)` 张量的逐元素运算和读写量不会随 d 增加。它支持“长序列的 s² 张量与数据搬运是主要压力”的判断，尚未通过 kernel profiler 精确划分各操作占比。要消除完整 s² 中间矩阵的保存成本，可以使用分块的 FlashAttention：在片上完成块内计算与在线 softmax，反向按需重算，而不是把完整矩阵一直存于显存。[FlashAttention 论文](https://arxiv.org/pdf/2205.14135#page=2)。仅降低精度可以减少每个元素的字节数，仍保留 s² 的增长。

## 复核与复现

原始记录：[CSV](../outputs/pytorch_attention/attention.csv)、[JSON](../outputs/pytorch_attention/attention.json)；较窄的报告表：[CSV](../outputs/pytorch_attention/summary.csv)、[LaTeX](../outputs/pytorch_attention/summary.tex)、[Typst](../outputs/pytorch_attention/summary.typ)。正式实验与诊断都正常退出（诊断捕获并记录预期 OOM），输入配置、100 个原始样本的均值/标准差、显存公式以及下载文件 SHA256 已核对。执行记录和源码快照位于 [execution.json](../outputs/pytorch_attention/execution.json) 与 `outputs/pytorch_attention/source_snapshot/`。

仅重建报告：在 assignment2-systems 中运行 `uv run python report_attention.py`。重新做正式实验：在 GPU 服务器运行 `uv run python -m cs336_systems.benchmark_attention`。本报告没有用诊断或此前 3 步检查的耗时替代正式数据。
"""
    (ROOT / "docs/pytorch_attention_report.md").write_text(report, encoding="utf-8")
    print(f"20 组配置已核对：{len(successful)} 成功，{len(failures)} OOM；报告及三张图已生成。")


if __name__ == "__main__":
    main()
