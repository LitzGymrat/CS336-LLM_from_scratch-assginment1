"""用已经采集的三组结果生成报告，不启动 GPU 实验。"""

import json
from pathlib import Path

from matplotlib.figure import Figure

from cs336_systems.reporting import save_table

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/gradient_checkpointing"
MIB = 1024**2
GIB = 1024**3


def main():
    names = ["medium_b4_s512_fp32_predicted", "medium_b4_s512_fp32_g2", "medium_b4_s512_fp32_g3"]
    results = [json.loads((OUTPUT / f"{name}.json").read_text()) for name in names]
    first = results[0]
    entry_bytes = first["single_checkpoint_bytes"]
    rows = []
    for result in results:
        selected = result["selected_theory"]
        start_bytes = result["memory"]["measurement_start_allocated_bytes"]
        # 正式起点还没有梯度，已包含参数、buffer、入口 x 和预先创建的输出梯度。
        model_start = result["parameter_bytes"] + result["buffer_bytes"] + 2 * entry_bytes
        extra_resident = start_bytes - model_start
        rows.append({
            "group_size": selected["group_size"],
            "checkpoint_count": selected["checkpoint_count"],
            "checkpoint_mib": selected["checkpoint_bytes"] / MIB,
            "recomputed_internal_mib": selected["recomputed_internal_bytes"] / MIB,
            "activation_estimate_mib": selected["activation_estimate_bytes"] / MIB,
            "model_only_estimate_gib": selected["total_estimate_bytes"] / GIB,
            "extra_resident_mib": extra_resident / MIB,
            "baseline_adjusted_estimate_gib": (selected["total_estimate_bytes"] + extra_resident) / GIB,
            "measured_peak_gib": result["memory"]["peak_allocated_bytes"] / GIB,
            "forward_end_gib": result["forward_end_allocated_bytes"] / GIB,
            "elapsed_ms": result["elapsed_ms"],
        })
    rows.sort(key=lambda row: row["group_size"])
    prefix = OUTPUT / "medium_b4_s512_fp32_analysis"
    save_table(rows, prefix)
    Path(str(prefix) + ".json").write_text(json.dumps(rows, indent=2) + "\n")

    figure = Figure(figsize=(8, 4.8))
    axes = figure.subplots()
    groups = [row["group_size"] for row in rows]
    axes.plot(groups, [row["model_only_estimate_gib"] for row in rows], "o:",
              color="#888888", label="Model-only estimate")
    axes.plot(groups, [row["baseline_adjusted_estimate_gib"] for row in rows], "o--",
              color="#2869a8", label="Estimate with measured resident baseline")
    axes.plot(groups, [row["measured_peak_gib"] for row in rows], "o-",
              color="#d47720", label="Measured forward + backward peak")
    axes.set(xlabel="Blocks per checkpoint region", ylabel="Memory (GiB)",
             title="medium: 24 independent blocks, B=4, S=512, FP32", xticks=groups)
    axes.grid(alpha=0.2)
    axes.legend()
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(Path(str(prefix) + "." + extension), dpi=180, bbox_inches="tight")

    table = "\n".join(
        f"| {r['group_size']} | {r['checkpoint_count']} | {r['checkpoint_mib']:.3f} | "
        f"{r['recomputed_internal_mib']:.3f} | {r['model_only_estimate_gib']:.3f} | "
        f"{r['baseline_adjusted_estimate_gib']:.3f} | **{r['measured_peak_gib']:.3f}** |"
        for r in rows
    )
    boundaries = "\n".join(
        f"| {r['group_size']} | {r['forward_end_gib']:.3f} | {r['measured_peak_gib']:.3f} |"
        for r in rows
    )
    snapshots = "\n".join(f"- [每段 {r['group_size']} 个 block 的快照](../outputs/gradient_checkpointing/{name}.pickle)"
                           for name, r in zip(names, rows, strict=True))
    num_layers = first["config"]["num_layers"]
    checkpoint_mib = entry_bytes / MIB
    internal_mib = first["single_block_internal_saved_bytes"] / MIB
    continuous_minimum_mib = 2 * first["continuous_optimal_checkpoint_count"] * checkpoint_mib
    derivative_at_max = checkpoint_mib - internal_mib / num_layers
    derivation = rf"""## 理论演算：用均值不等式选择 K

目标是在不嵌套、均匀分组的近似模型里，选择段数 K，使激活峰值最小。设 N 为总 block 数，M(C) 为一份入口检查点大小，M(B) 为一个 block 内部的保存存储大小。
K 段的入口要保存 K×M(C)；反向处理一段时，最多重建 N/K 个 block 的中间值，所以使用你提出的估计：

\[
M_{{\mathrm{{act}}}}(K)\approx K M(C)+\frac{{N}}{{K}}M(B).
\]

两项均为正，且乘积与 K 无关：

\[
\left[K M(C)\right]\left[\frac{{N}}{{K}}M(B)\right]=N M(C)M(B).
\]

由均值不等式，有

\[
K M(C)+\frac{{N}}{{K}}M(B)\ge 2\sqrt{{N M(C)M(B)}}.
\]

等号当且仅当两项相等：

\[
K M(C)=\frac{{N}}{{K}}M(B)
\quad\Longrightarrow\quad
K^2=\frac{{N M(B)}}{{M(C)}}
\quad\Longrightarrow\quad
K^*=\sqrt{{\frac{{N M(B)}}{{M(C)}}}}.
\]

本次入口是形状 (4,512,1024) 的 FP32 张量，因此

\[
M(C)=\frac{{4\times512\times1024\times4}}{{1024^2}}={checkpoint_mib:.3f}\ \mathrm{{MiB}}.
\]

M(B) 使用当前实现的保存存储校准值 **{internal_mib:.6f} MiB**，不是从参数数量推出来的，也不是一个 block 的完整显存峰值。代入 N={num_layers}：

\[
M_{{\mathrm{{act}}}}(K)\approx {checkpoint_mib:g}K+\frac{{{num_layers}\times{internal_mib:.6f}}}{{K}}
={checkpoint_mib:g}K+\frac{{{num_layers*internal_mib:.3f}}}{{K}}\quad(\mathrm{{MiB}}),
\]

\[
K^*=\sqrt{{\frac{{{num_layers}\times{internal_mib:.6f}}}{{{checkpoint_mib:g}}}}}
\approx {first['continuous_optimal_checkpoint_count']:.3f}.
\]

均值不等式给出的连续下界是 {continuous_minimum_mib:.3f} MiB，但合法范围是 1≤K≤{num_layers}；每段至少包含一个完整 block，所以本次无法取到 K≈{first['continuous_optimal_checkpoint_count']:.3f}。
需要再检查受约束的最优点，不能直接把无法达到的等号条件当作答案。对上述近似表达式求导，在合法范围内：

\[
M'_{{\mathrm{{act}}}}(K)={checkpoint_mib:g}-\frac{{{num_layers*internal_mib:.3f}}}{{K^2}}
\le {checkpoint_mib:g}-\frac{{{num_layers*internal_mib:.3f}}}{{{num_layers}^2}}
={derivative_at_max:.6f}<0.
\]

因此它在合法范围内随 K 增大而减小，选 K={num_layers}，也就是每段 1 个 block。
这说明“两项相等”是能够达到时的内点最优条件；受边界约束时，最优点可以不相等。本次最优点的两项分别为 {rows[0]['checkpoint_mib']:.3f} 和 {rows[0]['recomputed_internal_mib']:.6f} MiB。

三种实测分组对应的激活估计为：

- 每段 1 个 block，K=24：24×8+1×360.890625=552.890625 MiB。
- 每段 2 个 block，K=12：12×8+2×360.890625=817.781250 MiB。
- 每段 3 个 block，K=8：8×8+3×360.890625=1146.671875 MiB。

这三种分组都能整除 24 层。加入本近似模型中的固定内存项不改变最优分组；完整峰值的估计与实测差异在下表中另外说明。
"""
    report = f"""# gradient_checkpointing (b)：分组与显存峰值

2026-10-08，RTX 4090，PyTorch {first['torch_version']}，CUDA {first['cuda_version']}。

## 可用于答题的解释

按本次选定的 medium 配置，单个检查点入口为 **{entry_bytes/MIB:.3f} MiB**，单 block 新增并保存的中间存储为 **{first['single_block_internal_saved_bytes']/MIB:.3f} MiB**。
在不嵌套、以完整 block 为单位分组的限制下，用 `K×M(C)+(24/K)×M(B)` 估计激活峰值，连续最优 K≈{first['continuous_optimal_checkpoint_count']:.3f} 超过 24 层，因此预测每段 1 个 block 最省显存。
该点已经是最小合法分组，没有每段 0 个 block 的左邻，故比较每段 1、2、3 个 block；完整前向、重算和反向的实测峰值分别为 **{rows[0]['measured_peak_gib']:.3f}、{rows[1]['measured_peak_gib']:.3f}、{rows[2]['measured_peak_gib']:.3f} GiB**。
每段 1 个 block 在这三组中最低，与理论预测一致；分组变大节省的入口存储不足以抵消段内重算时更多中间值同时存在的开销。

## 配置与测量边界

使用你自己的 TransformerBlock：d_model=1024、d_ff=4096、24 层、16 头、batch=4、context=512、FP32、PreNorm、SwiGLU、RoPE；24 层的权重相互独立。
每组一个独立 GPU 进程，固定种子 0，预热一次，然后清空参数和输入梯度、同步 GPU、清除预热历史并重置峰值，正式运行一次完整前向与反向。
输入需要梯度，输出梯度提前创建；采集包含反向中的重算，保留输出到反向结束，没有 embedding、LM head、交叉熵或优化器。
这三组验证的是用户指定的 medium/512 配置，原题的 xl/batch=4/context=2048 没有运行；也没有穷举所有分组的 GPU 测量。

峰值使用 `torch.cuda.max_memory_allocated()`，单位 GiB=1024³ 字节。它记录 PyTorch 管理的分配峰值，包括参数、梯度、激活和临时分配；不是 allocator 保留显存，也不是 `nvidia-smi` 的整进程占用。[PyTorch 峰值统计说明](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.memory.max_memory_allocated.html)

{derivation}

## 理论与实测

参数为 {first['parameter_bytes']//4:,} 个 FP32 元素，共 {first['parameter_bytes']/GIB:.6f} GiB；最终参数梯度字节数相同。表中的激活估计沿用你的两项公式：检查点入口存储加上一个分组的段内保存存储。
“模型估计”再加参数、全部参数梯度、buffer，以及输出、上游梯度和输入梯度；入口 x 已算在检查点项中，不重复加。

| 每段 block 数 | K：段数 | 检查点（MiB） | 段内中间存储估计（MiB） | 模型估计（GiB） | 常驻基线校正后估计（GiB） | 实测完整峰值（GiB） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

正式测量起点没有参数梯度，但实际分配比“参数+buffer+输入+上游梯度”多 **{rows[0]['extra_resident_mib']:.3f} MiB**，三组一致。
这部分是预热后仍常驻、未被模型元素计数覆盖的分配；本次没有按调用栈进一步归属，不能直接指定它来自哪个库。
“常驻基线校正后估计”将这部分加回原模型估计，让实际存在的共同基线进入对比。
两种估计都不是精确峰值公式：它们假定全部梯度、全部入口检查点和一整段保存存储同时存在，却没有逐时刻描述它们的释放，也没有完整计入反向临时张量。
因此理论主要用于预测最优分组和变化方向，实际峰值仍以采集为准；不能拿总峰值减去全部梯度大小，直接宣称剩余值就是激活峰值。

![分组的理论估计与实测峰值](../outputs/gradient_checkpointing/medium_b4_s512_fp32_analysis.png)

图中灰色点线是模型估计，蓝色虚线是在模型估计上加入实测额外常驻基线，橙色实线是完整前向/反向的实测峰值。

| 每段 block 数 | 前向结束分配（GiB） | 完整过程峰值（GiB） |
| ---: | ---: | ---: |
{boundaries}

这里最容易混淆的地方是：**前向结束的保存量随分组变大而下降，完整峰值却上升**。
更多 block 放在同一段时入口检查点减少，但反向处理该段前要重建更多中间值，所以优化对象应是整个过程的最大值。

## 重算量与耗时

在每段完整重算一次的理想模型里，每个 block 都额外前向一次，分组大小不改变总重算量，只改变中间值同时存活的数量。
代价来自开启 checkpoint 本身；已经付出这次额外前向后，调整分组可以继续降低峰值，而不增加这个理想模型的计算量。
本次使用 `use_reentrant=False`，PyTorch 默认会在反向所需中间值已经重建后提前停止，因此实际不一定执行每段最后所有前向操作。[PyTorch checkpoint 说明](https://docs.pytorch.org/docs/2.11/checkpoint.html)
本次分别记录了 {rows[0]['elapsed_ms']:.1f}、{rows[1]['elapsed_ms']:.1f}、{rows[2]['elapsed_ms']:.1f} ms，但每组只有一个正式样本且开启分配历史采集，这些数值只是运行记录，不足以给出可靠的性能差异结论，也不能量化相对不开 checkpoint 的额外耗时。

## 核对与文件

三组 GPU 运行都正常退出，所有参数梯度均存在且有限，FP32 参数梯度字节数与参数完全一致，输入梯度也有限。
一个 CPU 小模型另外核对了每段 1、2、3 个 block：输出、输入梯度、全部参数梯度均与普通前向/反向匹配，其中每段 2 个 block 覆盖了最后一段不满的情况。
三组模型源码哈希相同，且与本地当前实现一致。采集日志出现 PyTorch 调用栈展开 warning，部分栈信息可能不完整；峰值读数与梯度核对正常。

- [整理后的数据表](../outputs/gradient_checkpointing/medium_b4_s512_fp32_analysis.csv)、[PDF 图](../outputs/gradient_checkpointing/medium_b4_s512_fp32_analysis.pdf)
- [原始三组汇总](../outputs/gradient_checkpointing/medium_b4_s512_fp32_comparison.json)、[实际执行命令与返回码](../outputs/gradient_checkpointing/execution.json)
- [正确性核对](../outputs/gradient_checkpointing/correctness_check.json)
- [下载校验清单](../outputs/gradient_checkpointing/download_manifest.json)：43 个原始结果文件均与服务器 SHA-256 一致。
{snapshots}

从已有结果重新生成本报告，不启动 GPU：

```bash
.venv/bin/python report_gradient_checkpointing.py
```

实验入口是 `run_gradient_checkpointing.py`，一次启动上述三组；它拒绝覆盖已有实验记录。
"""
    path = ROOT / "docs/gradient_checkpointing_report.md"
    path.write_text(report, encoding="utf-8")
    print(f"已生成报告：{path}")


if __name__ == "__main__":
    main()
