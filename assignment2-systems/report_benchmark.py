"""汇总本次 small / medium 基准实验；不启动新的测量。"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cs336_systems.reporting import plot_bars, save_table


ROOT = Path(__file__).resolve().parent
SIZES = ["small", "medium"]
STAGES = ["forward_with_loss", "backward", "optimizer"]
LABELS = ["前向 + loss", "反向传播", "AdamW 更新"]


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def mean_std(row):
    return f"{row['mean_ms']:.2f} ± {row['std_ms']:.2f}"


def main():
    source = ROOT / "outputs/benchmark"
    output = ROOT / "outputs/benchmark_summary"
    figures = ROOT / "docs/figures"
    figures.mkdir(parents=True, exist_ok=True)

    expected_files = set()
    for size in SIZES:
        for mode in ["forward", "forward_backward", "train", "train_stages"]:
            expected_files.add(f"{size}_{mode}.csv")
        for warmup in [0, 1, 2]:
            expected_files.add(f"{size}_train_stages_w{warmup}.csv")
    files = sorted(source.glob("*.csv"))
    assert {path.name for path in files} == expected_files

    records = []
    for path in files:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frame["model_size"] = path.name.split("_")[0]
        records.extend(frame.to_dict("records"))
    table = pd.DataFrame(records)
    assert len(table) == 30
    assert not table.duplicated(["model_size", "mode", "warmup", "stage"]).any()
    shared = {
        "vocab_size": 10000, "batch_size": 4, "seq_len": 512,
        "device": "cuda", "precision": "torch.float32", "steps": 10,
        "norm_position": "pre", "use_rope": True, "ffn_type": "swiglu",
        "rope_theta": 10000.0, "lr": 0.001,
    }
    for column, value in shared.items():
        assert table[column].eq(value).all(), column
    configs = {"small": (768, 3072, 12, 12), "medium": (1024, 4096, 24, 16)}
    for size, values in configs.items():
        data = table[table.model_size == size]
        for column, value in zip(["d_model", "d_ff", "num_layers", "num_heads"], values):
            assert data[column].eq(value).all(), (size, column)
        assert set(data[data["mode"] == "train_stages"].warmup) == {0, 1, 2, 5}
        assert data[data["mode"] != "train_stages"].warmup.eq(5).all()
    assert table[["mean_ms", "std_ms"]].gt(0).all().all()
    save_table(records, output / "measurements")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (output / "source_sha256.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8"
    )

    # 沿用已有的表格与柱状图工具；中文字体由本机 matplotlib 查找。
    plt.rcParams.update({
        "font.family": ["Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 11, "pdf.fonttype": 42,
    })
    baseline = table[(table["mode"] == "train_stages") & (table.warmup == 5)]
    plot_records = []
    for size in SIZES:
        indexed = baseline[baseline.model_size == size].set_index("stage")
        for stage, label in zip(STAGES, LABELS):
            plot_records.append({
                "模型": size, "阶段": label,
                "mean_ms": indexed.loc[stage, "mean_ms"],
                "std_ms": indexed.loc[stage, "std_ms"],
            })
    plot_bars(pd.DataFrame(plot_records), "模型", "mean_ms", "阶段",
              figures / "benchmark_stages", xlabel="模型规模", ylabel="耗时（ms）",
              title="训练各阶段耗时：5 次预热、10 次测量", error="std_ms")

    sweep = table[table["mode"] == "train_stages"]
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    for column, (stage, label) in enumerate(zip(STAGES, LABELS)):
        for size, color in zip(SIZES, ["#2675b5", "#de7a22"]):
            data = sweep[(sweep.model_size == size) & (sweep.stage == stage)].sort_values("warmup")
            axes[0, column].plot(data.warmup, data.mean_ms, "o-", color=color, label=size)
            axes[1, column].plot(data.warmup, data.std_ms, "o-", color=color, label=size)
        axes[0, column].set_title(label)
        axes[0, column].set_ylim(bottom=0)
        axes[1, column].set_yscale("log")
        axes[1, column].set_xlabel("预热次数")
        for row in [0, 1]:
            axes[row, column].set_xticks([0, 1, 2, 5])
            axes[row, column].grid(alpha=0.25)
    axes[0, 0].set_ylabel("平均耗时（ms）")
    axes[1, 0].set_ylabel("标准差（ms，对数轴）")
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, 0.96), frameon=False)
    figure.suptitle("预热次数对平均耗时与波动的影响（每个设置测量 10 次）", y=1.01)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(figures / "benchmark_warmup.png", dpi=300, bbox_inches="tight")
    figure.savefig(figures / "benchmark_warmup.pdf", bbox_inches="tight")
    plt.close(figure)

    indexed = table.set_index(["model_size", "mode", "warmup", "stage"])
    whole_rows, stage_rows, warmup_rows = [], [], []
    for size in SIZES:
        whole_rows.append([size] + [mean_std(indexed.loc[(size, mode, 5, "total")])
                                   for mode in ["forward", "forward_backward", "train"]])
        stage_rows.append([size] + [mean_std(indexed.loc[(size, "train_stages", 5, stage)])
                                   for stage in STAGES])
        for warmup in [0, 1, 2, 5]:
            warmup_rows.append([size, warmup] + [
                mean_std(indexed.loc[(size, "train_stages", warmup, stage)]) for stage in STAGES
            ])
    train_ratio = (indexed.loc[("medium", "train", 5, "total"), "mean_ms"] /
                   indexed.loc[("small", "train", 5, "total"), "mean_ms"])
    backward_shares = []
    for size in SIZES:
        data = baseline[baseline.model_size == size].set_index("stage")
        backward_shares.append(100 * data.loc["backward", "mean_ms"] / data.mean_ms.sum())
    small_optimizer = indexed.loc[("small", "train_stages", 5, "optimizer")]
    optimizer_cv = 100 * small_optimizer.std_ms / small_optimizer.mean_ms

    report = f"""# CS336 A2 基准测量小报告

整理日期：2026-10-07。本次使用自己的 A1 Transformer 实现，在 AutoDL 单卡 RTX 4090（24 GB）上测量 small、medium 两种规模，共 14 组实验、30 条阶段或整步记录。主要观察是：一次预热后，前向与反向的均值已经接近五次预热；反向传播是三个训练阶段中耗时最多的部分。

## 实验配置与计时范围

公共配置：词表 10,000，batch size 4，序列长度 512，FP32，Pre-Norm、SwiGLU、RoPE（θ = 10,000），自写 AdamW（学习率 0.001）。模型和输入、目标均随机生成，初始化与数据生成在计时外；每组测量 10 步，基线预热 5 步，预热对照为 0、1、2、5 步。

{markdown_table(["规模", "d_model", "d_ff", "层数", "头数"], [[size, *configs[size]] for size in SIZES])}

整步模式在每步计时前、后同步 CUDA；分阶段模式在各阶段边界同步。下表均为 **平均耗时 ± 样本标准差，单位 ms**，标准差来自同一次实验的 10 步，表示波动幅度，不是置信区间。前向模式保留自动求导，测量的是训练前向。

## 五次预热的基线

{markdown_table(["规模", "仅模型前向", "前向 + loss + 反向", "完整训练一步"], whole_rows)}

完整训练一步包含清空梯度、前向、loss、反向及优化器更新。medium 的完整训练耗时约为 small 的 **{train_ratio:.2f} 倍**。

{markdown_table(["规模", "前向 + loss", "反向传播", "AdamW 更新"], stage_rows)}

分阶段的前向包含交叉熵计算，清空梯度在三个阶段计时外。反向在三阶段耗时总和中的占比分别为 **{backward_shares[0]:.1f}%、{backward_shares[1]:.1f}%**。这些阶段由实际计时得到；额外同步改变执行组织，三阶段之和不能直接当作另一轮完整训练的耗时。

![五次预热后的训练各阶段耗时](figures/benchmark_stages.png)

图 1：柱高为平均耗时，黑色误差条为 ±1 个样本标准差。small 的 AdamW 更新波动较大，标准差约为均值的 **{optimizer_cv:.1f}%**。

## 预热次数对比

此处比较的全部是 `train_stages`；每次预热执行完整训练一步。

{markdown_table(["规模", "预热次数", "前向 + loss", "反向传播", "AdamW 更新"], warmup_rows)}

**一次预热已经带来主要改善。** 两种规模的前向与反向均值，相比五次预热的差异均小于 1.1%；零次预热时，前向标准差约 95 ms、反向约 13 ms，明显高于预热后的结果。medium 从一次增加到两次预热，部分阶段的波动仍进一步减小，因此“均值已接近”不等于“波动也完全相同”。

![预热次数与阶段平均耗时及标准差](figures/benchmark_warmup.png)

图 2：蓝线为 small，橙线为 medium；上排显示平均耗时，下排显示标准差。下排纵轴为对数轴，方便同时观察 0.1 ms 与近 100 ms 的波动；连线仅帮助比较离散实验点，不代表中间预热次数已经测量。

零次预热的异常符合首次执行存在额外开销的推测。[PyTorch 官方基准测试教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)也展示了首次调用加载 cuBLAS 导致明显变慢的例子；现有结果文件只保存了汇总数据，尚不能确认异常来自哪个具体初始化步骤，或集中在哪一个测量步。

**优化器受预热的影响相对较小，且结果不单调。** medium 的 AdamW 均值约 31 ms，零次与五次预热的差别约 1.5%；small 则在两次预热时测得 10.97 ± 0.90 ms，五次时为 13.20 ± 3.37 ms。因此不能由这次结果得出“预热越多，每个阶段都越快或越稳定”。

## 本次覆盖范围

结果覆盖 small、medium 的三种整步模式及训练阶段预热对照；large、xl、10B，以及整步模式的 0/1/2 次预热尚无测量记录。每个设置只有一轮 10 步测量，结果文件未记录随机种子和逐步耗时，结论限于本次配置；这些时间包含 CPU 调度与同步等待，不是纯 GPU kernel 耗时。

原始结果保留在 [outputs/benchmark](../outputs/benchmark/)，[汇总 CSV](../outputs/benchmark_summary/measurements.csv)保留原始精度及来源文件名，另有同名 LaTeX/Typst 表格与输入 CSV 的 SHA256 清单。两幅图均提供 [基线 PDF](figures/benchmark_stages.pdf) 与 [预热 PDF](figures/benchmark_warmup.pdf)。在 assignment2-systems 目录运行 `uv run python report_benchmark.py` 可重新生成报告与图表，不会启动基准测量。
"""
    destination = ROOT / "docs/benchmark_report.md"
    destination.write_text(report, encoding="utf-8")
    print(f"已汇总 {len(files)} 组实验、{len(table)} 条记录：{destination}")


if __name__ == "__main__":
    main()
