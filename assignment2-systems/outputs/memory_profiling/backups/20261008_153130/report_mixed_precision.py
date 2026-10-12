"""读取两套 BF16 阶段测量，与历史 FP32 结果比较；不启动新实验。"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from cs336_systems.reporting import plot_bars, save_table


ROOT = Path(__file__).resolve().parent
STAGES = {"forward_with_loss": "前向＋loss", "backward": "反向", "optimizer": "优化器"}
MATCH_FIELDS = [
    "vocab_size", "d_model", "d_ff", "num_layers", "num_heads", "rope_theta",
    "batch_size", "seq_len", "device", "norm_position", "use_rope", "ffn_type",
    "lr", "mode", "warmup", "steps",
]


def main():
    records = []
    inputs = {}
    execution = json.loads((ROOT / "outputs/mixed_precision/stage_comparison_execution.json").read_text())
    assert execution["benchmark_sha256"] == hashlib.sha256((ROOT / "cs336_systems/benchmark.py").read_bytes()).hexdigest()
    assert not execution["nvtx"] and all(run["exit_code"] == 0 for run in execution["runs"])
    plt.rcParams.update({"font.family": ["Arial Unicode MS", "DejaVu Sans"], "pdf.fonttype": 42})

    for model in ["small", "medium"]:
        old_path = ROOT / f"outputs/benchmark/{model}_train_stages.csv"
        new_path = ROOT / f"outputs/mixed_precision/{model}_bf16_train_stages.csv"
        old = pd.read_csv(old_path).set_index("stage")
        new = pd.read_csv(new_path).set_index("stage")
        assert set(old.index) == set(new.index) == set(STAGES)
        assert len(old) == len(new) == 3
        for path in [old_path, new_path]:
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()

        plot_records = []
        for stage, label in STAGES.items():
            baseline = old.loc[stage]
            mixed = new.loc[stage]
            assert all(baseline[field] == mixed[field] for field in MATCH_FIELDS)
            assert baseline["precision"] == "torch.float32"
            assert mixed["precision"] == "torch.bfloat16"
            assert mixed["parameter_dtype"] == "torch.float32" and mixed["autocast"]
            assert not mixed["tf32_matmul"]
            speedup = baseline["mean_ms"] / mixed["mean_ms"]
            records.append({
                "model": model, "stage": stage,
                "fp32_mean_ms": baseline["mean_ms"], "fp32_std_ms": baseline["std_ms"],
                "bf16_mean_ms": mixed["mean_ms"], "bf16_std_ms": mixed["std_ms"],
                "speedup": speedup, "time_reduction_pct": (1 - 1 / speedup) * 100,
            })
            for precision, row in [("FP32（历史）", baseline), ("BF16", mixed)]:
                plot_records.append({"阶段": label, "精度": precision,
                                     "mean_ms": row["mean_ms"], "std_ms": row["std_ms"]})

        plot_bars(pd.DataFrame(plot_records), "阶段", "mean_ms", "精度",
                  ROOT / f"docs/figures/mixed_precision_{model}",
                  xlabel="测量阶段", ylabel="耗时（ms）", title=f"{model}：FP32 与 BF16", error="std_ms")

    table = save_table(records, ROOT / "outputs/mixed_precision/stage_comparison")
    (ROOT / "outputs/mixed_precision/comparison_inputs.json").write_text(json.dumps(inputs, indent=2))

    lines = [
        "# 混合精度分阶段性能对照", "",
        "2026-10-08 在 RTX 4090 上新增 small、medium 各一次 BF16 `train_stages` 实验，"
        "与 `outputs/benchmark/` 中对应的历史 FP32 测量比较。两次新实验均正常完成，未开启 Nsight 或 NVTX。", "",
        "共用配置：batch=4、序列长度=512、词表=10000、SwiGLU、PreNorm、RoPE、lr=0.001；预热 5 次、正式测量 10 次。"
        "small 为 d_model=768、d_ff=3072、12 层、12 头；medium 为 d_model=1024、d_ff=4096、24 层、16 头。"
        "新实验参数仍为 FP32，开启 BF16 autocast，seed=0，matmul TF32=False，PyTorch 2.11.0+cu128。", "",
        "前向列包含 loss。反向是单独围绕 `loss.backward()` 测得，未用两次总耗时相减。"
        "各阶段边界同步 CUDA；清空梯度不在三个阶段的计时内。这些是同步边界下的阶段耗时，不是纯 kernel 时间。", "",
        "## 结果", "",
        "时间为均值 ± step 间样本标准差，单位 ms；加速比 = FP32 均值 / BF16 均值。", "",
        "| 模型 | 阶段 | 历史 FP32 | BF16 autocast | 加速比 | 耗时减少 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(f"| {row['model']} | {STAGES[row['stage']]} | "
                     f"{row['fp32_mean_ms']:.2f} ± {row['fp32_std_ms']:.2f} | "
                     f"{row['bf16_mean_ms']:.2f} ± {row['bf16_std_ms']:.2f} | "
                     f"{row['speedup']:.2f}× | {row['time_reduction_pct']:.1f}% |")
    lines.extend(["", "![small 阶段耗时](figures/mixed_precision_small.png)", "",
                  "![medium 阶段耗时](figures/mixed_precision_medium.png)", "",
                  "图中柱高是平均耗时，误差线是 step 间样本标准差，不是置信区间。", "",
                  "## 观察与解释", ""])
    for model in ["small", "medium"]:
        rows = table[table["model"] == model].set_index("stage")
        forward = rows.loc["forward_with_loss", "speedup"]
        backward = rows.loc["backward", "speedup"]
        selected = rows.loc[["forward_with_loss", "backward"]]
        combined = selected["fp32_mean_ms"].sum() / selected["bf16_mean_ms"].sum()
        lines.append(f"- {model}：前向约加速 {forward:.2f}×，反向约加速 {backward:.2f}×。"
                     f"前向与反向阶段均值之和约加速 {combined:.2f}×；这个和不称作另测的整步总时间。")
    lines.extend([
        "- 从 small 到 medium，本轮观察到前向与反向的加速比均上升；优化器未显示稳定的加速收益。"
        "两种精度下优化器均处理 FP32 参数、梯度和动量，因而它没有直接获得低精度矩阵乘法的收益。"
        "small 的历史优化器标准差较大，不能把其均值的小幅下降解释为明确加速。", "",
        "一种合理解释是：较大的矩阵乘法更容易充分使用 GPU，计算工作也更能摊薄 CPU 调度与启动开销，"
        "因此低精度矩阵运算的收益在 medium 上更明显；与此同时，FP32 归一化、loss、逐元素操作和转换开销仍在。"
        "这是根据计算结构与公开性能规律作出的解释，本轮没有采集 BF16 的逐 kernel 时间来确认各项贡献。"
        "参考 [NVIDIA 矩阵乘法性能说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。", "",
        "## 对照范围", "",
        "本轮按选定范围只比较 small 与 medium。宽度和层数同时变化，所以结果是这两套配置的观察，"
        "不能单独归因于层数或保证加速比随所有模型规模单调增长，也未覆盖 Table 1 的 large、xl、10B。"
        "旧 FP32 结果未记录随机种子，本轮是沿用历史结果的对照；模型形状与已有配置字段已逐项核对一致。"
        "BF16 前向的标准差比旧 FP32 高，正式结论保留全部 10 次测量，不挑选最快 step。", "",
        "原始日志、CSV、精确运行命令及执行信息在 `outputs/mixed_precision/`；"
        "输入文件 SHA256 在 `comparison_inputs.json`。运行 `uv run python report_mixed_precision.py` 可重新生成表格和图片，"
        "该命令不启动 GPU 测量。", "",
    ])
    (ROOT / "docs/mixed_precision_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))
    print("已生成 docs/mixed_precision_report.md、比较表和两组 PNG/PDF。")


if __name__ == "__main__":
    main()
