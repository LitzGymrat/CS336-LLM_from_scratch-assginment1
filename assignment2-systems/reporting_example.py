"""结果整理工具的使用示例。数字是演示数据，不是 benchmark 结果。

在 A2 目录运行：uv run python reporting_example.py
输出到 outputs/reporting_demo/。
"""

from cs336_systems.reporting import plot_bars, plot_lines, save_table


def main():
    records = [
        {"seq_len": 256, "implementation": "pytorch", "mean_ms": 5.1264, "std_ms": 0.2},
        {"seq_len": 256, "implementation": "compiled", "mean_ms": 3.4126, "std_ms": 0.1},
        {"seq_len": 512, "implementation": "pytorch", "mean_ms": 12.3264, "std_ms": 0.4},
        {"seq_len": 512, "implementation": "compiled", "mean_ms": 7.1126, "std_ms": 0.2},
        {"seq_len": 1024, "implementation": "pytorch", "mean_ms": 35.4264, "std_ms": 0.8},
        {"seq_len": 1024, "implementation": "compiled", "mean_ms": 19.5126, "std_ms": 0.5},
    ]

    table = save_table(records, "outputs/reporting_demo/demo_table")
    plot_lines(
        table, x="seq_len", y="mean_ms", group="implementation",
        output_prefix="outputs/reporting_demo/demo_lines",
        xlabel="Sequence length (tokens)", ylabel="Latency (ms)",
        title="Reporting demo (synthetic data)", error="std_ms", log_x=True,
    )
    plot_bars(
        table, x="seq_len", y="mean_ms", group="implementation",
        output_prefix="outputs/reporting_demo/demo_bars",
        xlabel="Sequence length (tokens)", ylabel="Latency (ms)",
        title="Reporting demo (synthetic data)", error="std_ms",
    )
    print("演示表格和图已保存到 outputs/reporting_demo/；数字不是实测结果。")


if __name__ == "__main__":
    main()
