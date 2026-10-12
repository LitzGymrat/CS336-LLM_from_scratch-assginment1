"""把实验记录导出成表格和图。

一条实验记录用一个字典表示，所有记录放进一个列表。
例如：{"seq_len": 512, "implementation": "pytorch", "mean_ms": 12.3}。
输出路径传文件名前缀，例如 results/attention，不带扩展名。
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def save_table(
    records: list[dict[str, object]],
    output_prefix: str | Path,
    *,
    decimals: int = 3,
) -> pd.DataFrame:
    """导出 CSV、LaTeX、Typst，返回可继续画图的 DataFrame。

    CSV 保留原始数值；报告表格按 decimals 显示小数。
    单元格使用字符串或数值。缺失测量用 None，另用 status 列说明原因。
    LaTeX 主文档需要加载 booktabs 包。
    """
    table = pd.DataFrame(records)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    table.to_csv(prefix.with_suffix(".csv"), index=False)
    table.to_latex(
        prefix.with_suffix(".tex"),
        index=False,
        escape=True,
        na_rep="--",
        float_format=f"%.{decimals}f",
    )

    # 当前环境的 pandas 没有 to_typst，直接生成 Typst table。
    # 使用字符串单元格，让下划线等符号按普通文本显示。
    headers = [json.dumps(str(column), ensure_ascii=False) for column in table.columns]
    lines = [
        "#table(",
        f"  columns: {len(table.columns)},",
        "  align: left,",
        "  table.header(" + ", ".join(headers) + "),",
    ]
    for row in table.round(decimals).itertuples(index=False, name=None):
        cells = []
        for value in row:
            text = "--" if pd.isna(value) else str(value)
            cells.append(json.dumps(text, ensure_ascii=False))
        lines.append("  " + ", ".join(cells) + ",")
    lines.append(")")
    prefix.with_suffix(".typ").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return table


def _save_figure(figure: Figure, output_prefix: str | Path) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_lines(
    table: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    output_prefix: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
    error: str | None = None,
    log_x: bool = False,
    log_y: bool = False,
) -> None:
    """每个 group 画一条线，适合序列长度与耗时/显存的关系。

    x、y、group、error 都是列名。error 可指定标准差列。
    传入已汇总、测量成功的记录，每个 (group, x) 对应一行。
    """
    figure, axes = plt.subplots(figsize=(7, 4))
    for name, data in table.groupby(group, sort=False):
        data = data.sort_values(x)
        errors = data[error] if error is not None else None
        axes.errorbar(data[x], data[y], yerr=errors, marker="o", capsize=3, label=str(name))

    if log_x:
        axes.set_xscale("log", base=2)
    if log_y:
        axes.set_yscale("log")
    axes.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axes.grid(alpha=0.25)
    axes.legend(title=group)
    _save_figure(figure, output_prefix)


def plot_bars(
    table: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    output_prefix: str | Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
    error: str | None = None,
) -> None:
    """按 x 的类别画分组柱状图，适合模型规模、精度、实现版本的对比。

    x 类别保留首次出现的顺序，例如 small、medium、large。
    传入已汇总、测量成功的记录，每个 (group, x) 对应一行。
    """
    categories = table[x].drop_duplicates().tolist()
    groups = list(table.groupby(group, sort=False))
    positions = np.arange(len(categories))
    width = 0.8 / len(groups)
    figure, axes = plt.subplots(figsize=(7, 4))

    for index, (name, data) in enumerate(groups):
        # 对齐类别后，同一个 x 的不同实现才会排在一起。
        data = data.set_index(x).reindex(categories)
        offset = (index - (len(groups) - 1) / 2) * width
        errors = data[error] if error is not None else None
        axes.bar(positions + offset, data[y], width=width, yerr=errors, capsize=3, label=str(name))

    axes.set_xticks(positions, labels=[str(value) for value in categories])
    axes.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axes.set_axisbelow(True)
    axes.grid(axis="y", alpha=0.25)
    axes.legend(title=group)
    _save_figure(figure, output_prefix)
