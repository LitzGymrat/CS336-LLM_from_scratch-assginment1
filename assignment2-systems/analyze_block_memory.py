"""离线分析 Nsight 2025.1 导出的单 block 分配记录，不启动 GPU 实验。"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from matplotlib.figure import Figure

from cs336_systems.reporting import save_table

ROOT = Path(__file__).resolve().parent
MIB = 1024**2


def read_trace(database):
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    ranges = list(db.execute(
        "SELECT n.start,n.end,COALESCE(n.text,s.value),n.globalTid "
        "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id"
    ))
    runtime = {
        correlation: (start, end, tid, name)
        for start, end, tid, correlation, name in db.execute(
            "SELECT r.start,r.end,r.globalTid,r.correlationId,s.value "
            "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId"
        )
    }
    events = list(db.execute(
        "SELECT start,address,bytes,memoryOperationType,correlationId,globalPid,contextId,memKind "
        "FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start,rowid"
    ))
    db.close()
    return ranges, runtime, events


def allocation_lifetimes(events):
    active = {}
    lifetimes = []
    for start, address, size, operation, correlation, pid, context, kind in events:
        if kind != 2:  # Device；不混入 Device Static、Managed 或主机内存。
            continue
        key = (pid, context, address)
        if operation == 0:
            assert key not in active, f"地址重复申请但未释放：{key}"
            allocation = {
                "start": start, "end": None, "address": address, "bytes": size,
                "alloc_correlation": correlation, "free_correlation": None,
            }
            active[key] = allocation
            lifetimes.append(allocation)
        else:
            assert operation == 1, operation
            allocation = active.pop(key)
            assert allocation["bytes"] == size, "申请、释放的大小不一致"
            allocation["end"] = start
            allocation["free_correlation"] = correlation
    return lifetimes


def allocating_operator(allocation, ranges, runtime):
    start, end, tid, api = runtime[allocation["alloc_correlation"]]
    # 用 CUDA API 的线程和完整调用区间归属，避免误配到同时运行的其他线程。
    enclosing = sorted(
        (range_end - range_start, label)
        for range_start, range_end, label, range_tid in ranges
        if range_end is not None and range_tid == tid
        and range_start <= start and end <= range_end and label.startswith("aten::")
    )
    assert enclosing, allocation
    allocation["allocation_label"] = enclosing[0][1]
    # empty 是 clone 等算子的存储申请子调用；同时保留原始标签，汇总到最近的实际算子。
    wrappers = {"aten::empty", "aten::empty_like", "aten::empty_strided"}
    meaningful = [label for duration, label in enclosing if label.split(",")[0] not in wrappers]
    assert meaningful, allocation
    allocation["operator_label"] = meaningful[0]
    allocation["operator"] = meaningful[0].split(",")[0]
    allocation["cuda_api"] = api


def write_csv(path, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def save_figure(figure, prefix):
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(prefix.with_suffix("." + extension), dpi=180, bbox_inches="tight")


def make_figures(events, markers, stages, levels, grouped, output_dir):
    origin = markers["before_forward"]
    finish = markers["after_backward"]
    current = 0
    times, usages = [0.0], [levels["before_forward"] / MIB]
    for start, address, size, operation, correlation, pid, context, kind in events:
        if kind == 2:
            current += size if operation == 0 else -size
            if origin < start <= finish:
                times.append((start - origin) / 1e6)
                usages.append(current / MIB)
    times.append((finish - origin) / 1e6)
    usages.append(levels["after_backward"] / MIB)
    figure = Figure(figsize=(10, 5.5))
    stage_axes, axes = figure.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1, 4]})
    for name, color in (("block_forward", "#d3e5fa"), ("block_backward", "#f7dac9")):
        start, end = stages[name]
        left, width = (start-origin)/1e6, (end-start)/1e6
        stage_axes.barh(0, width, left=left, height=0.8, color=color, edgecolor="white")
        stage_axes.text(left+width/2, 0, name+f"\n{width:.3f} ms", ha="center", va="center")
        axes.axvspan(left, left+width, color=color, alpha=0.35)
    stage_axes.set(yticks=[], ylim=(-0.6, 0.6))
    stage_axes.tick_params(axis="x", bottom=False, labelbottom=False)
    axes.step(times, usages, where="post", color="#157535", linewidth=1.5)
    axes.fill_between(times, usages, step="post", color="#157535", alpha=0.10)
    for index, name in enumerate(("before_forward", "after_forward", "after_backward")):
        x, y = (markers[name]-origin)/1e6, levels[name]/MIB
        axes.scatter([x], [y], color="black", s=22, zorder=4)
        offset = (6, 12) if index < 2 else (-6, 12)
        axes.annotate(f"M{index}: {y:.3f} MiB", (x, y), xytext=offset,
                      textcoords="offset points", ha="left" if index < 2 else "right")
    axes.set(xlabel="Time since before_forward (ms)", ylabel="CUDA device memory usage (MiB)",
             xlim=(-2, times[-1]+2), ylim=(0, max(usages)*1.17))
    axes.grid(alpha=0.2)
    figure.suptitle("One small block: FP32, batch=4, sequence=1024\nReconstructed from Nsight allocation events")
    save_figure(figure, output_dir / "block_memory_timeline")

    figure = Figure(figsize=(8, 4.5))
    axes = figure.subplots()
    top = grouped[:6]  # 第五名有并列，两个都画出来。
    axes.barh([row["operator"] for row in top], [row["saved_mib"] for row in top], color="#367cb5")
    for index, row in enumerate(top):
        axes.text(row["saved_mib"]+3, index, f"{row['saved_mib']:.0f} MiB / {row['percent']:.2f}%", va="center")
    axes.invert_yaxis()
    axes.set(xlabel="Saved allocation storage (MiB)", xlim=(0, 300),
             title="Largest contributing operator types (fifth-place tie included)")
    axes.grid(axis="x", alpha=0.2)
    save_figure(figure, output_dir / "block_saved_operators")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_prefix", default="outputs/block_memory/small_b4_s1024_fp32")
    args = parser.parse_args()
    prefix = ROOT / args.input_prefix
    metadata = json.loads(Path(str(prefix)+".json").read_text())
    execution = json.loads(Path(str(prefix)+".command.json").read_text())
    assert execution["exit_code"] == 0
    assert not metadata["cuda_memory_caching"] and metadata["precision"] == "torch.float32"
    for name, expected in metadata["transformer_source_sha256"].items():
        source = ROOT.parent / "assignment1-basics/cs336_basics/Transformer" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name

    ranges, runtime, events = read_trace(Path(str(prefix)+".sqlite"))
    marker_names = ("before_forward", "after_forward", "after_backward")
    markers = {}
    for name in marker_names:
        matches = [start for start, end, label, tid in ranges if label == name]
        assert len(matches) == 1, (name, matches)
        markers[name] = matches[0]
    stages = {}
    for name in ("block_measurement", "block_forward", "block_backward"):
        matches = [(start, end) for start, end, label, tid in ranges if label == name]
        assert len(matches) == 1, (name, matches)
        stages[name] = matches[0]
    assert markers["before_forward"] < markers["after_forward"] < markers["after_backward"]
    levels = {
        name: sum(size if operation == 0 else -size
                  for start, address, size, operation, correlation, pid, context, kind in events
                  if kind == 2 and start <= timestamp)
        for name, timestamp in markers.items()
    }
    lifetimes = allocation_lifetimes(events)
    forward_retained = [
        a for a in lifetimes
        if markers["before_forward"] < a["start"] < markers["after_forward"]
        and (a["end"] is None or a["end"] > markers["after_forward"])
    ]
    saved = [a for a in forward_retained if a["end"] is not None and a["end"] < markers["after_backward"]]
    outputs = [a for a in forward_retained if a not in saved]
    final_gradients = [
        a for a in lifetimes if markers["after_forward"] < a["start"] < markers["after_backward"]
        and (a["end"] is None or a["end"] > markers["after_backward"])
    ]
    saved_bytes = sum(a["bytes"] for a in saved)
    output_bytes = sum(a["bytes"] for a in outputs)
    final_gradient_bytes = sum(a["bytes"] for a in final_gradients)
    assert output_bytes == metadata["output_bytes"]
    assert saved_bytes + output_bytes == levels["after_forward"] - levels["before_forward"]
    expected_gradient_bytes = metadata["parameter_gradient_bytes"] + metadata["input_gradient_bytes"]
    inferred_gradient_bytes = levels["after_backward"] - levels["before_forward"] - output_bytes
    assert final_gradient_bytes == inferred_gradient_bytes == expected_gradient_bytes

    totals = defaultdict(lambda: {"bytes": 0, "count": 0})
    instances = defaultdict(lambda: {"bytes": 0, "count": 0})
    for allocation in saved:
        allocating_operator(allocation, ranges, runtime)
        for mapping, key in ((totals, allocation["operator"]), (instances, allocation["operator_label"])):
            mapping[key]["bytes"] += allocation["bytes"]
            mapping[key]["count"] += 1
    grouped = [
        {"operator": name, "saved_bytes": data["bytes"], "saved_mib": data["bytes"]/MIB,
         "percent": 100*data["bytes"]/saved_bytes, "allocation_count": data["count"]}
        for name, data in sorted(totals.items(), key=lambda item: (-item[1]["bytes"], item[0]))
    ]
    individual = [
        {"nvtx_label": label, "saved_bytes": data["bytes"], "saved_mib": data["bytes"]/MIB,
         "percent": 100*data["bytes"]/saved_bytes, "allocation_count": data["count"]}
        for label, data in sorted(instances.items(), key=lambda item: -item[1]["bytes"])
    ]
    sequence_pattern = re.compile(r"seq\s*=\s*(\d+)")
    forward_sequences, backward_sequences = set(), set()
    for start, end, label, tid in ranges:
        match = sequence_pattern.search(label or "")
        if match:
            if stages["block_forward"][0] <= start <= stages["block_forward"][1]:
                forward_sequences.add(int(match[1]))
            if stages["block_backward"][0] <= start <= stages["block_backward"][1] and "Backward" in label:
                backward_sequences.add(int(match[1]))
    assert backward_sequences and backward_sequences <= forward_sequences
    result = {
        "config": metadata["config"], "marker_time_ns": markers, "marker_memory_bytes": levels,
        "stage_ranges_ns": stages, "stage_ms": {name: (end-start)/1e6 for name, (start, end) in stages.items()},
        "saved_allocation_count": len(saved), "saved_bytes": saved_bytes, "output_bytes": output_bytes,
        "final_gradient_allocation_count": len(final_gradients), "inferred_gradient_bytes": inferred_gradient_bytes,
        "parameter_gradient_bytes": metadata["parameter_gradient_bytes"],
        "input_gradient_bytes": metadata["input_gradient_bytes"],
        "forward_sequence_count": len(forward_sequences), "backward_sequence_count": len(backward_sequences),
        "operator_groups": grouped, "operator_instances": individual,
        "saved_lifetimes": saved, "output_lifetimes": outputs, "gradient_lifetimes": final_gradients,
        "checks": {"source_hashes_match": True, "allocation_free_sizes_match": True,
                   "saved_sum_matches_memory_difference": True, "gradient_sum_matches_tensor_bytes": True,
                   "backward_sequence_ids_match": True},
        "raw_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sorted(prefix.parent.glob(prefix.name+".*"))},
    }
    output_dir = prefix.parent / "analysis"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "block_memory_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    save_table(grouped, output_dir / "saved_by_operator")
    save_table(individual, output_dir / "saved_by_operator_instance")
    boundaries = [{"marker": name, "time_s": timestamp/1e9, "memory_bytes": levels[name],
                   "memory_mib": levels[name]/MIB} for name, timestamp in markers.items()]
    save_table(boundaries, output_dir / "memory_boundaries")
    write_csv(output_dir / "saved_allocations.csv", saved)
    make_figures(events, markers, stages, levels, grouped, output_dir)
    print(f"保存量：{saved_bytes/MIB:.6f} MiB，{len(saved)} 块存储")
    for row in grouped[:6]:
        print(f"{row['operator']}: {row['saved_mib']:.6f} MiB, {row['percent']:.4f}%")
    print(f"梯度：{inferred_gradient_bytes/MIB:.6f} MiB，与实际张量大小一致")
    print(f"分析已保存：{output_dir}")


if __name__ == "__main__":
    main()
