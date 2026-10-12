"""medium / batch=4 / context=512 / FP32，测预测最优分组和两种相邻分组。"""

import json
import subprocess
import sys
from pathlib import Path

from matplotlib.figure import Figure

from cs336_systems.reporting import save_table

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/gradient_checkpointing"
GIB = 1024**3


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    execution = {"scope": __doc__, "runs": []}
    records = []
    results = []

    def run(name, group_size):
        prefix = OUTPUT / name
        if Path(str(prefix) + ".json").exists() or Path(str(prefix) + ".log").exists():
            raise FileExistsError(f"已有实验记录：{prefix}")
        command = [sys.executable, "-m", "cs336_systems.profile_gradient_checkpointing",
                   "--output_prefix", str(prefix)]
        if group_size is not None:
            command += ["--group_size", str(group_size)]
        print(f"运行：{name}", flush=True)
        with Path(str(prefix) + ".log").open("w") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       check=False)
        execution["runs"].append({"command": command, "exit_code": completed.returncode,
                                  "log": str(prefix) + ".log"})
        (OUTPUT / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
        if completed.returncode != 0:
            raise RuntimeError(f"采集失败，请查看 {prefix}.log")
        result = json.loads(Path(str(prefix) + ".json").read_text())
        selected = result["selected_theory"]
        results.append(result)
        records.append({
            "group_size": selected["group_size"],
            "checkpoint_count": selected["checkpoint_count"],
            "checkpoint_mib": selected["checkpoint_bytes"] / 1024**2,
            "recomputed_internal_mib": selected["recomputed_internal_bytes"] / 1024**2,
            "activation_estimate_mib": selected["activation_estimate_bytes"] / 1024**2,
            "total_estimate_gib": selected["total_estimate_bytes"] / GIB,
            "peak_allocated_gib": result["memory"]["peak_allocated_bytes"] / GIB,
            "peak_reserved_gib": result["memory"]["peak_reserved_bytes"] / GIB,
            "forward_peak_gib": result["forward_peak_allocated_bytes"] / GIB,
            "elapsed_ms": result["elapsed_ms"],
        })
        print(f"每段 {selected['group_size']} 个 block，实测峰值 "
              f"{result['memory']['peak_allocated_bytes']/GIB:.6f} GiB", flush=True)
        return result

    first = run("medium_b4_s512_fp32_predicted", None)
    best_size = first["predicted_optimum"]["group_size"]
    num_layers = first["config"]["num_layers"]
    # 内点比较左右相邻值；边界没有一侧的合法值时，比较同侧最近的两个值。
    if best_size == 1:
        neighbors = [2, 3]
    elif best_size == num_layers:
        neighbors = [num_layers - 2, num_layers - 1]
    else:
        neighbors = [best_size - 1, best_size + 1]
    for size in neighbors:
        run(f"medium_b4_s512_fp32_g{size}", size)

    for result in results:
        assert result["transformer_source_sha256"] == first["transformer_source_sha256"]
        assert result["single_block_internal_saved_bytes"] == first["single_block_internal_saved_bytes"]
        for key in ("d_model", "d_ff", "num_layers", "num_heads", "batch_size", "context_len", "seed", "warmup"):
            assert result["config"][key] == first["config"][key]
    records.sort(key=lambda row: row["group_size"])
    prefix = OUTPUT / "medium_b4_s512_fp32_comparison"
    save_table(records, prefix)

    figure = Figure(figsize=(8, 4.5))
    axes = figure.subplots()
    axes.plot([row["group_size"] for row in records], [row["total_estimate_gib"] for row in records],
              "o--", label="Analytical estimate (includes full gradient buffers)")
    axes.plot([row["group_size"] for row in records], [row["peak_allocated_gib"] for row in records],
              "o-", color="#d47720", label="Measured forward + backward")
    axes.set(xlabel="Blocks per checkpoint region", ylabel="Peak tensor memory (GiB)",
             title="medium: N=24, B=4, S=512, FP32")
    axes.set_xticks([row["group_size"] for row in records])
    axes.grid(alpha=0.2)
    axes.legend()
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(Path(str(prefix) + "." + extension), dpi=180, bbox_inches="tight")
    summary = {
        "measured_strategy_count": len(records), "records": records,
        "predicted_group_size": best_size,
        "lowest_measured_group_size": min(records, key=lambda row: row["peak_allocated_gib"])["group_size"],
        "claim_scope": "minimum among these three measured strategies, not an exhaustive GPU search",
    }
    Path(str(prefix) + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"三组对照完成：{prefix}", flush=True)


if __name__ == "__main__":
    main()
