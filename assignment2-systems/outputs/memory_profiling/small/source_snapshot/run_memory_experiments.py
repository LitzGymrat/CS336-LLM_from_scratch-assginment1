"""依次采集显存快照，每个配置在新进程中运行，避免互相遗留张量。"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import torch

from cs336_systems.reporting import save_table


ROOT = Path(__file__).resolve().parent
MODELS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="small")
    parser.add_argument("--context_lengths", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output_dir", default="outputs/memory_profiling/small")
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    execution = {
        "model": args.model, "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__, "runs": [], "source_sha256": {},
    }
    for name in ["cs336_systems/benchmark.py", "cs336_systems/memory_profiling.py",
                 "run_memory_experiments.py"]:
        execution["source_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        archived = output_dir / "source_snapshot" / name
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, archived)
    basics = ROOT.parent / "assignment1-basics/cs336_basics"
    for folder in ["Transformer", "Training_modules"]:
        for path in sorted((basics / folder).glob("*.py")):
            name = str(path.relative_to(ROOT.parent))
            execution["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archived = output_dir / "source_snapshot" / name
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, archived)

    records = []
    for seq_len in args.context_lengths:
        for precision in ["fp32", "bf16"]:
            for mode in ["forward", "forward_backward", "train"]:
                name = f"{args.model}_b{args.batch_size}_s{seq_len}_{precision}_{mode}"
                prefix = output_dir / name
                command = [sys.executable, "-m", "cs336_systems.benchmark", "--device", "cuda",
                           "--vocab_size", "10000", "--batch_size", str(args.batch_size),
                           "--context_len", str(seq_len), "--precision", precision,
                           "--mode", mode, "--warmup", str(args.warmup), "--steps", str(args.steps),
                           "--memory_profile", "--output_prefix", str(prefix)]
                for field, value in MODELS[args.model].items():
                    command.extend([f"--{field}", str(value)])
                if mode == "forward":
                    command.append("--inference")

                print(f"开始：{name}", flush=True)
                with prefix.with_suffix(".log").open("w") as log:
                    process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                metrics = json.loads(prefix.with_suffix(".memory.json").read_text())
                if process.returncode != 0 and metrics["status"] != "oom":
                    raise RuntimeError(f"实验失败，请查看 {prefix.with_suffix('.log')}")

                row = {"model": args.model, "batch_size": args.batch_size, "seq_len": seq_len,
                       "precision": precision, "mode": mode, "inference": mode == "forward",
                       "warmup": args.warmup, "steps": args.steps, **metrics,
                       "peak_allocated_gib": metrics["peak_allocated_bytes"] / 1024**3,
                       "peak_reserved_gib": metrics["peak_reserved_bytes"] / 1024**3}
                records.append(row)
                execution["runs"].append({"name": name, "command": shlex.join(command),
                                          "exit_code": process.returncode, "status": metrics["status"]})
                (output_dir / "execution.json").write_text(json.dumps(execution, indent=2))
                save_table(records, output_dir / "peak_memory")
                print(f"完成：{metrics['status']}；已分配峰值 {row['peak_allocated_gib']:.3f} GiB", flush=True)

    hashes = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256.json":
            hashes[str(path.relative_to(output_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output_dir / "sha256.json").write_text(json.dumps(hashes, indent=2))
    print(f"快照和汇总：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
