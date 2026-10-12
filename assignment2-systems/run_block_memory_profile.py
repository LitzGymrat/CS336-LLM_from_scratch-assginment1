"""一条命令采集一个 block 的 Nsight 显存报告。"""

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from cs336_systems.profile_block_memory import parse_args

ROOT = Path(__file__).resolve().parent


def main():
    args = parse_args()
    prefix = Path(args.output_prefix)
    if not prefix.is_absolute():
        prefix = ROOT / prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # 不覆盖已经采集的报告；需要重跑时使用新的 output_prefix。
    for suffix in (".nsys-rep", ".sqlite", ".json", ".command.json", ".log"):
        if Path(str(prefix) + suffix).exists():
            raise FileExistsError(f"已有 {prefix}{suffix}，请换一个 --output_prefix")

    command = [
        "nsys", "profile", "--trace=cuda,nvtx", "--cuda-memory-usage=true",
        "--sample=none", "--cpuctxsw=none", "--export=sqlite",
        f"--output={prefix}",
        sys.executable, "-m", "cs336_systems.profile_block_memory",
    ]
    for name, value in vars(args).items():
        command.extend([f"--{name}", str(prefix) if name == "output_prefix" else str(value)])
    # 必须在新的 Python 进程启动、CUDA 初始化之前设置。
    environment = os.environ.copy()
    environment["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    source_hashes = {}
    for path in (ROOT / "run_block_memory_profile.py", ROOT / "cs336_systems/profile_block_memory.py"):
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "command": command,
        "cwd": str(ROOT),
        "environment_overrides": {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1", "PYTHONUNBUFFERED": "1"},
        "harness_sha256": source_hashes,
    }
    record_path = Path(str(prefix) + ".command.json")
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(shlex.join(command), flush=True)
    with Path(str(prefix) + ".log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    record["exit_code"] = result.returncode
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"采集失败，请查看 {prefix}.log")
    for suffix in (".nsys-rep", ".sqlite", ".json"):
        if not Path(str(prefix) + suffix).is_file():
            raise RuntimeError(f"缺少 {prefix}{suffix}，请查看采集日志")
    print(f"完成：{prefix}.nsys-rep")


if __name__ == "__main__":
    main()
