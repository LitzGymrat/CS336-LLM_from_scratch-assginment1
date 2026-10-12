"""串行运行选定的 8 组 compile 对照；每组独立进程、独立日志。"""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    output = root / "outputs/torch_compile"
    output.mkdir(parents=True, exist_ok=True)
    commands = [("verification", [sys.executable, "-u", "verify_compile.py"])]
    attention_args = ["--d_models", "128", "--seq_lens", "8192", "--steps", "100",
                      "--warmup", "5", "--seed", "0", "--device", "cuda", "--measure_first_call"]
    model_args = ["--d_model", "1024", "--d_ff", "4096", "--num_layers", "24",
                  "--num_heads", "16", "--vocab_size", "10000", "--batch_size", "4",
                  "--context_len", "512", "--steps", "10", "--warmup", "5",
                  "--seed", "0", "--device", "cuda", "--precision", "fp32", "--measure_first_call"]
    for implementation in ("eager", "compiled"):
        name = f"attention_{implementation}"
        command = [sys.executable, "-u", "-m", "cs336_systems.benchmark_attention", *attention_args,
                   "--output_prefix", f"outputs/torch_compile/{name}"]
        if implementation == "compiled":
            command.append("--compile")
        commands.append((name, command))
    for mode in ("forward", "forward_backward", "train"):
        for implementation in ("eager", "compiled"):
            name = f"medium_{mode}_{implementation}"
            command = [sys.executable, "-u", "-m", "cs336_systems.benchmark", *model_args,
                       "--mode", mode, "--output_prefix", f"outputs/torch_compile/{name}"]
            if implementation == "compiled":
                command.append("--compile")
            commands.append((name, command))

    sources = ["cs336_systems/benchmark.py", "cs336_systems/benchmark_attention.py",
               "run_compile_experiments.py", "verify_compile.py"]
    manifest = {"source_sha256": {}, "runs": [],
                "compile_cache": "Inductor 默认缓存；首次执行可能复用缓存，不等于纯编译耗时"}
    for name in sources:
        manifest["source_sha256"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()

    for name, command in commands:
        print(f"开始：{name}", flush=True)
        started = time.time()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        manifest["runs"].append({"name": name, "command": command,
                                 "started_unix": started, "elapsed_s": time.time() - started,
                                 "exit_code": result.returncode})
        (output / "execution.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"完成：{name}，exit={result.returncode}", flush=True)
        # 正确性失败或运行异常时停止，不将失败包装成 compiled 的测速结果。
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
