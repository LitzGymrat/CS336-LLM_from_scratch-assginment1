"""只对比两种前向；每项单独启动进程，让 PyTorch OOM 不影响 Triton。"""

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from cs336_systems.benchmark_flash_attention import measure_call, regular_attention
from cs336_systems.flash_attention2_triton import FlashAttention2_with_triton
from cs336_systems.reporting import save_table


def check_forward(output, inputs, case):
    """短序列检查所有行，长序列抽查行；参考从不生成完整 N×N 矩阵。"""
    q, k, v = inputs
    n = q.shape[1]
    if n <= 512:
        rows = torch.arange(n, device=q.device)
    else:
        spaced = torch.linspace(0, n - 1, 16, device=q.device).long()
        boundaries = torch.tensor([0, 1, 15, 16, 17, 63, 64, 127, 128], device=q.device)
        rows = torch.cat((spaced, boundaries)).unique()

    with torch.no_grad():
        torch.backends.cuda.matmul.allow_tf32 = False
        scores = q[:, rows].float() @ k.float().transpose(-1, -2) / q.shape[-1] ** 0.5
        keys = torch.arange(n, device=q.device)
        scores = scores.masked_fill(keys[None, :] > rows[:, None], float("-inf"))
        reference = torch.softmax(scores, dim=-1) @ v.float()
        actual = output[:, rows].float()
        difference = actual - reference
        torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)
        assert torch.isfinite(output).all(), "Forward contains NaN/Inf"
        check = {
            "checked_rows": rows.numel(),
            "check_scope": "all_rows" if n <= 512 else "sampled_rows",
            "max_abs_error": difference.abs().max().item(),
            "relative_l2_error": (difference.norm() / reference.norm()).item(),
            "finite_output": True,
            "within_tolerance": True,
        }
        if case == "triton_forward":
            lse = output.grad_fn.saved_tensors[-1][:, rows]
            reference_lse = torch.logsumexp(scores, dim=-1)
            torch.testing.assert_close(lse, reference_lse, atol=1e-2, rtol=1e-2)
            check["lse_max_abs_error"] = (lse - reference_lse).abs().max().item()
    torch.backends.cuda.matmul.allow_tf32 = True
    return check


def run_case(args):
    n, d = args.seq_lens[0], args.head_dims[0]
    precision = args.precisions[0]
    dtype = torch.float32 if precision == "fp32" else torch.bfloat16
    record = {
        "case": args.case,
        "seq_len": n,
        "head_dim": d,
        "precision": precision,
        "batch_size": 1,
        "causal": True,
        "autograd_enabled": True,
        "autocast": False,
        "q_tile": 16,
        "k_tile": 16,
        "status": "ok",
        "one_score_matrix_gib": n * n * torch.empty((), dtype=dtype).element_size() / 1024**3,
    }
    times = []
    stage = "inputs"
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        # 同一形状和 seed 的两个进程得到相同输入；随机数生成不计时。
        inputs = [torch.randn(1, n, d, device="cuda").to(dtype).requires_grad_(True)
                  for _ in range(3)]
        if args.case == "pytorch_forward":
            stage = "causal_mask"
            positions = torch.arange(n, device="cuda")
            allowed = positions[:, None] >= positions[None, :]
            fn = lambda: regular_attention(*inputs, allowed)
        else:
            fn = lambda: FlashAttention2_with_triton.apply(*inputs, True)

        stage = "first_forward"
        fn()  # 完成首次调用/JIT，输出立即释放。
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # 单独测一次前向显存。校验、do_bench 的 L2 清理缓冲均在此范围外。
        stage = "memory_forward"
        torch.cuda.reset_peak_memory_stats()
        record["input_allocated_mib"] = torch.cuda.memory_allocated() / 1024**2
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        torch.cuda.synchronize()
        warmed_ms = start.elapsed_time(end)
        record["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        record["forward_extra_mib"] = record["peak_allocated_mib"] - record["input_allocated_mib"]
        record["output_dtype"] = str(output.dtype)
        if args.case == "triton_forward":
            record["lse_dtype"] = str(output.grad_fn.saved_tensors[-1].dtype)

        stage = "validation"
        record.update(check_forward(output, inputs, args.case))
        del output  # 不能在下一次计时期间保留上一份前向计算图。
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        stage = "timing"
        # 长序列一次调用可能超过原 100 ms 总预算；扩大预算以保留多个样本。
        rep_ms = max(args.rep_ms, 20 * warmed_ms)
        summary, times = measure_call(fn, inputs, args.warmup_ms, rep_ms)
        record.update(summary)
        record["warmup_ms"] = args.warmup_ms
        record["rep_ms"] = rep_ms
    except torch.OutOfMemoryError as error:
        record.update(status="oom", failed_stage=stage, error=str(error))
        record["allocated_at_oom_mib"] = torch.cuda.memory_allocated() / 1024**2
    return {"record": record, "raw_times_ms": times}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[2**i for i in range(7, 17)])
    parser.add_argument("--head_dims", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "bf16"], default=["fp32", "bf16"])
    parser.add_argument("--warmup_ms", type=float, default=25)
    parser.add_argument("--rep_ms", type=float, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_prefix", default="outputs/flash_attention/forward_sweep/benchmark")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=["triton_forward", "pytorch_forward"], help=argparse.SUPPRESS)
    parser.add_argument("--result_file", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        payload = run_case(args)
        Path(args.result_file).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    worker_dir = prefix.parent / "cases"
    worker_dir.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__).resolve(), root / "cs336_systems/benchmark_flash_attention.py",
                    root / "cs336_systems/flash_attention2_triton.py", root / "cs336_systems/reporting.py"]
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True).strip()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": vars(args),
        "gpu": gpu,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "batch_size": 1,
        "causal": True,
        "autograd_enabled": True,
        "autocast": False,
        "tile_sizes": {"query": 16, "key": 16},
        "pytorch_tf32_allowed": True,
        "triton_fp32_dot_precision": "default tf32",
        "timing": "do_bench CUDA events; forward wrapper/ops; excludes input/mask generation and validation",
        "memory": "single warmed forward; allocated tensor peak; includes inputs/prebuilt PyTorch mask; excludes validation and do_bench cache buffer",
        "validation": "strict FP32 reference: all rows at N<=512, sampled query rows otherwise; all output elements checked finite",
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
    }
    configs = list(itertools.product(args.seq_lens, args.head_dims, args.precisions,
                                    ("triton_forward", "pytorch_forward")))
    records, raw_timings = [], []
    print(f"GPU: {gpu}; planned forward measurements: {len(configs)}", flush=True)
    for index, (n, d, precision, case) in enumerate(configs, start=1):
        result_file = worker_dir / f"s{n}_d{d}_{precision}_{case}.json"
        command = [sys.executable, "-m", "cs336_systems.benchmark_flash_forward",
                   "--worker", "--case", case, "--seq_lens", str(n), "--head_dims", str(d),
                   "--precisions", precision, "--seed", str(args.seed),
                   "--warmup_ms", str(args.warmup_ms), "--rep_ms", str(args.rep_ms),
                   "--result_file", str(result_file.resolve())]
        subprocess.run(command, check=True, cwd=root)
        payload = json.loads(result_file.read_text())
        record = payload["record"]
        records.append(record)
        raw_timings.append({"seq_len": n, "head_dim": d, "precision": precision,
                            "case": case, "times_ms": payload["raw_times_ms"]})
        save_table(records, prefix, decimals=5)
        prefix.with_suffix(".json").write_text(
            json.dumps({"metadata": metadata, "records": records, "raw_timings": raw_timings}, indent=2),
            encoding="utf-8")
        result = "OOM" if record["status"] == "oom" else (
            f"{record['median_ms']:.5f} ms, peak {record['peak_allocated_mib']:.2f} MiB")
        print(f"[{index}/{len(configs)}] s={n} d={d} {precision} {case}: {result}", flush=True)
    print(f"Saved {len(records)} forward records to {prefix}", flush=True)


if __name__ == "__main__":
    main()
