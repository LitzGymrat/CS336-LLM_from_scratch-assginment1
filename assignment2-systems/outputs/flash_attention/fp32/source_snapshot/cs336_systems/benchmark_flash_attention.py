"""测三项：Triton 前向、普通 PyTorch 前向、未 compile 的自定义反向。"""

import argparse
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.testing

from cs336_systems.flash_attention2_triton import (
    FlashAttention2_with_triton,
    flash_backward,
)
from cs336_systems.reporting import save_table


# 仅在测量入口关闭 backward 的 compile，不修改学生实现。
eager_backward = flash_backward.__wrapped__


class FlashAttentionEagerBackward(FlashAttention2_with_triton):
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        dq, dk, dv = eager_backward(q, k, v, output, grad_output, lse, ctx.causal)
        return dq, dk, dv, None


def regular_attention(q, k, v, allowed):
    """显式生成分数和概率矩阵，不使用 SDPA / FlashAttention。"""
    scores = q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5
    scores = scores.masked_fill(~allowed, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return probabilities @ v


def make_inputs(batch_size, seq_len, d_model, dtype, seed):
    torch.manual_seed(seed)
    # 现有 kernel 无边界检查地读取 16 行 Q。底层存储至少留 16 行，
    # 逻辑形状仍为 (batch, seq_len, d_model)，不把短序列算作长度 16。
    storage_rows = max(seq_len, 16)
    inputs = []
    for _ in range(3):
        storage = torch.randn(batch_size, storage_rows, d_model, device="cuda")
        storage = storage.to(dtype)
        tensor = storage[:, :seq_len, :].detach().requires_grad_(True)
        inputs.append(tensor)
    grad_output = torch.randn(batch_size, seq_len, d_model, device="cuda").to(dtype)
    positions = torch.arange(seq_len, device="cuda")
    allowed = positions[:, None] >= positions[None, :]
    return inputs, grad_output, allowed, storage_rows


def error_metrics(actual, expected):
    difference = actual.detach().float() - expected.detach().float()
    return {
        "max_abs": difference.abs().max().item(),
        "relative_l2": (difference.norm() / expected.detach().float().norm()).item(),
        "close_1e2": torch.allclose(actual.detach().float(), expected.detach().float(),
                                    rtol=1e-2, atol=1e-2),
    }


def check_case(inputs, grad_output, allowed):
    # 独立参考：已量化的输入转回 FP32，关闭 TF32，检查真实误差。
    torch.backends.cuda.matmul.allow_tf32 = False
    reference_inputs = [tensor.detach().float().requires_grad_(True) for tensor in inputs]
    reference_output = regular_attention(*reference_inputs, allowed)
    reference_grads = torch.autograd.grad(reference_output, reference_inputs, grad_output.float())
    # 当前 Triton 的 tl.dot 默认使用 TF32；普通 PyTorch 也允许 TF32。
    torch.backends.cuda.matmul.allow_tf32 = True
    output = FlashAttentionEagerBackward.apply(*inputs, True)
    grads = torch.autograd.grad(output, inputs, grad_output, retain_graph=True)
    regular_output = regular_attention(*inputs, allowed)
    tensors = [output, regular_output, *grads]
    for tensor in tensors:
        assert torch.isfinite(tensor).all(), "Output or gradient contains NaN/Inf"
    checks = {
        "triton_forward": error_metrics(output, reference_output),
        "pytorch_forward": error_metrics(regular_output, reference_output),
        "triton_backward_eager": [error_metrics(a, b) for a, b in zip(grads, reference_grads)],
        "input_dtype": str(inputs[0].dtype),
        "output_dtype": str(output.dtype),
        "lse_dtype": str(output.grad_fn.saved_tensors[-1].dtype),
        "gradient_dtypes": [str(gradient.dtype) for gradient in grads],
    }
    return output, checks


def measure_call(fn, inputs, warmup_ms, rep_ms, is_backward=False):
    times = triton.testing.do_bench(
        fn,
        warmup=warmup_ms,
        rep=rep_ms,
        grad_to_none=inputs if is_backward else None,
        return_mode="all",
    )
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "samples": len(times),
    }, times


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[2**i for i in range(1, 10)])
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "bf16"], default=["fp32", "bf16"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--warmup_ms", type=float, default=25)
    parser.add_argument("--rep_ms", type=float, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_prefix", default="outputs/flash_attention/benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__).resolve(), root / "cs336_systems/flash_attention2_triton.py",
                    root / "cs336_systems/reporting.py"]
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "arguments": vars(args),
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "source_model": {"name": "medium", "d_model": 1024, "num_heads": 16},
        "causal": True,
        "autograd_enabled": True,
        "autocast": False,
        "backward_compiled": False,
        "tile_sizes": {"query": 16, "key": 16},
        "pytorch_tf32_allowed": True,
        "triton_fp32_dot_precision": "default tf32",
        "short_sequence_storage": "At least 16 physical rows; logical sequence length unchanged",
        "timing": "do_bench CUDA events, L2 cache cleared; backward only retains its graph",
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
    }
    records, raw_timings, checks = [], [], []
    expected_cases = len(args.seq_lens) * len(args.precisions) * 3
    print(f"GPU: {metadata['gpu']}; planned cases: {expected_cases}", flush=True)
    for seq_len in args.seq_lens:
        for precision in args.precisions:
            dtype = torch.float32 if precision == "fp32" else torch.bfloat16
            inputs, grad_output, allowed, storage_rows = make_inputs(
                args.batch_size, seq_len, args.head_dim, dtype, args.seed)
            output, check = check_case(inputs, grad_output, allowed)
            checks.append({"seq_len": seq_len, "precision": precision, **check})
            calls = {
                "triton_forward": lambda: FlashAttentionEagerBackward.apply(*inputs, True),
                "pytorch_forward": lambda: regular_attention(*inputs, allowed),
                "triton_backward_eager": lambda: output.backward(grad_output, retain_graph=True),
            }
            for case, fn in calls.items():
                summary, times = measure_call(fn, inputs, args.warmup_ms, args.rep_ms,
                                              is_backward=case == "triton_backward_eager")
                case_checks = check[case]
                if case == "triton_backward_eager":
                    max_abs_error = max(item["max_abs"] for item in case_checks)
                    relative_l2_error = max(item["relative_l2"] for item in case_checks)
                    within_fp32_tolerance = all(item["close_1e2"] for item in case_checks)
                else:
                    max_abs_error = case_checks["max_abs"]
                    relative_l2_error = case_checks["relative_l2"]
                    within_fp32_tolerance = case_checks["close_1e2"]
                record = {"seq_len": seq_len, "precision": precision, "case": case,
                          "batch_size": args.batch_size, "head_dim": args.head_dim,
                          "storage_rows": storage_rows, "causal": True,
                          "warmup_ms": args.warmup_ms, "rep_ms": args.rep_ms,
                          "max_abs_error": max_abs_error, "relative_l2_error": relative_l2_error,
                          "within_fp32_tolerance": within_fp32_tolerance, **summary}
                records.append(record)
                raw_timings.append({"seq_len": seq_len, "precision": precision,
                                    "case": case, "times_ms": times})
                print(f"[{len(records):02}/{expected_cases}] s={seq_len:3} {precision:4} "
                      f"{case:24} {summary['median_ms']:.5f} ms (median)", flush=True)
            save_table(records, prefix, decimals=5)
            payload = {"metadata": metadata, "records": records,
                       "checks": checks, "raw_timings": raw_timings}
            prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            del calls, output, inputs, grad_output, allowed
    print(f"Saved {len(records)} cases to {prefix}", flush=True)


if __name__ == "__main__":
    main()
