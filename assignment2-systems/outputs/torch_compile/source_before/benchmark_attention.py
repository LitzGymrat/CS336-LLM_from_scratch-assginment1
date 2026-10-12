"""测量 A1 单头 attention 的前向、反向耗时，以及反向开始前的显存。"""

import argparse
import gc
import inspect
import json
import statistics
from pathlib import Path
from timeit import default_timer

import torch

from cs336_basics.Transformer.Scaled_dot_product_function import scaled_dot_product_attention
from cs336_systems.memory_profiling import begin_memory_measurement, profile_memory
from cs336_systems.reporting import save_table


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d_models", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memory_profile", action="store_true",
                        help="计时结束后另跑一步，复用显存 harness 保存 .pickle")
    parser.add_argument("--output_prefix", default="outputs/pytorch_attention/attention")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0:
        parser.error("steps 必须 >= 1，warmup 必须 >= 0")
    if min(args.d_models + args.seq_lens) < 1:
        parser.error("维度和序列长度必须 >= 1")
    if args.memory_profile and args.device != "cuda":
        parser.error("显存快照需要 --device cuda")
    return args


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(function, device, *args):
    """只计这次调用，返回结果供后续反向使用；CUDA 调用前后均同步。"""
    synchronize(device)
    start = default_timer()
    result = function(*args)
    synchronize(device)
    elapsed_ms = (default_timer() - start) * 1000
    return result, elapsed_ms


def clear_gradients(q, k, v):
    q.grad = None
    k.grad = None
    v.grad = None


def run_pass(q, k, v, output_gradient, device):
    output = scaled_dot_product_attention(q, k, v)
    synchronize(device)
    output.backward(output_gradient)
    synchronize(device)
    # 不把这一轮的计算图和输入梯度留到下一轮。
    del output
    clear_gradients(q, k, v)


def attention_accounting(batch_size, seq_len, d_model):
    """当前 FP32 手写 softmax 的主要张量估算，见 docs/pytorch_attention.md。"""
    matrix_bytes = 4 * batch_size * seq_len**2
    qkv_bytes = 4 * 3 * batch_size * seq_len * d_model
    # saved_tensors_hooks 核对过：exp 和概率矩阵、行和、max 的 int64 索引。
    saved_extra_bytes = 2 * matrix_bytes + 12 * batch_size * seq_len
    return {
        "attention_matrix_mib": matrix_bytes / 1024**2,
        "qkv_mib": qkv_bytes / 1024**2,
        "saved_extra_estimate_mib": saved_extra_bytes / 1024**2,
        "forward_retained_extra_estimate_mib": (4 * batch_size * seq_len * d_model + saved_extra_bytes) / 1024**2,
        # Q/K/V、外部输出梯度、输出，加上上述新保存的张量。
        "before_backward_estimate_mib": (4 * 5 * batch_size * seq_len * d_model + saved_extra_bytes) / 1024**2,
        "forward_matmul_flops": 4 * batch_size * seq_len**2 * d_model,
        "backward_matmul_flops": 8 * batch_size * seq_len**2 * d_model,
    }


def benchmark_case(args, record, device):
    torch.manual_seed(args.seed)
    shape = (8, record["seq_len"], record["d_model"])
    record["phase"] = "inputs"
    q = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=torch.float32, requires_grad=True)
    # 模拟后续层传回来的梯度，避免在计时中额外计算一个 loss。
    output_gradient = torch.randn_like(q)

    record["phase"] = "warmup"
    for _ in range(args.warmup):
        run_pass(q, k, v, output_gradient, device)

    before_backward_bytes = []
    forward_peak_bytes = []
    backward_peak_bytes = []
    synchronize(device)
    if device.type == "cuda":
        record["baseline_allocated_mib"] = torch.cuda.memory_allocated(device) / 1024**2

    for _ in range(args.steps):
        clear_gradients(q, k, v)
        record["phase"] = "forward"
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        output, forward_ms = time_call(scaled_dot_product_attention, device, q, k, v)
        record["forward_samples_ms"].append(forward_ms)

        if device.type == "cuda":
            # 此时前向已经完成，output 和反向所需的中间值仍然活跃。
            before_backward_bytes.append(torch.cuda.memory_allocated(device))
            forward_peak_bytes.append(torch.cuda.max_memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)

        record["phase"] = "backward"
        _, backward_ms = time_call(output.backward, device, output_gradient)
        record["backward_samples_ms"].append(backward_ms)
        if device.type == "cuda":
            backward_peak_bytes.append(torch.cuda.max_memory_allocated(device))
        del output

    clear_gradients(q, k, v)
    for stage in ("forward", "backward"):
        times = record[f"{stage}_samples_ms"]
        record[f"{stage}_mean_ms"] = statistics.mean(times)
        record[f"{stage}_std_ms"] = statistics.pstdev(times)
    if device.type == "cuda":
        record["before_backward_allocated_mib"] = max(before_backward_bytes) / 1024**2
        record["forward_retained_extra_mib"] = record["before_backward_allocated_mib"] - record["baseline_allocated_mib"]
        record["forward_peak_allocated_mib"] = max(forward_peak_bytes) / 1024**2
        record["backward_peak_allocated_mib"] = max(backward_peak_bytes) / 1024**2

    # 快照另跑一步，不把记录调用栈的开销混进普通计时。
    if args.memory_profile:
        record["phase"] = "snapshot"
        prefix = f"{args.output_prefix}_d{record['d_model']}_s{record['seq_len']}"
        with profile_memory(prefix, device) as memory_result:
            begin_memory_measurement(memory_result, device)
            run_pass(q, k, v, output_gradient, device)
        record["snapshot_path"] = memory_result["snapshot_path"]
    record["phase"] = "complete"


def save_results(records, output_prefix):
    # 原始耗时保存在 JSON；表格只放汇总数值。
    table_records = []
    for record in records:
        row = record.copy()
        del row["forward_samples_ms"]
        del row["backward_samples_ms"]
        table_records.append(row)
    save_table(table_records, output_prefix)
    Path(output_prefix).with_suffix(".json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    records = []
    for d_model in args.d_models:
        for seq_len in args.seq_lens:
            record = {
                "d_model": d_model, "seq_len": seq_len, "batch_size": 8,
                "device": str(device), "gpu": gpu, "dtype": "torch.float32",
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "attention_source": inspect.getfile(scaled_dot_product_attention),
                "causal": False, "warmup": args.warmup, "steps": args.steps, "seed": args.seed,
                "status": "ok", "phase": "inputs", "error": None,
                "forward_mean_ms": None, "forward_std_ms": None,
                "backward_mean_ms": None, "backward_std_ms": None,
                "baseline_allocated_mib": None, "before_backward_allocated_mib": None,
                "forward_retained_extra_mib": None,
                "forward_peak_allocated_mib": None, "backward_peak_allocated_mib": None,
                "forward_samples_ms": [], "backward_samples_ms": [],
            }
            record.update(attention_accounting(8, seq_len, d_model))
            print(f"d_model={d_model}, seq_len={seq_len}", flush=True)
            try:
                benchmark_case(args, record, device)
            except torch.OutOfMemoryError as error:
                # 题目明确要求报告 OOM；只处理这一种失败，继续其他配置。
                record["status"] = "oom"
                record["error"] = str(error)
            records.append(record)
            print(f"  status={record['status']}, forward={record['forward_mean_ms']} ms, "
                  f"backward={record['backward_mean_ms']} ms, phase={record['phase']}", flush=True)
            # 每完成一组就落盘；异常退出后也保留已完成的配置。
            save_results(records, args.output_prefix)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
