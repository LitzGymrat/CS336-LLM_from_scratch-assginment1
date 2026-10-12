"""测量一种非嵌套 checkpoint 分组的完整前向、重算和反向显存峰值。"""

import argparse
import hashlib
import inspect
import json
import math
import os
import time
from pathlib import Path

import torch
from cs336_basics.Transformer.Transformer_block import Transformer_blk
from matplotlib.figure import Figure
from torch.utils.checkpoint import checkpoint

from cs336_systems.memory_profiling import begin_memory_measurement, profile_memory
from cs336_systems.reporting import save_table

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2
GIB = 1024**3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d_model", type=int, default=1024)
    parser.add_argument("--d_ff", type=int, default=4096)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_len", type=int, default=512)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group_size", type=int, default=None,
                        help="每段的 block 数；省略时选择理论预测最好的分组")
    parser.add_argument("--output_prefix", default="outputs/gradient_checkpointing/medium_b4_s512_fp32")
    return parser.parse_args()


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def storage_key(tensor):
    return tensor.device, tensor.untyped_storage().data_ptr()


def block_saved_bytes(block, x):
    # 参数的 view 也共享参数存储；用存储地址排除，不能只判断 Parameter 类型。
    resident = {storage_key(x)}
    for tensor in list(block.parameters()) + list(block.buffers()):
        resident.add(storage_key(tensor))
    saved = {}

    def pack_hook(tensor):
        key = storage_key(tensor)
        if key not in resident:
            saved[key] = tensor.untyped_storage().nbytes()
        return tensor

    def unpack_hook(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        output = block(x)
    del output  # 校准用的计算图不进入正式峰值测量。
    return sum(saved.values()), len(saved)


def run_group(x, blocks, start, end):
    for index in range(start, end):
        x = blocks[index](x)
    return x


def forward_checkpointed(blocks, x, group_size):
    for start in range(0, len(blocks), group_size):
        end = min(start + group_size, len(blocks))
        # start/end 作为参数传入，反向重算时仍使用这次调用对应的范围。
        x = checkpoint(run_group, x, blocks, start, end, use_reentrant=False)
    return x


def theoretical_groups(num_layers, checkpoint_bytes, internal_bytes, fixed_bytes):
    records = []
    for group_size in range(1, num_layers + 1):
        count = math.ceil(num_layers / group_size)
        activation_bytes = count * checkpoint_bytes + group_size * internal_bytes
        records.append({
            "group_size": group_size,
            "checkpoint_count": count,
            "checkpoint_bytes": count * checkpoint_bytes,
            "recomputed_internal_bytes": group_size * internal_bytes,
            "activation_estimate_bytes": activation_bytes,
            "total_estimate_bytes": fixed_bytes + activation_bytes,
        })
    best = min(records, key=lambda row: row["activation_estimate_bytes"])
    return records, best


def save_plot(records, best, measured_bytes, config, prefix):
    figure = Figure(figsize=(8, 4.5))
    axes = figure.subplots()
    axes.plot([r["group_size"] for r in records],
              [r["total_estimate_bytes"] / GIB for r in records],
              label="Analytical estimate (includes full gradient buffers)")
    axes.scatter([best["group_size"]], [measured_bytes / GIB],
                 color="#d47720", marker="*", s=140, zorder=3,
                 label="Measured forward + backward (one run)")
    axes.set(xlabel="Blocks per checkpoint region", ylabel="Peak tensor memory (GiB)",
             title=f"N={config.num_layers}, d={config.d_model}, B={config.batch_size}, S={config.context_len}, FP32")
    axes.grid(alpha=0.2)
    axes.legend()
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(Path(str(prefix) + "." + extension), dpi=180, bbox_inches="tight")


def main():
    args = parse_args()
    prefix = ROOT / args.output_prefix
    if Path(str(prefix) + ".json").exists():
        raise FileExistsError("已有实验结果，请指定新的 --output_prefix")
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") == "1":
        raise ValueError("本次用 PyTorch allocator 测峰值，需要启用正常的 CUDA memory caching。")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    # 各层参数独立，不复用同一个 block 来降低参数显存。
    blocks = torch.nn.ModuleList([
        Transformer_blk(d_model=args.d_model, d_ff=args.d_ff, num_heads=args.num_heads,
                        max_seq_len=args.context_len, theta=args.rope_theta,
                        device=device, dtype=torch.float32)
        for _ in range(args.num_layers)
    ])
    x = torch.randn(args.batch_size, args.context_len, args.d_model,
                    device=device, dtype=torch.float32, requires_grad=True)
    output_gradient = torch.randn_like(x)

    internal_bytes, saved_count = block_saved_bytes(blocks[0], x)
    checkpoint_bytes = tensor_bytes(x)
    parameter_bytes = sum(tensor_bytes(p) for p in blocks.parameters())
    buffer_bytes = sum(tensor_bytes(b) for b in blocks.buffers())
    # 完整峰值的粗略估计：参数、全部参数梯度、buffer，以及上游梯度/输出/输入梯度。
    # 激活项已包含入口 x，因此这里不再为 x 加一份。
    fixed_bytes = 2 * parameter_bytes + buffer_bytes + 3 * checkpoint_bytes
    theory, best = theoretical_groups(args.num_layers, checkpoint_bytes, internal_bytes, fixed_bytes)
    group_size = best["group_size"] if args.group_size is None else args.group_size
    if not 1 <= group_size <= args.num_layers:
        raise ValueError("group_size 必须在 1 到 num_layers 之间")
    selected = theory[group_size - 1]
    print(f"M(C)={checkpoint_bytes/MIB:.6f} MiB，M(B)={internal_bytes/MIB:.6f} MiB", flush=True)
    print(f"本进程测一组：每段 {group_size} 个 block，共 {selected['checkpoint_count']} 段", flush=True)

    with profile_memory(str(prefix), device) as memory:
        for _ in range(args.warmup):
            blocks.zero_grad(set_to_none=True)
            x.grad = None
            output = forward_checkpointed(blocks, x, group_size)
            output.backward(output_gradient)
            del output
        blocks.zero_grad(set_to_none=True)
        x.grad = None
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        begin_memory_measurement(memory, device)

        start = time.perf_counter()
        output = forward_checkpointed(blocks, x, group_size)
        torch.cuda.synchronize(device)
        forward_end_bytes = torch.cuda.memory_allocated(device)
        forward_peak_bytes = torch.cuda.max_memory_allocated(device)
        output.backward(output_gradient)
        torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        backward_end_bytes = torch.cuda.memory_allocated(device)

    # 核对放在正式采集之外，避免检查本身的临时张量改变峰值。
    parameter_gradient_bytes = 0
    for parameter in blocks.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all().item()
        parameter_gradient_bytes += tensor_bytes(parameter.grad)
    assert parameter_gradient_bytes == parameter_bytes
    assert x.grad is not None and torch.isfinite(x.grad).all().item()
    source_directory = Path(inspect.getfile(Transformer_blk)).parent
    record = {
        "config": vars(args),
        "scope": "independent Transformer blocks; forward/backward including recomputation; no optimizer, embedding or LM head",
        "measured_strategy_count": 1,
        "precision": "torch.float32",
        "use_reentrant": False,
        "nested": False,
        "compiled": False,
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameter_bytes": parameter_bytes,
        "parameter_gradient_bytes": parameter_gradient_bytes,
        "input_gradient_bytes": tensor_bytes(x.grad),
        "buffer_bytes": buffer_bytes,
        "single_checkpoint_bytes": checkpoint_bytes,
        "single_block_internal_saved_bytes": internal_bytes,
        "single_block_saved_storage_count": saved_count,
        "continuous_optimal_checkpoint_count": math.sqrt(args.num_layers * internal_bytes / checkpoint_bytes),
        "selected_theory": selected,
        "predicted_optimum": best,
        "theory_fixed_bytes": fixed_bytes,
        "forward_end_allocated_bytes": forward_end_bytes,
        "forward_peak_allocated_bytes": forward_peak_bytes,
        "backward_end_allocated_bytes": backward_end_bytes,
        "elapsed_ms": elapsed_ms,
        "memory": memory,
        "checks": {"all_parameter_gradients_present": True, "all_gradients_finite": True,
                   "fp32_gradient_bytes_match_parameters": True},
        "transformer_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(source_directory.glob("*.py"))
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    Path(str(prefix) + ".json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    save_table(theory, Path(str(prefix) + "_theory"))
    save_table([{
        "num_layers": args.num_layers, "group_size": group_size,
        "checkpoint_count": selected["checkpoint_count"],
        "single_checkpoint_mib": checkpoint_bytes / MIB,
        "single_block_internal_mib": internal_bytes / MIB,
        "activation_estimate_mib": selected["activation_estimate_bytes"] / MIB,
        "total_estimate_gib": selected["total_estimate_bytes"] / GIB,
        "peak_allocated_gib": memory["peak_allocated_bytes"] / GIB,
        "peak_reserved_gib": memory["peak_reserved_bytes"] / GIB,
    }], prefix)
    save_plot(theory, selected, memory["peak_allocated_bytes"], args, prefix)
    print(f"理论总峰值估计={selected['total_estimate_bytes']/GIB:.6f} GiB", flush=True)
    print(f"实测完整前向/反向峰值={memory['peak_allocated_bytes']/GIB:.6f} GiB", flush=True)
    print(f"已保存结果：{prefix}", flush=True)


if __name__ == "__main__":
    main()
