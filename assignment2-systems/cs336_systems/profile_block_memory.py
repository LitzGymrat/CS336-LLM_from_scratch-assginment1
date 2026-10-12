"""给一个真实 Transformer block 的前向、反向添加显存分析标签。"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch

from cs336_basics.Transformer.Transformer_block import Transformer_blk


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # 与之前的 small 实验保持一致；这里仅实例化其中一个 block。
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_len", type=int, default=1024)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_prefix", default="outputs/block_memory/small_b4_s1024_fp32")
    return parser.parse_args()


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def main():
    args = parse_args()
    if os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING") != "1":
        raise ValueError("请通过 run_block_memory_profile.py 启动，关闭缓存后 Nsight 才能看清张量释放。")

    torch.manual_seed(args.seed)
    block = Transformer_blk(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        max_seq_len=args.context_len,
        theta=args.rope_theta,
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    block.train()
    x = torch.randn(args.batch_size, args.context_len, args.d_model,
                    device="cuda", dtype=torch.float32, requires_grad=True)
    # 模拟后续层传回来的 dL/doutput，在正式前向之前就创建，避免混入梯度分配量。
    output_gradient = torch.randn_like(x)

    with torch.cuda.nvtx.range("warmup"):
        for _ in range(args.warmup):
            output = block(x)
            output.backward(output_gradient)
            del output
            block.zero_grad(set_to_none=True)
            x.grad = None
    torch.cuda.synchronize()

    # emit_nvtx 自动标注每个算子、输入形状，以及前向/反向对应的 seq。
    with torch.autograd.profiler.emit_nvtx(record_shapes=True):
        with torch.cuda.nvtx.range("block_measurement"):
            torch.cuda.nvtx.mark("before_forward")
            with torch.cuda.nvtx.range("block_forward"):
                output = block(x)
                torch.cuda.synchronize()
            torch.cuda.nvtx.mark("after_forward")

            with torch.cuda.nvtx.range("block_backward"):
                output.backward(output_gradient)
                torch.cuda.synchronize()
            torch.cuda.nvtx.mark("after_backward")

    # 只在区间外记录结果；不持有额外的激活张量，也不读取失效的 allocator 统计量。
    parameter_bytes = sum(tensor_bytes(p) for p in block.parameters())
    gradient_records = []
    for name, parameter in block.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"参数 {name} 没有产生梯度")
        gradient_records.append({
            "parameter": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.grad.dtype),
            "bytes": tensor_bytes(parameter.grad),
        })
    parameter_gradient_bytes = sum(record["bytes"] for record in gradient_records)
    if parameter_gradient_bytes != parameter_bytes:
        raise RuntimeError("FP32 参数梯度大小与参数大小不一致")

    source_directory = Path(inspect.getfile(Transformer_blk)).parent
    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(source_directory.glob("*.py"))
    }
    result = {
        "config": vars(args),
        "scope": "one isolated block, forward and backward, no optimizer",
        "precision": "torch.float32",
        "cuda_memory_caching": False,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "model_source": str(source_directory / "Transformer_block.py"),
        "transformer_source_sha256": source_hashes,
        "parameter_count": sum(p.numel() for p in block.parameters()),
        "parameter_bytes": parameter_bytes,
        "parameter_gradient_bytes": parameter_gradient_bytes,
        "input_bytes": tensor_bytes(x),
        "output_bytes": tensor_bytes(output),
        "input_gradient_bytes": tensor_bytes(x.grad),
        "upstream_gradient_bytes": tensor_bytes(output_gradient),
        "parameter_gradients": gradient_records,
    }
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    result_path = Path(str(prefix) + ".json")
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"参数梯度：{parameter_gradient_bytes / 1024**2:.6f} MiB")
    print(f"输入梯度：{result['input_gradient_bytes'] / 1024**2:.6f} MiB")
    print(f"已保存核对数据：{result_path}")


if __name__ == "__main__":
    main()
