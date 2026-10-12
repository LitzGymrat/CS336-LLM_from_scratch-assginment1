"""Run the existing benchmark and print whole-process CUDA allocator peaks.

Use the same CLI arguments as cs336_systems.benchmark, with --device cuda.
This includes model initialization, warm-up and measurement in the peak.
"""
import torch
from cs336_systems.benchmark import main


if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    main()
    torch.cuda.synchronize()
    print(f"peak_allocated_GiB={torch.cuda.max_memory_allocated() / 2**30:.3f}")
    print(f"peak_reserved_GiB={torch.cuda.max_memory_reserved() / 2**30:.3f}")
