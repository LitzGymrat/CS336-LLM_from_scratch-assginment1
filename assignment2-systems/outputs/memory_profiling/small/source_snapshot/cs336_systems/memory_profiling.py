"""记录 CUDA 分配历史，导出供 memory_viz 打开的快照。"""

import json
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def profile_memory(output_prefix: str, device: torch.device):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path = prefix.with_suffix(".pickle")
    result = {"status": "ok", "profile_phase": "warmup"}

    # 先覆盖预热，防止长序列在预热时 OOM 而没有留下快照。
    # 预热成功后，benchmark 会清掉这段历史，重新记录正式步骤。
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        yield result
    except torch.OutOfMemoryError:
        result["status"] = "oom"
        raise
    finally:
        torch.cuda.synchronize(device)
        result.update({
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "snapshot_path": str(snapshot_path),
        })
        torch.cuda.memory._dump_snapshot(str(snapshot_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        prefix.with_suffix(".memory.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"显存快照：{snapshot_path}", flush=True)


def begin_memory_measurement(result: dict, device: torch.device):
    """保留当前张量，清除预热事件和峰值，开始记录正式步骤。"""
    torch.cuda.memory._record_memory_history(max_entries=1_000_000, clear_history=True)
    torch.cuda.reset_peak_memory_stats(device)
    result["profile_phase"] = "measurement"
    result["measurement_start_allocated_bytes"] = torch.cuda.memory_allocated(device)
