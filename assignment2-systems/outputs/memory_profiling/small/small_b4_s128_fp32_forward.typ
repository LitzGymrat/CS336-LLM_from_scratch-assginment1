#table(
  columns: 32,
  align: left,
  table.header("vocab_size", "d_model", "d_ff", "num_layers", "num_heads", "rope_theta", "batch_size", "seq_len", "device", "precision", "parameter_dtype", "autocast", "tf32_matmul", "seed", "norm_position", "use_rope", "ffn_type", "lr", "mode", "warmup", "steps", "memory_profile", "inference", "status", "profile_phase", "measurement_start_allocated_bytes", "peak_allocated_bytes", "peak_reserved_bytes", "snapshot_path", "stage", "mean_ms", "std_ms"),
  "10000", "768", "3072", "12", "12", "10000.0", "4", "128", "cuda", "torch.float32", "torch.float32", "False", "False", "0", "pre", "True", "swiglu", "0.001", "forward", "1", "2", "True", "True", "ok", "measurement", "524897280", "556354560", "614465536", "/root/autodl-tmp/cs336/assignment2-systems/outputs/memory_profiling/small/small_b4_s128_fp32_forward.pickle", "total", "16.325", "0.157",
)
