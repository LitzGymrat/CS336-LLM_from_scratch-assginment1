#table(
  columns: 34,
  align: left,
  table.header("d_model", "seq_len", "batch_size", "device", "gpu", "dtype", "torch_version", "cuda_version", "tf32_matmul", "attention_source", "causal", "warmup", "steps", "seed", "status", "phase", "error", "forward_mean_ms", "forward_std_ms", "backward_mean_ms", "backward_std_ms", "baseline_allocated_mib", "before_backward_allocated_mib", "forward_retained_extra_mib", "forward_peak_allocated_mib", "backward_peak_allocated_mib", "attention_matrix_mib", "qkv_mib", "saved_extra_estimate_mib", "forward_retained_extra_estimate_mib", "before_backward_estimate_mib", "forward_matmul_flops", "backward_matmul_flops", "snapshot_path"),
  "16", "256", "8", "cuda", "NVIDIA GeForce RTX 4090", "torch.float32", "2.11.0+cu128", "12.8", "False", "/root/autodl-tmp/cs336/assignment1-basics/cs336_basics/Transformer/Scaled_dot_product_function.py", "False", "1", "3", "0", "ok", "complete", "--", "0.663", "0.429", "0.785", "0.213", "16.75", "20.898", "4.148", "24.773", "29.023", "2.0", "0.375", "4.023", "4.148", "4.648", "33554432", "67108864", "outputs/pytorch_attention/smoke_cuda_d16_s256.pickle",
)
