#table(
  columns: 38,
  align: left,
  table.header("d_model", "seq_len", "batch_size", "device", "gpu", "dtype", "torch_version", "cuda_version", "tf32_matmul", "attention_source", "causal", "warmup", "steps", "seed", "status", "phase", "error", "implementation", "compile_backend", "compile_mode", "first_call_ms", "memory_estimate_scope", "forward_mean_ms", "forward_std_ms", "backward_mean_ms", "backward_std_ms", "baseline_allocated_mib", "before_backward_allocated_mib", "forward_retained_extra_mib", "forward_peak_allocated_mib", "backward_peak_allocated_mib", "attention_matrix_mib", "qkv_mib", "saved_extra_estimate_mib", "forward_retained_extra_estimate_mib", "before_backward_estimate_mib", "forward_matmul_flops", "backward_matmul_flops"),
  "8", "16", "8", "cpu", "--", "torch.float32", "2.11.0", "--", "False", "/Users/mileszhu/Documents/个人项目/cs336/assignment1-basics/cs336_basics/Transformer/Scaled_dot_product_function.py", "False", "1", "2", "0", "ok", "complete", "--", "eager", "--", "--", "178.685", "eager 手写 softmax；compiled 以实测显存为准", "0.089", "0.0", "0.102", "0.012", "--", "--", "--", "--", "--", "0.008", "0.012", "0.017", "0.021", "0.037", "65536", "131072",
)
