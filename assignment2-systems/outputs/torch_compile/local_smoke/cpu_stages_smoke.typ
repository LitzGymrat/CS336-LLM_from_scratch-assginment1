#table(
  columns: 33,
  align: left,
  table.header("vocab_size", "d_model", "d_ff", "num_layers", "num_heads", "rope_theta", "batch_size", "seq_len", "device", "precision", "parameter_dtype", "autocast", "tf32_matmul", "seed", "norm_position", "use_rope", "ffn_type", "lr", "mode", "warmup", "steps", "memory_profile", "inference", "implementation", "compile_backend", "compile_mode", "first_call_ms", "torch_version", "cuda_version", "gpu", "stage", "mean_ms", "std_ms"),
  "64", "32", "128", "1", "4", "10000.0", "2", "8", "cpu", "torch.float32", "torch.float32", "False", "False", "0", "pre", "True", "swiglu", "0.001", "train_stages", "1", "2", "False", "False", "eager", "--", "--", "27.665", "2.11.0", "--", "--", "forward_with_loss", "0.355", "0.03",
  "64", "32", "128", "1", "4", "10000.0", "2", "8", "cpu", "torch.float32", "torch.float32", "False", "False", "0", "pre", "True", "swiglu", "0.001", "train_stages", "1", "2", "False", "False", "eager", "--", "--", "27.665", "2.11.0", "--", "--", "backward", "0.357", "0.024",
  "64", "32", "128", "1", "4", "10000.0", "2", "8", "cpu", "torch.float32", "torch.float32", "False", "False", "0", "pre", "True", "swiglu", "0.001", "train_stages", "1", "2", "False", "False", "eager", "--", "--", "27.665", "2.11.0", "--", "--", "optimizer", "0.232", "0.032",
)
