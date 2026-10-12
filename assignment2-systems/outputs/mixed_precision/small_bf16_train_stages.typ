#table(
  columns: 24,
  align: left,
  table.header("vocab_size", "d_model", "d_ff", "num_layers", "num_heads", "rope_theta", "batch_size", "seq_len", "device", "precision", "parameter_dtype", "autocast", "tf32_matmul", "seed", "norm_position", "use_rope", "ffn_type", "lr", "mode", "warmup", "steps", "stage", "mean_ms", "std_ms"),
  "10000", "768", "3072", "12", "12", "10000.0", "4", "512", "cuda", "torch.bfloat16", "torch.float32", "True", "False", "0", "pre", "True", "swiglu", "0.001", "train_stages", "5", "10", "forward_with_loss", "19.363", "1.433",
  "10000", "768", "3072", "12", "12", "10000.0", "4", "512", "cuda", "torch.bfloat16", "torch.float32", "True", "False", "0", "pre", "True", "swiglu", "0.001", "train_stages", "5", "10", "backward", "34.806", "1.173",
  "10000", "768", "3072", "12", "12", "10000.0", "4", "512", "cuda", "torch.bfloat16", "torch.float32", "True", "False", "0", "pre", "True", "swiglu", "0.001", "train_stages", "5", "10", "optimizer", "12.219", "1.305",
)
