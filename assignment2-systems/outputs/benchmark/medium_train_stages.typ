#table(
  columns: 20,
  align: left,
  table.header("vocab_size", "d_model", "d_ff", "num_layers", "num_heads", "rope_theta", "batch_size", "seq_len", "device", "precision", "norm_position", "use_rope", "ffn_type", "lr", "mode", "warmup", "steps", "stage", "mean_ms", "std_ms"),
  "10000", "1024", "4096", "24", "16", "10000.0", "4", "512", "cuda", "torch.float32", "pre", "True", "swiglu", "0.001", "train_stages", "5", "10", "forward_with_loss", "81.184", "0.069",
  "10000", "1024", "4096", "24", "16", "10000.0", "4", "512", "cuda", "torch.float32", "pre", "True", "swiglu", "0.001", "train_stages", "5", "10", "backward", "167.384", "0.472",
  "10000", "1024", "4096", "24", "16", "10000.0", "4", "512", "cuda", "torch.float32", "pre", "True", "swiglu", "0.001", "train_stages", "5", "10", "optimizer", "30.929", "0.081",
)
