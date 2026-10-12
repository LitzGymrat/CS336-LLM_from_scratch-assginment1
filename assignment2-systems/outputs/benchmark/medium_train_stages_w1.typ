#table(
  columns: 20,
  align: left,
  table.header("vocab_size", "d_model", "d_ff", "num_layers", "num_heads", "rope_theta", "batch_size", "seq_len", "device", "precision", "norm_position", "use_rope", "ffn_type", "lr", "mode", "warmup", "steps", "stage", "mean_ms", "std_ms"),
  "10000", "1024", "4096", "24", "16", "10000.0", "4", "512", "cuda", "torch.float32", "pre", "True", "swiglu", "0.001", "train_stages", "1", "10", "forward_with_loss", "81.778", "1.568",
  "10000", "1024", "4096", "24", "16", "10000.0", "4", "512", "cuda", "torch.float32", "pre", "True", "swiglu", "0.001", "train_stages", "1", "10", "backward", "167.513", "0.755",
  "10000", "1024", "4096", "24", "16", "10000.0", "4", "512", "cuda", "torch.float32", "pre", "True", "swiglu", "0.001", "train_stages", "1", "10", "optimizer", "31.283", "0.668",
)
