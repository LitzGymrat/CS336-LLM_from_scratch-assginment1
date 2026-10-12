#table(
  columns: 20,
  align: left,
  table.header("vocab_size", "d_model", "d_ff", "num_layers", "num_heads", "rope_theta", "batch_size", "seq_len", "device", "precision", "norm_position", "use_rope", "ffn_type", "lr", "mode", "warmup", "steps", "stage", "mean_ms", "std_ms"),
  "10000", "768", "3072", "12", "12", "10000.0", "4", "512", "cuda", "torch.float32", "pre", "True", "swiglu", "0.001", "train_stages", "1", "10", "forward_with_loss", "26.752", "0.145",
  "10000", "768", "3072", "12", "12", "10000.0", "4", "512", "cuda", "torch.float32", "pre", "True", "swiglu", "0.001", "train_stages", "1", "10", "backward", "58.382", "0.637",
  "10000", "768", "3072", "12", "12", "10000.0", "4", "512", "cuda", "torch.float32", "pre", "True", "swiglu", "0.001", "train_stages", "1", "10", "optimizer", "12.102", "2.898",
)
