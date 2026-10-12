#table(
  columns: 6,
  align: left,
  table.header("seq_len", "precision", "forward_allocated_gib", "forward_reserved_gib", "train_allocated_gib", "train_reserved_gib"),
  "128", "fp32", "0.518", "0.572", "2.191", "2.418",
  "128", "bf16", "0.759", "0.805", "2.187", "2.418",
  "1024", "fp32", "1.302", "1.418", "10.904", "11.566",
  "1024", "bf16", "1.324", "1.402", "8.527", "8.943",
)
