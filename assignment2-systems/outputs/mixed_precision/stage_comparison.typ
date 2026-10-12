#table(
  columns: 8,
  align: left,
  table.header("model", "stage", "fp32_mean_ms", "fp32_std_ms", "bf16_mean_ms", "bf16_std_ms", "speedup", "time_reduction_pct"),
  "small", "forward_with_loss", "27.037", "0.068", "19.363", "1.433", "1.396", "28.382",
  "small", "backward", "58.473", "0.226", "34.806", "1.173", "1.68", "40.475",
  "small", "optimizer", "13.195", "3.369", "12.219", "1.305", "1.08", "7.401",
  "medium", "forward_with_loss", "81.184", "0.069", "43.068", "3.983", "1.885", "46.951",
  "medium", "backward", "167.384", "0.472", "89.553", "0.547", "1.869", "46.499",
  "medium", "optimizer", "30.929", "0.081", "31.011", "0.195", "0.997", "-0.263",
)
