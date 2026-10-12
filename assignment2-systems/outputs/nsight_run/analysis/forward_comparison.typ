#table(
  columns: 9,
  align: left,
  table.header("model", "seq_len", "prior_python_mean_ms", "prior_python_std_ms", "nsight_gpu_span_mean_ms", "nsight_gpu_span_std_ms", "kernel_sum_mean_ms", "profiled_python_mean_ms", "profiled_python_std_ms"),
  "small", "256", "--", "--", "21.616", "3.04", "10.774", "22.073", "3.079",
  "small", "512", "25.853", "0.056", "26.335", "0.279", "24.048", "26.468", "0.335",
  "small", "1024", "--", "--", "80.719", "0.103", "74.125", "80.937", "0.131",
  "medium", "256", "--", "--", "42.821", "3.821", "31.596", "44.28", "3.968",
  "medium", "512", "80.619", "0.227", "81.805", "0.287", "75.787", "82.098", "0.29",
)
