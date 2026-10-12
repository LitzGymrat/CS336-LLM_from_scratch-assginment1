#table(
  columns: 11,
  align: left,
  table.header("model", "seq_len", "forward_top_kernel", "forward_total_ms_10_steps", "forward_calls_per_step", "forward_kernel_percent", "forward_backward_top_kernel", "forward_backward_total_ms_10_steps", "forward_backward_calls_per_step", "forward_backward_kernel_percent", "same_top_kernel"),
  "small", "256", "ampere_sgemm_128x64_tn", "74.317", "85.0", "68.981", "ampere_sgemm_128x64_tn", "74.097", "85.0", "21.741", "True",
  "small", "512", "ampere_sgemm_128x64_tn", "133.083", "84.0", "55.341", "ampere_sgemm_128x64_tn", "135.533", "84.0", "16.915", "True",
  "small", "1024", "ampere_sgemm_128x64_tn", "259.067", "84.0", "34.95", "void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)", "278.849", "171.0", "12.1", "False",
  "medium", "256", "ampere_sgemm_128x64_tn", "236.187", "169.0", "74.751", "ampere_sgemm_128x64_tn", "238.069", "169.0", "25.244", "True",
  "medium", "512", "ampere_sgemm_128x64_tn", "427.232", "168.0", "56.372", "ampere_sgemm_128x64_tn", "430.421", "168.0", "18.25", "True",
)
