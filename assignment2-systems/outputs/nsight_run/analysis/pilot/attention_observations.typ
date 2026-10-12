#table(
  columns: 16,
  align: left,
  table.header("config_id", "model", "seq_len", "batch_size", "num_layers", "d_model", "num_heads", "scope", "softmax_gpu_mean_ms", "msa_matmul_gpu_mean_ms", "softmax_flops", "msa_matmul_flops", "flops_counting_convention", "time_ratio_softmax_to_matmul", "flops_ratio_softmax_to_matmul", "source_report"),
  "C2", "small", "512", "4", "12", "768", "12", "all MSA layers within one forward", "2.796", "5.038", "754384896", "154618822656", "softmax: max compare/sub/exp/sum/div each 1 -> L*B*H*S*(5S-2); matmul: multiply+add 2", "0.555", "0.005", "outputs/nsight_run/profiles/small_s512_forward.nsys-rep",
)
