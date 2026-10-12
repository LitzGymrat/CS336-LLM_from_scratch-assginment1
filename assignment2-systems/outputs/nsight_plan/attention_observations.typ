#table(
  columns: 16,
  align: left,
  table.header("config_id", "model", "seq_len", "batch_size", "num_layers", "d_model", "num_heads", "scope", "softmax_gpu_mean_ms", "msa_matmul_gpu_mean_ms", "softmax_flops", "msa_matmul_flops", "flops_counting_convention", "time_ratio_softmax_to_matmul", "flops_ratio_softmax_to_matmul", "source_report"),
  "C2", "small", "512", "4", "12", "768", "12", "all MSA layers within one forward", "2.796", "5.038", "754384896", "154618822656", "softmax: max compare/sub/exp/sum/div each 1 -> L*B*H*S*(5S-2); matmul: multiply+add 2", "0.555", "0.005", "outputs/nsight_run/profiles/small_s512_forward.nsys-rep",
  "C1", "small", "256", "4", "12", "768", "12", "all MSA layers within one forward", "0.49", "2.498", "188448768", "67645734912", "softmax: max compare/sub/exp/sum/div each 1 -> L*B*H*S*(5S-2); matmul: multiply+add 2", "0.196", "0.003", "outputs/nsight_run/profiles/small_s256_forward.nsys-rep",
  "C3", "small", "1024", "4", "12", "768", "12", "all MSA layers within one forward", "21.045", "14.267", "3018719232", "386547056640", "softmax: max compare/sub/exp/sum/div each 1 -> L*B*H*S*(5S-2); matmul: multiply+add 2", "1.475", "0.008", "outputs/nsight_run/profiles/small_s1024_forward.nsys-rep",
  "C5", "medium", "512", "4", "24", "1024", "16", "all MSA layers within one forward", "10.992", "15.548", "2011693056", "515396075520", "softmax: max compare/sub/exp/sum/div each 1 -> L*B*H*S*(5S-2); matmul: multiply+add 2", "0.707", "0.004", "outputs/nsight_run/profiles/medium_s512_forward.nsys-rep",
  "C4", "medium", "256", "4", "24", "1024", "16", "all MSA layers within one forward", "1.194", "6.792", "502530048", "231928233984", "softmax: max compare/sub/exp/sum/div each 1 -> L*B*H*S*(5S-2); matmul: multiply+add 2", "0.176", "0.002", "outputs/nsight_run/profiles/medium_s256_forward.nsys-rep",
)
