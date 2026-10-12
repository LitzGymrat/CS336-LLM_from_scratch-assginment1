#table(
  columns: 3,
  align: left,
  table.header("case", "implementation", "first_execution_s"),
  "注意力前向+反向", "eager", "0.652",
  "注意力前向+反向", "compiled", "2.269",
  "模型前向", "eager", "0.355",
  "模型前向", "compiled", "47.741",
  "模型前向+反向", "eager", "0.625",
  "模型前向+反向", "compiled", "56.763",
  "模型完整训练", "eager", "0.611",
  "模型完整训练", "compiled", "36.994",
)
