# 2026-10-12 发布记录

本次在已有个人仓库 main 分支上增加 A2 与当前 2026 版 A1，不覆盖原有 A1 / KV-cache / AttnRes 代码。

## 来源

- `assignment1-basics-2026/`：本地当前 A1 个人实现及官方起始文件。
- `assignment2-systems/`：本地当前 A2 实现、测试适配、实验 harness、报告与较小结果。
- [source_manifest.json](source_manifest.json)：逐文件记录原始源码路径、发布路径、SHA-256 和大小。
- [verification.json](verification.json)：本次发布副本检查；历史 CUDA 测试与性能记录另保留在 outputs。

为保证 A2 使用正确版本的模型，发布副本只调整 `pyproject.toml` 与 `uv.lock` 中的本地依赖目录、`run_memory_experiments.py` 的源码归档目录。README/.gitignore 为发布说明。算法与测量逻辑未修改。

## 工件

Git 保存代码、报告、图表、CSV、JSON 样本、校验与较小日志。大 Nsight/SQLite、逐事件表和 memory_viz 快照放在 `cs336-a2-raw-results-20261012.zip`，包含当前 A2 outputs 下全部 881 个文件：

- 压缩包 SHA-256：`5f5011bcf37baae8f7c47bab9f27dc8aa987b52c712b28979f22b843aef241dc`。
- 解压目标是仓库根目录，路径为 `assignment2-systems/outputs/...`。
- 不包含虚拟环境、缓存、完整训练语料或训练权重。
- 旧 A1 原有文件与历史分支保持原样；该仓库先前已提交的历史数据与 W&B 记录仍在原位置。

## 证据边界

自写 kernel 是 FlashAttention-2 风格的融合前向。反向采用 PyTorch 重算并支持 torch.compile；Triton 分块反向尚未实现。完整 Transformer 训练的 compile 对照使用普通注意力，不能与自写 FlashAttention 的算子收益合并。

长序列 FlashAttention 数值校验在 S≤512 检查全部 query 行，较长序列抽查均匀分布和 tile 边界附近的行；成功配置检查 NaN/Inf，不能描述为长序列逐元素完整参考验证。正式计时前构造输入，并排除 JIT。

模型部署到端侧、量化、多模态训练和完整分布式训练不属于当前已完成工作的声明。
