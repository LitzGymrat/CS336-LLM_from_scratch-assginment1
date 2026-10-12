# Transformer 部分：开始前的准备

核对日期：2026-09-19。本地作业为 Spring 2026、Version 26.0.3，以本地手册、接口与测试为准。本文整理参考、模块顺序和学习起点，尚未实现模型。

## 当前状态和范围

- 本次已读取 [作业手册](../cs336_assignment1_basics.pdf)第 3 节（第 13–28 页）、[模型测试](../tests/test_model.py)、[数值工具测试](../tests/test_nn_utils.py)及 [adapters](../tests/adapters.py)。
- 尚未发现 Transformer 实现文件；模型与 softmax 的测试入口仍为 `NotImplementedError`。本次没有运行这些待实现测试，也没有更改已有代码。
- 已验证环境能导入 PyTorch、einops 并建立 CPU 张量：Python 3.12.14、PyTorch 2.11.0、einops 0.8.2。MPS 可用；当前先在 CPU 上做小例子和模块检查，训练时再评估设备。
- 这一阶段完成模型前向计算及参数量/FLOPs 分析。交叉熵、AdamW 和训练循环属于后续章节，待模型模块完成后进入。

## 已查阅的参考及用途

| 来源 | 当前用途 | 使用边界 |
| --- | --- | --- |
| [本地作业手册](../cs336_assignment1_basics.pdf)与[官方起始仓库](https://github.com/stanford-cs336/assignment1-basics) | 确定目标架构、接口、初始化和测试要求 | 官方仓库是起始代码，不是完整解答；不同版本不能直接混用。 |
| PyTorch 2.11 的 [Module](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Module.html) 与 [Parameter](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parameter.Parameter.html) | 理解继承、`forward`、参数注册、`parameters()` 和 `load_state_dict()` | 查询框架机制；课程要求自写的层仍由我们实现。 |
| [einops 官方基础教程](https://einops.rocks/1-einops-basics/) | 理解轴的合并、拆分、调换，辅助阅读张量形状 | 作业推荐学习 einsum；按当前模块需要逐步引入。 |
| [Karpathy 的 llama2.c/model.py](https://github.com/karpathy/llama2.c/blob/d7704bdeaa7142300edeabf179e6b1e41c637608/model.py) | 对照 RMSNorm、RoPE、Attention、FeedForward、TransformerBlock 的组织方式 | 架构接近，但不是本作业答案；以下差异必须保留。 |

最后一个链接固定到本次查到的 `model.py` 最近修改提交 `d7704bdeaa7142300edeabf179e6b1e41c637608`。已阅读其相关模块实现，本次未运行该仓库测试，因此不把它当作已验证的本地替代实现。

### 外部实现与作业的关键差异

以下对照依据上述 `model.py`、本地手册和 adapters：

| 外部实现的做法 | 本次采用的要求 |
| --- | --- |
| 使用 PyTorch 内置 Linear、Embedding、SiLU 和注意力等操作 | 按作业逐个实现相应模块；`nn.Module`、`nn.Parameter` 等允许使用，手册还明确允许初始化函数和 `torch.sigmoid`。 |
| 共享输入 embedding 与输出投影权重 | 作业 adapter 分别提供两套权重，按独立参数处理。 |
| 未传 targets 时，只输出最后位置的 logits | 作业要求输出每个输入位置的 logits。 |
| 带有可选的 GQA、dropout 和快速注意力分支 | 先实现作业规定的因果多头注意力，按各模块要求处理。 |
| 使用自己的初始化方案与 FFN 维度取整约定 | 初始化按手册第 17 页；测试显式传入的 `d_ff` 要被正确使用。 |

## 整体数据流

```text
token IDs                 (batch, seq)
    ↓ Embedding
每个 token 的向量          (batch, seq, d_model)
    ↓ 多个 TransformerBlock
结合上下文后的向量          (batch, seq, d_model)
    ↓ 最后的 RMSNorm
    ↓ Linear / LM head
下一 token 的 logits       (batch, seq, vocab_size)
```

例如，2 段文本、每段 3 个 token，向量维度为 4，词表大小为 8：输入形状是 `(2, 3)`，内部主要形状是 `(2, 3, 4)`，最终输出是 `(2, 3, 8)`。每个位置的 8 个分数对应“该位置之后的 token”候选。

第 3 节开头用概率分布描述语言模型，但具体 `transformer_lm` 交付和 adapter 要求返回未归一化的 logits。需要概率时再使用 softmax。

## 按模块推进

顺序基本跟随手册，注意力先完成不带 RoPE 的接口，再接入已实现的 RoPE。每个模块都讲目的、输入输出和小例子，再由用户实现并检查。

| 顺序 | 模块与目的 | 小例子或形状 | 对应测试 |
| --- | --- | --- | --- |
| 1 | Linear：对每个位置做同一个可学习的线性变换 | `(2, 3, 4)` → `(2, 3, 6)`；前两维不变 | `test_linear` |
| 2 | Embedding：把 ID 查表变成向量 | `[2, 5, 2]` 查询第 2、5、2 行；相同 ID 得到相同初始向量 | `test_embedding` |
| 3 | RMSNorm：沿特征维调整数值尺度 | 每个 4 维向量单独归一化；整体形状不变 | `test_rmsnorm` |
| 4 | SiLU 与 SwiGLU：逐位置进行非线性特征变换 | 内部可由 4 维扩到 64 维再回到 4 维；这里仅作形状示例 | `test_silu_matches_pytorch`、`test_swiglu` |
| 5 | RoPE：通过旋转为 Q、K 引入位置信息 | 位置 0 的旋转角为 0；输入输出维度相同 | `test_rope` |
| 6 | Softmax：沿指定轴把分数转成概率 | `[0, 0]` → `[0.5, 0.5]` | `test_softmax_matches_pytorch` |
| 7 | 缩放点积注意力：按相关程度汇总 V | 一个 query 对三个 key 产生三个权重，再加权汇总 value | `test_scaled_dot_product_attention`、`test_4d_scaled_dot_product_attention` |
| 8 | 因果多头自注意力：并行使用多个头，限制未来信息 | 第 2 个位置只能看第 1、2 个位置；4 维可拆成 2 个 2 维头 | `test_multihead_self_attention`、`test_multihead_self_attention_with_rope` |
| 9 | TransformerBlock：组合归一化、注意力、FFN 和残差 | `(2, 3, 4)` → `(2, 3, 4)`，方便连续堆叠 | `test_transformer_block` |
| 10 | TransformerLM：连接 Embedding、多个块、最终 Norm 与输出层 | `(2, 3)` → `(2, 3, 8)` | `test_transformer_lm`、`test_transformer_lm_truncated_input` |
| 11 | 参数量与 FLOPs 分析 | 从实际矩阵形状数参数、数乘加 | 手册 `transformer_accounting`，书面题 |

Softmax 测试在 `test_nn_utils.py`，其余上表代码测试在 `test_model.py`。每完成一个模块再接它的 adapter；adapter 只负责加载给定权重并调用实现，不放算法主体。

## 开始 Linear 时要讲清楚的内容

第一步以手册第 17–18 页、`run_linear` 和 `test_linear` 为依据。先讲三件事：

1. **目的与形状。** 输入最后一维是 `in_features`，输出最后一维是 `out_features`，前面的批次和序列维保留。
2. **参数与普通变量的区别。** 权重是训练要学习的对象；结合官方文档解释 `nn.Module`、`nn.Parameter` 和 `forward`，不用一开始讲完整训练框架。
3. **矩阵方向。** 权重保存为 `(out_features, in_features)`；用一个 2 维输入、3 维输出的小例子，区分数学列向量写法与代码最后一维存特征的约定。

手册要求该层没有 bias，使用指定的截断正态初始化，并支持构造时传入 `device`、`dtype`。这些是当前接口要求。先看懂小矩阵，再写构造函数和前向计算。

Linear 的验收除了已给测试，还需核对参数形状、注册和初始化，因为官方前向测试会覆盖为给定权重，单靠它不能验证随机初始化。到这个模块时做对应检查即可。

## 后面容易混淆的约定

- `d_model` 是整个 token 向量宽度，`d_k = d_model / num_heads` 是一个注意力头的宽度。
- 作业的布尔注意力 mask 中，`True` 表示允许关注该位置。
- RoPE 应作用于每个头的 Q、K，V 不参与旋转；位置由传入的 `token_positions` 确定。
- RMSNorm 按手册在 float32 中做归一化，再恢复输入 dtype。
- `pre-norm` 的归一化在子层之前；最后一个 TransformerBlock 后还有一次 RMSNorm。
- `seq_len` 可以小于最大 `context_length`，不能把最大长度当成每次输入的实际长度。

这些约定留到对应模块时配例子解释，本次不要求预先掌握全部内容。
