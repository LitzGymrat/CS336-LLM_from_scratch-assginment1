# FlashAttention-2 PyTorch 前向：实现 Follow-up

## 这次要完成什么

在 [flash_attention2.py](../cs336_systems/flash_attention2.py) 中，补完你已经创建的 `FlashAttention2_with_tiling` 类。它接收 Q、K、V，按块计算注意力，返回完整输出 O，并保存反向需要的 L、Q、K、V、O。

我们把这项工作拆成五个实现任务。每一项都说明：这段代码要得到什么、写在哪里、具体处理哪些数据，以及它如何交给下一项。各项连续写进同一个 `forward()`，不用另外建立辅助函数。

依据是作业 PDF 第 25 页 Algorithm 1、第 26 页 `flash_forward (a)`，以及 [官方前向测试](../tests/test_attention.py)。这份文件是学习用的拆解。

本次选择 query 块大小 `Bq = 16`、key 块大小 `Bk = 16`。题目保证测试维度是至少 16 的二次幂，因此这组块大小能整除测试中的序列长度。按这个约定实现即可。

贯穿下面的小例子：Q、K、V 的形状都是 `(2, 32, 16)`。有两个样本，每个样本 32 个 token，每个 token 16 个特征。

## 任务 1：给最终结果分配存放位置

**目标：创建完整的 O、L 张量，让后面每一块算完后有地方写入结果。**

**写在哪里：`forward()` 开头，所有循环之前。**

先把接口中的 `is_casual` 改成题目使用的 `is_causal`。保留默认值 `False`；这一问允许暂时忽略这个参数。

接下来做两件事：读取输入尺寸，创建结果张量。

从 `Q.shape` 读取 `batch_size`、`n_queries`、`d`；从 `K.shape[1]` 读取 `n_keys`。它们都是整数，用来决定张量大小和后面的循环范围。例子中分别是 `2、32、16、32`。

| 要创建什么 | 形状 | 为什么是这个形状 |
| --- | --- | --- |
| O | `(batch_size, n_queries, d)` | 每个 query 对应一个 d 维输出 |
| L | `(batch_size, n_queries)` | 每个 query 对应一个 logsumexp 标量 |

创建张量使用 `torch.empty`，形状按照上表填写，`device`、`dtype` 使用 Q 的对应属性。比如，创建一个同设备、同类型的两行三列张量，API 写法是：

```python
result = torch.empty((2, 3), device=Q.device, dtype=Q.dtype)
```

将这个 API 中的形状换成上表的形状，分别用于 O、L。`empty` 分配的是尚未初始化数值的张量；后面的计算会填入所有位置。[torch.empty 文档](https://docs.pytorch.org/docs/2.11/generated/torch.empty.html)

**做完这一项时：**小例子中已经有 `(2,32,16)` 的 O 和 `(2,32)` 的 L，后面要往里面填结果。同时记下 `Bq = 16`、`Bk = 16`，后面的切片和循环会用到它们。

## 任务 2：取出一块 query，为它开始一次独立计算

**目标：选出当前要计算输出的 16 个 query，并建立这 16 个 query 的累计状态。**

**写在哪里：任务 1 后面，建立“样本循环 → query 块循环”。**

先遍历样本编号 `b`：从 0 到 `batch_size - 1`。在每个样本内部，再遍历 query 块的起始位置 `q_start`：从 0 开始，每次增加 Bq，直到覆盖全部 query。

小例子里，b 依次是 `0、1`；每个 b 内，q_start 依次是 `0、16`。

在 query 块循环内，用下面这类切片取出当前块：

```python
q_block = Q[b, q_start:q_start + Bq, :]
```

例如 `b=0、q_start=16`，取的是第一个样本的第 16～31 行，形状是 `(16,16)`。整数索引 b 已经选定一个样本，所以 q_block 是二维张量。

接着为当前 q_block 创建三个张量，都使用 Q 的设备和类型：

| 变量名 | 形状 | 初始值 | 后面拿它做什么 |
| --- | --- | --- | --- |
| `o_acc` | `(Bq,d)` | 全 0 | 累计各 key 块的加权输出，最后再归一化 |
| `l` | `(Bq,)` | 全 0 | 累计每一行的 softmax 分母 |
| `m` | `(Bq,)` | 全负无穷 | 记录每一行已经见过的最大分数 |

全 0 使用 `torch.zeros`；全负无穷使用 `torch.full`，填充值为 `float("-inf")`。形状按照上表填写。

这三个张量的每一行都对应 q_block 的一个 query。它们在处理后面的所有 key 块时持续更新；开始计算下一块 query 时，重新创建它们。

**做完这一项时：**已经取出一个 q_block，并且有它自己的 o_acc、l、m。下一项要让这块 query 依次与所有 key 块计算。

## 任务 3：遍历所有 key 块，累计当前 query 块的结果

**目标：让当前 16 个 query 看完这个样本中的全部 key/value，得到覆盖完整上下文的累计结果。**

**写在哪里：query 块循环内部，在创建 o_acc、l、m 之后，再建立 key 块循环。**

key 块的起始位置叫 `k_start`，从 0 开始，每次增加 Bk，直到覆盖全部 key。小例子里，它依次是 `0、16`。

每次取出同一个样本 b 中的 K 块、V 块：行范围都是 `k_start:k_start + Bk`，特征维保留全部。得到 `k_block`、`v_block`，形状都是 `(Bk,d)`。

于是，第一个样本的计算顺序是：

| 当前 query 行 | 本次 key/value 行 | 本次工作 |
| --- | --- | --- |
| 0～15 | 0～15 | 开始累计前 16 个 query 的结果 |
| 0～15 | 16～31 | 接着累计，前 16 个 query 已经看完全部 key |
| 16～31 | 0～15 | 开始累计后 16 个 query 的结果 |
| 16～31 | 16～31 | 接着累计，后 16 个 query 已经看完全部 key |

**下面四件事都写在 key 块循环里面，每取出一对 K/V 块就执行一次。**

### 3.1 算这两个块之间的注意力分数

计算 `S = q_block × k_block转置 / sqrt(d)`。

这里是二维矩阵乘法：`(Bq,d) × (d,Bk) → (Bq,Bk)`。可以使用你已经熟悉的 `@` 或 `torch.matmul`。S 的每一行对应一个 query，每一列对应当前块中的一个 key。

### 3.2 更新每个 query 的最大值，算本块的指数权重

先求 S 每一行的最大值，得到 `(Bq,)`；再把它与旧 m 逐元素比较，得到 `m_new`。行最大值使用 `torch.amax(S, dim=-1)`，逐元素比较使用 `torch.maximum`。[torch.amax 文档](https://docs.pytorch.org/docs/2.11/generated/torch.amax.html)

接下来计算两种指数值：

- **旧累计量的修正系数 alpha：**`exp(m - m_new)`，形状 `(Bq,)`。旧结果原先使用旧 m，现在要把它换到 m_new 的标准下。
- **本块的未归一化权重 p_tilde：**`exp(S - m_new的列形式)`，形状 `(Bq,Bk)`。每一行减去这一行的 m_new。

“列形式”就是你前面用过的 `m_new[:, None]`：把 `(Bq,)` 变成 `(Bq,1)`，让每一行的一个值广播到该行的所有列。

此时旧 m 还要用于计算 alpha。等当前块更新结束，再用 m_new 替换 m。

### 3.3 更新分母累计值 l

现在 l 已经包含前面各 key 块的贡献。本次更新需要：

1. 旧 l 乘 alpha，修正到新的最大值标准。
2. p_tilde 沿每一行求和，得到当前块新增的分母贡献。
3. 把这两项相加，作为新的 l。

对应你已经理解的公式：`l_new = alpha * l + rowsum(p_tilde)`。行求和使用 `sum(dim=-1)`，这里每一项的形状都是 `(Bq,)`。

### 3.4 更新输出累计值 o_acc

这次同样有“修正旧贡献 + 添加新贡献”两部分：

1. 旧 o_acc 的每一行乘该行的 alpha。
2. 矩阵乘法 `p_tilde × v_block`，得到本块新增的加权输出。
3. 两项相加，作为新的 o_acc。

第一项需要把 alpha 变成 `(Bq,1)` 再广播；第二项是 `(Bq,Bk) × (Bk,d) → (Bq,d)`。这就是伪代码里对角矩阵缩放各行的操作，直接用逐元素乘法表达即可。

用本次的新值更新 o_acc、l、m，随后进入下一个 key 块。可以给新结果分别起名 `o_acc_new`、`l_new`、`m_new`，最后再替换旧变量，方便跟踪新旧值。

**做完全部 key 块时：**o_acc、l、m 已经包含当前 query 块对全部 key 的计算结果。接下来就能得到这块 query 的最终输出。

## 任务 4：完成当前 query 块，把结果填回 O、L

**目标：把累计值转换为最终结果，填入任务 1 创建的对应位置。**

**写在哪里：退出 key 块循环之后，仍然位于 query 块循环内部。**

做两项计算：

| 要得到什么 | 怎么算 | 形状 |
| --- | --- | --- |
| `o_block`：这块 query 的最终输出 | o_acc 的每一行除以该行的 l，分母用 `l[:, None]` 广播 | `(Bq,d)` |
| `l_block`：这块 query 的 logsumexp | `m + log(l)` | `(Bq,)` |

这里小写 l 是累计分母；完整大写 L 存的是 logsumexp，两者内容不同。

现在填回结果。o_block 写入 O 的 `b, q_start:q_start + Bq, :` 位置；l_block 写入 L 的 `b, q_start:q_start + Bq` 位置。

小例子：算完 `b=0、q_start=16` 后，o_block 填入 `O[0,16:32,:]`，l_block 填入 `L[0,16:32]`。这就是前面创建完整 O、L 的用途。

随后 query 块循环继续处理下一块；当前样本的 query 全部处理完后，样本循环继续处理下一个样本。

**做完全部样本时：**O、L 的所有位置都已经填好。

## 任务 5：保存反向所需张量，完成函数出口

**目标：让调用者拿到 O，同时让 PyTorch 能在后面的 backward 中取回所需数据。**

**写在哪里：退出全部循环后，放在 `forward()` 最后。**

按题目要求，用 `ctx.save_for_backward` 一次保存 L、Q、K、V、O，然后返回 O。L 保存在 ctx 中，调用者收到的是 O。[save_for_backward 文档](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.function.FunctionCtx.save_for_backward.html)

在同一个类中添加静态方法 `backward(ctx, grad_output)`，当前函数体只抛出 `NotImplementedError`。这是题目允许的阶段性接口，后面的反向任务再填写计算内容。

最后在 [tests/adapters.py](../tests/adapters.py) 的 `get_flashattention_autograd_function_pytorch()` 中，导入并返回你写的类 `FlashAttention2_with_tiling`。返回类本身；测试会调用它的 `.apply`。

整段 forward 的位置关系是：

| 位置 | 执行的任务 | 执行次数 |
| --- | --- | --- |
| 所有循环之前 | 任务 1：读取尺寸、创建完整 O/L | 每次 forward 一次 |
| 每个样本的每个 query 块开头 | 任务 2：取 Q 块，创建累计状态 | 每个 query 块一次 |
| 当前 query 块的每个 key 块内部 | 任务 3：计算分数，更新累计状态 | 每对 Q/K 块一次 |
| 当前 query 块的全部 key 处理完后 | 任务 4：归一化，填回 O/L | 每个 query 块一次 |
| 全部样本处理完后 | 任务 5：保存、返回 | 每次 forward 一次 |

写完后，在 `assignment2-systems/` 目录运行作业自带的这一项测试：

```sh
uv run pytest tests/test_attention.py -k test_flash_forward_pass_pytorch
```

它会比较 O 和保存的 L。这个测试在 CPU 上运行，当前 PyTorch 实现可以在 Mac 上完成。Triton 前向和实际反向计算属于接下来的任务。

## 参考位置

- [作业 PDF](../cs336_assignment2_systems.pdf)：第 19～20 页 WeightedSum 的 autograd 包装、第 25 页分块算法、第 26 页本问接口。
- WeightedSum 中“接收张量 → 创建输出 → 保存反向数据 → 返回输出”的结构在这里继续使用；本次 forward 的计算主体是任务 2～4 的 PyTorch 分块计算。
- [你的 CS106B 逐讲练习清单](../../../DSA/cs106B/逐讲练习清单.md)：本文沿用“指定动手位置、分配明确任务、在对应任务中完成所需操作”的组织方式。
