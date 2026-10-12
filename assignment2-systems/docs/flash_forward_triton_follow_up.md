# FlashAttention-2 Triton 前向：实现 Follow-up

## 这次要写出什么

你已经写过 [PyTorch 分块版](../cs336_systems/flash_attention2.py)。现在把同一套计算放进 Triton kernel，让 GPU 执行。

实现写在已有的 [flash_attention2_triton.py](../cs336_systems/flash_attention2_triton.py) 中，分成两部分：

| 写什么 | 接收什么 | 具体做什么 |
| --- | --- | --- |
| `flash_fwd_kernel` | Q/K/V/O/L 的指针，以及尺寸、strides 等数值 | 读取 Q/K/V 块，执行在线 softmax，向 O/L 的对应位置写入结果 |
| `FlashAttention2_with_triton` | PyTorch 张量 Q/K/V、`is_causal` | 分配完整 O/L，指定启动多少个 program，调用 kernel，保存张量，返回 O |

**本次完成 (b) 的无掩码前向。** 包装类保留 `is_causal=False` 参数；处理 True 属于 (c)。当前 backward 可以只抛出 `NotImplementedError`。

下面的任务 1～5 写在 kernel 中，任务 6 写包装类。依据是 [作业 PDF](../cs336_assignment2_systems.pdf) 第 25～27 页的 Algorithm 1、[现有测试](../tests/test_attention.py)，以及各步骤链接的官方 Triton API。块指针接口按 [Triton 3.6.0 官方源码](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/core.py#L2063-L2087) 说明。

先固定 `Bq=16`、`Bk=16`。题目保证测试维度是至少 16 的二次幂，所以这两个块大小能整除测试的 query/key 数量；这份实现按该约定编写。

## 贯穿全篇的小例子

假设连续的 Q/K/V 都是 `(2,32,16)`：两个样本，每个样本 32 个 query/key，每个 token 有 16 个特征。

| 名字 | 本例取值 | 含义 |
| --- | --- | --- |
| `batch_size` | 2 | 样本数 |
| `n_queries` / `N_QUERIES` | 32 | 每个样本的 query 数 |
| `n_keys` / `N_KEYS` | 32 | 每个样本的 key/value 数 |
| `d` / `D` | 16 | 特征维度 |
| `Bq` / `Q_TILE_SIZE` | 16 | 一个 program 负责的 query 行数 |
| `Bk` / `K_TILE_SIZE` | 16 | 每轮读取的 key/value 行数 |
| `Tq` | 2 | 每个样本有几个 query 块 |
| `Tk` | 2 | 每个 program 要遍历几个 key/value 块 |

小写尺寸名用于包装类，接口中的大写名用于 kernel。它们是调用时传进去的同一个数，例如 `Q_TILE_SIZE=Bq`。

完整 O 的形状是 `(2,32,16)`，完整 L 是 `(2,32)`。**小写 l 是一个 program 内部的累计分母；大写 L 是最终保存的 logsumexp 张量。**

## 任务 1：保留题目接口，明确每组参数的用途

**写在哪里：`flash_attention2_triton.py` 中，类定义外面的 kernel 声明。**

文件导入 `torch`、`triton`、`triton.language as tl`。你已开始写 kernel，沿用题目接口即可：

```python
@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):
    # 任务 2～5 的代码写在这个函数体中。
    pass
```

| 参数 | 表示什么 | 后面用在哪里 |
| --- | --- | --- |
| 五个 `*_ptr` | 各张量第一个元素的地址 | 算出当前样本的起始地址，建立块指针 |
| `stride_qb` | Q 的样本索引增加 1，地址跨过多少个元素 | 选中当前样本 |
| `stride_qq` | Q 的 query 索引增加 1，地址跨过多少个元素 | 定位 query 行 |
| `stride_qd` | Q 的特征索引增加 1，地址跨过多少个元素 | 定位一行中的特征 |
| K/V/O/L 的 strides | 各张量对应维度的地址间隔 | 对它们完成相同的定位 |
| `N_QUERIES`、`N_KEYS` | 一个样本内的 query/key 总数 | 描述完整矩阵，确定 key 循环次数 |
| `scale` | `1/sqrt(d)` | 对 QKᵀ 分数做缩放 |
| 三个 `tl.constexpr` 参数 | 编译时确定的特征维度和块大小 | 决定 kernel 内各块的形状 |

**stride 的单位是元素，不是字节。** 本例连续 Q 的三个 stride 是 `(512,16,1)`：跨一个样本跳 512 个元素，跨一行跳 16 个元素，跨一个特征跳 1 个元素。

接下来根据 program 的编号，确定当前 program 处理哪一部分数据。

## 任务 2：选定样本和 query 块，建立五个块指针

**目标：让当前 program 找到自己的 Q、全部 K/V，以及自己的 O/L 写入位置。**

**写在哪里：kernel 函数体开头，key 循环之前。**

### 2.1 读取两个编号，算出 query 块的起始行

包装类会用 `grid=(Tq,batch_size)` 启动 kernel。grid 指定每个方向有多少个 program；每个 program 执行同一个函数体，但取得自己的编号。

在 kernel 开头写：

```python
query_tile_index = tl.program_id(0)
batch_index = tl.program_id(1)
q_start = query_tile_index * Q_TILE_SIZE
```

`tl.program_id(0)` 的 0 是 grid 的第一个轴，取得 query 块编号；`tl.program_id(1)` 的 1 是第二个轴，取得样本编号。[program_id 官方文档](https://triton-lang.org/main/python-api/generated/triton.language.program_id.html)

本例 grid 为 `(2,2)`，四个 program 的分工如下。编号顺序始终是 **`(query_tile_index,batch_index)`**：

| 两个编号 | 当前样本 | q_start | 负责的 Q | 要遍历的 K/V |
| --- | --- | --- | --- | --- |
| `(0,0)` | 0 | 0 | `Q[0,0:16,:]` | 样本 0 的全部 K/V |
| `(1,0)` | 0 | 16 | `Q[0,16:32,:]` | 样本 0 的全部 K/V |
| `(0,1)` | 1 | 0 | `Q[1,0:16,:]` | 样本 1 的全部 K/V |
| `(1,1)` | 1 | 16 | `Q[1,16:32,:]` | 样本 1 的全部 K/V |

因此 kernel 内只需要后面的 key 循环。样本和 query 块的分工已经由 grid 完成。

### 2.2 算出当前样本的起始地址，作为各块指针的 base

**现在要写的就是五个地址计算。** 为了容易跟踪，先存成变量，再填进下一步 `tl.make_block_ptr` 的 base 参数：

```python
q_base = Q_ptr + batch_index * stride_qb
k_base = K_ptr + batch_index * stride_kb
v_base = V_ptr + batch_index * stride_vb
o_base = O_ptr + batch_index * stride_ob
l_base = L_ptr + batch_index * stride_lb
```

以 Q 为例：`Q_ptr` 指向 `Q[0,0,0]`，跨一个样本需要跳过 stride_qb 个元素。因此 q_base 指向 `Q[batch_index,0,0]`。

本例 batch_index=1 时，`q_base=Q_ptr+512`，指向 `Q[1,0,0]`；`l_base=L_ptr+32`，指向 `L[1,0]`。L 每个样本只有 32 个元素，所以它使用自己的 stride。

这一步算的是地址，数据仍在原处。**它只选中了样本，还没有选 query 块，也还没有读取数据。** 下一步的 offsets 负责选块，之后的 tl.load 才读取数值。

### 2.3 建立 Q 块指针，指定当前样本中要读取的 query 行

在五个 base 变量之后，写 Q 的块指针：

```python
Q_block_ptr = tl.make_block_ptr(
    base=q_base,
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(q_start, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
)
```

每个参数回答一个具体问题。以 program `(1,1)` 为例：

| 参数 | 回答什么问题 | 本例的答案 |
| --- | --- | --- |
| base | 当前样本的矩阵从哪里开始？ | `Q[1,0,0]` 的地址 |
| shape | 从这个 base 开始的完整矩阵有多大？ | `(32,16)`，即整个 `Q[1,:,:]` |
| strides | 行或列索引增加 1，地址怎么变？ | `(16,1)` |
| offsets | 当前块从矩阵的哪一行、哪一列开始？ | `(16,0)` |
| block_shape | 本次要读多少行、多少列？ | `(16,16)` |
| order | 告诉编译器怎样的维度顺序符合布局 | 沿用题目的 `(1,0)`，最后一维连续 |

所以这个指针描述的是 `Q[1,16:32,:]` 的位置。order 不改变行列含义，所有形状仍按“query 行、特征列”理解。

块内第 `(r,c)` 个元素的地址可以展开成：

```text
q_base + (q_start + r) * stride_qq + c * stride_qd
```

本例第 `(0,0)` 个元素的地址为 `Q_ptr+512+16*16`，对应 `Q[1,16,0]`。这就是 base、offsets 和 strides 各自起作用的位置。[make_block_ptr 官方接口](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/core.py#L2063-L2073)

### 2.4 依照下面的参数，建立 K/V/O/L 四个块指针

继续写四次 tl.make_block_ptr，分别赋给下面四个变量。每次调用的参数由同一行给出：

| 变量 | base | shape | strides | offsets | block_shape |
| --- | --- | --- | --- | --- | --- |
| `K_block_ptr` | `k_base` | `(N_KEYS,D)` | `(stride_kk,stride_kd)` | `(0,0)` | `(K_TILE_SIZE,D)` |
| `V_block_ptr` | `v_base` | `(N_KEYS,D)` | `(stride_vk,stride_vd)` | `(0,0)` | `(K_TILE_SIZE,D)` |
| `O_block_ptr` | `o_base` | `(N_QUERIES,D)` | `(stride_oq,stride_od)` | `(q_start,0)` | `(Q_TILE_SIZE,D)` |
| `L_block_ptr` | `l_base` | `(N_QUERIES,)` | `(stride_lq,)` | `(q_start,)` | `(Q_TILE_SIZE,)` |

K/V/O 是二维块，`order=(1,0)`；L 是一维块，`order=(0,)`。L 的 shape、strides、offsets、block_shape 都是一元素元组，保留末尾逗号。

**为什么 offsets 不一样？** 当前 program 要让自己的 query 块与整个样本的 K/V 计算，所以 K/V 从第 0 行开始，之后逐块遍历。O/L 要写回当前 query 块的位置，因此与 Q 使用相同的 q_start。

例如 program `(1,1)` 建好五个块指针后：

| 指针 | 接下来读取或写入的位置 |
| --- | --- |
| Q_block_ptr | 读取 `Q[1,16:32,:]` |
| K_block_ptr | 首轮读取 `K[1,0:16,:]` |
| V_block_ptr | 首轮读取 `V[1,0:16,:]` |
| O_block_ptr | 最后写入 `O[1,16:32,:]` |
| L_block_ptr | 最后写入 `L[1,16:32]` |

O 的 offsets 也必须包含 q_start。如果所有 program 都用 `(0,0)`，同一样本的多个 program 会向相同的前 16 行写结果，后面的行得不到自己的结果。

**任务 2 完成时，你得到的是五个描述位置的块指针。** 下一步通过 Q 块指针读取实际数值。

## 任务 3：读取当前 Q 块，初始化三个累计量

**目标：得到当前 program 的 Q 数值，创建用于汇总所有 key 块的状态。**

**写在哪里：五个块指针之后、key 循环之前。**

### 3.1 用 Q 块指针读取数据

写：

```python
q_block = tl.load(Q_block_ptr)
```

Q_block_ptr 描述位置，q_block 才是读取后的数值，形状 `(Q_TILE_SIZE,D)`。本例 program `(1,1)` 读到 `Q[1,16:32,:]`，形状 `(16,16)`。[tl.load 官方文档](https://triton-lang.org/main/python-api/generated/triton.language.load.html)

这个 program 的所有 key 轮次都使用同一块 Q，因此把读取放在循环外。

### 3.2 创建 o_acc、l、m

在 q_block 后面创建三个累计量。初始化写法是：

```python
o_acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
m = tl.full((Q_TILE_SIZE,), float("-inf"), dtype=tl.float32)
```

| 变量 | 累计什么 | 本例形状 | 后面如何使用 |
| --- | --- | --- | --- |
| o_acc | 当前 query 块的未归一化加权输出 | `(16,16)` | 每轮加入当前 V 块的贡献，最后除以 l |
| l | 每个 query 的、减去当前最大值后的指数和 | `(16,)` | 每轮加入当前 key 块的分母贡献 |
| m | 每个 query 到目前为止见过的最大分数 | `(16,)` | 算新权重，修正旧累计量 |

这是 Algorithm 1 的初始化。题目要求这三个 kernel 内的累计量使用 FP32；它们和包装类分配的完整 O/L 是不同的对象。[tl.full 官方文档](https://triton-lang.org/main/python-api/generated/triton.language.full.html)

**任务 3 完成时，Q 数值和初始状态已经就绪。** 下一步每轮读取一块 K/V，给三个累计量添加贡献。

## 任务 4：遍历所有 key 块，逐轮更新累计量

**目标：计算当前 query 块对完整 K/V 的注意力结果。**

**写在哪里：任务 3 后面。下面 4.1～4.4 全部放在同一个 key 循环体内。**

循环次数为 `tl.cdiv(N_KEYS,K_TILE_SIZE)`，即向上取整的 key 块数。写一个以这个数为上界的 for 循环。本例为 `32/16=2` 轮。

循环编号只控制执行次数；每轮读取哪一块 K/V，由 K_block_ptr、V_block_ptr 当前的位置决定。

### 4.1 读取当前 K/V 块，算出分数 s_block

每轮先分别用 `tl.load(K_block_ptr)`、`tl.load(V_block_ptr)`，结果存入 k_block、v_block，形状都是 `(K_TILE_SIZE,D)`。

本例 program `(1,1)` 的两轮输入如下：

| 轮次 | 一直使用的 Q | 本轮 K | 本轮 V |
| --- | --- | --- | --- |
| 第一轮 | `Q[1,16:32,:]` | `K[1,0:16,:]` | `V[1,0:16,:]` |
| 第二轮 | `Q[1,16:32,:]` | `K[1,16:32,:]` | `V[1,16:32,:]` |

现在要得到 query/key 两两之间的分数：`s_block = q_block × k_blockᵀ × scale`。

用 `tl.trans(k_block)` 转置 K，通过 tl.dot 做矩阵乘法，最后乘传入的 scale：

```text
q_block         k_block 转置        s_block
(Bq,d)     ×       (d,Bk)      →     (Bq,Bk)
```

第一轮 `s_block[r,c]` 是全局 query `16+r` 与 key c 的分数；第二轮是 query `16+r` 与 key `16+c` 的分数。s 已经包含 `1/sqrt(d)` 缩放。[tl.trans](https://triton-lang.org/main/python-api/generated/triton.language.trans.html)、[tl.dot](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)

第一版对齐 FP32 参考时，两个 tl.dot 可指定 `input_precision="ieee"`。这是本份的精度建议，题目没有强制这个参数；NVIDIA 上 FP32 的 dot 默认可采用 TF32。性能比较时再单独调整它。

### 4.2 算新最大值、旧值修正系数和当前块权重

先保留旧 m，按下面的依赖顺序得到三个新量：

| 要赋值的变量 | 怎么算 | 形状 | 用来干什么 |
| --- | --- | --- | --- |
| m_new | `tl.maximum(m,tl.max(s_block,axis=1))` | `(Bq,)` | 合并旧最大值和当前块最大值 |
| alpha | `tl.exp(m-m_new)` | `(Bq,)` | 将旧 l、旧 o_acc 修正到新的最大值基准 |
| p_block | `tl.exp(s_block-m_new[:,None])` | `(Bq,Bk)` | 当前 key 块的未归一化指数权重 |

axis=1 沿 key 列求最大值，每个 query 留一个数。`m_new[:,None]` 把 `(Bq,)` 变成 `(Bq,1)`，让同一行的所有分数减去该 query 的最大值。

本例广播是 `(16,16)-(16,1) → (16,16)`，展开为 `p_block[r,c]=exp(s_block[r,c]-m_new[r])`。

**现在先不要把 m 改成 m_new。** alpha 需要旧 m；到 4.3 完成后再更新 m。

### 4.3 更新 l、o_acc，最后更新 m

先用 `tl.sum(p_block,axis=1)` 算当前块的分母贡献，赋给 p_block_sum，形状 `(Bq,)`。随后将 l 赋为：

```text
新的 l = alpha * 旧 l + p_block_sum
```

这是逐 query 更新，每个 query 留一个累计分母。[tl.sum 官方文档](https://triton-lang.org/main/python-api/generated/triton.language.sum.html)

接下来要计算的仍然是你 PyTorch 版的原公式：

```text
新的 o_acc = alpha[:,None] * 旧 o_acc + p_block × v_block
```

**这一份实现采用题目给出的精度安排：o_acc/l/m 用 FP32，做 PV 前将 P 转为 V 的类型，并通过 acc 累计输出。** 它们不改变上述更新公式。先用原来的 FP32 p_block 完成上面的分母求和，再写：

```python
o_acc = o_acc * alpha[:, None]
p_for_dot = p_block.to(v_block.dtype)
o_acc = tl.dot(p_for_dot, v_block, acc=o_acc)
```

这三行分别负责：

| 行 | 在原公式中完成什么 | 类型安排 |
| --- | --- | --- |
| 第一行 | 修正旧的累计输出 | o_acc 继续为 FP32 |
| 第二行 | 为本轮 PV 准备权重 | p_for_dot 与 V 同类型；p_block 保留原来的 FP32 |
| 第三行 | 本轮 PV 加上修正后的旧输出 | 用 FP32 的 o_acc 作为累计量 |

**acc 对应原公式中的加法。** `tl.dot(a,b,acc=c)` 返回 `a×b+c`。所以先把旧输出乘 alpha，再将修正后的输出作为 acc 传入，把 dot 返回值赋回 o_acc。acc 不会自动替你乘 alpha。[tl.dot 的 acc 参数](https://triton-lang.org/main/python-api/generated/triton.language.dot.html)

**乘法输入的类型与累计量的类型可以不同。** 类型转换在不同输入下的效果如下：

| 当前输入类型 | p_block 的类型 | v_block 的类型 | `.to(v_block.dtype)` 做什么 |
| --- | --- | --- | --- |
| FP32 | FP32 | FP32 | 类型和值不变，保留同一写法 |
| FP16/BF16 | FP32 | FP16/BF16 | 把用于 PV 的 P 转成 V 的类型，o_acc 仍以 FP32 累计 |

p_for_dot 只是转换结果的变量名，形状仍为 `(Bq,Bk)`。例如 V 是 BF16 时，做的是“BF16 的 P × BF16 的 V，加到 FP32 的 o_acc”；分母 l 则仍使用原来的 FP32 P 计算。它没有添加异常处理或额外分支。

形状分别是：

```text
alpha * l                 (Bq,) × (Bq,)    → (Bq,)
alpha[:,None] * o_acc      (Bq,1) × (Bq,D)  → (Bq,D)
p_for_dot × v_block        (Bq,Bk) × (Bk,D) → (Bq,D)
```

l、o_acc 都更新后，执行 `m=m_new`，为下一轮保留新的最大值。

### 4.4 让 K/V 块指针指向下一组 token 行

在循环体末尾写：

```python
K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
```

两个数表示“行索引增加 Bk、列索引不变”。本例起始行由 0 变为 16，下一次 load 就读取 16～31 行。

这是坐标增量；跨一行在内存中要跳多少元素，已经由建指针时的 stride 指定。本例行 stride 为 16，跨 16 行对应地址增加 `16×16=256` 个元素。

**把返回的新指针赋回原变量。** advance 返回描述新位置的指针，下一轮的 load 才会使用新位置。[advance 官方接口](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/core.py#L2075-L2087)

Q 始终是本 program 的 query 块，O/L 始终是该 query 块的结果位置。循环中移动 K/V 就够了。

**退出循环时：**o_acc、l、m 都包含完整上下文的累计结果。最终归一化写在循环外。

## 任务 5：计算最终 O/L，写回对应 query 行

**目标：把当前 program 的累计状态变成最终结果，填入完整 O/L。**

**写在哪里：key 循环之后，kernel 函数体的最后。**

### 5.1 得到当前块的最终输出和 logsumexp

分别得到下面两个量：

| 变量 | 计算表达式 | 形状 | 含义 |
| --- | --- | --- | --- |
| o_block | `o_acc / l[:,None]` | `(Bq,D)` | 归一化后的注意力输出 |
| l_block | `m + tl.log(l)` | `(Bq,)` | 每个 query 的最终 logsumexp |

l 此时等于 `sum(exp(S-m))`，S 是这个 query 对全部 key 的缩放后分数。完整分母为 `exp(m)*l`，取 log 才得到 `m+log(l)`。

l_block 保存了 m 和 l 合起来的归一化信息；之后重算概率可直接使用 `exp(S-L)`。这里恢复的是减去 m 之前的分数基准，S 仍然带有 `1/sqrt(d)` 缩放。

### 5.2 转换输出类型，通过 O/L 块指针写回

写 O 前，把 o_block 转为完整 O 张量的元素类型。使用原始输出指针取得标量类型：

```python
o_block = o_block.to(O_ptr.type.element_ty)
```

如果包装类的 O 是 BF16，这行就将 FP32 累计后的结果转为 BF16。包装类的 L 在本份中分配为 FP32，l_block 保持 FP32。

随后分别调用：

```python
tl.store(O_block_ptr, o_block)
tl.store(L_block_ptr, l_block)
```

store 的第一个参数给出写入位置，第二个参数给出要写的数值。[tl.store 官方接口](https://github.com/triton-lang/triton/blob/v3.6.0/python/triton/language/core.py#L2017-L2060)

本例 program `(1,1)` 写入 `O[1,16:32,:]` 和 `L[1,16:32]`。另外三个 program 写各自区域，四个 program 合起来填完完整 O/L。

**任务 5 完成时，kernel 主体结束。** 它通过 store 写入传入的 O/L；接下来写负责分配和传入这些张量的包装类。

## 任务 6：写 PyTorch 包装类，把张量接到 kernel 上

**目标：调用 `FlashAttention2_with_triton.apply(Q,K,V,False)` 时，启动 kernel，返回填好的 O。**

**写在哪里：同一个文件中，kernel 定义之后的类定义。**

类名用 `FlashAttention2_with_triton`，继承 `torch.autograd.Function`。forward 使用 `@staticmethod`，接口为 `forward(ctx,Q,K,V,is_causal=False)`。

### 6.1 读取输入尺寸，分配完整输出

在 forward 开头，从 Q.shape 取出 batch_size、n_queries、d，从 K.shape[1] 取出 n_keys。

然后创建两个实际的 PyTorch 显存张量：

| 变量 | 创建方式 | 形状 | 用来干什么 |
| --- | --- | --- | --- |
| O | `torch.empty_like(Q)` | `(batch_size,n_queries,d)` | 提供 kernel 要填入的完整输出空间 |
| L | `torch.empty((batch_size,n_queries),device=Q.device,dtype=torch.float32)` | `(batch_size,n_queries)` | 提供 kernel 要填入的完整 logsumexp 空间 |

empty 分配的元素尚未初始化，后面的 program 按各自的 query 范围将它们全部填入。

### 6.2 计算块大小、scale 和 grid

设置 Bq/Bk 为 16，计算 `scale=1/(d**0.5)`。然后写：

```python
grid = (triton.cdiv(n_queries, Bq), batch_size)
```

它决定启动 `Tq×batch_size` 个 program。本例 `(2,2)` 启动四个，与任务 2 的两个 program_id 轴一一对应。

### 6.3 传入张量、strides 和尺寸，启动 kernel

O/L、scale、grid 都创建之后，使用下面的调用形式：

```python
flash_fwd_kernel[grid](
    Q, K, V, O, L,
    Q.stride(0), Q.stride(1), Q.stride(2),
    K.stride(0), K.stride(1), K.stride(2),
    V.stride(0), V.stride(1), V.stride(2),
    O.stride(0), O.stride(1), O.stride(2),
    L.stride(0), L.stride(1),
    n_queries, n_keys,
    scale,
    D=d,
    Q_TILE_SIZE=Bq,
    K_TILE_SIZE=Bk,
)
```

这里 `[grid]` 指定启动规模，后面的圆括号传 kernel 参数。参数顺序与任务 1 的声明一致。

调用时传入 Q 张量，kernel 中的 Q_ptr 就收到它的数据指针。O/L 也是同样的传法，只是它们提供的是要写入的空间。

**结果留在原来的 O/L 张量里。** kernel 内的 tl.store 已负责写入，包装类通过 O/L 获取结果。

### 6.4 保存反向需要的张量，返回 O

kernel 调用之后，用 `ctx.save_for_backward(Q,K,V,O,L)` 保存这五个张量，随后 `return O`。之后写 backward 时，按相同顺序解包。

这里返回完整 O。L 放在 ctx 中供反向和测试使用，调用者不会收到 `(O,L)` 元组。

类中再声明带 `@staticmethod` 的 `backward(ctx,grad_output)`，当前只抛出 `NotImplementedError`。

### 6.5 让官方测试拿到这个类

找到 [tests/adapters.py](../tests/adapters.py) 中的 `get_flashattention_autograd_function_triton()`。

在这个函数内部，从 `cs336_systems.flash_attention2_triton` 导入新类，返回 `FlashAttention2_with_triton` **类本身**。测试拿到类后会调用它的 `.apply`。

Triton 相关导入和 kernel 留在单独的 Triton 文件中，adapter 在这个入口内导入该类。Mac 上调用 PyTorch 测试入口时只需加载 PyTorch 版本；CUDA 测试调用 Triton 入口时再加载 Triton 文件。

## 写完后运行这一项测试

在 AutoDL 的 CUDA 环境、`assignment2-systems/` 目录中运行：

```sh
uv run pytest 'tests/test_attention.py::test_flash_forward_pass_triton[False]'
```

它使用 `(batch_size,n_queries,d)=(4,128,64)` 的输入，对照普通注意力比较 O 和保存的 L；它不调用 backward。

整个 test_flash_forward_pass_triton 包含 False、True 两个用例。当前无掩码版本先跑 False，完成 (c) 后再跑 True。Mac 缺少 CUDA 时，该 CUDA 用例会被跳过，跳过不表示 kernel 已通过。

## 最后沿着一个 program 复习数据流

仍取 program `(1,1)`：

| 位置 | 当前具体完成什么 |
| --- | --- |
| 包装类 forward | 分配完整 O/L，用 grid 启动四个 program |
| kernel 开头 | 取得样本 1、query 块 1，算出 q_start=16 |
| base 计算 | 将五个 base 定位到样本 1 的起始位置 |
| 建块指针 | Q/O/L 选 16～31 行，K/V 先选 0～15 行 |
| 循环前 | load Q，初始化 o_acc/l/m |
| 第一轮 | load K/V 的 0～15 行，更新累计量，advance K/V |
| 第二轮 | load K/V 的 16～31 行，再次更新累计量 |
| 循环后 | 求最终 o_block/l_block，store 到样本 1 的 16～31 行 |
| 包装类 forward 最后 | 保存完整 Q/K/V/O/L，返回 O |

这份 follow-up 是逐段实现的指导。实际 kernel 按任务填写后，用上面的 CUDA 测试验证。
