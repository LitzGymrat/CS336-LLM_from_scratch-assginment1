# BPE tokenizer 参考版

实现文件：[`reference_tokenizer.py`](../reference_tokenizer.py)。你正在写的 `BPE_tokenizer.py` 保持原样。
对应[作业说明](../../../cs336_assignment1_basics.pdf)的 2.4～2.6 节。

这是功能完整的 **naive 学习版**：训练时每一轮重新统计相邻对，并遍历所有不同的 pre-token 应用合并。
尚未实现多进程预分词或增量频次更新。下面的默认演示只使用仓库约 4 KB 的小样本，不会读取 2.23 GB 的训练集。

参考代码按合法输入编写，不额外做特殊 token 去重、异常参数校验或非连续词表编号兼容。

## 先读哪些函数

按下面的顺序读，可以先跳过 `Tokenizer` 类及文件最末尾的演示。

| 函数 | 输入 → 输出 | 对应你刚学的内容 |
| --- | --- | --- |
| `initialize_vocab` | 特殊 token 列表 → 编号到字节串的字典 | 先放 256 种单字节，再添加特殊 token |
| `pre_tokenize_and_count` | 文本、特殊 token → 片段频次字典 | 正则切分、UTF-8 编码、汇总重复片段 |
| `count_pairs` | 片段频次字典 → 相邻对频次字典 | 每个相邻位置都计数，并乘片段频次 |
| `merge_pair` | 一个片段、选中的相邻对 → 合并后的片段 | 从左往右合并，已使用的位置不重复使用 |
| `train_from_counts` | 片段频次、目标大小、特殊 token → 词表与合并规则 | 用循环把上面的步骤连接起来 |
| `train_bpe` | 文件路径、目标大小、特殊 token → 词表与合并规则 | 从文件读取文本的训练入口 |

训练时最重要的三个数据结构：

```python
# 编号 → token 的字节串；一个 token 可以包含多个字节。
vocab = {108: b"l", 111: b"o", 119: b"w"}  # 这里只展示其中三项。

# 当前切分状态 → 这个完整片段出现的次数。
words = {(b"l", b"o", b"w"): 5}

# 按学习顺序记录每一轮合并；编码时需要这个顺序。
merges = [(b"o", b"w"), (b"l", b"ow")]
```

训练会改变 `words` 的切分状态并扩充词表。编码时只应用已经学好的规则，不重新训练。

## 运行演示

在 `assignment1-basics/` 目录的终端中运行：

```sh
uv run --frozen python cs336_basics/BPE_tokenizer/reference_tokenizer.py
```

也可以在 VS Code 中打开该文件，选择项目 `.venv` 解释器后点击运行。
演示会输出每轮合并及频次、词表大小、token IDs，以及解码后的文字。
需要更换输入文件、词表大小、特殊 token 或演示文字时，直接修改 `main()` 中对应的值。

## 使用已经训练好的 tokenizer

```python
from cs336_basics.BPE_tokenizer.reference_tokenizer import Tokenizer, train_bpe

specials = ["<|endoftext|>"]
vocab, merges = train_bpe("tests/fixtures/tinystories_sample.txt", 300, specials)
tokenizer = Tokenizer(vocab, merges, specials)

ids = tokenizer.encode("你好，黎子姗！<|endoftext|>")
text = tokenizer.decode(ids)
print(ids)
print(text)
```

`save(vocab_filepath, merges_filepath)` 保存两个 JSON；字节使用十六进制表示，避免丢失无效 UTF-8 字节。
`Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)` 读取相同格式。
重新加载时传入相同的特殊 token 列表。这个文件格式是本参考版的格式，不是 GPT-2 的 remapped-Unicode 文件格式。

`encode_iterable` 接受逐行文件或其他字符串迭代器，并按原始顺序拼接理解输入。
它保留尚未确定的片段尾部，支持单词、缩写和特殊 token 跨输入块，避免简单逐行编码造成不同结果。
缓冲空间取决于输入块和最长未完成片段；极长的无边界单词或空白串仍可能需要较大的缓冲。

## 作业要求中的细节

- 相邻对同频时，取字典序更大的 **pair 元组**，不是比较拼接后的字符串。
- 特殊 token 优先匹配更长的标记；训练时排除它们，并保留两侧的硬边界。
- 特殊 token 计入目标词表大小，但不计入合并次数。
- GPT-2 预分词保留空格，因此 `low` 和 ` low` 是不同的片段。
- 手册 `bpe_example` 特意使用空白切分，不能拿正则切分的结果与它直接比较。
- 解码时先拼接所有字节，再按 UTF-8 解码；无效序列按要求替换为 `U+FFFD`。
- 文件读取保留原文 CRLF，不把它自动转换成 LF。
- 没有相邻对可合并时提前停止，词表大小是上限，不一定恰好达到。

## 验证

2026-09-16 起，`tests/adapters.py` 中的 `run_train_bpe` 和 `get_tokenizer` 均已连接到用户的 `BPE_tokenizer.py`。
官方训练和 tokenizer 测试验证用户实现；`tests/test_reference_tokenizer.py` 单独验证本参考文件。

```sh
uv run --frozen pytest tests/test_train_bpe.py tests/test_tokenizer.py tests/test_reference_tokenizer.py -q
```

官方测试覆盖训练结果、特殊 token、速度、与 GPT-2/tiktoken 的编码对照及文件迭代编码。
补充测试覆盖手册六轮合并、重叠相邻对、任意分块一致性、序列化往返和无效字节解码。
Linux 专用内存测试在 macOS 上按官方设置跳过。
