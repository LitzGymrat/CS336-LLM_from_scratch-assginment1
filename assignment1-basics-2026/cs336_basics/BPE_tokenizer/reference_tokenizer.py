"""CS336 字节级 BPE：先读训练函数，再读 Tokenizer。

按模块阅读：构建词表 → 预切分计数 → 统计相邻对 → 合并 → 训练循环。
假定调用者传入合法参数；不做额外的输入修复、去重或兼容处理。
直接运行本文件，会在仓库的小样本上演示训练和编码、解码。
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PRETOKEN_PATTERN = regex.compile(PAT)


def initialize_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    """模块 1：0～255 是单字节，特殊 token 从 256 开始依次添加。"""
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf-8")
    return vocab


def pre_tokenize_and_count(text: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """模块 2：按特殊 token 隔开文本，再正则切分、转换为字节元组、计数。"""
    if special_tokens:
        ordered = sorted(special_tokens, key=len, reverse=True)
        delimiter = "|".join(regex.escape(token) for token in ordered)
        sections = regex.split(delimiter, text)
    else:
        sections = [text]

    counts = {}
    for section in sections:
        for match in PRETOKEN_PATTERN.finditer(section):
            piece = match.group()
            word = tuple(bytes([value]) for value in piece.encode("utf-8"))
            counts[word] = counts.get(word, 0) + 1
    return counts


def count_pairs(words: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """模块 3：汇总所有片段内部的相邻对，乘上片段的出现次数。"""
    counts = {}
    for word, frequency in words.items():
        for pair in zip(word, word[1:]):
            # 如 aaaa 内的 aa 有 3 个相邻位置，不能先对 pair 去重。
            counts[pair] = counts.get(pair, 0) + frequency
    return counts


def merge_pair(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """模块 4：从左到右合并。aaa 合并 aa 得到 aa,a，同一位置不重复使用。"""
    result = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            result.append(pair[0] + pair[1])
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)


def train_from_counts(
    words: dict[tuple[bytes, ...], int],
    vocab_size: int,
    special_tokens: list[str],
    *,
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """模块 5：每轮重新统计所有相邻对，再遍历全部片段应用合并。"""
    vocab = initialize_vocab(special_tokens)
    merges = []

    while len(vocab) < vocab_size:
        pair_counts = count_pairs(words)
        if not pair_counts:
            break
        # 先比频次；同频时比 pair 本身，取字典序更大的那个。
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        new_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = new_token
        if verbose:
            print(f"第 {len(merges):3d} 轮：{best_pair!r} → {new_token!r}，频次 {pair_counts[best_pair]}")

        updated = {}
        for word, frequency in words.items():
            new_word = merge_pair(word, best_pair)
            updated[new_word] = updated.get(new_word, 0) + frequency
        words = updated
    return vocab, merges


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    *,
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练入口。Naive 版一次读取全文，先用小样本理解、验证流程。"""
    # newline="" 防止 Python 把原文的 CRLF 自动替换成 LF。
    with open(input_path, encoding="utf-8", newline="") as stream:
        text = stream.read()
    counts = pre_tokenize_and_count(text, special_tokens)
    return train_from_counts(counts, vocab_size, special_tokens, verbose=verbose)


def iter_units(text: str, specials) -> Iterator[tuple[int, int, bool]]:
    """编码阶段使用：给出每个片段的起止位置及其是否为特殊 token。"""
    start = 0
    if specials is not None:
        for match in specials.finditer(text):
            for piece in PRETOKEN_PATTERN.finditer(text[start : match.start()]):
                yield start + piece.start(), start + piece.end(), False
            yield match.start(), match.end(), True
            start = match.end()
    for piece in PRETOKEN_PATTERN.finditer(text[start:]):
        yield start + piece.start(), start + piece.end(), False


class Tokenizer:
    """使用已经学好的 vocab 和 merges 编码、解码；这里不再统计语料频次。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [] if special_tokens is None else special_tokens
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        # 作业明确要求：构造 tokenizer 时，补入词表中尚不存在的特殊 token。
        for token in self.special_tokens:
            value = token.encode("utf-8")
            if value not in self.token_to_id:
                next_id = len(self.vocab)
                self.vocab[next_id] = value
                self.token_to_id[value] = next_id
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.specials = None
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self.specials = regex.compile("|".join(regex.escape(token) for token in ordered))

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        word = tuple(bytes([value]) for value in piece.encode("utf-8"))
        while len(word) > 1:
            pairs = [pair for pair in zip(word, word[1:]) if pair in self.merge_ranks]
            if not pairs:
                break
            best_pair = min(pairs, key=lambda pair: self.merge_ranks[pair])
            word = merge_pair(word, best_pair)
        return tuple(self.token_to_id[token] for token in word)

    def encode(self, text: str) -> list[int]:
        ids = []
        for start, end, is_special in iter_units(text, self.specials):
            piece = text[start:end]
            if is_special:
                ids.append(self.token_to_id[piece.encode("utf-8")])
            else:
                ids.extend(self._encode_piece(piece))
        return ids

    def decode(self, ids: list[int]) -> str:
        # 必须先拼接全部字节再解码，不能逐个 token 单独解码后拼字符串。
        data = b"".join(self.vocab[token_id] for token_id in ids)
        return data.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """逐块处理，并保留未确定的尾部；效果与 encode(全部文本) 相同。

        末尾的词可能还没结束，空白的前瞻与 'll 等缩写还可能改变最后两段。
        特殊 token 也可能跨块，所以其最长长度范围内的尾部不能过早输出。
        缓冲取决于输入块大小和最长未完成片段，不读取整份文件。
        """
        pending = ""
        keep_chars = max((len(token) - 1 for token in self.special_tokens), default=0)
        for chunk in iterable:
            pending += chunk
            safe_limit = len(pending) - keep_chars
            recent = deque(maxlen=3)
            stable_end = 0
            for unit in iter_units(pending, self.specials):
                recent.append(unit)
                if len(recent) == 3 and recent[0][1] <= safe_limit:
                    start, end, is_special = recent[0]
                    piece = pending[start:end]
                    # 使用已在完整上下文中确定的边界，不能对截断的前缀重新预分词。
                    # 否则末尾的空白会被误当成全文的尾随空白，改变 \s+(?!\S) 的匹配。
                    if is_special:
                        yield self.token_to_id[piece.encode("utf-8")]
                    else:
                        yield from self._encode_piece(piece)
                    stable_end = end
            if stable_end:
                pending = pending[stable_end:]
        if pending:
            yield from self.encode(pending)

    def save(self, vocab_filepath: str | Path, merges_filepath: str | Path) -> None:
        """以 JSON 保存；bytes 用十六进制保存，任何字节（含无效 UTF-8）都能往返。"""
        Path(vocab_filepath).write_text(
            json.dumps({str(i): token.hex() for i, token in self.vocab.items()}, indent=2) + "\n",
            encoding="utf-8",
        )
        Path(merges_filepath).write_text(
            json.dumps([[a.hex(), b.hex()] for a, b in self.merges], indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """读取本实现 save() 写出的两个 JSON；特殊 token 列表由调用者传入。"""
        raw_vocab = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        raw_merges = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))
        vocab = {int(i): bytes.fromhex(value) for i, value in raw_vocab.items()}
        merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in raw_merges]
        return cls(vocab, merges, special_tokens)


def main() -> None:
    """固定的小样本演示；需要换参数时，直接修改这里。"""
    input_path = Path(__file__).resolve().parents[2] / "tests/fixtures/tinystories_sample.txt"
    specials = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, 300, specials, verbose=True)
    tokenizer = Tokenizer(vocab, merges, specials)
    print(f"词表大小：{len(vocab)}；合并次数：{len(merges)}")
    text = "Hello, 黎子姗！😀<|endoftext|>"
    ids = tokenizer.encode(text)
    print(f"原文：{text!r}")
    print(f"token IDs：{ids}")
    print(f"解码：{tokenizer.decode(ids)!r}")


if __name__ == "__main__":
    main()
