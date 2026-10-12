"""补充官方测试未覆盖的序列化、任意分块和小样例边界检查。"""

import random

import pytest

from cs336_basics.BPE_tokenizer.reference_tokenizer import (
    Tokenizer,
    initialize_vocab,
    pre_tokenize_and_count,
    train_bpe,
    train_from_counts,
)


def as_word(text):
    return tuple(bytes([value]) for value in text.encode("utf-8"))


def test_handout_whitespace_example():
    # 手册的这个示例特意使用空白切分，不要把它误当作 GPT-2 正则的输出。
    words = {as_word(word): count for word, count in {"low": 5, "lower": 2, "widest": 3, "newest": 6}.items()}
    vocab, merges = train_from_counts(words, 263, ["<|endoftext|>"])
    assert merges == [(b"s", b"t"), (b"e", b"st"), (b"o", b"w"), (b"l", b"ow"), (b"w", b"est"), (b"n", b"e")]
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    assert [vocab[i] for i in tokenizer.encode("newest")] == [b"ne", b"west"]


def test_repeated_pairs_count_overlaps_but_merge_does_not():
    vocab, merges = train_from_counts({as_word("aaaaa"): 1, as_word("bc"): 3}, 257, [])
    # aa 的相邻位置有 4 个，所以应胜过 bc 的 3 次。
    assert merges == [(b"a", b"a")]
    tokenizer = Tokenizer(vocab, merges)
    assert [vocab[i] for i in tokenizer.encode("aaaaa")] == [b"aa", b"aa", b"a"]


def test_special_boundaries_and_regex_metacharacters():
    specials = ["<|x|>", "<|x|><|x|>", "a.*[b]"]
    counts = pre_tokenize_and_count("low<|x|><|x|>lowa.*[b]low", specials)
    assert counts == {as_word("low"): 3}


def test_empty_corpus_stops_without_merges():
    vocab, merges = train_from_counts({}, 500, ["<eot>"])
    assert len(vocab) == 257
    assert merges == []


def test_decode_concatenates_bytes_before_replacing_invalid_sequences():
    tokenizer = Tokenizer(initialize_vocab([]), [])
    assert tokenizer.decode([0xE4, 0xB8, 0xAD]) == "中"
    assert tokenizer.decode([0xE4]) == "\ufffd"
    assert tokenizer.decode([0, 255]) == "\x00\ufffd"


def test_tokenizer_adds_missing_special_tokens():
    vocab = initialize_vocab(["<existing>"])
    tokenizer = Tokenizer(vocab, [], ["<existing>", "<new>"])
    assert tokenizer.encode("<existing>") == [256]
    assert tokenizer.encode("<new>") == [257]
    assert len(tokenizer.vocab) == 258


@pytest.fixture(scope="module")
def small_tokenizer():
    specials = ["<|x|>", "<|x|><|x|>", "a.*[b]", "hello world"]
    text = "cat! dog? hello world\r\n黎子姗 😀 isn't we'll they're I'd cat?  \t\n" * 3
    vocab, merges = train_from_counts(pre_tokenize_and_count(text, specials), 310, specials)
    return Tokenizer(vocab, merges, specials)


@pytest.mark.parametrize(
    "text",
    [
        "cat! dog?",
        "abc\r\n\r\n  def\t  \n",
        "'ll 've 're 's 't 'd 'm isn't we'll they're I'd",
        "你好，黎子姗😀e\u0301\x00!",
        "a<|x|><|x|>b<|x|>c",
        "hello world! a.*[b]hello world",
        "<|x|><|x|>",
        "  ",
        "",
    ],
)
def test_every_two_chunk_boundary_matches_whole_text(small_tokenizer, text):
    expected = small_tokenizer.encode(text)
    assert small_tokenizer.decode(expected) == text
    for split in range(len(text) + 1):
        assert list(small_tokenizer.encode_iterable([text[:split], "", text[split:]])) == expected
    assert list(small_tokenizer.encode_iterable(iter(text))) == expected


def test_deterministic_random_chunk_boundaries(small_tokenizer):
    rng = random.Random(336)
    alphabet = ["cat", " hello world", "'", "ll", "'r", "e", "  ", "\n", "\r", "\t", "中", "😀", "<|x|>", "a.*[b]"]
    for _ in range(80):
        text = "".join(rng.choices(alphabet, k=20))
        chunks = []
        start = 0
        while start < len(text):
            width = rng.randint(1, 9)
            chunks.append(text[start : start + width])
            start += width
        assert list(small_tokenizer.encode_iterable(chunks)) == small_tokenizer.encode(text)


def test_stream_does_not_consume_all_input_before_first_token(small_tokenizer):
    def source():
        yield "cat! dog? " * 30
        raise AssertionError("尚未输出第一个 token，就读取了后面的输入")

    stream = small_tokenizer.encode_iterable(source())
    assert next(stream) == small_tokenizer.encode("cat!")[0]
    stream.close()


def test_save_and_from_files_roundtrip(tmp_path, small_tokenizer):
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.json"
    small_tokenizer.save(vocab_path, merges_path)
    loaded = Tokenizer.from_files(vocab_path, merges_path, small_tokenizer.special_tokens)
    assert loaded.vocab == small_tokenizer.vocab
    assert loaded.merges == small_tokenizer.merges
    text = "hello world 黎子姗😀\x00<|x|><|x|>"
    assert loaded.encode(text) == small_tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_training_preserves_crlf_bytes(tmp_path):
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"a\r\nb\r\n")
    vocab, _ = train_bpe(path, 257, [])
    assert b"\r\n" in vocab.values()
