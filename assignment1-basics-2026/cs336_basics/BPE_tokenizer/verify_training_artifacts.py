"""结合 OWT 下载校验记录，检查训练词表、合并规则和分块边界。"""

import argparse
import hashlib
import json
from pathlib import Path

from cs336_basics.BPE_tokenizer.BPE_tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    metrics = json.loads((output_dir / "metrics.json").read_text())
    vocab_data = json.loads((output_dir / "vocab.json").read_text())
    merges_data = json.loads((output_dir / "merges.json").read_text())
    vocab = {int(token_id): bytes.fromhex(value) for token_id, value in vocab_data.items()}
    merges = [(bytes.fromhex(first), bytes.fromhex(second)) for first, second in merges_data]
    target_size = metrics["requested_vocab_size"]
    specials = [token.encode("utf-8") for token in metrics["special_tokens"]]
    initial_size = 256 + len(specials)

    assert len(vocab) == target_size
    assert set(vocab) == set(range(target_size))
    assert len(set(vocab.values())) == target_size
    assert len(merges) == target_size - initial_size
    for token_id in range(256):
        assert vocab[token_id] == bytes([token_id])
    for token_id, special in enumerate(specials, start=256):
        assert vocab[token_id] == special

    available = {vocab[token_id] for token_id in range(initial_size)}
    for rank, (first, second) in enumerate(merges):
        assert first in available and second in available
        merged = first + second
        assert merged == vocab[initial_size + rank]
        assert all(special not in merged for special in specials)
        available.add(merged)

    input_path = Path(metrics["input_path"])
    assert input_path.stat().st_size == metrics["input_bytes"]
    download_metadata = json.loads(input_path.with_suffix(".download.json").read_text())
    assert metrics["input_sha256"] == download_metadata["text_sha256"]
    boundaries = metrics["boundaries"]
    assert boundaries[0] == 0 and boundaries[-1] == metrics["input_bytes"]
    assert boundaries == sorted(set(boundaries))
    with input_path.open("rb") as source:
        for boundary in boundaries[1:-1]:
            source.seek(boundary)
            assert source.read(len(specials[0])) == specials[0]

    tokenizer = Tokenizer.from_files(
        output_dir / "vocab.json", output_dir / "merges.json", metrics["special_tokens"]
    )
    assert tokenizer.vocab == vocab and tokenizer.merges == merges
    samples = ["", "hello world", "黎子姗🙂", "OWT\n<|endoftext|>test<|endoftext|>"]
    for sample in samples:
        ids = tokenizer.encode(sample)
        assert tokenizer.decode(ids) == sample
        assert ids.count(256) == sample.count("<|endoftext|>")
    assert tokenizer.decode([255]) == "\ufffd"

    longest_length = max(len(token) for token in vocab.values())
    longest_tokens = [
        {"id": token_id, "bytes": len(token), "hex": token.hex(), "repr": repr(token)}
        for token_id, token in vocab.items()
        if len(token) == longest_length
    ]
    report = {
        "passed": True,
        "checks": [
            "requested vocabulary size and contiguous IDs",
            "unique token bytes, base bytes and special IDs",
            "merge count, parent availability and vocabulary order",
            "special tokens excluded from learned merges",
            "input hash matches verified download",
            "all chunk boundaries begin at the special token",
            "from_files preserves vocabulary and merge order",
            "empty, ASCII, Unicode and special token roundtrips",
            "malformed UTF-8 replacement",
        ],
        "roundtrip_sample_count": len(samples),
        "vocab_sha256": hashlib.sha256((output_dir / "vocab.json").read_bytes()).hexdigest(),
        "merges_sha256": hashlib.sha256((output_dir / "merges.json").read_bytes()).hexdigest(),
        "longest_token_bytes": longest_length,
        "longest_tokens": longest_tokens,
        "scope": "Structural and small roundtrip checks. Full-corpus IDs are not compared to an independent reference tokenizer.",
    }
    (output_dir / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"PASS: {target_size:,} tokens, {len(merges):,} merges, {len(boundaries) - 1} chunks")
    print(f"Longest token: {longest_length} bytes; candidates: {longest_tokens[:5]}")


if __name__ == "__main__":
    main()
