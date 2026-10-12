"""固定抽取验证集文档，测量用户 Tokenizer 的压缩率和编码吞吐量。"""

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

from cs336_basics.BPE_tokenizer import BPE_tokenizer as implementation
from cs336_basics.BPE_tokenizer.reference_tokenizer import Tokenizer as ReferenceTokenizer


SPECIAL = "<|endoftext|>"


def iter_documents(path):
    separator = SPECIAL.encode("utf-8")
    pending = b""
    document_index = 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            parts = (pending + block).split(separator)
            pending = parts.pop()
            for document in parts:
                if document.strip():
                    yield document_index, document
                document_index += 1
        if pending.strip():
            yield document_index, pending


def sample_documents(path, count, seed):
    # Reservoir sampling gives each nonempty document equal inclusion probability.
    rng = random.Random(seed)
    reservoir = []
    population = 0
    for document_index, document in iter_documents(path):
        population += 1
        item = (document_index, document)
        if len(reservoir) < count:
            reservoir.append(item)
        else:
            position = rng.randrange(population)
            if position < count:
                reservoir[position] = item
    assert len(reservoir) == count
    documents = []
    for document_index, document in sorted(reservoir):
        documents.append({
            "document_index_zero_based": document_index,
            "utf8_bytes": len(document),
            "sha256": hashlib.sha256(document).hexdigest(),
            "text": document.decode("utf-8"),
        })
    with path.open("rb") as source:
        corpus_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    return {
        "input_path": str(path),
        "input_sha256": corpus_sha256,
        "nonempty_documents_in_corpus": population,
        "seed": seed,
        "sampling": "Uniform reservoir sample of nonempty documents; sorted by source index after selection",
        "document_policy": "Split at endoftext; exclude the separator, preserve all document whitespace; exclude whitespace-only records",
        "documents": documents,
    }


def measure_case(name, tokenizer, reference, sample, output_dir):
    tokenizer.encode("Hello, world!")
    total_bytes = 0
    total_tokens = 0
    total_seconds = 0.0
    records = []
    for index, document in enumerate(sample["documents"], start=1):
        text = document["text"]
        print(f"{name}: 开始文档 {index}/{len(sample['documents'])}，{document['utf8_bytes']:,} bytes", flush=True)
        started = perf_counter()
        ids = tokenizer.encode(text)
        elapsed = perf_counter() - started
        assert tokenizer.decode(ids) == text
        assert ids == reference.encode(text)
        assert all(0 <= token_id < len(tokenizer.vocab) for token_id in ids)
        total_bytes += document["utf8_bytes"]
        total_tokens += len(ids)
        total_seconds += elapsed
        records.append({
            "document_index_zero_based": document["document_index_zero_based"],
            "document_sha256": document["sha256"],
            "utf8_bytes": document["utf8_bytes"],
            "token_count": len(ids),
            "encode_seconds": elapsed,
            "ids": ids,
        })
        print(f"{name}: 文档 {index} 完成，{len(ids):,} tokens，{elapsed:.3f} 秒，往返及参考 ID 检查通过", flush=True)
    ids_filename = name + "_ids.json"
    (output_dir / ids_filename).write_text(json.dumps(records, indent=2) + "\n")
    throughput = total_bytes / total_seconds
    result = {
        "name": name,
        "vocab_size": len(tokenizer.vocab),
        "document_count": len(records),
        "utf8_bytes": total_bytes,
        "token_count": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "encode_seconds": total_seconds,
        "bytes_per_second": throughput,
        "pile_825_decimal_gb_estimated_seconds": 825 * 10**9 / throughput,
        "pile_825_decimal_gb_estimated_days": 825 * 10**9 / throughput / 86400,
        "ids_file": ids_filename,
        "all_roundtrips_passed": True,
        "all_ids_match_reference": True,
    }
    (output_dir / (name + "_metrics.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--documents", type=int, default=10)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "data/tokenizer_experiments" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True)
    source = Path(implementation.__file__).read_bytes()
    (output_dir / "implementation.py").write_bytes(source)
    (output_dir / "tokenizer_experiments.py").write_bytes(Path(__file__).read_bytes())
    print(f"输出目录：{output_dir}", flush=True)
    paths = {
        "tinystories": root / "data/TinyStoriesV2-GPT4-valid.txt",
        "owt": root / "data/owt_valid.txt",
    }
    samples = {}
    for name, path in paths.items():
        sample = sample_documents(path, args.documents, args.seed)
        samples[name] = sample
        (output_dir / (name + "_sample.json")).write_text(json.dumps(sample, ensure_ascii=False, indent=2) + "\n")
        print(f"{name}: 从 {sample['nonempty_documents_in_corpus']:,} 篇文档抽取 {args.documents} 篇，共 {sum(doc['utf8_bytes'] for doc in sample['documents']):,} bytes", flush=True)

    models = {
        "tinystories": root / "data/bpe_benchmarks/20260915-193145-068347-multiprocess-8p-8c-train-10000-baseline",
        "owt": root / "data/bpe_benchmarks/20260915-212155-070170-multiprocess-8p-128c-owt-train-32000-baseline",
    }
    tokenizers = {}
    references = {}
    artifact_hashes = {}
    for name, directory in models.items():
        vocab_path = directory / "vocab.json"
        merges_path = directory / "merges.json"
        tokenizers[name] = implementation.Tokenizer.from_files(vocab_path, merges_path, [SPECIAL])
        references[name] = ReferenceTokenizer.from_files(vocab_path, merges_path, [SPECIAL])
        artifact_hashes[name] = {
            "directory": str(directory),
            "vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
            "merges_sha256": hashlib.sha256(merges_path.read_bytes()).hexdigest(),
        }
    assert len(tokenizers["tinystories"].vocab) == 10_000
    assert len(tokenizers["owt"].vocab) == 32_000

    cases = {}
    for case, corpus, model in [
        ("tinystories_with_tinystories", "tinystories", "tinystories"),
        ("owt_with_owt", "owt", "owt"),
        ("owt_with_tinystories", "owt", "tinystories"),
    ]:
        cases[case] = measure_case(case, tokenizers[model], references[model], samples[corpus], output_dir)
    result = {
        "seed": args.seed,
        "documents_per_corpus": args.documents,
        "python": sys.version,
        "platform": platform.platform(),
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "model_artifacts": artifact_hashes,
        "timing_scope": "Sum of individual encode calls after a short warmup; excludes loading, sampling, decoding, reference checks and serialization. No profiler, one measured pass per case.",
        "cases": cases,
        "owt_extra_tokens_with_tinystories_percent": (
            cases["owt_with_tinystories"]["token_count"] / cases["owt_with_owt"]["token_count"] - 1
        ) * 100,
        "scope": "Ten held-out documents per corpus. Full training/development corpora have not been tokenized by this script.",
    }
    assert hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest() == result["implementation_sha256"]
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("全部抽样实验完成。", flush=True)


if __name__ == "__main__":
    main()
