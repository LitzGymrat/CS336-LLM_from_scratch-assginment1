"""比较真实多进程入口的预分词阶段；到达合并入口后结束，不重复训练。"""

import argparse
import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter

from cs336_basics.BPE_tokenizer import BPE_tokenizer as implementation
from cs336_basics.BPE_tokenizer.benchmark_training import sample_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, required=True)
    parser.add_argument("--chunks", type=int, default=128)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    input_path = root / "data/owt_valid.txt"
    label = f"pretokenize-{args.processes}p-{args.chunks}c-owt-valid"
    output_dir = root / "data/bpe_benchmarks" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-" + label)
    output_dir.mkdir(parents=True)
    source = Path(implementation.__file__).read_bytes()
    (output_dir / "implementation.py").write_bytes(source)
    (output_dir / "benchmark_pretokenization.py").write_bytes(Path(__file__).read_bytes())
    captured = {}

    def capture_counts(vocab, vocab_size, counts):
        captured["counts"] = counts
        return vocab, []

    # 仅本次测量进程替换合并入口；分块、进程池、预分词和汇总均执行用户代码。
    implementation.train_bpe_loop = capture_counts
    measurement = {"peak_rss_sum_bytes": 0, "max_process_count": 0, "samples": 0}
    stop = threading.Event()
    monitor = threading.Thread(target=sample_memory, args=(stop, measurement), daemon=True)
    print(f"输出目录：{output_dir}", flush=True)
    monitor.start()
    started = perf_counter()
    implementation.train_bpe_multiprocesses(
        str(input_path), 32_000, ["<|endoftext|>"], args.processes, args.chunks
    )
    elapsed = perf_counter() - started
    stop.set()
    monitor.join()
    counts = captured["counts"]

    # 长度、原始字节、频次组成无歧义的记录；排序后计算完整频次表的 SHA-256。
    digest = hashlib.sha256()
    for word in sorted(counts):
        word_bytes = b"".join(word)
        digest.update(len(word_bytes).to_bytes(8, "big"))
        digest.update(word_bytes)
        digest.update(counts[word].to_bytes(8, "big"))
    with input_path.open("rb") as source_file:
        input_sha256 = hashlib.file_digest(source_file, "sha256").hexdigest()
    metadata = {
        "stage": "pretokenization only; no merges trained",
        "num_processes": args.processes,
        "requested_chunks": args.chunks,
        "seconds": elapsed,
        "input_path": str(input_path),
        "input_sha256": input_sha256,
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "unique_pretokens": len(counts),
        "pretoken_occurrences": sum(counts.values()),
        "initial_counts_sha256": digest.hexdigest(),
        "counts_hash_format": "Sorted byte tuples, each serialized as byte_length_u64be + bytes + frequency_u64be",
        "process_tree_memory": measurement,
        "timing_scope": "Initialization, file boundaries, pool startup, pretokenization and aggregation; excludes hashing and exports.",
        "memory_scope": "RSS sum sampled every 0.25 seconds; shared pages may be counted repeatedly.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
