"""在 TinyStories 验证集上测量当前训练实现；--profile 另跑一次 cProfile。"""

import argparse
import cProfile
import hashlib
import json
import platform
import pstats
import resource
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time

from cs336_basics.BPE_tokenizer import BPE_tokenizer as implementation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="current", help="本次实现的版本标签")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    input_path = root / "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10_000
    special_tokens = ["<|endoftext|>"]
    profile_enabled = args.profile
    mode = "profile" if profile_enabled else "baseline"
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{args.label}-valid-10000-" + mode
    output_dir = root / "data/bpe_benchmarks" / run_name
    output_dir.mkdir(parents=True)
    source = Path(implementation.__file__).read_bytes()
    (output_dir / "implementation.py").write_bytes(source)

    print(f"输出目录：{output_dir}", flush=True)
    print(f"模式：{mode}；输入：{input_path.stat().st_size:,} bytes；词表目标：{vocab_size}", flush=True)

    profiler = cProfile.Profile()
    if profile_enabled:
        profiler.enable()

    # 与 train_bpe 的步骤一致，分开调用是为了测量各阶段耗时。
    start = perf_counter()
    cpu_start = process_time()
    vocab = implementation.initialize_vocab(special_tokens)
    initialized = perf_counter()

    with open(input_path, encoding="utf-8", newline="") as file:
        raw_text = file.read()
    read_done = perf_counter()
    print(f"读取完成：{read_done - initialized:.3f} 秒", flush=True)

    counts = implementation.pretokenize_and_count(raw_text, special_tokens)
    pretokenize_done = perf_counter()
    print(f"预分词完成：{pretokenize_done - read_done:.3f} 秒；不同片段：{len(counts):,}", flush=True)
    print("开始合并训练……", flush=True)

    vocab, merges = implementation.train_bpe_loop(vocab, vocab_size, counts)
    train_done = perf_counter()
    cpu_seconds = process_time() - cpu_start
    if profile_enabled:
        profiler.disable()

    # macOS 的 ru_maxrss 单位是字节；记录的是进程从启动至此的峰值 RSS。
    peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    longest_id = max(vocab, key=lambda token_id: len(vocab[token_id]))
    longest = vocab[longest_id]
    metrics = {
        "implementation_label": args.label,
        "mode": mode,
        "input_path": str(input_path),
        "input_bytes": input_path.stat().st_size,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "requested_vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "unique_pretokens": len(counts),
        "pretoken_occurrences": sum(counts.values()),
        "seconds": {
            "initialize": initialized - start,
            "read": read_done - initialized,
            "pretokenize": pretokenize_done - read_done,
            "merge_training": train_done - pretokenize_done,
            "total_training": train_done - start,
            "cpu": cpu_seconds,
        },
        "peak_rss_bytes": peak_rss_bytes,
        "longest_token": {
            "id": longest_id,
            "byte_length": len(longest),
            "hex": longest.hex(),
            "repr": repr(longest),
            "text": longest.decode("utf-8", errors="replace"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # JSON 中用十六进制保存原始字节，与参考版的 save/from_files 格式一致。
    vocab_data = {}
    for token_id, token in vocab.items():
        vocab_data[str(token_id)] = token.hex()
    merge_data = []
    for first, second in merges:
        merge_data.append([first.hex(), second.hex()])
    (output_dir / "vocab.json").write_text(json.dumps(vocab_data, indent=2) + "\n", encoding="utf-8")
    (output_dir / "merges.json").write_text(json.dumps(merge_data, indent=2) + "\n", encoding="utf-8")

    if profile_enabled:
        profiler.dump_stats(str(output_dir / "training.prof"))
        with open(output_dir / "profile.txt", "w", encoding="utf-8") as report:
            stats = pstats.Stats(profiler, stream=report)
            stats.strip_dirs().sort_stats("cumulative").print_stats(25)
            stats.sort_stats("tottime").print_stats(25)

    print(f"合并训练：{train_done - pretokenize_done:.3f} 秒", flush=True)
    print(f"总训练耗时：{train_done - start:.3f} 秒；峰值 RSS：{peak_rss_bytes / 1024**2:.2f} MiB", flush=True)
    print(f"词表：{len(vocab)}；合并规则：{len(merges)}；最长 token：{len(longest)} bytes，{longest!r}", flush=True)
    print(f"已保存 metrics.json、vocab.json、merges.json：{output_dir}", flush=True)


if __name__ == "__main__":
    main()
