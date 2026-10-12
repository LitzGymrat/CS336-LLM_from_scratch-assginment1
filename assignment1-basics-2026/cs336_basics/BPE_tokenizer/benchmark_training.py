"""测量实际训练入口；--processes 0 为串行，--profile 同时分析主进程和分块任务。"""

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import resource
import sys
import threading
from datetime import datetime
from functools import partial
from pathlib import Path
from time import perf_counter

import psutil

from cs336_basics.BPE_tokenizer import BPE_tokenizer as implementation
from cs336_basics import pretokenization_example


def profile_chunk(input_path, start, end, special_tokens, profile_dir):
    # 子进程重新导入模块，implementation 中仍是用户原始的分块函数。
    profiler = cProfile.Profile()
    started = perf_counter()
    counts = profiler.runcall(
        implementation.pretokenize_chunk, input_path, start, end, special_tokens
    )
    elapsed = perf_counter() - started
    name = f"worker-{start}-{end}"
    output_dir = Path(profile_dir)
    profiler.dump_stats(str(output_dir / f"{name}.prof"))
    metadata = {
        "pid": os.getpid(),
        "start": start,
        "end": end,
        "seconds_with_profile": elapsed,
        "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (output_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return counts


def write_profile_report(stats, path):
    with path.open("w", encoding="utf-8") as report:
        stats.stream = report
        stats.strip_dirs().sort_stats("cumulative").print_stats(25)
        stats.sort_stats("tottime").print_stats(25)


def sample_memory(stop, measurement):
    process = psutil.Process()
    while not stop.is_set():
        members = [process] + process.children(recursive=True)
        total_rss = 0
        for member in members:
            try:
                total_rss += member.memory_info().rss
            except psutil.NoSuchProcess:
                # 进程池退出时，子进程可能在本轮采样中途正常结束。
                continue
        measurement["peak_rss_sum_bytes"] = max(measurement["peak_rss_sum_bytes"], total_rss)
        measurement["max_process_count"] = max(measurement["max_process_count"], len(members))
        measurement["samples"] += 1
        stop.wait(0.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, default=4, help="0 为串行预分词")
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dataset", choices=["tinystories", "owt"], default="tinystories")
    parser.add_argument("--split", choices=["valid", "train"], default="valid")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--skip-counts-export", action="store_true", help="不复制和导出初始频次表，减少大语料的测量开销")
    parser.add_argument("--sample-memory", action="store_true", help="每 0.25 秒采样整个进程组的 RSS 总和")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    if args.dataset == "owt":
        input_path = root / f"data/owt_{args.split}.txt"
    else:
        input_path = root / f"data/TinyStoriesV2-GPT4-{args.split}.txt"
    mode = "profile" if args.profile else "baseline"
    label = "serial" if args.processes == 0 else f"multiprocess-{args.processes}p-{args.chunks}c"
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"-{label}-{args.dataset}-{args.split}-{args.vocab_size}-{mode}"
    output_dir = root / "data/bpe_benchmarks" / run_name
    output_dir.mkdir(parents=True)
    source = Path(implementation.__file__).read_bytes()
    helper_source = Path(pretokenization_example.__file__).read_bytes()
    (output_dir / "implementation.py").write_bytes(source)
    (output_dir / "pretokenization_example.py").write_bytes(helper_source)
    (output_dir / "benchmark_training.py").write_bytes(Path(__file__).read_bytes())

    # 只在本次测量进程中包一层计时；用户文件和训练算法不变。
    original_loop = implementation.train_bpe_loop
    original_boundaries = implementation.find_chunk_boundaries
    stage = {}

    def timed_loop(vocab, vocab_size, counts):
        stage["merge_start"] = perf_counter()
        stage["unique_pretokens"] = len(counts)
        stage["pretoken_occurrences"] = sum(counts.values())
        if not args.skip_counts_export:
            stage["initial_counts"] = counts.copy()
        print(f"预分词及汇总完成：{stage['merge_start'] - started:.3f} 秒；不同片段：{len(counts):,}。开始合并训练……", flush=True)
        result = original_loop(vocab, vocab_size, counts)
        stage["merge_end"] = perf_counter()
        return result

    def recorded_boundaries(file, desired_num_chunks, split_special_token):
        boundaries = original_boundaries(file, desired_num_chunks, split_special_token)
        stage["boundaries"] = boundaries
        return boundaries

    implementation.train_bpe_loop = timed_loop
    implementation.find_chunk_boundaries = recorded_boundaries
    if args.profile and args.processes > 0:
        implementation.pretokenize_chunk = partial(profile_chunk, profile_dir=str(output_dir))

    print(f"输出目录：{output_dir}", flush=True)
    print(f"输入：{input_path}；大小：{input_path.stat().st_size:,} bytes", flush=True)
    print(f"模式：{mode}；进程数：{args.processes}（0 表示串行）；目标块数：{args.chunks}", flush=True)
    memory_measurement = {"peak_rss_sum_bytes": 0, "max_process_count": 0, "samples": 0}
    if args.sample_memory:
        memory_stop = threading.Event()
        memory_thread = threading.Thread(target=sample_memory, args=(memory_stop, memory_measurement), daemon=True)
        memory_thread.start()
    profiler = cProfile.Profile()
    if args.profile:
        profiler.enable()
    started = perf_counter()
    if args.processes == 0:
        vocab, merges = implementation.train_bpe(str(input_path), args.vocab_size, ["<|endoftext|>"])
    else:
        vocab, merges = implementation.train_bpe_multiprocesses(
            str(input_path), args.vocab_size, ["<|endoftext|>"],
            num_processes=args.processes, desired_num_chunks=args.chunks,
        )
    finished = perf_counter()
    if args.profile:
        profiler.disable()
    parent_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if args.sample_memory:
        memory_stop.set()
        memory_thread.join()

    counts_sha256 = None
    if not args.skip_counts_export:
        counts = stage["initial_counts"]
        counts_data = []
        for word in sorted(counts):
            word_hex = []
            for token in word:
                word_hex.append(token.hex())
            counts_data.append([word_hex, counts[word]])
        counts_json = json.dumps(counts_data, separators=(",", ":")).encode("utf-8")
        (output_dir / "initial_counts.json").write_bytes(counts_json)
        counts_sha256 = hashlib.sha256(counts_json).hexdigest()
    vocab_data = {}
    for token_id, token in vocab.items():
        vocab_data[str(token_id)] = token.hex()
    merge_data = []
    for first, second in merges:
        merge_data.append([first.hex(), second.hex()])
    (output_dir / "vocab.json").write_text(json.dumps(vocab_data, indent=2) + "\n")
    (output_dir / "merges.json").write_text(json.dumps(merge_data, indent=2) + "\n")

    with input_path.open("rb") as input_file:
        input_sha256 = hashlib.file_digest(input_file, "sha256").hexdigest()
    metrics = {
        "implementation_label": label,
        "mode": mode,
        "dataset": args.dataset,
        "dataset_split": args.split,
        "input_path": str(input_path),
        "input_bytes": input_path.stat().st_size,
        "input_sha256": input_sha256,
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "chunk_helper_sha256": hashlib.sha256(helper_source).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "num_processes": args.processes,
        "requested_chunks": args.chunks,
        "boundaries": stage.get("boundaries"),
        "requested_vocab_size": args.vocab_size,
        "special_tokens": ["<|endoftext|>"],
        "vocab_size": len(vocab),
        "merge_count": len(merges),
        "unique_pretokens": stage["unique_pretokens"],
        "pretoken_occurrences": stage["pretoken_occurrences"],
        "initial_counts_exported": not args.skip_counts_export,
        "initial_counts_sha256": counts_sha256,
        "seconds": {
            "preparation": stage["merge_start"] - started,
            "merge_training": stage["merge_end"] - stage["merge_start"],
            "total_training": finished - started,
        },
        "parent_peak_rss_bytes": parent_peak_rss,
        "memory_scope": "Main process peak RSS only; not total multiprocessing memory.",
        "timing_scope": "Training call only. Preparation includes initialization, I/O, pretokenization, pool startup and count aggregation. Merge timing includes pretoken statistics and, when exported, a shallow counts snapshot. Excludes imports, exports and report generation.",
    }
    if args.sample_memory:
        metrics["process_tree_memory"] = memory_measurement
        metrics["process_tree_memory_scope"] = "RSS sum of the parent and recursive children sampled every 0.25 seconds during training. Approximate peak; shared pages may be counted in multiple processes."
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")

    if args.profile:
        profiler.dump_stats(str(output_dir / "parent.prof"))
        write_profile_report(pstats.Stats(profiler), output_dir / "parent_profile.txt")
        if args.processes > 0:
            worker_files = sorted(output_dir.glob("worker-*.prof"))
            worker_stats = pstats.Stats(str(worker_files[0]))
            for path in worker_files[1:]:
                worker_stats.add(str(path))
            worker_stats.dump_stats(str(output_dir / "workers_aggregate.prof"))
            write_profile_report(worker_stats, output_dir / "workers_profile.txt")

    print(f"准备与预分词：{metrics['seconds']['preparation']:.3f} 秒；合并：{metrics['seconds']['merge_training']:.3f} 秒；总计：{metrics['seconds']['total_training']:.3f} 秒", flush=True)
    print(f"片段种类：{stage['unique_pretokens']:,}；片段出现次数：{stage['pretoken_occurrences']:,}；词表：{len(vocab)}；合并：{len(merges)}", flush=True)
    if args.sample_memory:
        print(f"进程组 RSS 总和的采样峰值：{memory_measurement['peak_rss_sum_bytes'] / 1024**3:.3f} GiB", flush=True)


if __name__ == "__main__":
    main()
