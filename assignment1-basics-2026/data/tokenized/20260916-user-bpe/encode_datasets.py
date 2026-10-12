"""Encode assignment datasets with the user's Tokenizer and save uint16 NumPy arrays."""

import argparse
import hashlib
import json
import math
import multiprocessing
import shutil
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import tiktoken

from cs336_basics.BPE_tokenizer import BPE_tokenizer as implementation
from cs336_basics.pretokenization_example import find_chunk_boundaries


ROOT = Path(__file__).resolve().parents[2]
SPECIAL = "<|endoftext|>"
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SAMPLES = ROOT / "data/tokenizer_experiments/20260916-083805-814679"
INPUTS = {
    "tinystories_valid": "TinyStoriesV2-GPT4-valid.txt",
    "owt_valid": "owt_valid.txt",
    "tinystories_train": "TinyStoriesV2-GPT4-train.txt",
    "owt_train": "owt_train.txt",
}


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def initialize_worker(model_dir):
    global tokenizer, reference, special_id
    model_dir = Path(model_dir)
    tokenizer = implementation.Tokenizer.from_files(
        model_dir / "vocab.json", model_dir / "merges.json", [SPECIAL]
    )
    special_id = tokenizer.reversed_vocab[SPECIAL.encode("utf-8")]
    assert min(tokenizer.vocab) == 0
    assert max(tokenizer.vocab) <= np.iinfo(np.uint16).max
    # The reference checks the output; all saved IDs come from the user's encode.
    reference = tiktoken.Encoding(
        "cs336_full_encoding_check",
        pat_str=PAT,
        mergeable_ranks={value: key for key, value in tokenizer.vocab.items() if key != special_id},
        special_tokens={SPECIAL: special_id},
        explicit_n_vocab=len(tokenizer.vocab),
    )


def encode_chunk(task):
    index, start, end, input_path, shard_dir = task
    with open(input_path, "rb") as source:
        source.seek(start)
        raw = source.read(end - start)
    assert len(raw) == end - start
    text = raw.decode("utf-8")
    started = perf_counter()
    ids = tokenizer.encode(text)
    encode_seconds = perf_counter() - started
    assert tokenizer.decode(ids) == text, f"Roundtrip failed in chunk {index}"
    assert ids == reference.encode(text, allowed_special="all"), f"Reference mismatch in chunk {index}"
    assert min(ids) >= 0 and max(ids) < len(tokenizer.vocab)
    values = np.asarray(ids, dtype=np.uint16)
    marker_count = raw.count(SPECIAL.encode("utf-8"))
    assert int(np.count_nonzero(values == special_id)) == marker_count
    np.save(Path(shard_dir) / f"{index:06d}.npy", values, allow_pickle=False)
    return {
        "index": index,
        "start": start,
        "end": end,
        "input_bytes": len(raw),
        "token_count": len(values),
        "special_token_count": marker_count,
        "min_id": int(values.min()),
        "max_id": int(values.max()),
        "encode_seconds": encode_seconds,
    }


def encode_dataset(name, model_dir, output_dir, processes, chunk_bytes):
    input_path = ROOT / "data" / INPUTS[name]
    input_bytes = input_path.stat().st_size
    started = perf_counter()
    with input_path.open("rb") as source:
        boundaries = find_chunk_boundaries(source, math.ceil(input_bytes / chunk_bytes), SPECIAL.encode())
        assert boundaries[0] == 0 and boundaries[-1] == input_bytes
        for boundary in boundaries[1:-1]:
            source.seek(boundary)
            assert source.read(len(SPECIAL)) == SPECIAL.encode()
    shard_dir = output_dir / (name + "_shards")
    shard_dir.mkdir()
    tasks = [
        (index, start, end, str(input_path), str(shard_dir))
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
    ]
    results = []
    processed_bytes = 0
    token_count = 0
    last_log = 0.0
    print(f"{name}: {input_bytes:,} bytes, {len(tasks)} chunks, {processes} workers", flush=True)
    with multiprocessing.get_context("spawn").Pool(
        processes, initializer=initialize_worker, initargs=(str(model_dir),)
    ) as pool:
        for result in pool.imap_unordered(encode_chunk, tasks, chunksize=1):
            results.append(result)
            processed_bytes += result["input_bytes"]
            token_count += result["token_count"]
            elapsed = perf_counter() - started
            status = {
                "phase": "encoding", "dataset": name,
                "chunks_done": len(results), "chunks_total": len(tasks),
                "processed_bytes": processed_bytes, "input_bytes": input_bytes,
                "token_count": token_count, "elapsed_seconds": elapsed,
                "bytes_per_second": processed_bytes / elapsed,
            }
            write_json(output_dir / "status.json", status)
            if elapsed - last_log >= 15 or len(results) == len(tasks):
                print(f"{name}: {processed_bytes / input_bytes:.1%}, {elapsed:.1f}s, {processed_bytes / elapsed / 1e6:.2f} MB/s", flush=True)
                last_log = elapsed
    encoding_wall_seconds = perf_counter() - started
    results.sort(key=lambda item: item["index"])
    assert processed_bytes == input_bytes
    write_json(output_dir / "status.json", {"phase": "assembling", "dataset": name, "token_count": token_count})
    partial_path = output_dir / (name + ".npy.partial")
    array = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.uint16, shape=(token_count,))
    payload_hash = hashlib.sha256()
    offset = 0
    for result in results:
        shard = np.load(shard_dir / f"{result['index']:06d}.npy", mmap_mode="r", allow_pickle=False)
        assert shard.dtype == np.uint16 and shard.shape == (result["token_count"],)
        array[offset:offset + len(shard)] = shard
        payload_hash.update(shard.tobytes())
        offset += len(shard)
        del shard
    assert offset == token_count
    array.flush()
    del array
    saved = np.load(partial_path, mmap_mode="r", allow_pickle=False)
    assert saved.dtype == np.uint16 and saved.shape == (token_count,)
    data_offset = saved.offset
    del saved
    with partial_path.open("rb") as file:
        file.seek(data_offset)
        saved_payload_hash = hashlib.file_digest(file, "sha256").hexdigest()
    assert saved_payload_hash == payload_hash.hexdigest()
    final_path = output_dir / (name + ".npy")
    partial_path.replace(final_path)
    with input_path.open("rb") as file:
        input_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    metrics = {
        "dataset": name, "input_path": str(input_path), "input_bytes": input_bytes,
        "input_sha256": input_sha256, "model_directory": str(model_dir),
        "output_path": str(final_path), "output_bytes": final_path.stat().st_size,
        "dtype": "uint16", "shape": [token_count], "token_count": token_count,
        "special_token_count": sum(item["special_token_count"] for item in results),
        "min_id": min(item["min_id"] for item in results),
        "max_id": max(item["max_id"] for item in results),
        "bytes_per_token": input_bytes / token_count,
        "encoding_wall_seconds": encoding_wall_seconds,
        "total_seconds": perf_counter() - started,
        "end_to_end_bytes_per_second": input_bytes / (perf_counter() - started),
        "payload_sha256": saved_payload_hash,
        "all_chunks_roundtrip_and_match_reference": True,
        "all_boundaries_at_special_tokens": True,
        "processes": processes, "target_chunk_bytes": chunk_bytes,
        "chunks": results,
    }
    write_json(output_dir / (name + "_metrics.json"), metrics)
    shutil.rmtree(shard_dir)
    print(f"{name}: saved {token_count:,} uint16 IDs to {final_path}; {metrics['total_seconds']:.1f}s", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=4)
    parser.add_argument("--datasets", nargs="+", choices=INPUTS, default=list(INPUTS))
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(implementation.__file__).read_bytes()
    (output_dir / "implementation.py").write_bytes(source)
    (output_dir / "encode_datasets.py").write_bytes(Path(__file__).read_bytes())
    baseline = json.loads((SAMPLES / "metrics.json").read_text())
    models = baseline["model_artifacts"]
    for artifact in models.values():
        for key in ("vocab", "merges"):
            path = Path(artifact["directory"]) / (key + ".json")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact[key + "_sha256"]
    manifest = {
        "started_at": datetime.now().astimezone().isoformat(),
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "model_artifacts": models,
        "policy": "Preserve source bytes, document order, whitespace and endoftext. Chunk only at document markers. Save user Tokenizer IDs; independently check every chunk with tiktoken and decode roundtrip.",
        "datasets": {},
    }
    for name in args.datasets:
        model = "tinystories" if name.startswith("tinystories") else "owt"
        metrics = encode_dataset(name, Path(models[model]["directory"]), output_dir, args.processes, args.chunk_mib * 1024**2)
        manifest["datasets"][name] = {key: value for key, value in metrics.items() if key != "chunks"}
        write_json(output_dir / "manifest.json", manifest)
    assert Path(implementation.__file__).read_bytes() == source
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "status.json", {"phase": "complete", "datasets": list(manifest["datasets"])})
    print("All requested datasets completed and verified.", flush=True)


if __name__ == "__main__":
    main()
