import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from cs336_basics.BPE_tokenizer.BPE_tokenizer import Tokenizer
from cs336_basics.Decoding.decode import decoding
from cs336_basics.Transformer.Transformer_lm import Transformer_lm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用训练好的 TinyStories 模型续写英文故事。")
    parser.add_argument("--prompt", default="Once upon a time, there was a little dragon who was afraid of fire.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.prompt or args.temperature <= 0 or not 0 < args.top_p <= 1 or args.max_new_tokens < 1:
        parser.error("prompt 不能为空；temperature > 0；0 < top-p <= 1；max-new-tokens >= 1。")

    root = Path(__file__).resolve().parent
    run_dir = root / "data/lm_runs/tinystories-train-20260929-124145-511697"
    tokenizer_dir = root / "data/bpe_benchmarks/20260915-193145-068347-multiprocess-8p-8c-train-10000-baseline"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    tokenizer = Tokenizer.from_files(
        tokenizer_dir / "vocab.json", tokenizer_dir / "merges.json",
        special_tokens=["<|endoftext|>"],
    )
    eos_id = tokenizer.reversed_vocab[b"<|endoftext|>"]
    model = Transformer_lm(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        num_heads=config["num_heads"], d_ff=config["d_ff"],
        num_layers=config["num_layers"], context_len=config["context_len"],
        rope_theta=config["rope_theta"], device=config["device"], dtype=torch.float32,
    )
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    iteration = checkpoint["iteration"]
    del checkpoint  # 推理不需要 checkpoint 中的优化器状态。

    prompt_ids = tokenizer.encode(args.prompt)
    torch.manual_seed(args.seed)
    print(f"已加载第 {iteration} 步模型；开始续写……", flush=True)
    torch.mps.synchronize()
    start = time.perf_counter()
    new_ids = decoding(
        prompt_ids, model, args.temperature, args.top_p,
        args.max_new_tokens, config["device"], eos_id,
    )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    completion = tokenizer.decode(new_ids)
    full_text = tokenizer.decode(prompt_ids + new_ids)
    print(full_text)
    print(f"\n新增 {len(new_ids)} token；生成耗时 {elapsed:.2f} 秒。")

    output_dir = run_dir / "generations"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{datetime.now():%Y%m%d-%H%M%S-%f}.json"
    record = {
        **vars(args), "checkpoint": str(run_dir / "checkpoint.pt"),
        "iteration": iteration, "tokenizer_dir": str(tokenizer_dir),
        "device": config["device"], "context_len": config["context_len"],
        "prompt_token_count": len(prompt_ids), "generated_token_count": len(new_ids),
        "stopped_on_eos": bool(new_ids and new_ids[-1] == eos_id),
        "generation_seconds": elapsed, "generated_token_ids": new_ids,
        "completion": completion, "full_text": full_text,
    }
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成记录：{output_path}")
