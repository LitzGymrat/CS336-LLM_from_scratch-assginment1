"""串行运行作业 7.3 的 TinyStories 消融；每个实验使用独立子进程。"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from datetime import datetime


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "data/lm_runs/tinystories-train-20260929-124145-511697"
EXPERIMENTS = [
    {"name": "no-rmsnorm-lr1e-3", "norm_position": "none", "max_lr": 1e-3},
    {"name": "post-norm", "norm_position": "post", "max_lr": 1e-3},
    {"name": "no-rope", "use_rope": False, "max_lr": 1e-3},
    {"name": "silu", "ffn_type": "silu", "max_lr": 1e-3},
    {"name": "no-rmsnorm-lr3e-4", "norm_position": "none", "max_lr": 3e-4},
    {"name": "no-rmsnorm-lr1e-4", "norm_position": "none", "max_lr": 1e-4},
]


def write_json(path, value):
    #读进度时不会读到写了一半的 JSON。
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_metrics(run_dir):
    path = run_dir / "metrics.jsonl"
    if not path.exists():  #尚未开始的实验还没有日志。
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def run_worker(queue_dir, index):
    import torch
    from cs336_basics.Training_loop.Training_loop import main

    manifest = json.loads((queue_dir / "manifest.json").read_text())
    experiment = manifest["experiments"][index]
    config = dict(manifest["baseline_config"])
    config.update(
        norm_position=experiment.get("norm_position", "pre"),
        use_rope=experiment.get("use_rope", True),
        ffn_type=experiment.get("ffn_type", "swiglu"),
        max_lr=experiment["max_lr"],
        min_lr=experiment["max_lr"] / 10,
        dtype=torch.float32,
        wandb_group=queue_dir.name,
        wandb_mode="offline" if manifest["smoke"] else "online",
    )
    if config["ffn_type"] == "silu":
        config["d_ff"] = 4 * config["d_model"]
    if manifest["smoke"]:
        config.update(training_steps=3, batch_size=2, context_len=32,
                      t_w=1, t_c=2, validate_interval=1,
                      checkpoint_interval=2, eval_batches=1)
    run_dir = queue_dir / experiment["name"]
    config.update(run_dir=str(run_dir), run_name=experiment["name"],
                  checkpoint_out=run_dir / "checkpoint.pt")
    try:
        main(**config)
    except FloatingPointError as error:
        write_json(run_dir / "failure.json", {"kind": "nonfinite", "message": str(error)})
        print(str(error), flush=True)
        return 2
    return 0


def summarize(queue_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = json.loads((queue_dir / "manifest.json").read_text())
    status = json.loads((queue_dir / "status.json").read_text())
    baseline = Path(manifest["baseline_dir"])
    baseline_rows = read_metrics(baseline)
    baseline_loss = baseline_rows[-1]["valid_loss"]
    curves = {"baseline": baseline_rows}
    lines = ["# TinyStories 消融队列结果", "",
             f"队列状态：`{status['state']}`。本文件在每组结束后自动更新。", "",
             f"正式基线验证损失 **{baseline_loss:.4f}**。下表与曲线中的训练损失来自固定 1 个训练 batch，验证损失来自固定 4 个验证 batch（正式实验为 32,768 token），不是全验证集遍历。", ""]
    if manifest["smoke"]:
        lines += ["**这是 3 步离线冒烟测试，不是正式消融结论，且不与完整基线比较优劣。**", ""]
    lines += ["| 实验 | 状态 | 最后记录步数 | 训练 loss | 验证 loss | 时间/分钟 |",
              "|---|---|---:|---:|---:|---:|"]
    completed = {}
    for experiment in status["experiments"]:
        name = experiment["name"]
        rows = read_metrics(queue_dir / name)
        curves[name] = rows
        if rows:
            last = rows[-1]
            lines.append(f"| [{name}]({name}/) | {experiment['state']} | {last['step']} | {last['train_loss']:.4f} | {last['valid_loss']:.4f} | {last['wall_seconds']/60:.2f} |")
            if experiment["state"] == "completed":
                completed[name] = last["valid_loss"]
        else:
            lines.append(f"| {name} | {experiment['state']} | — | — | — | — |")
        if experiment.get("error"):
            lines += ["", f"`{name}`：{experiment['error']}", ""]

    panels = [
        ("Remove RMSNorm", ["no-rmsnorm-lr1e-3", "no-rmsnorm-lr3e-4", "no-rmsnorm-lr1e-4"]),
        ("Pre-norm vs. post-norm", ["post-norm"]),
        ("RoPE vs. NoPE", ["no-rope"]),
        ("SwiGLU vs. SiLU", ["silu"]),
    ]
    for key, filename, xlabel, scale in (
        ("step", "loss_by_step.png", "Completed updates", 1),
        ("wall_seconds", "loss_by_time.png", "Wall-clock minutes", 60),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (title, names) in zip(axes.flat, panels):
            for name in ["baseline", *names]:
                rows = [row for row in curves[name] if all(math.isfinite(row[k]) for k in ("train_loss", "valid_loss"))]
                if not rows:
                    continue
                xs = [row[key] / scale for row in rows]
                line, = ax.plot(xs, [row["valid_loss"] for row in rows], label=name, linewidth=1.8)
                ax.plot(xs, [row["train_loss"] for row in rows], color=line.get_color(), linestyle="--", alpha=0.65)
            ax.set(title=title, xlabel=xlabel, ylabel="Cross entropy (nats/token, log scale)")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle("TinyStories ablations — solid: validation; dashed: training", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(queue_dir / filename, dpi=160)
        plt.close(fig)
    lines += ["", "## 学习曲线", "", "![按更新步数](loss_by_step.png)", "", "![按墙钟时间](loss_by_time.png)", "",
              "实线为验证损失，虚线为训练损失；纵轴使用对数刻度。未完成或失败实验只绘制已有的有限值记录，不外推后续表现。", ""]
    if completed and not manifest["smoke"]:
        lines += ["## 本轮观察", ""]
        no_norm = {name: loss for name, loss in completed.items() if name.startswith("no-rmsnorm")}
        if no_norm:
            best = min(no_norm, key=no_norm.get)
            lines += [f"移除 RMSNorm 的已完成候选中，最低最终验证损失来自 `{best}`：**{no_norm[best]:.4f}**。这里的最好仅指本轮有限候选，不代表全局最优学习率。", ""]
        for name in ("post-norm", "no-rope", "silu"):
            if name in completed:
                delta = completed[name] - baseline_loss
                lines += [f"- `{name}`：最终验证损失 {completed[name]:.4f}，相对基线变化 {delta:+.4f}（越低越好）。"]
        lines += ["", "以上是相同 token 预算、单个随机种子下的观察；改变结构后，最合适的学习率可能也改变。本轮仅对移除 RMSNorm 额外比较了较低学习率。", ""]
    lines += ["## 实验约定与产物", "",
              "- 正式实验均从头初始化：batch=32、context=256、5,000 步，共 40.96M token；seed=42，采样顺序与固定评估样本沿用基线。",
              "- no-rmsnorm 移除块内以及 LM head 前的全部 RMSNorm；post-norm 只改变块内归一化位置，保留输出前归一化。",
              "- NoPE 仅去掉 Q/K 的 RoPE，保留因果掩码。SiLU 去掉第三个门控投影，d_ff 从 1344 改为 2048，近似匹配 FFN 参数量。",
              "- 每组的 config.json、metrics.jsonl、run_info.json、checkpoint.pt 位于对应目录。失败原因保存在日志及 failure.json（非有限损失时）。",
              "- 完整队列与代码哈希见 [manifest.json](manifest.json)，运行源码快照在 source/，各组控制台输出在同名 .log 中。",
              "- 不覆盖原来的基线 checkpoint；本队列没有进行学习率发散边界、batch size sweep 或 OWT 实验。", ""]
    (queue_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_queue(queue_dir, smoke):
    queue_dir.mkdir(parents=True, exist_ok=False)
    source = queue_dir / "source"
    package = source / "cs336_basics"
    package.mkdir(parents=True)
    shutil.copy2(__file__, source / Path(__file__).name)
    shutil.copy2(ROOT / "cs336_basics/__init__.py", package / "__init__.py")
    for directory in ("Transformer", "Training_modules", "Training_loop"):
        shutil.copytree(ROOT / "cs336_basics" / directory, package / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in source.rglob("*.py")}
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "baseline_dir": str(BASELINE),
        "baseline_config": json.loads((BASELINE / "config.json").read_text()),
        "experiments": EXPERIMENTS,
        "source_sha256": hashes,
        "smoke": smoke,
    }
    write_json(queue_dir / "manifest.json", manifest)
    status = {"pid": os.getpid(), "state": "running", "experiments": [
        {"name": experiment["name"], "state": "pending"} for experiment in EXPERIMENTS
    ]}
    write_json(queue_dir / "status.json", status)
    snapshot_script = source / Path(__file__).name

    def refresh_report():
        with (queue_dir / "report.log").open("a") as log:
            result = subprocess.run(
                ["uv", "run", "--no-project", "--with", "matplotlib", "python",
                 str(snapshot_script), "--summarize", "--queue-dir", str(queue_dir)],
                stdout=log, stderr=subprocess.STDOUT,
            )
        status["report_exit_code"] = result.returncode
        write_json(queue_dir / "status.json", status)

    def stop_queue(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_queue)
    refresh_report()
    for index, experiment in enumerate(status["experiments"]):
        experiment.update(state="running", started_at=datetime.now().astimezone().isoformat())
        print(f"START {index+1}/{len(EXPERIMENTS)} {experiment['name']}", flush=True)
        with (queue_dir / f"{experiment['name']}.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-B", "-u", str(snapshot_script), "--worker", str(index),
                 "--queue-dir", str(queue_dir)],
                cwd=source, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            experiment["pid"] = process.pid
            write_json(queue_dir / "status.json", status)
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                experiment["state"] = "interrupted"
                status["state"] = "interrupted"
                write_json(queue_dir / "status.json", status)
                refresh_report()
                return 130
        experiment.update(returncode=returncode, finished_at=datetime.now().astimezone().isoformat())
        experiment["state"] = "completed" if returncode == 0 else "nonfinite" if returncode == 2 else "failed"
        run_dir = queue_dir / experiment["name"]
        failure = run_dir / "failure.json"
        if failure.exists():
            experiment["error"] = json.loads(failure.read_text())["message"]
        elif returncode != 0:
            experiment["error"] = f"退出码 {returncode}；请查看 {experiment['name']}.log。"
        rows = read_metrics(run_dir)
        if rows:
            experiment["last_metrics"] = rows[-1]
        write_json(queue_dir / "status.json", status)
        refresh_report()
        print(f"END {experiment['name']}: {experiment['state']}", flush=True)
    status["state"] = "finished" if all(e["state"] == "completed" for e in status["experiments"]) else "finished_with_failures"
    status["finished_at"] = datetime.now().astimezone().isoformat()
    write_json(queue_dir / "status.json", status)
    refresh_report()
    print(f"QUEUE {status['state']}: {queue_dir / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path)
    parser.add_argument("--smoke", action="store_true", help="仅运行每组 3 步，离线检查队列")
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--summarize", action="store_true", help="从已保存的日志更新报告")
    args = parser.parse_args()
    queue_dir = args.queue_dir
    if queue_dir is None:
        if args.worker is not None or args.summarize:
            parser.error("worker/summarize 需要 --queue-dir")
        label = "ablation-smoke" if args.smoke else "ablations"
        queue_dir = ROOT / "data/lm_runs" / f"{label}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    queue_dir = queue_dir.resolve()
    if args.worker is not None:
        sys.exit(run_worker(queue_dir, args.worker))
    elif args.summarize:
        summarize(queue_dir)
    else:
        sys.exit(run_queue(queue_dir, args.smoke))
