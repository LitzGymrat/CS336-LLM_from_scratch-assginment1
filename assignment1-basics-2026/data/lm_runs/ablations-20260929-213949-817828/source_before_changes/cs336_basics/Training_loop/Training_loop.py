import torch
import numpy as np
import os
import typing
import wandb
import json
import time
from pathlib import Path
from cs336_basics.Transformer.Transformer_lm import Transformer_lm
from cs336_basics.Transformer.Linear_module import Linear
from cs336_basics.Transformer.Embedding_module import Embedding
from cs336_basics.Transformer.RMSNorm_module import RMSNorm
from cs336_basics.Transformer.Swiglu_module import SWIGLU
from cs336_basics.Transformer.RoPE_module import RoPE
from cs336_basics.Transformer.SoftMax_function import softmax
from cs336_basics.Transformer.Scaled_dot_product_function import scaled_dot_product_attention
from cs336_basics.Transformer.Multihead_self_attention_module import Multihead_self_attention
from cs336_basics.Transformer.Transformer_block import Transformer_blk
from cs336_basics.Training_modules.Crossentropyloss import cross_entropy
from cs336_basics.Training_modules.AdamW import AdamW
from cs336_basics.Training_modules.Learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.Training_modules.Gradient_clipping import gradient_clipping
from cs336_basics.Training_loop.Data_loader import data_loader
from cs336_basics.Training_loop.Checkpoint import save_checkpoint,load_checkpoint

#模型评估
@torch.no_grad()
def evaluate_loss(model, batches):
    model.eval()
    losses = []

    for x, y in batches:
        losses.append(cross_entropy(model(x), y).item())

    model.train()
    return sum(losses) / len(losses)


def main(
    train_src: os.PathLike | typing.BinaryIO | typing.IO[bytes],
    valid_src: os.PathLike | typing.BinaryIO | typing.IO[bytes],
    vocab_size: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    batch_size:int,
    context_len: int,
    rope_theta: float,
    max_lr: float,
    min_lr: float,
    t_w: int,
    t_c: int,
    max_l2_norm:float,
    training_steps:int,
    validate_interval:int,
    checkpoint_interval:int,
    checkpoint_out: os.PathLike | typing.BinaryIO | typing.IO[bytes],
    run_dir: str,
    run_name: str,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.01,
    eps: float = 1e-8,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    seed: int = 42,
    eval_batches: int = 4,
    overfit_one_batch: bool = False,
    wandb_mode: str = "online",
):

    #0.配置这次运行
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    #1.加载数据（使用mmap懒加载）
    train_data = np.load(train_src,mmap_mode="r")
    valid_data = np.load(valid_src,mmap_mode="r")
    #固定评估样本
    fixed_train_batch = data_loader(
        train_data, batch_size, context_len, device
    )

    fixed_valid_batches = [
        data_loader(valid_data, batch_size, context_len, device)
        for _ in range(eval_batches)
    ]
    #2.创建模型#&优化器
    model = Transformer_lm(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        context_len=context_len,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )
    optimizer = AdamW(
        params=model.parameters(),
        betas=betas,
        weight_decay=weight_decay,
        eps=eps,
    )
    run = wandb.init(
        project="cs336-assignment1",
        name=run_name,
        dir=str(run_dir),
        mode=wandb_mode,
        config={
            "vocab_size": vocab_size,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "context_len": context_len,
            "rope_theta": rope_theta,
            "batch_size": batch_size,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "t_w": t_w,
            "t_c": t_c,
            "betas": betas,
            "weight_decay": weight_decay,
            "eps": eps,
            "max_l2_norm": max_l2_norm,
            "training_steps": training_steps,
        },
    )
    run.config.update({
        "seed": seed,
        "eval_batches": eval_batches,
        "overfit_one_batch": overfit_one_batch,
        "train_src": str(train_src),
        "valid_src": str(valid_src),
        "device": str(device),
        "dtype": str(dtype),
        "validate_interval": validate_interval,
        "checkpoint_interval": checkpoint_interval,
    })

    (run_dir / "config.json").write_text(
        json.dumps(dict(run.config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics_file = (run_dir / "metrics.jsonl").open(
        "w", encoding="utf-8"
    )

    run.define_metric("train_loss", step_metric="step")
    run.define_metric("valid_loss", step_metric="step")
    run.define_metric("loss_vs_time/*", step_metric="wall_seconds")

    start_time = time.perf_counter()


    def record_metrics(step, lr):
        train_eval_loss = evaluate_loss(model, [fixed_train_batch])
        valid_eval_loss = evaluate_loss(model, fixed_valid_batches)

        row = {
            "step": step,
            "wall_seconds": time.perf_counter() - start_time,
            "train_loss": train_eval_loss,
            "valid_loss": valid_eval_loss,
            "lr": lr,
            "tokens_seen": step * batch_size * context_len,
        }

        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()

        run.log({
            **row,
            "loss_vs_time/train": train_eval_loss,
            "loss_vs_time/valid": valid_eval_loss,
        }, step=step)

        print(
            f"step={step} time={row['wall_seconds']:.1f}s "
            f"train={train_eval_loss:.4f} "
            f"valid={valid_eval_loss:.4f} lr={lr:.6g}"
        )


    record_metrics(0, 0.0)
    for t in range(training_steps):
        #1.获取学习率
        lr_t = get_lr_cosine_schedule(t,max_lr,min_lr,t_w,t_c)
        #交给本轮优化器
        for group in optimizer.param_groups:
            group["lr"] = lr_t #按照参数组去调整
        #optimizer.load_state_dict({"lr": lr_t})不行，是用来恢复完整状态的(比如训练中断后),需要param_groups和state
        #2.清空梯度
        optimizer.zero_grad()
        #3.进入前向
        if overfit_one_batch:
            sample_batches,next_token_predictions = fixed_train_batch
        else:
            sample_batches,next_token_predictions = data_loader(train_data,batch_size,context_len,device)
        logits = model.forward(sample_batches)       
        #4.计算损失(区分训练和验证损失)
        train_loss = cross_entropy(logits,next_token_predictions)
        #5.反向
        train_loss.backward()
        #更新之前来个梯度截断
        gradient_clipping(model.parameters(),max_l2_norm,eps)
        #6.更新参数
        optimizer.step()
        #处理valid:t表示完成了t+1次训练
        if (t+1) % validate_interval == 0 or t+1 == training_steps:
            record_metrics(t+1, lr_t)
        #处理checkpoint
        if (t+1) % checkpoint_interval == 0 or t+1 == training_steps:
            save_checkpoint(model,optimizer,t+1,checkpoint_out)

    metrics_file.close()
    run.finish()
