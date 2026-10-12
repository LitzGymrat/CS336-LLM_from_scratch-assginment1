from datetime import datetime
from pathlib import Path

import torch

from cs336_basics.Training_loop.Training_loop import main


if __name__ == "__main__":
    #设为 True 可重复进行单 batch 过拟合测试。
    overfit_one_batch = False
    training_steps = 40 if overfit_one_batch else 5000
    run_name = "tinystories-overfit" if overfit_one_batch else "tinystories-train"

    root = Path(__file__).resolve().parent
    data_dir = root / "data" / "tokenized" / "20260916-user-bpe"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = root / "data" / "lm_runs" / f"{run_name}-{timestamp}"

    main(
        train_src=data_dir / "tinystories_train.npy",
        valid_src=data_dir / "tinystories_valid.npy",
        vocab_size=10000,
        d_model=512,
        num_heads=16,
        d_ff=1344,
        num_layers=4,
        batch_size=2 if overfit_one_batch else 32,
        context_len=64 if overfit_one_batch else 256,
        rope_theta=10000,
        max_lr=1e-3,
        min_lr=1e-3 if overfit_one_batch else 1e-4,
        t_w=5 if overfit_one_batch else 100,
        t_c=training_steps-1,
        max_l2_norm=1.0,
        training_steps=training_steps,
        validate_interval=10 if overfit_one_batch else 100,
        checkpoint_interval=1000,
        checkpoint_out=run_dir / "checkpoint.pt",
        run_dir=str(run_dir),
        run_name=run_name,
        betas=(0.9, 0.95),
        weight_decay=0.0 if overfit_one_batch else 0.1,
        eps=1e-8,
        device="mps",
        dtype=torch.float32,
        seed=42,
        eval_batches=4,
        overfit_one_batch=overfit_one_batch,
        wandb_mode="online",
    )
