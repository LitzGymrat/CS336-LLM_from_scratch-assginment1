"""作业第 30 页给出的 SGD 示例，以及 learning_rate_tuning 实验。"""

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    """沿用作业示例：第 t 次更新使用 lr / sqrt(t + 1)。"""

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


def run_experiment():
    # 所有学习率从同一份初始权重出发，便于比较。
    torch.manual_seed(0)
    initial_weights = 5 * torch.randn((10, 10), dtype=torch.float32, device="cpu")
    learning_rates = [1, 10, 100, 1000]
    rows = []

    for lr in learning_rates:
        weights = torch.nn.Parameter(initial_weights.clone())
        opt = SGD([weights], lr=lr)
        print(f"\nlr={lr}")

        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            loss_value = loss.cpu().item()
            # 与作业一致：先记录 loss，再执行本轮更新。
            rows.append((lr, t, loss_value))
            print(f"已完成 {t:2d} 次更新，loss={loss_value:.10e}")
            loss.backward()
            opt.step()

        # 额外观察第 10 次更新后的结果；不再反向传播或更新。
        with torch.no_grad():
            final_loss = (weights**2).mean().cpu().item()
        rows.append((lr, 10, final_loss))
        print(f"已完成 10 次更新，loss={final_loss:.10e}")

    output_path = Path(__file__).resolve().parents[2] / "docs" / "sgd_learning_rate_tuning.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["learning_rate", "updates_completed", "loss"])
        writer.writerows(rows)
    print(f"\n逐步结果已保存：{output_path}")


if __name__ == "__main__":
    run_experiment()
