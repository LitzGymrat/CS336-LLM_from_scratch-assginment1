"""用题目的 ToyModel 观察 autocast 下的数据类型，并测一次完整训练 step。"""

import argparse
import statistics
import timeit

import torch
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        #只记录 dtype，不保留中间张量。
        self.fc1_output_dtype = x.dtype
        x = self.relu(x)
        x = self.ln(x)
        self.ln_output_dtype = x.dtype
        return self.fc2(x)


def train_step(model, inputs, targets, optimizer, scaler, amp_dtype):
    optimizer.zero_grad()
    with torch.autocast(device_type=inputs.device.type, dtype=amp_dtype):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)

    #FP16 配合梯度缩放；BF16 时 scaler 关闭，这三句等价于普通训练。
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return logits, loss


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--in_features", type=int, default=64)
    parser.add_argument("--out_features", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)
    device = torch.device(args.device)
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    torch.manual_seed(0)

    #参数与原始输入保持 FP32，类别标签是整数。
    model = ToyModel(args.in_features, args.out_features).to(device)
    inputs = torch.randn(args.batch_size, args.in_features, device=device, dtype=torch.float32)
    targets = torch.randint(0, args.out_features, (args.batch_size,), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=args.precision == "fp16")

    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        print("模型参数（autocast 内）：")
        for name, parameter in model.named_parameters():
            print(f"  {name}: {parameter.dtype}")

    for _ in range(args.warmup):
        train_step(model, inputs, targets, optimizer, scaler, amp_dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(args.steps):
        start = timeit.default_timer()
        logits, loss = train_step(model, inputs, targets, optimizer, scaler, amp_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((timeit.default_timer() - start) * 1000)

    #打印放在计时循环外。
    print("输入：", inputs.dtype)
    print("标签：", targets.dtype)
    print("fc1 输出：", model.fc1_output_dtype)
    print("LayerNorm 输出：", model.ln_output_dtype)
    print("logits：", logits.dtype)
    print("loss：", loss.dtype)
    print("参数梯度（缩放已还原）：")
    for name, parameter in model.named_parameters():
        print(f"  {name}: {parameter.grad.dtype}")
    print(f"loss 值：{loss.item():.6f}")
    print(f"完整训练 step：{statistics.mean(times):.3f} ± {statistics.stdev(times):.3f} ms")


if __name__ == "__main__":
    main()
