"""在正式实验的形状上核对 eager / compiled 的输出和梯度。"""

import gc
import json
from pathlib import Path

import torch

from cs336_basics.Transformer.Scaled_dot_product_function import scaled_dot_product_attention
from cs336_basics.Transformer.Transformer_lm import Transformer_lm
from cs336_basics.Training_modules.Crossentropyloss import cross_entropy


def compare(actual, expected, *, rtol, atol):
    actual = actual.detach().cpu()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return (actual - expected).abs().max().item()


def verify_attention():
    torch.manual_seed(0)
    shape = (8, 8192, 128)
    q, k, v = [torch.randn(shape, device="cuda", requires_grad=True) for _ in range(3)]
    output_gradient = torch.randn_like(q)
    output = scaled_dot_product_attention(q, k, v)
    expected_output = output.detach().cpu()
    output.backward(output_gradient)
    expected_gradients = [tensor.grad.cpu() for tensor in (q, k, v)]
    del output
    for tensor in (q, k, v):
        tensor.grad = None

    attention = torch.compile(scaled_dot_product_attention, backend="inductor", mode="default")
    output = attention(q, k, v)
    output_error = compare(output, expected_output, rtol=1e-4, atol=1e-5)
    output.backward(output_gradient)
    gradient_errors = []
    for tensor, expected in zip((q, k, v), expected_gradients):
        gradient_errors.append(compare(tensor.grad, expected, rtol=1e-3, atol=1e-6))
    return {"case": "attention_b8_s8192_d128", "status": "passed",
            "output_max_abs_error": output_error, "gradient_max_abs_errors": gradient_errors}


def verify_transformer():
    torch.manual_seed(0)
    model = Transformer_lm(vocab_size=10000, d_model=1024, d_ff=4096,
                           num_layers=24, num_heads=16, context_len=512,
                           rope_theta=10000.0, device=torch.device("cuda"), dtype=torch.float32)
    inputs = torch.randint(10000, (4, 512), device="cuda")
    targets = torch.randint(10000, (4, 512), device="cuda")
    logits = model(inputs)
    loss = cross_entropy(logits, targets)
    expected_logits = logits.detach().cpu()
    expected_loss = loss.detach().cpu()
    loss.backward()
    expected_gradients = {name: parameter.grad.cpu() for name, parameter in model.named_parameters()}
    del logits, loss
    model.zero_grad(set_to_none=True)

    compiled_model = torch.compile(model, backend="inductor", mode="default")
    logits = compiled_model(inputs)
    loss = cross_entropy(logits, targets)
    logits_error = compare(logits, expected_logits, rtol=1e-4, atol=1e-5)
    loss_error = compare(loss, expected_loss, rtol=1e-4, atol=1e-5)
    loss.backward()
    largest_gradient_error = 0.0
    for name, parameter in model.named_parameters():
        error = compare(parameter.grad, expected_gradients[name], rtol=1e-3, atol=1e-6)
        largest_gradient_error = max(largest_gradient_error, error)
    return {"case": "medium_b4_s512", "status": "passed",
            "logits_max_abs_error": logits_error, "loss_abs_error": loss_error,
            "gradient_max_abs_error": largest_gradient_error,
            "parameter_tensors_checked": len(expected_gradients)}


def main():
    torch.set_float32_matmul_precision("highest")
    results = []
    output = Path("outputs/torch_compile/verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    for verify in (verify_attention, verify_transformer):
        print(f"开始核对：{verify.__name__}", flush=True)
        result = verify()
        results.append(result)
        output.write_text(json.dumps(results, indent=2) + "\n")
        print(result, flush=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
