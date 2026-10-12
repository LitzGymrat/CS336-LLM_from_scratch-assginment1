"""只在采集期间给已有注意力调用贴 NVTX 标签，不改变计算。"""

from contextlib import ExitStack, contextmanager
from functools import wraps
from unittest.mock import patch

import torch.cuda.nvtx as nvtx
import cs336_basics.Transformer.Scaled_dot_product_function as attention_function
from cs336_basics.Transformer.Multihead_self_attention_module import Multihead_self_attention


def labelled_call(function, name):
    @wraps(function)
    def call(*args, **kwargs):
        with nvtx.range(name):
            return function(*args, **kwargs)
    return call


@contextmanager
def attention_ranges(model, enabled):
    # benchmark 已检查 CUDA；关闭标注时直接执行原来的调用。
    with ExitStack() as stack:
        if enabled:
            for module in model.modules():
                if isinstance(module, Multihead_self_attention):
                    stack.enter_context(patch.object(
                        module, "forward", labelled_call(module.forward, "attention")
                    ))
            stack.enter_context(patch.object(
                attention_function, "softmax",
                labelled_call(attention_function.softmax, "attention_softmax")
            ))
        yield
