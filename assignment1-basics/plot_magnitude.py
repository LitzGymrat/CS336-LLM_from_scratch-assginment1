import torch
import numpy as np
import matplotlib.pyplot as plt
from Transformer_lm.transformer_lm import Transformer_lm as Base_Model
from Transformer_lm.transformer_lm_attn_res import Transformer_lm as AttnRes_Model
# 这里假设你把两个模型放在了不同的文件里，注意导入路径

@torch.no_grad()
def get_layer_magnitudes(model, dummy_input):
    model.eval()
    # 我们需要稍微侵入一下模型的 forward 过程，把 hidden_states 拿出来
    # 对于 Base 模型，如果你没有保存 hidden_states，你可以快速重写一个只用于测试的 forward
    
    # 假设你的模型 forward 能返回 hidden_states 列表
    # hidden_states 是一个 list，包含每一层的输出 [batch, seq, d_model]
    
    # 因为你只有 4 层，我们记录每层的 L2 Norm 的平均值
    magnitudes = []
    
    # 伪代码：你需要修改你的模型，让它 return hidden_states
    _, hidden_states = model(dummy_input, return_hidden_states=True) 
    
    for h in hidden_states:
        # 计算 L2 Norm，并在 batch 和 seq 维度上求均值
        mag = torch.norm(h, p=2, dim=-1).mean().item()
        magnitudes.append(mag)
        
    return magnitudes

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randint(0, 10000, (1, 256)).to(device) # 随便造一个句子

    # 1. 测 Baseline
    base_model = Base_Model(10000, 256, 512, 4, 16, 1344, 10000).to(device)
    base_model.load_state_dict(torch.load("/root/autodl-tmp/CS336-LLM_from_scratch-assginment1/outputs/baseline/checkpoints/ckpt_step_5000.pt")['model'])
    base_mags = get_layer_magnitudes(base_model, dummy_input)

    # 2. 测 AttnRes
    attn_model = AttnRes_Model(10000, 256, 512, 4, 16, 1344, 10000).to(device)
    attn_model.load_state_dict(torch.load("/root/autodl-tmp/CS336-LLM_from_scratch-assginment1/assignment1-basics/outputs/attnres_exp/checkpoints_attn_res/ckpt_step_5000.pt")['model'])
    attn_mags = get_layer_magnitudes(attn_model, dummy_input)

    # 3. 画图！复刻 Kimi 论文的 Figure 5(b)
    plt.figure(figsize=(8, 5))
    layers = np.arange(len(base_mags))
    plt.plot(layers, base_mags, marker='o', label='Baseline', color='#3498db', linewidth=2)
    plt.plot(layers, attn_mags, marker='s', label='Attention Residuals', color='#e74c3c', linewidth=2)
    
    plt.title("Output Magnitude across Transformer Blocks")
    plt.xlabel("Transformer Block Index")
    plt.ylabel("Output Magnitude (L2 Norm)")
    plt.xticks(layers)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig("magnitude_comparison.png", dpi=300)
    print("图表已保存为 magnitude_comparison.png！")

if __name__ == "__main__":
    main()