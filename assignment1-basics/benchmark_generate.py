import torch
import time
# 导入你两个版本的 generate 函数（注意调整导入路径）
from generation.generation_optimized import generate as generate_base
from generation.generation_kv import generate as generate_kv

def main():
    prompt = "Once upon a time, there was a little girl named Lily. She loved to"
    max_tokens = 256 # 生成得越长，KV Cache 的加速优势越恐怖
    temperature = 0.8
    top_p = 0.9

    print("========== 1. 测试 Baseline (O(N^2) 重复计算) ==========")
    start_time = time.time()
    # 你的 base 生成函数
    output_base = generate_base(prompt, max_tokens, temperature, top_p)
    base_time = time.time() - start_time
    
    print(f"\n[生成结果]:\n{output_base}\n")
    print(f"[耗时]: {base_time:.2f} 秒")
    print(f"[生成速度]: {max_tokens / base_time:.2f} tokens/s\n")

    print("========== 2. 测试 KV Cache (O(N) 极速缓存) ==========")
    start_time = time.time()
    # 你新写的 kv 生成函数
    output_kv = generate_kv(prompt, max_tokens, temperature, top_p)
    kv_time = time.time() - start_time
    
    print(f"\n[生成结果]:\n{output_kv}\n")
    print(f"[耗时]: {kv_time:.2f} 秒")
    print(f"[生成速度]: {max_tokens / kv_time:.2f} tokens/s")
    
    print("=======================================================")
    print(f"🚀 结论: KV Cache 使长文本生成速度提升了 {base_time / kv_time:.2f} 倍！")

if __name__ == "__main__":
    main()