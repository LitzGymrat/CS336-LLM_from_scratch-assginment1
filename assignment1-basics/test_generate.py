from generation.generation_optimized import generate

prompt = "Once upon a time, there was a little girl named Lily. She loved to"
print(generate(prompt, max_tokens=100, temperature=0.8, top_p=0.9))