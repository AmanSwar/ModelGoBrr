import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "Write a concise, friendly summary of why distributed training matters for large models.\n"

def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    print("Loading model... this can take a minute")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",   
        trust_remote_code=True,  
    )
    return tokenizer, model

def benchmark_generation(tokenizer, model, prompt=PROMPT, warmup=1, iters=8, max_new_tokens=128):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer) 
    print("Warming up...")
    for _ in range(warmup):
        _ = pipe(prompt, max_new_tokens=32, do_sample=False)

    t0 = time.monotonic()
    outputs = []
    for i in range(iters):
        s = time.monotonic()
        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        e = time.monotonic()
        latency = e - s
        in_len = len(tokenizer(prompt)["input_ids"])
        gen_len = len(out[0]["generated_text"].strip().split())  # rough word count (coarse)
        print(f"iter {i+1:2d}: latency={latency:.3f}s")
        outputs.append((latency, out[0]["generated_text"]))
    t1 = time.monotonic()
    avg_latency = sum([l for l,_ in outputs]) / len(outputs)
    print(f"\nAverage latency per generation: {avg_latency:.3f}s")
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
        print("torch.cuda.max_memory_allocated:", torch.cuda.max_memory_allocated()/1024**2, "MB")
    return outputs

if __name__ == "__main__":
    tokenizer, model = load_model()
    outs = benchmark_generation(tokenizer, model)
    print("Sample output (truncated):\n", outs[-1][1][:600])
