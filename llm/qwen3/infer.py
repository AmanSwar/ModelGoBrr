import torch
import torch.nn as nn

from qwen3.qwen_torch import Qwen3
from qwen3.config import QwenConfig
from utils import model_mem_size
from qwen3.load import load_weights
from qwen3.qwen_token import Qwen3Tokenizer


from pathlib import Path
import os
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import time



torch.manual_seed(696969)
device = torch.device("cuda")


config = QwenConfig()
model = Qwen3(config).to(device)

# checking the model
# print(model(torch.tensor([1,2,3]).unsqueeze(0).to(device=device)))

print(f"float32 (PyTorch default): {model_mem_size(model, input_dtype=torch.float32):.2f} GB")
print(f"bfloat16: {model_mem_size(model, input_dtype=torch.bfloat16):.2f} GB")

repo_dir = "/home/aman/code/model_go_brr/llm/Qwen3-0.6B"
single_file_path = os.path.join(repo_dir, "model.safetensors")
weights_dict = load_file(single_file_path)
load_weights(model, config, weights_dict)
model.to(device)
print("Model loaded successfully!")

tokenizer_file_path = "/home/aman/code/model_go_brr/llm/Qwen3-0.6B/tokenizer.json"

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    add_gen_prompt=True,
    add_thinking=True,
)
PROMPT = "Write a concise, friendly summary of why distributed training matters for large models.\n"


def generate(model, tokenizer, prompt, max_new_tokens=128):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    start_pos = tokens.shape[1]
    total_len = start_pos + max_new_tokens

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)
        
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        tokens = torch.cat((tokens, next_token), dim=1)

    decoded = tokenizer.decode(tokens.squeeze(0).tolist())
    return decoded

def benchmark_generation(model, tokenizer, prompt=PROMPT, warmup=1, iters=8, max_new_tokens=128):
    print("Warming up...")
    for _ in range(warmup):
        _ = generate(model, tokenizer, prompt, max_new_tokens=32)

    t0 = time.monotonic()
    outputs = []
    for i in range(iters):
        s = time.monotonic()
        out = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        e = time.monotonic()
        latency = e - s
        
        in_len = len(tokenizer.encode(prompt))
        gen_len = len(tokenizer.encode(out)) - in_len
        
        print(f"iter {i+1:2d}: latency={latency:.3f}s, tokens/s={gen_len/latency:.2f}")
        outputs.append((latency, out))
        
    t1 = time.monotonic()
    avg_latency = sum([l for l,_ in outputs]) / len(outputs)
    
    total_tokens = sum(len(tokenizer.encode(o)) - len(tokenizer.encode(prompt)) for _, o in outputs)
    total_time = sum(l for l, _ in outputs)
    avg_tokens_per_sec = total_tokens / total_time
    
    print(f"\nAverage latency per generation: {avg_latency:.3f}s")
    print(f"Average tokens/s: {avg_tokens_per_sec:.2f}")
    
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated:", torch.cuda.memory_allocated()/1024**2, "MB")
        print("torch.cuda.max_memory_allocated:", torch.cuda.max_memory_allocated()/1024**2, "MB")
        
    return outputs

if __name__ == "__main__":
    outs = benchmark_generation(model, tokenizer)
    print("Sample output (truncated):\n", outs[-1][1][:600])
