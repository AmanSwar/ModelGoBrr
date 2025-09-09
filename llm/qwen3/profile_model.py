import torch
import torch.nn as nn

from llm.qwen3.qwen_torch import Qwen3
from llm.qwen3.qwen_optim import FastQwen3
from llm.qwen3.config import QwenConfig

from llm.qwen3.load import load_weights_fastqwen , load_weights_qwen
from llm.qwen3.qwen_token import Qwen3Tokenizer


from pathlib import Path
import os
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
import time
import torch.profiler

torch.manual_seed(696969)
device = torch.device("cuda")

config = QwenConfig()
# model : FastQwen3 = FastQwen3(config ,device=device)
model = Qwen3(config)

repo_dir = "/home/aman/code/model_go_brr/Qwen3-0.6B"
single_file_path = os.path.join(repo_dir, "model.safetensors")
weights_dict = load_file(single_file_path)
# load_weights_fastqwen(model, config, weights_dict)
load_weights_qwen(model, config, weights_dict)
model.to(device)
print("Model loaded successfully!")

tokenizer_file_path = "/home/aman/code/model_go_brr/Qwen3-0.6B/tokenizer.json"

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

if __name__ == "__main__":
    # Option 1: Simple profiling with explicit control
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        ) as prof:
            print("warmup")
            for i in range(2):
                generate(model, tokenizer, PROMPT, max_new_tokens=2)

            print("profile")
            for i in range(3):
                generate(model, tokenizer, PROMPT, max_new_tokens=2)

        # Export after context manager closes
        print("saving")
        prof.export_chrome_trace("profile_trace_qwen.json")
        print("Profiling complete. Chrome trace saved to profile_trace.json")

    except Exception as e:
        print(f"Profiling error: {e}")
