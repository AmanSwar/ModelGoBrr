import torch

import time


def generate(model, tokenizer, prompt, max_new_tokens=128):
    tokens = tokenizer.encode(prompt)
    tokens = (
        torch.tensor(tokens).unsqueeze(0).to(device)
    )  # unsequeze to include the batch dimension

    start_pos = tokens.shape[1]
    total_len = start_pos + max_new_tokens
    st = time.monotonic()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)
            # print(f"Pass {_}")

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        tokens = torch.cat((tokens, next_token), dim=1)
    et = time.monotonic() - st
    print(f"End time", et)
    decoded = tokenizer.decode(tokens.squeeze(0).tolist())
    return decoded



def benchmark_generation(
    model, tokenizer, prompt, warmup=1, iters=5, max_new_tokens=20
):
    print("Warming up...")
    for _ in range(warmup):
        _ = generate(model, tokenizer, prompt, max_new_tokens=32)

    total_latency = 0
    total_token_sec = 0
    t0 = time.monotonic()
    outputs = []
    for i in range(iters):
        s = time.monotonic()
        out = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        e = time.monotonic()
        latency = e - s

        in_len = len(tokenizer.encode(prompt))
        gen_len = len(tokenizer.encode(out)) - in_len

        total_latency += latency
        total_token_sec += (gen_len / latency)
        print(
            f"iter {i+1:2d}: latency={latency:.3f}s, tokens/s={gen_len/latency:.2f}"
        )
        outputs.append((latency, out))
    
    print(f"Token : {max_new_tokens}: average latency = {total_latency / iters}  | average token/s = {total_token_sec / iters}")

    return (total_latency/iters , total_token_sec/iters)

if __name__ == "__main__":

    from llm.qwen3.qwen_torch import Qwen3
    from llm.qwen3.fast_qwen_cuda import FastQwen3
    from llm.qwen3.qwen_token import Qwen3Tokenizer
    from llm.qwen3.config import QwenConfig_bfloat16 , QwenConfig_float16

    device = torch.device("cuda")
    tokenizer_file_path = "/home/aman/code/model_go_brr/Qwen3-0.6B/tokenizer.json"

    config_qwen_bf16 =QwenConfig_bfloat16()
    config_qwen_fp16 = QwenConfig_float16()
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        add_gen_prompt=True,
        add_thinking=True,
    )

    model_torch = Qwen3(config_qwen_bf16).to(device)
    model_fast = FastQwen3(config_qwen_fp16).to(device)


    max_new_token_range = [32 , 64 , 128 , 256 , 512 , 1024]

    latency = []
    token_per_sec = []

    print("Start benchmarking")
    for token_size in max_new_token_range:
        pass


    
