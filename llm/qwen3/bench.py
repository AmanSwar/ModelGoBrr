import torch

import time

device = torch.device("cuda")

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

    t0 = time.monotonic()
    outputs = []
    for i in range(iters):
        s = time.monotonic()
        out = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        e = time.monotonic()
        latency = e - s

        in_len = len(tokenizer.encode(prompt))
        gen_len = len(tokenizer.encode(out)) - in_len

        print(
            f"iter {i+1:2d}: latency={latency:.3f}s, tokens/s={gen_len/latency:.2f}"
        )
        outputs.append((latency, out))