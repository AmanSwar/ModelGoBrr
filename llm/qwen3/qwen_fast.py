import torch
import torch.nn as nn

from llm.qwen3.config import QwenConfig

from kernels.gqa import GQA_Triton
from kernels.fusedffn import ffn_silu_fwd_triton
from kernels.rmsnorm import RMSNormTriton


class FFN(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()

        self.linear_layer1 = nn.Linear(
            in_features=in_dim, out_features=hidden_dim, bias=False
        )
        self.linear_layerP = nn.Linear(
            in_features=in_dim, out_features=hidden_dim, bias=False
        )
        self.silu = nn.SiLU()
        self.linear_layer2 = nn.Linear(
            in_features=hidden_dim, out_features=in_dim, bias=False
        )

    def forward(self, x):
        x_l = self.linear_layer1(x)
        x_p = self.linear_layerP(x)
        x = self.silu(x_l)
        x = x * x_p
        x = self.linear_layer2(x)
        return x


def compute_rope_params(head_dim, theta_base, context_length, dtype=torch.float32):
    assert head_dim % 2 == 0, "head dim must be divisible by 2"

    inv_freq = 1 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: head_dim // 2].float()
            / head_dim
        )
    )

    position = torch.arange(context_length, dtype=dtype)
    angles = position[:, None] * inv_freq[None, :]

    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


class FusedFFNSilu(nn.Module):

    def __init__(self, embed_dim, hidden_dim, dtype, device):
        super().__init__()
        self.linear_layer1 = (
            nn.Parameter(torch.empty((hidden_dim, embed_dim))).to(dtype).to(device)
        )
        self.linear_layerP = (
            nn.Parameter(torch.empty((hidden_dim, embed_dim))).to(dtype).to(device)
        )
        self.linear_layer2 = (
            nn.Parameter(torch.empty((embed_dim, hidden_dim))).to(dtype).to(device)
        )

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        reshaped_x = x.reshape(-1, embed_dim)
        N = reshaped_x.shape[0]
        out_reshaped = ffn_silu_fwd_triton(
            input_matrix=reshaped_x,
            weight1_matrix=self.linear_layer1.T,
            weight2_matrix=self.linear_layerP.T,
            weight3_matrix=self.linear_layer2.T,
            N=N,
            HIDDEN_DIM=self.hidden_dim,
            EMBED_DIM=self.embed_dim,
        )
        out = out_reshaped.view(batch_size, seq_len, self.embed_dim)
        return out


class Transformer(nn.Module):
    """
    RMS -> attn -> rms -> ffnaaa
    """

    def __init__(self, cfg: QwenConfig , device):
        super().__init__()

        self.attn = GQA_Triton(
            d_in=cfg.embed_dim,
            num_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            n_kv_heads=cfg.n_kv_heads,
            qk_norm=cfg.qk_norm,
            dtype=cfg.dtype,
        )

        self.rms_norm1 = nn.RMSNorm(cfg.embed_dim, eps=1e-6)
        self.rms_norm2 = nn.RMSNorm(cfg.embed_dim, eps=1e-6)

        self.ffn = FFN(cfg.embed_dim, cfg.hidden_dim)

    def forward(self, x, cos, sin):
        x_res = x
        x = self.rms_norm1(x)
        # print("check1")
        x = self.attn(x, cos, sin)
        # print("check2")
        x = x + x_res

        x_res = x
        x = self.rms_norm2(x)
        # print("check3")

        x = self.ffn(x)
        # print("check4")
        x = x + x_res

        return x


class FastQwen3(nn.Module):

    def __init__(self, cfg: QwenConfig , device):

        super().__init__()

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)

        self.transformer_blocs = nn.ModuleList(
            [Transformer(cfg=cfg , device=device) for _ in range(cfg.n_layers)]
        )

        self.final_rmsnorm = nn.RMSNorm(cfg.embed_dim)

        self.out_head = nn.Linear(
            cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype
        )

        if cfg.head_dim is None:
            head_dim = cfg.embed_dim // cfg.n_head

        else:
            head_dim = cfg.head_dim

        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
        )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.cfg = cfg

    def forward(self, x):

        token_embed: torch.Tensor = self.tok_embed(x)
        # print("check5")
        x = token_embed
        num_tokens = x.shape[1]

        for block in self.transformer_blocs:
            x = block(x, self.cos, self.sin)
        # print("check6")
        x = self.final_rmsnorm(x)   
        # print("check7")
        logits = self.out_head(x.to(self.cfg.dtype))
        # print("check8")
        return logits


if __name__ == "__main__":

    from llm.qwen3.qwen_token import Qwen3Tokenizer
    from llm.qwen3.load import load_weights_fastqwen
    from safetensors.torch import load_file
    import os
    import sys
    import time
    torch.manual_seed(696969)
    device = torch.device("cuda")
    repo_dir = "/home/aman/code/model_go_brr/Qwen3-0.6B"
    single_file_path = os.path.join(repo_dir, "model.safetensors")

    config = QwenConfig()
    model : FastQwen3 = FastQwen3(config ,device=device)

    weights_dict = load_file(single_file_path)
    # load_weights_fastqwen(model, config, weights_dict)

    model = model.to(device=device)
    print("model loaded")
    print(model.out_head.weight.dtype)
    # sys.exit(0)
    tokenizer_file_path = "/home/aman/code/model_go_brr/Qwen3-0.6B/tokenizer.json"

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        add_gen_prompt=True,
        add_thinking=True,
    )
    print("tokenizer done")
    PROMPT = "Write a concise, friendly summary of why distributed training matters for large models.\n"

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
                # print(f"pass {_}")

            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)
        et = time.monotonic() - st
        print(f"End time {et}")
        decoded = tokenizer.decode(tokens.squeeze(0).tolist())
        return decoded


    def benchmark_generation(
        model, tokenizer, prompt=PROMPT, warmup=1, iters=5, max_new_tokens=20
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

            print(f"iter {i+1:2d}: latency={latency:.3f}s, tokens/s={gen_len/latency:.2f}")
            outputs.append((latency, out))
            
    print("Start benchmarking")
    benchmark_generation(model , tokenizer)
