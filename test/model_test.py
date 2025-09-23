import torch
import torch.nn as nn

from llm.qwen3.config import QwenConfig

# kernels
from rmsnorm_kernel_vectorized import rmsnorm_kernel_vectorized as rmsnorm_kernel
from rope_cuda import rope_apply_cuda
from flash_attn import flash_attn_func

import sys

# def compute_rope_params(head_dim, theta_base, context_length, dtype=torch.float32):
#     assert head_dim % 2 == 0, "head dim must be divisible by 2"

#     inv_freq = 1 / (
#         theta_base
#         ** (
#             torch.arange(0, head_dim, 2, dtype=dtype)[: head_dim // 2].float()
#             / head_dim
#         )
#     )

#     position = torch.arange(context_length, dtype=dtype)
#     angles = position[:, None] * inv_freq[None, :]

#     angles = torch.cat([angles, angles], dim=1)

#     cos = torch.cos(angles).to(torch.float16)
#     sin = torch.sin(angles).to(torch.float16)

#     return cos, sin


def compute_rope_params(
    head_dim, theta_base, context_length, device="cuda", dtype=torch.float32
):
    assert head_dim % 2 == 0, "head dim must be divisible by 2"
    ar = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    inv_freq = 1.0 / (theta_base ** (ar / head_dim))
    pos = torch.arange(context_length, device=device, dtype=dtype)
    angles = pos[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=1)  # [context_length, head_dim]
    cos = torch.cos(angles).to(torch.float16).contiguous()
    sin = torch.sin(angles).to(torch.float16).contiguous()
    return cos, sin


class RMSNorm_CUDA(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(size=(dimension,), dtype=torch.float16))
        self.eps = eps

    def forward(self, x):
        out = rmsnorm_kernel(x, self.weight, self.eps)

        return out


class GQA_FlashAttn(nn.Module):
    """Grouped Query Attention - Reference Implementation"""

    def __init__(
        self,
        d_in: int,
        num_heads: int,
        n_kv_heads: int,
        head_dim: int | None = None,
        qk_norm: bool = True,
        dtype=None,
    ):
        super().__init__()

        assert num_heads % n_kv_heads == 0, "Num heads is not divisible by num kv grps"

        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads
        self.num_kv_grps = num_heads // n_kv_heads

        if head_dim is None:
            assert (
                d_in % num_heads == 0
            ), "in dimension must be divisible by number of heads"
            head_dim = d_in // num_heads

        self.head_dim: int = head_dim
        self.d_out = self.head_dim * self.num_heads

        self.Wq = nn.Linear(
            in_features=d_in, out_features=self.d_out, bias=False, dtype=dtype
        )
        self.Wk = nn.Linear(
            in_features=d_in,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        self.Wv = nn.Linear(
            in_features=d_in,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )

        self.out_projection = nn.Linear(
            in_features=self.d_out, out_features=d_in, bias=False, dtype=dtype
        )

        if qk_norm:
            # FIXED: Use LayerNorm to match FlashAttn implementation
            self.q_norm = RMSNorm_CUDA(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm_CUDA(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, cos, sin):
        bs, seq_len, _ = x.shape

        Q: torch.Tensor = self.Wq(x)
        K: torch.Tensor = self.Wk(x)
        V: torch.Tensor = self.Wv(x)

        Q = Q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm:
            Q = self.q_norm(Q)

        if self.k_norm:
            K = self.k_norm(K)

        Q = rope_apply_cuda(Q, cos, sin)
        K = rope_apply_cuda(K, cos, sin)

        K = K.repeat_interleave(self.num_kv_grps, dim=1)
        V = V.repeat_interleave(self.num_kv_grps, dim=1)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        attn_out = flash_attn_func(Q, K, V, causal=True)

        attn_out = attn_out.reshape(bs, seq_len, self.d_out)
        return self.out_projection(attn_out)


class FFN(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()

        self.linear_layer1 = nn.Linear(
            in_features=in_dim, out_features=hidden_dim, bias=False ,dtype=torch.float16
        )
        self.linear_layerP = nn.Linear(
            in_features=in_dim, out_features=hidden_dim, bias=False, dtype=torch.float16
        )
        self.silu = nn.SiLU()
        self.linear_layer2 = nn.Linear(
            in_features=hidden_dim, out_features=in_dim, bias=False, dtype=torch.float16
        )

    def forward(self, x):
        x_l = self.linear_layer1(x)
        x_p = self.linear_layerP(x)
        x = self.silu(x_l)
        x = x * x_p
        x = self.linear_layer2(x)
        return x


class Transformer(nn.Module):
    """
    RMS -> attn -> rms -> ffnaaa
    """

    def __init__(self, cfg: QwenConfig):
        super().__init__()

        self.attn = GQA_FlashAttn(
            d_in=cfg.embed_dim,
            num_heads=cfg.n_heads,
            head_dim=cfg.head_dim,
            n_kv_heads=cfg.n_kv_heads,
            qk_norm=cfg.qk_norm,
            dtype=cfg.dtype,
        )

        self.rms_norm1 = RMSNorm_CUDA(cfg.embed_dim, eps=1e-6)
        self.rms_norm2 = RMSNorm_CUDA(cfg.embed_dim, eps=1e-6)

        self.ffn = FFN(cfg.embed_dim, cfg.hidden_dim)

    def forward(self, x, cos, sin):
        # print(x.shape)
        x_res = x
        x = self.rms_norm1(x)
        x = self.attn(x, cos, sin)

        x = x + x_res

        x_res = x
        x = self.rms_norm2(x)
        # print(x.shape)
        x = self.ffn(x)
        x = x + x_res

        return x


class Qwen3(nn.Module):

    def __init__(self, cfg: QwenConfig):

        super().__init__()

        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim, dtype=cfg.dtype)

        self.transformer_blocs = nn.ModuleList(
            [Transformer(cfg=cfg) for _ in range(cfg.n_layers)]
        )

        self.final_rmsnorm = RMSNorm_CUDA(cfg.embed_dim)

        self.out_head = nn.Linear(
            cfg.embed_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype
        )

        if cfg.head_dim is None:
            head_dim = cfg.embed_dim // cfg.n_head

        else:
            head_dim = cfg.head_dim

        self.cos, self.sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
        )


        # self.register_buffer("cos", self.cos, persistent=False)
        # self.register_buffer("sin", self.sin, persistent=False)

        self.cfg = cfg

    def forward(self, x):
        # print(self.cos.shape)
        # print(x.shape)
        N = x.shape[-1]
        # print(x.shape[-1])
        cos_cache, sin_cache = self.cos[:N], self.sin[:N]
        # print(cos_cache.shape)
        # sys.exit(0)

        token_embed: torch.Tensor = self.tok_embed(x)

        x = token_embed
        num_tokens = x.shape[1]

        mask = torch.triu(
            torch.ones(
                num_tokens, num_tokens, device=token_embed.device, dtype=torch.bool
            ),
            diagonal=1,
        )

        # assert x.dtype ==     6 , "input not in bfloat16"
        # x = x.to(torch.bfloat16)
        for block in self.transformer_blocs:
            x = block(x, cos_cache, sin_cache)

        x = self.final_rmsnorm(x)

        logits = self.out_head(x)
        return logits


if __name__ == "__main__":

    from llm.qwen3.qwen_token import Qwen3Tokenizer
    import time
    # from llm.qwen3.load import load_weights_qwen
    import os
    # from llm.qwen3.hf_load import load_file

    repo_dir = "/home/aman/code/model_go_brr/Qwen3-0.6B"
    torch.manual_seed(696969)
    device = torch.device("cuda")
    single_file_path = os.path.join(repo_dir, "model.safetensors")

    # weights_dict = load_file(single_file_path)
    config = QwenConfig()
    model = Qwen3(config).to(device)
    # load_weights_qwen(model, config, weights_dict)
    tokenizer_file_path = "/home/aman/code/model_go_brr/Qwen3-0.6B/tokenizer.json"

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        add_gen_prompt=True,
        add_thinking=True,
    )
    model = model.to(device)

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
                # print(f"Pass {_}")

            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            tokens = torch.cat((tokens, next_token), dim=1)
        et = time.monotonic() - st
        print(f"End time", et)
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

            print(
                f"iter {i+1:2d}: latency={latency:.3f}s, tokens/s={gen_len/latency:.2f}"
            )
            outputs.append((latency, out))

    print("Start benchmarking")
    benchmark_generation(model, tokenizer)
