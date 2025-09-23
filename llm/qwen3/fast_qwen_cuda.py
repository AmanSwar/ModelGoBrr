import torch
import torch.nn as nn

from llm.qwen3.config import QwenConfig_float16

# kernels
from rmsnorm_kernel_vectorized import rmsnorm_kernel_vectorized as rmsnorm_kernel
from rope_cuda import rope_apply_cuda
from flash_attn import flash_attn_func

import sys
from typing import Optional, Tuple, List


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


class KVCache:

    def __init__(
        self,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_seq_len = max_seq_len

        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        self.dtype = dtype
        self.device = device

        self.k_cache = torch.zeros(
            (1, n_kv_heads, max_seq_len, head_dim), dtype=dtype, device=device
        )
        self.v_cache = torch.zeros(
            (1, n_kv_heads, max_seq_len, head_dim), dtype=dtype, device=device
        )
        self.cache_len = 0

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, n_heads, seq_len, head_dim = k.shape

        end_pos = self.cache_len + seq_len
        self.k_cache[:, :, self.cache_len : end_pos] = k
        self.v_cache[:, :, self.cache_len : end_pos] = v

        self.cache_len = end_pos

        return (
            self.k_cache[:, :, : self.cache_len].contiguous(),
            self.v_cache[:, :, : self.cache_len].contiguous(),
        )

    def reset(self):

        self.cache_len = 0

        self.k_cache.zero_()
        self.v_cache.zero_()


class RMSNorm_CUDA(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(size=(dimension,), dtype=torch.float16))
        self.eps = eps

    def forward(self, x):
        out = rmsnorm_kernel(x, self.weight, self.eps)
        return out


class GQA_FlashAttn(nn.Module):

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
            self.q_norm = RMSNorm_CUDA(self.head_dim, eps=1e-6)
            self.k_norm = RMSNorm_CUDA(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self,
        x,
        cos,
        sin,
        kv_cache: Optional[KVCache] = None,
        cache_position: Optional[int] = None,
    ):
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

        if cache_position is not None and kv_cache is not None:
            pos_cos = cos[cache_position : cache_position + seq_len]
            pos_sin = sin[cache_position : cache_position + seq_len]
            Q = rope_apply_cuda(Q, pos_cos, pos_sin)
            K = rope_apply_cuda(K, pos_cos, pos_sin)
        else:
            Q = rope_apply_cuda(Q, cos, sin)
            K = rope_apply_cuda(K, cos, sin)

        if kv_cache is not None:
            K, V = kv_cache.update(K, V)

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
            in_features=in_dim, out_features=hidden_dim, bias=False, dtype=torch.float16
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
    Transformer block with KV cache support
    """

    def __init__(self, cfg: QwenConfig_float16):
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

    def forward(
        self,
        x,
        cos,
        sin,
        kv_cache: Optional[KVCache] = None,
        cache_position: Optional[int] = None,
    ):
        x_res = x
        x = self.rms_norm1(x)
        x = self.attn(x, cos, sin, kv_cache, cache_position)

        x = x + x_res

        x_res = x
        x = self.rms_norm2(x)
        x = self.ffn(x)
        x = x + x_res

        return x


class Qwen3(nn.Module):

    def __init__(self, cfg: QwenConfig_float16):

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
            head_dim = cfg.embed_dim // cfg.n_heads
        else:
            head_dim = cfg.head_dim

        self.cos, self.sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length,
        )

        self.cfg = cfg
        self.kv_caches: Optional[List[KVCache]] = None

    def setup_kv_cache(self, max_seq_len: Optional[int] = None):
        """Initialize KV caches for all transformer blocks"""
        if max_seq_len is None:
            max_seq_len = self.cfg.context_length

        self.kv_caches = []
        for _ in range(len(self.transformer_blocs)):
            kv_cache = KVCache(
                max_seq_len=max_seq_len,
                n_kv_heads=self.cfg.n_kv_heads,
                head_dim=(
                    self.cfg.head_dim
                    if self.cfg.head_dim
                    else self.cfg.embed_dim // self.cfg.n_heads
                ),
                dtype=self.cfg.dtype,
                device=next(self.parameters()).device,
            )
            self.kv_caches.append(kv_cache)

    def reset_kv_cache(self):
        """Reset all KV caches"""
        if self.kv_caches:
            for cache in self.kv_caches:
                cache.reset()

    def forward(self, x, use_cache: bool = False, cache_position: Optional[int] = None):
        N = x.shape[-1]

        # Get RoPE parameters
        if use_cache and cache_position is not None:
            # For cached inference, we need the full cos/sin tensors
            cos_cache, sin_cache = self.cos, self.sin
        else:
            cos_cache, sin_cache = self.cos[:N], self.sin[:N]

        token_embed: torch.Tensor = self.tok_embed(x)
        x = token_embed

        # Forward through transformer blocks
        for i, block in enumerate(self.transformer_blocs):
            if use_cache and self.kv_caches:
                x = block(x, cos_cache, sin_cache, self.kv_caches[i], cache_position)
            else:
                x = block(x, cos_cache, sin_cache)

        x = self.final_rmsnorm(x)
        logits = self.out_head(x)
        return logits

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        """
        Generate text using KV cache for efficient autoregressive generation
        """
        self.setup_kv_cache()
        self.reset_kv_cache()

        batch_size, seq_len = input_ids.shape

        # Process initial prompt (prefill phase)
        logits = self.forward(input_ids, use_cache=True, cache_position=0)
        next_token_logits = logits[:, -1, :] / temperature

        # Apply top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float("-inf")

        next_token = torch.multinomial(
            torch.softmax(next_token_logits, dim=-1), num_samples=1
        )
        generated_tokens = [next_token]

        # Autoregressive generation (decode phase)
        for i in range(max_new_tokens - 1):
            cache_position = seq_len + i
            logits = self.forward(
                next_token, use_cache=True, cache_position=cache_position
            )
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-p sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")

            next_token = torch.multinomial(
                torch.softmax(next_token_logits, dim=-1), num_samples=1
            )
            generated_tokens.append(next_token)

        # Concatenate all generated tokens
        generated_sequence = torch.cat([input_ids] + generated_tokens, dim=1)
        return generated_sequence


if __name__ == "__main__":

    from llm.qwen3.qwen_token import Qwen3Tokenizer
    from llm.qwen3.bench import benchmark_generation
    import time

    # from llm.qwen3.load import load_weights_qwen
    import os

    # from llm.qwen3.hf_load import load_file

    repo_dir = "/home/aman/code/model_go_brr/Qwen3-0.6B"
    torch.manual_seed(696969)
    device = torch.device("cuda")
    single_file_path = os.path.join(repo_dir, "model.safetensors")

    # weights_dict = load_file(single_file_path)
    config = QwenConfig_float16()
    model = Qwen3(config).to(device)
    # load_weights_qwen(model, config, weights_dict)
    tokenizer_file_path = "/home/aman/code/model_go_brr/Qwen3-0.6B/tokenizer.json"

    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        add_gen_prompt=True,
        add_thinking=True,
    )
    model = model.to(device)
    # compiled_model = torch.compile(model)

    PROMPT = "Write a concise, friendly summary of why distributed training matters for large models.\n"

    print("Start benchmarking original model")
    benchmark_generation(
        model, tokenizer, prompt=PROMPT, warmup=2, iters=5, max_new_tokens=128
    )
