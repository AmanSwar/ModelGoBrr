from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

import torch
import torch.nn as nn


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    batch_size, num_heads, seq_len, head_dim = x.shape

    assert head_dim % 2 == 0, "Head dim is not divisible by 2"

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # The unsqueezing logic is correct for broadcasting across batch and head dimensions
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # -> (1 , 1 , seq_len , head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)  # -> (1 , 1 , seq_len , head_dim)

    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


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
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype)
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

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        K = K.repeat_interleave(self.num_kv_grps, dim=1)
        V = V.repeat_interleave(self.num_kv_grps, dim=1)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        attn_out = flash_attn_func(Q, K, V, causal=True)

        attn_out = attn_out.reshape(bs, seq_len, self.d_out)
        return self.out_projection(attn_out)


class GQA(nn.Module):
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
            self.q_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype)
            self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype)
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

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        K = K.repeat_interleave(self.num_kv_grps, dim=1)
        V = V.repeat_interleave(self.num_kv_grps, dim=1)

        scores = Q @ K.transpose(2, 3)
        scores = torch.softmax(scores / (self.head_dim**0.5), dim=-1)
        scores = scores.to(torch.float16)

        attn_out = (scores @ V).transpose(1, 2).reshape(bs, seq_len, self.d_out)

        return self.out_projection(attn_out)


seq_len = 512
embed_dim = 1024
head_dim = 128


device = torch.device("cuda")

x = torch.rand(size=(1, seq_len, embed_dim), dtype=torch.float16).to(device)
cos = torch.rand(size=(seq_len, head_dim), dtype=torch.float16).to(device)
sin = torch.rand(size=(seq_len, head_dim), dtype=torch.float16).to(device)


# Create models
model_flash = GQA_FlashAttn(
    d_in=embed_dim, num_heads=16, n_kv_heads=8, head_dim=head_dim, dtype=torch.float16
).to(device)

model_torch = GQA(
    d_in=embed_dim, num_heads=16, n_kv_heads=8, head_dim=head_dim, dtype=torch.float16
).to(device)

# CRITICAL: Share the same weights between both models for fair comparison
model_torch.Wq.weight.data = model_flash.Wq.weight.data.clone()
model_torch.Wk.weight.data = model_flash.Wk.weight.data.clone()
model_torch.Wv.weight.data = model_flash.Wv.weight.data.clone()
model_torch.out_projection.weight.data = model_flash.out_projection.weight.data.clone()

if model_flash.q_norm is not None:
    model_torch.q_norm.weight.data = model_flash.q_norm.weight.data.clone()
    model_torch.q_norm.bias.data = model_flash.q_norm.bias.data.clone()
    model_torch.k_norm.weight.data = model_flash.k_norm.weight.data.clone()
    model_torch.k_norm.bias.data = model_flash.k_norm.bias.data.clone()

import time
import torch
import time

# Test with much larger problem sizes
test_configs = [
    {"batch": 1, "seq_len": 512, "desc": "Original (small)"},
    {"batch": 2, "seq_len": 1024, "desc": "Medium batch/seq"},
    {"batch": 4, "seq_len": 2048, "desc": "Large batch/seq"},
    {"batch": 8, "seq_len": 4096, "desc": "Very large"},
]

embed_dim = 1024
head_dim = 128
device = torch.device("cuda")


def benchmark_models(batch_size, seq_len, num_iters=50):
    print(f"\n=== Testing B={batch_size}, SeqLen={seq_len} ===")

    x = torch.rand(size=(batch_size, seq_len, embed_dim), dtype=torch.float16).to(
        device
    )
    cos = torch.rand(size=(seq_len, head_dim), dtype=torch.float16).to(device)
    sin = torch.rand(size=(seq_len, head_dim), dtype=torch.float16).to(device)

    # Create models (assuming your models are already defined)
    model_flash = GQA_FlashAttn(
        d_in=embed_dim,
        num_heads=16,
        n_kv_heads=8,
        head_dim=head_dim,
        dtype=torch.float16,
    ).to(device)

    model_torch = GQA(
        d_in=embed_dim,
        num_heads=16,
        n_kv_heads=8,
        head_dim=head_dim,
        dtype=torch.float16,
    ).to(device)

    # Share weights
    model_torch.Wq.weight.data = model_flash.Wq.weight.data.clone()
    model_torch.Wk.weight.data = model_flash.Wk.weight.data.clone()
    model_torch.Wv.weight.data = model_flash.Wv.weight.data.clone()
    model_torch.out_projection.weight.data = (
        model_flash.out_projection.weight.data.clone()
    )

    if model_flash.q_norm is not None:
        model_torch.q_norm.weight.data = model_flash.q_norm.weight.data.clone()
        model_torch.q_norm.bias.data = model_flash.q_norm.bias.data.clone()
        model_torch.k_norm.weight.data = model_flash.k_norm.weight.data.clone()
        model_torch.k_norm.bias.data = model_flash.k_norm.bias.data.clone()

    # Warmup
    for _ in range(10):
        _ = model_flash(x, cos, sin)
        _ = model_torch(x, cos, sin)

    torch.cuda.synchronize()

    # Benchmark FlashAttention
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(num_iters):
        out_flash = model_flash(x, cos, sin)
    torch.cuda.synchronize()
    flash_time = time.monotonic() - start

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(num_iters):
        out_torch = model_torch(x, cos, sin)
    torch.cuda.synchronize()
    torch_time = time.monotonic() - start

    # Memory usage
    flash_memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    _ = model_torch(x, cos, sin)
    torch_memory = torch.cuda.max_memory_allocated()

    speedup = torch_time / flash_time
    memory_ratio = flash_memory / torch_memory

    print(
        f"FlashAttention: {flash_time:.4f}s ({flash_time/num_iters*1000:.2f}ms per iter)"
    )
    print(
        f"PyTorch Naive:  {torch_time:.4f}s ({torch_time/num_iters*1000:.2f}ms per iter)"
    )
    print(f"Speedup: {speedup:.2f}x {'🚀' if speedup > 1 else '😭'}")
    print(f"Memory ratio: {memory_ratio:.2f}x")

    # Check correctness
    diff = torch.max(torch.abs(out_flash - out_torch)).item()
    print(f"Max difference: {diff:.2e}")

    return speedup, flash_time, torch_time


# Run benchmarks
results = []
for config in test_configs:
    try:
        speedup, flash_time, torch_time = benchmark_models(
            config["batch"], config["seq_len"]
        )
        results.append((config["desc"], speedup, flash_time, torch_time))
    except RuntimeError as e:
        print(f"❌ {config['desc']}: {e}")
        results.append((config["desc"], 0, 0, 0))

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for desc, speedup, flash_time, torch_time in results:
    if speedup > 0:
        status = (
            "🚀 FASTER"
            if speedup > 1.1
            else "😭 SLOWER" if speedup < 0.9 else "🤔 SIMILAR"
        )
        print(f"{desc:20} | Speedup: {speedup:5.2f}x | {status}")
    else:
        print(f"{desc:20} | ❌ FAILED")
