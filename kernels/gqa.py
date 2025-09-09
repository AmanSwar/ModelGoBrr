from typing import Any
import triton
import triton.language as tl

import torch
from torch.autograd.function import Function
import torch.nn as nn

from .rope import rope_embedding_triton
from .flashattn import launch_attn
from .matmul import triton_matmul
from .qk_norm import qknorm_triton

import sys
"""
For GQA we need to do: 
1) get Q ,K and V matrices from Wq , Wk and Wv
2) change the view
3) RMSNorm to q and k
3) apply rope
4) repeat_interleave
5) attn
"""

def gqa(
    Q : torch.Tensor ,
    K : torch.Tensor,
    V : torch.Tensor,
    cos, sin, #cos and sin,
    weight_q, # weight for RMSNorm
    weight_k, #weight for RMSNorm
    softmax_scale : float,
    num_kv_groups : int
):
    K = K.repeat_interleave(num_kv_groups, dim=1)
    V = V.repeat_interleave(num_kv_groups, dim=1)
    Q , K = qknorm_triton(
        q=Q,
        k=K,
        weight_q=weight_q,
        weight_k=weight_k,
        eps=1e-6
    ) # type: ignore

    Q , K = rope_embedding_triton(
        Q=Q,
        K=K,
        cos=cos,
        sin=sin
    )

    attn_out = launch_attn(
        Q=Q,
        K=K,
        V=V,
        causal=True,
        softmax_scale=softmax_scale
    )

    return attn_out


class GQAFunction(Function):
    """
    PyTorch autograd function for GQA, wrapping the Triton implementation.
    """

    @staticmethod
    def forward(
        ctx: Any,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        weight_q: torch.Tensor,
        weight_k: torch.Tensor,
        num_kv_groups: int,
    ) -> torch.Tensor:
        """
        Forward pass for GQA.
        """
        head_dim = Q.shape[-1]
        softmax_scale = head_dim**-0.5

        attn_out = gqa(
            Q=Q,
            K=K,
            V=V,
            cos=cos,
            sin=sin,
            weight_q=weight_q,
            weight_k=weight_k,
            softmax_scale=softmax_scale,
            num_kv_groups=num_kv_groups,
        )

        ctx.save_for_backward(Q, K, V, cos, sin, weight_q, weight_k, attn_out)
        ctx.num_kv_groups = num_kv_groups

        return attn_out

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        """
        me have not implemented yet :)
        """
        raise NotImplementedError("Backward pass for GQAFunction is not implemented.")

class GQA_Triton(nn.Module):
    """GQA Module that uses the optimized Triton kernels in its forward pass."""

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
        assert (
            num_heads % n_kv_heads == 0
        ), "Num heads is not divisible by num kv grps"

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
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        
        else:
            self.q_norm = self.k_norm = None

    def forward(
        self, x, mask, cos, sin
    ): 
        bs, seq_len, _ = x.shape
        Q: torch.Tensor = self.Wq(x)
        K: torch.Tensor = self.Wk(x)
        V: torch.Tensor = self.Wv(x)
        
        Q = Q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        attn_out = gqa(
            Q=Q,
            K=K,
            V=V,
            cos=cos,
            sin=sin,
            weight_q=self.q_norm.weight,
            weight_k=self.k_norm.weight,
            softmax_scale=(self.head_dim**-0.5),
            num_kv_groups=self.num_kv_grps,
        )

        attn_out = attn_out.transpose(1, 2).reshape(bs, seq_len, self.d_out)
        return self.out_projection(attn_out)


if __name__ == "__main__":

    import torch.nn as nn

    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        batch_size, num_heads, seq_len, head_dim = x.shape

        assert head_dim % 2 == 0, "Head dim is not divisible by 2"

        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]

        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # -> (1 , 1 , seq_len , head_dim)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)  # -> (1 , 1 , seq_len , head_dim)

        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)

        return x_rotated.to(dtype=x.dtype)

    class GQA(nn.Module):
        """Grouped Query Attentio"""

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

            assert (
                num_heads % n_kv_heads == 0
            ), "Num heads is not divisible by num kv grps"

            self.num_heads = num_heads
            self.n_kv_heads = n_kv_heads
            # self.grp_size = num_heads // num_kv_grps
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
                self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
                self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

            else:
                self.q_norm = self.k_norm = None

        def forward(self, x, mask, cos, sin):

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
            scores = scores.masked_fill(mask, -torch.inf).to(x.dtype)

            scores = torch.softmax(scores / (self.head_dim**0.5), dim=-1)

            attn_out = (scores @ V).transpose(1, 2).reshape(bs, seq_len, self.d_out)

            return self.out_projection(attn_out)

    def compute_rope_params(
        head_dim,
        theta_base,
        context_length,
        dtype= torch.float32
    ):
        assert head_dim % 2 == 0 , "head dim must be divisible by 2"

        inv_freq = 1 / (theta_base ** (torch.arange(0 , head_dim  , 2 , dtype=dtype)[: head_dim//2].float() / head_dim))

        position = torch.arange(context_length , dtype=dtype)
        angles = position[: , None] * inv_freq[None, :]

        angles = torch.cat([angles , angles] , dim=1)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos , sin

    def verify_gqa_implementations(
        BATCH, SEQ_LEN, N_HEADS, N_KV_HEADS, D_HEAD, DTYPE=torch.float16
    ):
        """
        Compares the output of the PyTorch and Triton GQA modules to ensure correctness.
        """
        print("\n--- Verifying GQA Implementations ---")
        print(
            f"Config: B={BATCH}, L={SEQ_LEN}, H={N_HEADS}, KV_H={N_KV_HEADS}, D_H={D_HEAD}, Dtype={DTYPE}"
        )

        D_IN = N_HEADS * D_HEAD
        DEVICE = "cuda"

        # 1. Instantiate models
        pytorch_model = (
            GQA(d_in=D_IN, num_heads=N_HEADS, n_kv_heads=N_KV_HEADS, dtype=DTYPE)
            .to(DEVICE)
            .eval()
        )
        triton_model = (
            GQA_Triton(d_in=D_IN, num_heads=N_HEADS, n_kv_heads=N_KV_HEADS, dtype=DTYPE)
            .to(DEVICE)
            .eval()
        )

        # 2. CRITICAL: Ensure both models have the exact same weights
        triton_model.load_state_dict(pytorch_model.state_dict())

        # 3. Create common inputs
        x = torch.randn(BATCH, SEQ_LEN, D_IN, dtype=DTYPE, device=DEVICE)
        mask = torch.triu(
            torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=1
        ).bool()
        cos, sin = compute_rope_params(D_HEAD, 1e6, SEQ_LEN)
        cos = cos.to(DEVICE)
        sin = sin.to(DEVICE)

        # 4. Run forward passes
        with torch.no_grad():
            output_pytorch = pytorch_model(x, mask, cos, sin)
            output_triton = triton_model(x, mask, cos, sin)

        # 5. Compare outputs
        try:
            # Use a tolerance appropriate for half-precision floating point numbers
            torch.testing.assert_close(
                output_pytorch, output_triton, rtol=1e-2, atol=1e-2
            )
            print("✅ Verification PASSED!")
        except AssertionError as e:
            print("❌ Verification FAILED!")
            # Print the full error for detailed debugging
            print(e)

    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=['SEQ_LEN'],
            x_vals=[128, 256, 512, 1024],
            line_arg='provider',
            line_vals=['pytorch', 'triton'],
            line_names=['PyTorch', 'Triton'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='ms',
            plot_name='gqa-performance-bf16',
            args={'D_HEAD': 64, 'BATCH': 2, 'N_HEADS': 32, 'N_KV_HEADS': 8, 'DTYPE': torch.float16}
        )
    ])
    def benchmark(SEQ_LEN, D_HEAD, BATCH, N_HEADS, N_KV_HEADS, DTYPE, provider):
        D_IN = N_HEADS * D_HEAD
        DEVICE = 'cuda'

        # Inputs
        x = torch.randn(BATCH, SEQ_LEN, D_IN, dtype=DTYPE, device=DEVICE)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool().to(DEVICE)

        # RoPE embeddings
        cos , sin = compute_rope_params(D_HEAD , 1e6, SEQ_LEN)
        cos = cos.to(DEVICE)
        sin = sin.to(DEVICE)
        # Instantiate models
        pytorch_model = GQA(d_in=D_IN, num_heads=N_HEADS, n_kv_heads=N_KV_HEADS, dtype=DTYPE).to(DEVICE)
        triton_model = GQA_Triton(d_in=D_IN, num_heads=N_HEADS, n_kv_heads=N_KV_HEADS, dtype=DTYPE).to(DEVICE)

        # Define quantiles for stable benchmarking
        quantiles = [0.5, 0.2, 0.8]
        if provider == 'pytorch':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: pytorch_model(x, mask, cos, sin), quantiles=quantiles)
        elif provider == 'triton':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_model(x, mask, cos, sin), quantiles=quantiles)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        return ms, min_ms, max_ms
    
    
    verify_gqa_implementations(
        BATCH=2, SEQ_LEN=256, N_HEADS=32, N_KV_HEADS=8, D_HEAD=64
    )
    # Run the benchmark
    benchmark.run(print_data=True, show_plots=True)
