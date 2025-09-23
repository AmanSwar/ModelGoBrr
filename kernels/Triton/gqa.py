from typing import Any
import triton
import triton.language as tl

import torch
from torch.autograd.function import Function
import torch.nn as nn

from .rope import rope_embedding_triton
from .flashattn import launch_attn
from .matmul import triton_matmul
from .rmsnorm import RMSNormTriton

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
            self.q_norm = RMSNormTriton(self.head_dim, eps=1e-6)
            self.k_norm = RMSNormTriton(self.head_dim, eps=1e-6)

        else:
            self.q_norm = self.k_norm = None

        self.softmax_scale = 1 / (self.head_dim **0.5)

    def forward(
        self, x, cos, sin
    ): 
        bs, seq_len, _ = x.shape
        Q: torch.Tensor = self.Wq(x)
        K: torch.Tensor = self.Wk(x)
        V: torch.Tensor = self.Wv(x)

        Q = Q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = V.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        Q = Q.to(torch.float32)
        K = K.to(torch.float32)

        if self.q_norm:
            Q = self.q_norm(Q)

        if self.k_norm:
            K = self.k_norm(K)
        # Q = Q.contiguous()
        # K = K.contiguous()
        # V = V.contiguous()
        cos = cos.to(torch.float32)
        sin = sin.to(torch.float32)

        Q, K = rope_embedding_triton(Q=Q, K=K, cos=cos, sin=sin)
        
        Q = Q.to(torch.float16).contiguous()
        K = K.to(torch.float16).contiguous()
        V = V.to(torch.float16).contiguous()


        K = K.repeat_interleave(self.num_kv_grps, dim=1).contiguous()
        V = V.repeat_interleave(self.num_kv_grps, dim=1).contiguous()
        
        
        attn_out = launch_attn(
            Q=Q,
            K=K,
            V=V,
            causal=True,
            softmax_scale=self.softmax_scale
        )

        attn_out = attn_out.transpose(1, 2).reshape(bs, seq_len, self.d_out)
        

        
        return self.out_projection(attn_out)


if __name__ == "__main__":
    """
    BENCHMARK AND VERIFYING FUNCTION MADE BY CHATGPT 
    """
    import torch.nn as nn
    import math

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

        triton_model.load_state_dict(pytorch_model.state_dict())

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
            output_triton = triton_model(x, cos, sin)

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

    def gqa_flops(BATCH, SEQ_LEN, N_HEADS, N_KV_HEADS, D_HEAD):
        D_IN = N_HEADS * D_HEAD

        # 1. FLOPs for Q, K, V projections
        q_proj_flops = 2 * BATCH * SEQ_LEN * D_IN * (N_HEADS * D_HEAD)
        k_proj_flops = 2 * BATCH * SEQ_LEN * D_IN * (N_KV_HEADS * D_HEAD)
        v_proj_flops = 2 * BATCH * SEQ_LEN * D_IN * (N_KV_HEADS * D_HEAD)

        # 2. FLOPs for attention mechanism
        # Q @ K.T -> (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
        qkT_flops = 2 * BATCH * N_HEADS * SEQ_LEN * D_HEAD * SEQ_LEN
        # Scores @ V -> (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
        scoresV_flops = 2 * BATCH * N_HEADS * SEQ_LEN * SEQ_LEN * D_HEAD

        # 3. FLOPs for output projection
        out_proj_flops = 2 * BATCH * SEQ_LEN * (N_HEADS * D_HEAD) * D_IN

        total_flops = (
            q_proj_flops + k_proj_flops + v_proj_flops +
            qkT_flops + scoresV_flops +
            out_proj_flops
        )
        return total_flops


    @triton.testing.perf_report([
        triton.testing.Benchmark(
            x_names=['SEQ_LEN'],
            x_vals=[128, 256, 512, 1024 , 2048],
            line_arg='provider',
            line_vals=['pytorch', 'triton'],
            line_names=['PyTorch', 'Triton'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GFLOPS',  # <-- CHANGED from 'ms'
            plot_name='Grouped Query Attention Perf Mixed Precision',  # <-- CHANGED plot name
            args={'D_HEAD': 64, 'BATCH': 2, 'N_HEADS': 32, 'N_KV_HEADS': 8, 'DTYPE': torch.float16}
        )
    ])
    def benchmark(SEQ_LEN, D_HEAD, BATCH, N_HEADS, N_KV_HEADS, DTYPE, provider):
        """
        Benchmarks GQA implementations and reports performance in GFLOPS.
        """
        D_IN = N_HEADS * D_HEAD
        DEVICE = 'cuda'

        # Inputs
        x = torch.randn(BATCH, SEQ_LEN, D_IN, dtype=DTYPE, device=DEVICE)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool().to(DEVICE)

        # RoPE embeddings
        cos, sin = compute_rope_params(D_HEAD, 1e6, SEQ_LEN)
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
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_model(x, cos, sin), quantiles=quantiles)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # Calculate GFLOPS from FLOPs and time
        flops = gqa_flops(BATCH, SEQ_LEN, N_HEADS, N_KV_HEADS, D_HEAD)
        gflops = flops / (ms * 1e6)

        # For the error bars, we need to convert min/max ms to min/max GFLOPS.
        # Note that max_ms corresponds to min_gflops and vice-versa.
        gflops_min = flops / (max_ms * 1e6)
        gflops_max = flops / (min_ms * 1e6)

        return gflops, gflops_min, gflops_max 

    verify_gqa_implementations(
        BATCH=2, SEQ_LEN=256, N_HEADS=32, N_KV_HEADS=8, D_HEAD=64
    )
    # Run the benchmark
    benchmark.run(print_data=True, show_plots=True)

    # ===============================

    def compare_tensor_stats(name, a, b, tol=1e-5):
        a_f = a.detach().to(torch.float32)
        b_f = b.detach().to(torch.float32)
        if a_f.shape != b_f.shape:
            print(f"[{name}] SHAPE MISMATCH {a_f.shape} vs {b_f.shape}")
            return False
        diff = (a_f - b_f).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        nonclose = int((diff > tol).sum())
        total = diff.numel()
        pct = 100.0 * nonclose / total
        print(f"[{name}] max_abs={max_abs:.6e}, mean_abs={mean_abs:.6e}, nonclose={nonclose}/{total} ({pct:.2f}%)")
        return nonclose == 0

    def print_tensor_meta(prefix, t):
        print(f"{prefix}: shape={tuple(t.shape)}, dtype={t.dtype}, contig={t.is_contiguous()}, strides={t.stride()}, storage_offset={t.storage_offset()}, data_ptr={hex(t.data_ptr())}")

    def debug_gqa_pipeline(BATCH=2, SEQ_LEN=64, N_HEADS=8, N_KV_HEADS=2, D_HEAD=32, DTYPE=torch.float32):
        DEVICE = "cuda"
        D_IN = N_HEADS * D_HEAD

        torch.manual_seed(0)
        x = torch.randn(BATCH, SEQ_LEN, D_IN, device=DEVICE, dtype=DTYPE)

        # instantiate models and sync state
        pytorch_model = GQA(d_in=D_IN, num_heads=N_HEADS, n_kv_heads=N_KV_HEADS, dtype=DTYPE).to(DEVICE).eval()
        triton_model = GQA_Triton(d_in=D_IN, num_heads=N_HEADS, n_kv_heads=N_KV_HEADS, dtype=DTYPE).to(DEVICE).eval()
        triton_model.load_state_dict(pytorch_model.state_dict())

        # compute RoPE params
        cos, sin = compute_rope_params(D_HEAD, 1e6, SEQ_LEN, dtype=torch.float32)
        cos, sin = cos.to(DEVICE), sin.to(DEVICE)

        # --------------------------
        # Build preprocessing exactly as in both forward paths (stop before attention)
        # --------------------------

        # Common linear outputs (these should be identical for both models since state dicts are loaded)
        Q_lin = pytorch_model.Wq(x)  # (B, L, D_out = H*D)
        K_lin = pytorch_model.Wk(x)
        V_lin = pytorch_model.Wv(x)

        # reshape & transpose to (B, H, L, D)
        Q_pre = Q_lin.view(BATCH, SEQ_LEN, N_HEADS, D_HEAD).transpose(1, 2)  # (B, H, L, D)
        K_pre = K_lin.view(BATCH, SEQ_LEN, N_KV_HEADS, D_HEAD).transpose(1, 2)
        V_pre = V_lin.view(BATCH, SEQ_LEN, N_KV_HEADS, D_HEAD).transpose(1, 2)

        print("\n--- After linear -> view -> transpose (pre-norm) ---")
        print_tensor_meta("Q_pre", Q_pre)
        print_tensor_meta("K_pre", K_pre)
        print_tensor_meta("V_pre", V_pre)

        # apply respective norms
        if pytorch_model.q_norm is not None:
            Q_normed = pytorch_model.q_norm(Q_pre)
        else:
            Q_normed = Q_pre
        if pytorch_model.k_norm is not None:
            K_normed = pytorch_model.k_norm(K_pre)
        else:
            K_normed = K_pre

        print("\n--- After RMSNorm ---")
        print_tensor_meta("Q_normed", Q_normed)
        print_tensor_meta("K_normed", K_normed)

        # PyTorch RoPE (reference)
        def apply_rope_local(x_t: torch.Tensor, cos_t: torch.Tensor, sin_t: torch.Tensor):
            b, h, l, d = x_t.shape
            assert d % 2 == 0
            x1 = x_t[..., : d // 2]
            x2 = x_t[..., d // 2 :]
            cos_s = cos_t[:l, :].unsqueeze(0).unsqueeze(0).to(x_t.dtype)
            sin_s = sin_t[:l, :].unsqueeze(0).unsqueeze(0).to(x_t.dtype)
            rotated = torch.cat((-x2, x1), dim=-1)
            return (x_t * cos_s) + (rotated * sin_s)

        Q_rope_ref = apply_rope_local(Q_normed, cos, sin)
        K_rope_ref = apply_rope_local(K_normed, cos, sin)
        V_ref = V_pre  # V not rope'd in GQA

        print("\n--- After PyTorch RoPE (reference) ---")
        print_tensor_meta("Q_rope_ref", Q_rope_ref)
        print_tensor_meta("K_rope_ref", K_rope_ref)

        # Now reproduce the *Triton* preprocessing path (calls rope_embedding_triton)
        # Note: use the tensors that your GQA_Triton would produce before calling launch_attn
        Q_t = Q_pre.clone()
        K_t = K_pre.clone()
        V_t = V_pre.clone()

        # apply RMSNorm using Triton RMS (to get exactly what triton path does)
        if triton_model.q_norm is not None:
            Q_t = triton_model.q_norm(Q_t)
        if triton_model.k_norm is not None:
            K_t = triton_model.k_norm(K_t)

        print("\n--- After Triton RMSNorm ---")
        print_tensor_meta("Q_t_after_norm", Q_t)
        print_tensor_meta("K_t_after_norm", K_t)

        # call rope_embedding_triton (the same call used in gqa)
        Q_rope_triton, K_rope_triton = rope_embedding_triton(Q=Q_t, K=K_t, cos=cos.to(Q_t.dtype), sin=sin.to(Q_t.dtype))

        print("\n--- After rope_embedding_triton ---")
        print_tensor_meta("Q_rope_triton", Q_rope_triton)
        print_tensor_meta("K_rope_triton", K_rope_triton)

        # repeat_interleave on KV as in gqa
        K_rep_triton = K_rope_triton.repeat_interleave(triton_model.num_kv_grps, dim=1)
        V_rep_triton = V_t.repeat_interleave(triton_model.num_kv_grps, dim=1)

        print("\n--- After repeat_interleave (Triton path) ---")
        print_tensor_meta("K_rep_triton", K_rep_triton)
        print_tensor_meta("V_rep_triton", V_rep_triton)

        # Also build the PyTorch reference repeated KV (what reference GQA does)
        K_rep_ref = K_rope_ref.repeat_interleave(triton_model.num_kv_grps, dim=1)
        V_rep_ref = V_ref.repeat_interleave(triton_model.num_kv_grps, dim=1)

        # Compare RoPE results
        print("\n--- Comparing RoPE/Q,K/V between PyTorch-ref and Triton-preproc ---")
        ok_q = compare_tensor_stats("Q_rope", Q_rope_ref.to(torch.float32), Q_rope_triton.to(torch.float32), tol=1e-4)
        ok_k = compare_tensor_stats("K_rope", K_rope_ref.to(torch.float32), K_rope_triton.to(torch.float32), tol=1e-4)
        ok_krep = compare_tensor_stats("K_rep", K_rep_ref.to(torch.float32), K_rep_triton.to(torch.float32), tol=1e-4)
        ok_vrep = compare_tensor_stats("V_rep", V_rep_ref.to(torch.float32), V_rep_triton.to(torch.float32), tol=1e-4)

        if not (ok_q and ok_k and ok_krep and ok_vrep):
            print("-> Mismatch detected in RoPE/repeat_interleave stage. Inspect above diffs.")
            return

        # If RoPE stage matches, check calling launch_attn variants
        print("\n--- Calling launch_attn with different memory variants and comparing to PyTorch reference attention ---")
        # Build PyTorch reference attention input (float32)
        scale = 1.0 / math.sqrt(D_HEAD)
        attn_ref = pytorch_attention(Q_rope_ref.to(torch.float32), K_rep_ref.to(torch.float32), V_rep_ref.to(torch.float32), causal=True, softmax_scale=scale)

        # Variant A: original (as Triton would be called in gqa)
        attn_triton_A = launch_attn(Q_rope_triton.to(torch.float16), K_rep_triton.to(torch.float16), V_rep_triton.to(torch.float16), causal=True, softmax_scale=scale)
        # Variant B: contiguous copies
        attn_triton_B = launch_attn(Q_rope_triton.contiguous().to(torch.float16), K_rep_triton.contiguous().to(torch.float16), V_rep_triton.contiguous().to(torch.float16), causal=True, softmax_scale=scale)
        # Variant C: clone + contiguous (guaranteed fresh storage)
        attn_triton_C = launch_attn(Q_rope_triton.detach().clone().contiguous().to(torch.float16), K_rep_triton.detach().clone().contiguous().to(torch.float16), V_rep_triton.detach().clone().contiguous().to(torch.float16), causal=True, softmax_scale=scale)

        print("\nCompare Attn Variant A (original) vs ref:")
        compare_tensor_stats("attn_A", attn_ref.to(torch.float32), attn_triton_A.to(torch.float32), tol=1e-2)
        print("\nCompare Attn Variant B (contiguous) vs ref:")
        compare_tensor_stats("attn_B", attn_ref.to(torch.float32), attn_triton_B.to(torch.float32), tol=1e-2)
        print("\nCompare Attn Variant C (clone+contig) vs ref:")
        compare_tensor_stats("attn_C", attn_ref.to(torch.float32), attn_triton_C.to(torch.float32), tol=1e-2)

        # Print final strides and meta to help trace
        print("\n--- Final meta of tensors used in launch_attn ---")
        print_tensor_meta("Q_rope_triton (before call)", Q_rope_triton)
        print_tensor_meta("K_rep_triton (before call)", K_rep_triton)
        print_tensor_meta("V_rep_triton (before call)", V_rep_triton)

    # debug_gqa_pipeline(
    #     BATCH=2,
    #     SEQ_LEN=128,
    #     N_HEADS=16,
    #     N_KV_HEADS=4,
    #     D_HEAD=64,
    #     DTYPE=torch.float32,
    # )
