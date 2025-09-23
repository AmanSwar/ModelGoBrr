import triton
import triton.language as tl
import torch


ROPE_GROUP_SIZE: int = 4


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_SIZE": 16}, num_warps=2),
#         triton.Config({"BLOCK_SIZE": 32}, num_warps=2),
#         triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
#         triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
#         triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
#         triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
#         triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
#     ],
#     key=["head_dim"],
# )

"""
best config
BLOCK_SIZE: 128, num_warps: 4, num_ctas: 1, num_stages: 2
"""

@triton.jit
def _rope_embedding(
    Q,
    Q_row_stride: tl.constexpr,
    cos,
    cos_row_stride: tl.constexpr,
    sin,
    sin_row_stride: tl.constexpr,
    seqlen,
    head_dim: tl.constexpr,
    n_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Calculates the RoPE Embedding quickly
    RoPE is Q * cos + rotate_half(Q) * sin
    """
    ROPE_GROUP_SIZE = 4

    row_position = tl.program_id(0)
    group_head_position = tl.program_id(1)

    col_offsets = tl.arange(0, BLOCK_SIZE)

    half_head_dim: tl.constexpr = head_dim // 2

    tl.static_assert(BLOCK_SIZE >= half_head_dim, "BLOCK_SIZE must be >= head_dim // 2")
    
    mask = col_offsets < half_head_dim


    sin1 = tl.load(
        sin
        + (row_position % seqlen) * sin_row_stride
        + half_head_dim * 0
        + col_offsets,
        mask=mask,
        other=0,
    )
    cos1 = tl.load(
        cos
        + (row_position % seqlen) * cos_row_stride
        + half_head_dim * 0
        + col_offsets,
        mask=mask,
        other=0,
    )

    head_start = group_head_position * ROPE_GROUP_SIZE

    head_end = min((head_start + ROPE_GROUP_SIZE), n_heads)

    for k in range(head_start, head_end):

        offs_q1 = row_position * Q_row_stride + k * head_dim + col_offsets
        offs_q2 = (
            row_position * Q_row_stride + k * head_dim + col_offsets + half_head_dim
        )

        Q1 = tl.load(Q + offs_q1, mask=mask, other=0).to(sin1.dtype)
        Q2 = tl.load(Q + offs_q2, mask=mask, other=0).to(sin1.dtype)

        tl.store(Q + offs_q1, Q1 * cos1 - Q2 * sin1, mask=mask)
        tl.store(Q + offs_q2, Q2 * cos1 + Q1 * sin1, mask=mask)


def rope_embedding_triton(Q, K, cos, sin):
    Q = Q.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, head_dim)
    K = K.transpose(1, 2).contiguous()  # (batch, seq_len, n_heads, head_dim)

    cos, sin = cos.squeeze(), sin.squeeze()

    batch, seq_len, n_heads, head_dim = Q.shape
    batch , seq_len , n_heads_k , head_dim_k = K.shape

    Q = Q.reshape(batch * seq_len, n_heads * head_dim)
    K = K.reshape(batch * seq_len, n_heads_k * head_dim_k)

    n_rows = batch * seq_len

    assert seq_len <= cos.shape[0]

    div, mod = divmod(n_heads, ROPE_GROUP_SIZE)
    n_groups = div + (1 if mod else 0)


    BLOCK_SIZE = 128
    num_warps = 4
    num_ctas = 1
    num_stages = 2

    _rope_embedding[(n_rows, n_groups)](
        Q,
        Q.stride(0),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        seq_len,
        head_dim,
        n_heads,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages,
    )

    # Launch kernel for K
    _rope_embedding[(n_rows, n_groups)](
        K,
        K.stride(0),
        cos,
        cos.stride(0),
        sin,
        sin.stride(0),
        seq_len,
        head_dim_k,
        n_heads_k,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages,
    )

    Q = Q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(batch, seq_len, n_heads_k, head_dim_k).transpose(1, 2)
    torch.cuda.current_stream(Q.device).synchronize()
    return Q, K


if __name__ == "__main__":
    import time

    print("Testing RoPE Embedding Implementation")
    print("=" * 50)

    def create_rope_params(
        head_dim: int, theta_base: float, context_length: int, device="cuda"
    ):
        """
        Create RoPE parameters (cos and sin tables)
        This replaces the incorrect Triton kernel version
        """
        inv_freq = 1.0 / (
            theta_base
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )

        pos = torch.arange(context_length, dtype=torch.float32, device=device)
        angles = pos[:, None] * inv_freq[None, :]

        angles = torch.cat([angles, angles], dim=1)

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        return cos, sin

    def apply_rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Reference implementation of RoPE for verification"""
        batch_size, num_heads, seq_len, head_dim = x.shape

        assert head_dim % 2 == 0, "Head dim is not divisible by 2"

        x1 = x[..., : head_dim // 2]
        x2 = x[..., head_dim // 2 :]

        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # -> (1, 1, seq_len, head_dim)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)  # -> (1, 1, seq_len, head_dim)

        rotated = torch.cat((-x2, x1), dim=-1)
        x_rotated = (x * cos) + (rotated * sin)

        return x_rotated.to(dtype=x.dtype)

    def verifier(Q, K, cos, sin):
        """Verify correctness of the Triton implementation"""
        print("Running verification...")

        # Make copies for comparison
        Q_copy = Q.clone()
        K_copy = K.clone()

        # Triton implementation
        kernel_out_q, kernel_out_k = rope_embedding_triton(Q, K, cos, sin)

        # Reference implementation
        q_out = apply_rope_reference(Q_copy, cos, sin)
        k_out = apply_rope_reference(K_copy, cos, sin)

        q_diff = (kernel_out_q - q_out).abs().max()
        k_diff = (kernel_out_k - k_out).abs().max()

        print(f"Q max diff: {q_diff.item():.6f}")
        print(f"K max diff: {k_diff.item():.6f}")

        tolerance = 1e-2
        if q_diff < tolerance and k_diff < tolerance:
            print("✅ Verification passed!")
        else:
            print("❌ Verification failed!")

        return q_diff < tolerance and k_diff < tolerance

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[128, 256, 512, 1024, 2048, 4096],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton RoPE", "PyTorch RoPE"],
            styles=[("blue", "-"), ("red", "--")],
            ylabel="Memory Bandwidth (GB/s)",
            plot_name="rope-bandwidth",
            args={"batch_size": 4, "n_heads": 32, "head_dim": 128},
        )
    )
    def benchmark_rope_bandwidth(seq_len, batch_size, n_heads, head_dim, provider):
        """Benchmark RoPE implementations - Memory Bandwidth"""
        device = "cuda"
        dtype = torch.float16

        # Create test tensors
        Q = torch.randn(
            batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        K = torch.randn(
            batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype
        )

        # Create RoPE parameters
        cos, sin = create_rope_params(head_dim, 10000.0, seq_len, device)

        # Calculate memory usage in bytes
        # Q and K: read + write, cos and sin: read only
        element_size = 2 if dtype == torch.float16 else 4  # bytes per element
        q_k_elements = batch_size * n_heads * seq_len * head_dim
        cos_sin_elements = seq_len * head_dim

        # Memory operations: read Q, K, cos, sin + write Q, K
        total_bytes = (
            2 * q_k_elements + 2 * cos_sin_elements + 2 * q_k_elements
        ) * element_size
        total_gb = total_bytes / (1024**3)

        if provider == "triton":

            def fn():
                return rope_embedding_triton(Q, K, cos, sin)

        else:  # torch

            def fn():
                q_out = apply_rope_reference(Q, cos, sin)
                k_out = apply_rope_reference(K, cos, sin)
                return q_out, k_out

        # Warmup
        for _ in range(10):
            fn()

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        end = time.time()

        time_per_iter = (end - start) / 100  # seconds per iteration
        bandwidth_gb_s = total_gb / time_per_iter

        return bandwidth_gb_s

    # TFLOPS Benchmarking
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["seq_len"],
            x_vals=[128, 256, 512, 1024, 2048, 4096],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton RoPE", "PyTorch RoPE"],
            styles=[("green", "-"), ("orange", "--")],
            ylabel="Throughput (TFLOPS)",
            plot_name="rope-tflops",
            args={"batch_size": 4, "n_heads": 32, "head_dim": 128},
        )
    )
    def benchmark_rope_tflops(seq_len, batch_size, n_heads, head_dim, provider):
        """Benchmark RoPE implementations - TFLOPS"""
        device = "cuda"
        dtype = torch.float16

        Q = torch.randn(
            batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype
        )
        K = torch.randn(
            batch_size, n_heads, seq_len, head_dim, device=device, dtype=dtype
        )

        cos, sin = create_rope_params(head_dim, 10000.0, seq_len, device)

        elements_per_tensor = batch_size * n_heads * seq_len * head_dim
        total_flops = 2 * elements_per_tensor * 4  # 2 tensors (Q, K) * 4 FLOPs each
        total_tflops = total_flops / (10**12)

        if provider == "triton":

            def fn():
                return rope_embedding_triton(Q, K, cos, sin)

        else:  # torch

            def fn():
                q_out = apply_rope_reference(Q, cos, sin)
                k_out = apply_rope_reference(K, cos, sin)
                return q_out, k_out

        # Warmup
        for _ in range(10):
            fn()

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            fn()
        torch.cuda.synchronize()
        end = time.time()

        time_per_iter = (end - start) / 100  # seconds per iteration
        tflops_achieved = total_tflops / time_per_iter

        return tflops_achieved

    batch_size = 2
    n_heads = 8
    seq_len = 512
    head_dim = 64
    device = "cuda"
    dtype = torch.float16

    Q = torch.randn(
        batch_size,
        n_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    K = torch.randn(
        batch_size,
        n_heads,
        seq_len,
        head_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    # Create RoPE parameters
    cos, sin = create_rope_params(head_dim, 10000.0, seq_len, device)

    print(f"Input shapes: Q={Q.shape}, K={K.shape}")
    print(f"RoPE params: cos={cos.shape}, sin={sin.shape}")

    # Run verification
    passed = verifier(Q, K, cos, sin)

    if passed:
        print("\n" + "=" * 50)
        print("Running benchmarks...")
        print("\nMemory Bandwidth Benchmark:")
        benchmark_rope_bandwidth.run(print_data=True, show_plots=True)
        print("\nTFLOPS Benchmark:")
        benchmark_rope_tflops.run(print_data=True, show_plots=True)

    else:
        print("\nSkipping benchmarks due to failed verification.")

    print("\nDone!")
