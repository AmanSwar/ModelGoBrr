import torch
import triton
import triton.language as tl
import torch.nn as nn

from .matmul import matmul_block, triton_matmul
from .activations import silu


def get_cuda_autotune_config():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_HID": 256,
                "BLOCK_SIZE_EMB": 64,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_HID": 256,
                "BLOCK_SIZE_EMB": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_HID": 128,
                "BLOCK_SIZE_EMB": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_HID": 128,
                "BLOCK_SIZE_EMB": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_HID": 32,
                "BLOCK_SIZE_EMB": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_HID": 32,
                "BLOCK_SIZE_EMB": 32,
            },
            num_stages=5,
            num_warps=2,
        ),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["N", "EMBED_DIM", "HIDDEN_DIM"],
)
@triton.jit
def _subffn_silu_fwd_kernel(
    X,
    Y,
    Weight1,
    Weight2,
    N,
    EMBED_DIM,
    HIDDEN_DIM,
    stride_x_n,
    stride_x_emb,
    stride_w1_emb,
    stride_w1_hid,
    stride_w2_emb,
    stride_w2_hid,
    stride_y_n,
    stride_y_hid,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_HID: tl.constexpr,
    BLOCK_SIZE_EMB: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hid = tl.program_id(1)

    offs_inp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_weight_hid = pid_hid * BLOCK_SIZE_HID + tl.arange(0, BLOCK_SIZE_HID)

    acc_weight1 = matmul_block(
        X,
        Weight1,
        EMBED_DIM,
        stride_x_n,
        stride_x_emb,
        stride_w1_emb,
        stride_w1_hid,
        offs_inp_n,
        offs_weight_hid,
        BLOCK_SIZE_N,
        BLOCK_SIZE_HID,
        BLOCK_SIZE_EMB,
    )
    acc_weight2 = matmul_block(
        X,
        Weight2,
        EMBED_DIM,
        stride_x_n,
        stride_x_emb,
        stride_w2_emb,
        stride_w2_hid,
        offs_inp_n,
        offs_weight_hid,
        BLOCK_SIZE_N,
        BLOCK_SIZE_HID,
        BLOCK_SIZE_EMB,
    )

    acc_weight1 = silu(acc_weight1)
    acc = acc_weight1 * acc_weight2
    acc = acc.to(tl.float16)

    offs_y_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_y_hid = pid_hid * BLOCK_SIZE_HID + tl.arange(0, BLOCK_SIZE_HID)

    y_ptrs = Y + stride_y_n * offs_y_n[:, None] + stride_y_hid * offs_y_hid[None, :]
    y_mask = (offs_y_n[:, None] < N) & (offs_y_hid[None, :] < HIDDEN_DIM)
    tl.store(y_ptrs, acc, mask=y_mask)


def ffn_silu_fwd_triton(
    input_matrix,
    weight1_matrix,
    weight2_matrix,
    weight3_matrix,
    N,
    HIDDEN_DIM,
    EMBED_DIM,
):
    intermediate_matrix = torch.zeros(
        size=(N, HIDDEN_DIM), device=input_matrix.device, dtype=torch.float16
    )

    grid = lambda CONFIG: (
        triton.cdiv(N, CONFIG["BLOCK_SIZE_N"]),
        triton.cdiv(HIDDEN_DIM, CONFIG["BLOCK_SIZE_HID"]),
    )

    _subffn_silu_fwd_kernel[grid](
        X=input_matrix,
        Y=intermediate_matrix,
        Weight1=weight1_matrix,
        Weight2=weight2_matrix,
        N=N,
        EMBED_DIM=EMBED_DIM,
        HIDDEN_DIM=HIDDEN_DIM,
        stride_x_n=input_matrix.stride(0),
        stride_x_emb=input_matrix.stride(1),
        stride_w1_emb=weight1_matrix.stride(0),
        stride_w1_hid=weight1_matrix.stride(1),
        stride_w2_emb=weight2_matrix.stride(0),
        stride_w2_hid=weight2_matrix.stride(1),
        stride_y_n=intermediate_matrix.stride(0),
        stride_y_hid=intermediate_matrix.stride(1),
    )

    output_matrix = triton_matmul(intermediate_matrix, weight3_matrix)
    return output_matrix


class FusedFFNSilu(nn.Module):

    def __init__(self, embed_dim , hidden_dim):
        super().__init__()
        self.linear_layer1 = nn.Parameter(torch.empty((embed_dim, hidden_dim)))
        self.linear_layer2 = nn.Parameter(torch.empty((embed_dim, hidden_dim)))
        self.linear_layer3 = nn.Parameter(torch.empty((hidden_dim , embed_dim)))

        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

    def forward(self , x):
        batch_size, seq_len, embed_dim = x.shape
        reshaped_x = x.reshape(-1, embed_dim)
        N = reshaped_x.shape[0]
        out_reshaped = ffn_silu_fwd_triton(
            input_matrix=reshaped_x,
            weight1_matrix=self.linear_layer1,
            weight2_matrix=self.linear_layer2,
            weight3_matrix=self.linear_layer3,
            N=N,
            HIDDEN_DIM=self.hidden_dim,
            EMBED_DIM=self.embed_dim,
        )
        out = out_reshaped.view(batch_size, seq_len, self.embed_dim)
        return out


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import time

    # Mock implementations for matmul and silu if they are not found
    # This allows the script to run for testing purposes.
    # Replace these with your actual, performant Triton implementations.
    if "matmul_block" not in globals():
        print("Using MOCK Triton implementations for testing.")

        @triton.jit
        def matmul_block(
            X,
            Weight,
            K,
            stride_x_m,
            stride_x_k,
            stride_w_k,
            stride_w_n,
            offs_m,
            offs_n,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        ):
            acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, K, BLOCK_SIZE_K):
                offs_k = k + tl.arange(0, BLOCK_SIZE_K)
                x_ptrs = X + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
                w_ptrs = (
                    Weight + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
                )
                x = tl.load(
                    x_ptrs,
                    mask=(offs_m[:, None] < BLOCK_SIZE_M) & (offs_k[None, :] < K),
                    other=0.0,
                )
                w = tl.load(
                    w_ptrs,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < BLOCK_SIZE_N),
                    other=0.0,
                )
                acc += tl.dot(x, w)
            return acc

        def triton_matmul(a, b):
            return torch.matmul(a, b)

        @triton.jit
        def silu(x):
            return x * tl.sigmoid(x)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        exit()

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

    # --- Test Parameters ---
    embed_dim = 1024
    hidden_dim = 4096  # SwiGLU often uses a larger hidden_dim
    batch_size = 4
    seq_len = 512

    print(
        f"Testing FFN with batch={batch_size}, seq_len={seq_len}, embed_dim={embed_dim}, hidden_dim={hidden_dim}"
    )

    ffn_torch = FFN(embed_dim, hidden_dim).to(DEVICE).half()

    # Create a 3D test input tensor
    input_tensor = torch.randn(
        batch_size, seq_len, embed_dim, device=DEVICE, dtype=torch.float16
    )

    # Extract weights for Triton version
    weight1 = ffn_torch.linear_layer1.weight.t().contiguous()
    weight2 = ffn_torch.linear_layerP.weight.t().contiguous()
    weight3 = ffn_torch.linear_layer2.weight.t().contiguous()

    # --- Verification ---
    def verify_outputs():
        print("\n=== Verification ===")

        # Create the FusedFFNSilu module and load weights
        fused_ffn = FusedFFNSilu(embed_dim, hidden_dim).to(DEVICE).half()
        fused_ffn.linear_layer1.data = weight1
        fused_ffn.linear_layer2.data = weight2
        fused_ffn.linear_layer3.data = weight3

        try:
            # PyTorch reference output
            with torch.no_grad():
                output_torch = ffn_torch(input_tensor)

            # Triton fused output
            with torch.no_grad():
                output_triton = fused_ffn(input_tensor)

            # Compare outputs
            max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
            is_close = torch.allclose(output_torch, output_triton, atol=1e-2, rtol=1e-2)

            print(
                f"Output shapes - PyTorch: {output_torch.shape}, Triton: {output_triton.shape}"
            )
            print(f"Max absolute difference: {max_diff:.6f}")
            print(f"torch.allclose result: {is_close}")

            if is_close:
                print("✓ Verification PASSED")
                return True
            else:
                print("✗ Verification FAILED")
                return False

        except Exception as e:
            import traceback

            print(f"✗ Triton implementation failed: {e}")
            traceback.print_exc()
            return False

    # Run tests
    print("Starting tests...")
    verify_outputs()

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],  
            x_vals=[
                32,
                64,
                128,
                256,
                512,
                1024,
            ],  
            line_arg="provider",  
            line_vals=["triton", "pytorch"],  
            line_names=["Triton", "PyTorch"], 
            styles=[("green", "-"), ("blue", "-")], 
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="ffn-performance-",  # Name for the plot
            args={"embed_dim": embed_dim, "hidden_dim": hidden_dim},
        )
    )
    def benchmark_ffn(batch_size, embed_dim, hidden_dim, provider):
        # Create test model and data for this batch size
        test_ffn = FFN(embed_dim, hidden_dim).to(DEVICE).half()
        test_input = torch.randn(
            batch_size, embed_dim, device=DEVICE, dtype=torch.float16
        )

        # Extract weights
        test_w1 = test_ffn.linear_layer1.weight.t().contiguous()
        test_w2 = test_ffn.linear_layerP.weight.t().contiguous()
        test_w3 = test_ffn.linear_layer2.weight.t().contiguous()

        quantiles = [0.5, 0.2, 0.8]

        if provider == "pytorch":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: test_ffn(test_input), quantiles=quantiles
            )
        elif provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: ffn_fwd_triton(
                    test_input,
                    test_w1,
                    test_w2,
                    test_w3,
                    batch_size,
                    hidden_dim,
                    embed_dim,
                ),
                quantiles=quantiles,
            )

        # Calculate TFLOPS
        # FFN operations: 2 * (batch_size * embed_dim * hidden_dim) + batch_size * hidden_dim * embed_dim
        # = 3 * batch_size * embed_dim * hidden_dim floating point operations
        total_flops = 3 * batch_size * embed_dim * hidden_dim
        perf = lambda ms: total_flops * 1e-12 / (ms * 1e-3)  # Convert to TFLOPS

        return perf(ms), perf(max_ms), perf(min_ms)

    # Benchmarking function using simple timing
    def benchmark_implementations():
        print("\n=== Simple Benchmarking ===")

        num_warmup = 10
        num_iter = 100

        # Warmup
        print("Warming up...")
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = ffn_torch(input_tensor)
                torch.cuda.synchronize()

        # Benchmark PyTorch
        print("Benchmarking PyTorch...")
        torch.cuda.synchronize()
        start_time = time.time()

        for _ in range(num_iter):
            with torch.no_grad():
                _ = ffn_torch(input_tensor)

        torch.cuda.synchronize()
        torch_time = (time.time() - start_time) / num_iter * 1000  # ms

        # Warmup Triton
        try:
            for _ in range(num_warmup):
                _ = ffn_fwd_triton(
                    input_tensor,
                    weight1,
                    weight2,
                    weight3,
                    batch_size,
                    hidden_dim,
                    embed_dim,
                )
                torch.cuda.synchronize()

            # Benchmark Triton
            print("Benchmarking Triton...")
            torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(num_iter):
                _ = ffn_fwd_triton(
                    input_tensor,
                    weight1,
                    weight2,
                    weight3,
                    batch_size,
                    hidden_dim,
                    embed_dim,
                )

            torch.cuda.synchronize()
            triton_time = (time.time() - start_time) / num_iter * 1000  # ms

            print(f"PyTorch time: {torch_time:.3f} ms")
            print(f"Triton time: {triton_time:.3f} ms")
            print(f"Speedup: {torch_time/triton_time:.2f}x")

        except Exception as e:
            print(f"Triton benchmark failed: {e}")

    # Run tests
    print("Starting tests...")

    if verify_outputs():
        benchmark_implementations()

        print("\n=== Running Triton Performance Report ===")
        benchmark_ffn.run(show_plots=True, print_data=True)
    else:
        print("Skipping benchmark due to verification failure")

    # Test with different sizes
    print("\n=== Testing different sizes ===")
    test_configs = [
        (64, 512, 1536),  # Small
        (128, 1024, 3072),  # Medium
        (256, 2048, 6144),  # Large
    ]

    for bs, emb_dim, hid_dim in test_configs:
        print(f"\nTesting: batch_size={bs}, embed_dim={emb_dim}, hidden_dim={hid_dim}")

        # Create test model and data
        test_ffn = FFN(emb_dim, hid_dim).to(DEVICE).half()
        test_input = torch.randn(bs, emb_dim, device=DEVICE, dtype=torch.float16)

        # Extract weights
        test_w1 = test_ffn.linear_layer1.weight.t().contiguous()
        test_w2 = test_ffn.linear_layerP.weight.t().contiguous()
        test_w3 = test_ffn.linear_layer2.weight.t().contiguous()

        # Quick verification
        try:
            with torch.no_grad():
                ref_out = test_ffn(test_input)
                triton_out = ffn_fwd_triton(
                    test_input, test_w1, test_w2, test_w3, bs, hid_dim, emb_dim
                )

                rel_diff = torch.mean(torch.abs(ref_out - triton_out)) / torch.mean(
                    torch.abs(ref_out)
                )
                print(f"  Relative difference: {rel_diff.item():.6f}")

                if rel_diff < 1e-2:
                    print("  ✓ PASSED")
                else:
                    print("  ✗ FAILED")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
