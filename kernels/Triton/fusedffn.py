import torch
import triton
import triton.language as tl
import torch.nn as nn

from .matmul import matmul_block, triton_matmul
from .activations import silu


# def get_cuda_autotune_config():
#     return [
#         triton.Config(
#             {
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_HID": 256,
#                 "BLOCK_SIZE_EMB": 64,
#             },
#             num_stages=3,
#             num_warps=8,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_HID": 256,
#                 "BLOCK_SIZE_EMB": 32,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_HID": 128,
#                 "BLOCK_SIZE_EMB": 32,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_HID": 128,
#                 "BLOCK_SIZE_EMB": 32,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_HID": 32,
#                 "BLOCK_SIZE_EMB": 32,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_N": 64,
#                 "BLOCK_SIZE_HID": 32,
#                 "BLOCK_SIZE_EMB": 32,
#             },
#             num_stages=5,
#             num_warps=2,
#         ),
#     ]


# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=["N", "EMBED_DIM", "HIDDEN_DIM"],
# )

"""
Best setting
BLOCK_SIZE_N: 64, 
BLOCK_SIZE_HID: 128, 
BLOCK_SIZE_EMB: 32, 
num_warps: 4, 
num_ctas: 1, 
num_stages: 4,
"""
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
    BLOCK_SIZE_N= 64
    BLOCK_SIZE_HID= 128 
    BLOCK_SIZE_EMB= 32 
    num_warps= 4 
    num_ctas= 1 
    num_stages= 4


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
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_EMB=BLOCK_SIZE_EMB,
        BLOCK_SIZE_HID=BLOCK_SIZE_HID,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages
    )

    output_matrix = triton_matmul(intermediate_matrix, weight3_matrix)
    return output_matrix


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import time

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
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

    embed_dim = 1024
    hidden_dim = 3072
    batch_size = 128

    print(
        f"Testing FFN with embed_dim={embed_dim}, hidden_dim={hidden_dim}, batch_size={batch_size}"
    )

    ffn_torch = FFN(embed_dim, hidden_dim).to(DEVICE).half()
    ffn_torch = torch.compile(ffn_torch , backend="inductor")
    # Create test input
    input_tensor = torch.randn(
        batch_size, embed_dim, device=DEVICE, dtype=torch.float16
    )

    # Extract weights for Triton version
    weight1 = ffn_torch.linear_layer1.weight.t().contiguous()  # (embed_dim, hidden_dim)
    weight2 = ffn_torch.linear_layerP.weight.t().contiguous()  # (embed_dim, hidden_dim)
    weight3 = ffn_torch.linear_layer2.weight.t().contiguous()  # (hidden_dim, embed_dim)

    print(f"Weight shapes: W1={weight1.shape}, W2={weight2.shape}, W3={weight3.shape}")
    print(f"Input shape: {input_tensor.shape}")

    # Verification function
    def verify_outputs():
        print("\n=== Verification ===")

        # PyTorch reference
        with torch.no_grad():
            output_torch = ffn_torch(input_tensor)

        try:
            output_triton = ffn_silu_fwd_triton(
                input_tensor,
                weight1,
                weight2,
                weight3,
                batch_size,
                hidden_dim,
                embed_dim,
            )

            # Compare outputs
            max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
            mean_diff = torch.mean(torch.abs(output_torch - output_triton)).item()
            rel_diff = mean_diff / torch.mean(torch.abs(output_torch)).item()

            print(
                f"Output shapes - PyTorch: {output_torch.shape}, Triton: {output_triton.shape}"
            )
            print(f"Max absolute difference: {max_diff:.6f}")
            print(f"Mean absolute difference: {mean_diff:.6f}")
            print(f"Relative difference: {rel_diff:.6f}")

            if rel_diff < 1e-2:
                print("✓ Verification PASSED")
                return True
            else:
                print("✗ Verification FAILED")
                return False

        except Exception as e:
            print(f"✗ Triton implementation failed: {e}")
            return False

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
                lambda: ffn_silu_fwd_triton(
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
                _ = ffn_silu_fwd_triton(
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
                _ = ffn_silu_fwd_triton(
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
                triton_out = ffn_silu_fwd_triton(
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
