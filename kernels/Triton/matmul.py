import torch
import triton
import triton.language as tl

# def get_cuda_autotune_config():
#     return [
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 256,
#                 "BLOCK_SIZE_K": 64,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=3,
#             num_warps=8,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 256,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 128,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 128,
#                 "BLOCK_SIZE_N": 32,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=4,
#             num_warps=4,
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": 64,
#                 "BLOCK_SIZE_N": 32,
#                 "BLOCK_SIZE_K": 32,
#                 "GROUP_SIZE_M": 8,
#             },
#             num_stages=5,
#             num_warps=2,
#         ),
#     ]


@triton.jit
def matmul_block(
    matrixA, #matrixA
    matrixB, #matrixB
    K, #the commond dimension
    stride_am, stride_ak,  # strides of A
    stride_bk, stride_bn,  # strides of B
    offs_am,  # row offsets for this block -> shape (BLOCK_M,)
    offs_bn,  # col offsets for this block -> shape (BLOCK_N,)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    offs_k = tl.arange(0, BLOCK_SIZE_K)  

    a_ptrs = matrixA + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = matrixB + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for kk in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - kk * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - kk * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    return acc


# @triton.autotune(
#     configs=get_cuda_autotune_config(),
#     key=["M", "N", "K"],
# )

"""
Best Setting
BLOCK_SIZE_M: 128
BLOCK_SIZE_N: 128
BLOCK_SIZE_K: 32
GROUP_SIZE_M: 8
num_warps: 4
num_ctas: 1 
num_stages: 4
"""
@triton.jit
def matmul_kernel(
    matrixA,
    matrixB,
    matrixC,
    M, N, K,
    stride_am, stride_ak, 
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # shape (BLOCK_M,)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # shape (BLOCK_N,)

    # call the portable block helper
    acc = matmul_block(
        matrixA,
        matrixB,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        offs_am,
        offs_bn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    if ACTIVATION == "leaky_relu":
        acc = leaky_relu(acc)

    c = acc.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_ptrs = matrixC + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    BLOCK_SIZE_M= 128
    BLOCK_SIZE_N= 128
    BLOCK_SIZE_K= 32
    GROUP_SIZE_M= 8
    num_warps= 4
    num_ctas= 1 
    num_stages= 4
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        ACTIVATION=activation,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages
    )
    return c


if __name__ == "__main__":
    DEVICE = torch.device("cuda")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[
                128 * i for i in range(2, 33)
            ],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            line_vals=["triton", "cublas"],  # Label name for the lines
            line_names=["Triton", "cublas"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-",  # Name for the plot, used also as a file name for saving the plot.
            args={},
        )
    )
    def benchmark(M, N, K, provider):
        a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
        b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

        quantiles = [0.5, 0.2, 0.8]
        if provider == "cuBLAS".lower():
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch.matmul(a, b), quantiles=quantiles
            )
        if provider == "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: triton_matmul(a, b), quantiles=quantiles
            )
        perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
        return perf(ms), perf(max_ms), perf(min_ms)

    benchmark.run(show_plots=True)
