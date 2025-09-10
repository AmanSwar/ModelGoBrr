from typing import Any
import triton
import triton.language as tl

import torch
from torch.autograd import Function

from .config_tool import tuner

# @triton.jit
# def _rmsnorm_fwd(
#     input_matrix,
#     output_matrix,
#     weight_matrix,
#     M , N,
#     eps,
#     BLOCK_SIZE : tl.constexpr

# ):

#     row_index = tl.program_id(0)

#     row_start = row_index * N

#     cols_offset = tl.arange(0 , BLOCK_SIZE)

#     input_ptrs = input_matrix + row_start + cols_offset

#     mask = cols_offset < N

#     row = tl.load(input_ptrs , mask=mask , other=0.0)
#     weight = tl.load(weight_matrix + cols_offset , mask=mask , other=1.0)

#     _rms = tl.rsqrt(((tl.sum(row * row))/N) + eps)

#     out_row = (row * _rms) * weight

#     output_ptrs = output_matrix + row_start + cols_offset
#     tl.store(output_ptrs , out_row , mask=mask)

# @triton.autotune(
#     [
#         triton.Config(
#             {"BLOCK_SIZE": BLOCK_SIZE},
#             num_stages=num_stages,
#             num_warps=num_warps,
#         )
#         for BLOCK_SIZE in [32 ,64, 128 , 256]
#         for num_stages in ([3, 4, 7])
#         for num_warps in [2, 4]
#     ],
#     key=["N"],
# )

"""
Best config :
BLOCK_SIZE : 256
num_warps: 2
num_ctas: 1
num_stages: 4
"""
@triton.jit
def _rmsnorm_fwd(
    input_matrix,
    output_matrix,
    weight_matrix,
    M , N,
    eps,
    BLOCK_SIZE : tl.constexpr
):
    row_index = tl.program_id(0)
    row_start = row_index * N
    _sum_sq = 0.0
    for block in range(tl.cdiv(N , BLOCK_SIZE)):
        col_ptrs = block * BLOCK_SIZE + tl.arange(0 , BLOCK_SIZE)
        input_ptrs = input_matrix + row_start + col_ptrs
        mask = col_ptrs < N
        row_block = tl.load(input_ptrs , mask=mask , other=0.0)
        _sum_sq += tl.sum(row_block * row_block)

    _rms = tl.rsqrt((_sum_sq/ N) + eps)

    for block in range(tl.cdiv(N , BLOCK_SIZE)):
        col_ptrs = block * BLOCK_SIZE + tl.arange(0 , BLOCK_SIZE)
        input_ptrs = input_matrix + row_start + col_ptrs
        output_ptrs = output_matrix + row_start + col_ptrs
        weight_ptrs = weight_matrix + col_ptrs

        mask = col_ptrs < N
        row_block = tl.load(input_ptrs, mask=mask , other=0.0)
        weight_block = tl.load(weight_ptrs , mask=mask , other=1.0)

        out_row = (row_block * _rms) * weight_block

        tl.store(output_ptrs , out_row , mask=mask)



def rmsnorm_triton(x , weight , eps=1e-6):
    assert x.shape[-1] == weight.shape[0], "Feature dimension mismatch"

    # get all dims
    *batch_dims, N = x.shape

    M = x.numel() // N

    x_2d_view = x.reshape(M , N)
    y = torch.empty_like(x_2d_view)

    grid = (M,)
    BLOCK_SIZE = 256
    num_warps= 2
    num_ctas= 1
    num_stages= 4
    _rmsnorm_fwd[grid](
        input_matrix=x_2d_view,
        output_matrix=y,
        weight_matrix=weight,
        M=M,
        N=N,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages
    )
    return y.view(x.shape)


class RMSNormTriton(torch.nn.Module):

    def __init__(self , embed_dim , eps=1e-6):

        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self,  x):
        return rmsnorm_triton(x, self.weight.to(device=x.device), self.eps)


if __name__ == "__main__":
    # TOTALLY COPIED FROM CLAUDE CUZ WHO WRITES BOILERPLATE BENCHMARKING

    def benchmark_rmsnorm():
        """Benchmark against PyTorch native implementation"""

        configs = [
            (1024, 4096),  # Small
            (2048, 4096),  # Medium
            (4096, 4096),  # Large
            (1024, 8192),  # Wide
        ]

        device = torch.device("cuda")

        for M, N in configs:
            print(f"\nBenchmarking M={M}, N={N}")

            x = torch.randn(M, N, device=device, requires_grad=True)
            weight = torch.randn(N, device=device, requires_grad=True)
            grad_out = torch.randn(M, N, device=device)

            rmsnorm = torch.nn.RMSNorm(N).to(device)
            rmsnormtriton = RMSNormTriton(N).to(device)

            for _ in range(10):
                y1 = rmsnormtriton(x)
                y2 = rmsnorm(x)

            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(100):
                y_triton = rmsnormtriton(x)
            end.record()
            torch.cuda.synchronize()
            triton_time = start.elapsed_time(end)

            start.record()
            for _ in range(100):
                y_torch = rmsnorm(x)
            end.record()
            torch.cuda.synchronize()
            torch_time = start.elapsed_time(end)

            print(f"Triton: {triton_time:.2f}ms")
            print(f"PyTorch: {torch_time:.2f}ms")
            print(f"Speedup: {torch_time/triton_time:.2f}x")

            max_diff = (y_triton - y_torch).abs().max().item()
            print(f"Max difference: {max_diff:.2e}")

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['M'],  
            x_vals=[128, 256, 512, 1024, 2048, 4096, 8192], 
            line_arg='provider', 
            line_vals=['triton', 'torch'],  
            line_names=['Triton RMSNorm', 'PyTorch RMSNorm'],
            styles=[('blue', '-'), ('red', '--')], 
            ylabel='GB/s',  
            plot_name='rmsnorm-performance', 
            args={'N': 4096}  
        )
    )
    def benchmark_rmsnorm_plot(M, N, provider):
        """Benchmark function for Triton's perf_report"""
        device = torch.device("cuda")

        
        x = torch.randn(M, N, device=device, dtype=torch.float32)

        
        if provider == 'torch':
            model = torch.nn.RMSNorm(N).to(device)
            fn = lambda: model(x)
        elif provider == 'triton':
            model = RMSNormTriton(N).to(device) 
            fn = lambda: model(x)

        
        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

        
        total_bytes = (2 * M * N + N) * 4
        gbps = total_bytes / (ms * 1e-3) / 1e9

        return gbps

    
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['N'],  
            x_vals=[1024, 2048, 4096, 8192, 16384],  
            line_arg='provider',
            line_vals=['triton', 'torch'],
            line_names=['Triton RMSNorm', 'PyTorch RMSNorm'],
            styles=[('blue', '-'), ('red', '--')],
            ylabel='GB/s',
            plot_name='rmsnorm-feature-dim-performance',
            args={'M': 2048}  
        )
    )
    def benchmark_rmsnorm_feature_dim(M, N, provider):
        """Benchmark varying feature dimensions"""
        device = torch.device("cuda")

        x = torch.randn(M, N, device=device, dtype=torch.float32)

        if provider == 'torch':
            model = torch.nn.RMSNorm(N).to(device)
            fn = lambda: model(x)
        elif provider == 'triton':
            model = RMSNormTriton(N).to(device)
            fn = lambda: model(x)

        ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=[0.5, 0.2, 0.8])

        # Calculate GB/s
        total_bytes = (2 * M * N + N) * 4
        gbps = total_bytes / (ms * 1e-3) / 1e9

        return gbps
    benchmark_rmsnorm()


    print("Running RMSNorm benchmark (varying batch size)...")
    benchmark_rmsnorm_plot.run(show_plots=True)

    print("Running RMSNorm benchmark (varying feature dimension)...")
    benchmark_rmsnorm_feature_dim.run(show_plots=True)
