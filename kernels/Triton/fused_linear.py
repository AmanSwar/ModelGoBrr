import triton
import triton.language as tl

import torch
from .matmul import matmul_block

from .activations import silu , relu

@triton.jit
def fused_linear_kernel(
    input_matrix,
    weight_matrix,
    bias,
    output_matrix,
    M,
    N,
    K,
    stride_am,
    stride_ak, 
    stride_wk,
    stride_wn, 
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1) 

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    acc = matmul_block(
        input_matrix,
        weight_matrix,
        K,
        stride_am,
        stride_ak,
        stride_wk,
        stride_wn,
        offs_am,
        offs_bn,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    bias_offs = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias_mask = bias_offs < N
    bias_vals = tl.load(bias + bias_offs, mask=bias_mask, other=0.0)
    acc += bias_vals[None, :]

    if ACTIVATION == "relu":
        acc = relu(acc)
    elif ACTIVATION == "silu":
        acc = silu(acc)

    out = acc.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = output_matrix + stride_om * offs_cm[:, None] + stride_on * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, out, mask=c_mask)


def launch_fused_linear_kernel(
    input_matrix: torch.Tensor,  
    weight: torch.Tensor, 
    bias: torch.Tensor, 
    activation: str = "none",
):
    M, K = input_matrix.shape
    K_weight, N = weight.shape

    assert (
        K == K_weight
    ), f"Input K dimension {K} must match weight K dimension {K_weight}"
    assert (
        bias.shape[0] == N
    ), f"Bias dimension {bias.shape[0]} must match output N dimension {N}"

    output_matrix = torch.empty(
        (M, N), dtype=input_matrix.dtype, device=input_matrix.device
    )

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    # Launch kernel
    fused_linear_kernel[grid](
        input_matrix,
        weight,
        bias,
        output_matrix,
        M,
        N,
        K,
        input_matrix.stride(0),
        input_matrix.stride(1),  
        weight.stride(0),
        weight.stride(1), 
        output_matrix.stride(0),
        output_matrix.stride(1), 
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ACTIVATION=activation,
    )

    return output_matrix


class TritonFusedLinear(torch.nn.Module):
    """Triton-accelerated fused linear layer with activation"""

    def __init__(self, in_features, out_features, bias=True, activation="none"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # Initialize weight and bias
        self.weight = torch.nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        return launch_fused_linear_kernel(x, self.weight, self.bias, self.activation)




if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    class PyTorchLinear(torch.nn.Module):
        """Standard PyTorch linear layer for comparison"""

        def __init__(self, in_features, out_features, bias=True, activation="none"):
            super().__init__()
            self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
            self.activation = activation

        def forward(self, x):
            out = self.linear(x)
            if self.activation == "relu":
                return torch.nn.functional.relu(out)
            elif self.activation == "silu":
                return torch.nn.functional.silu(out)
            return out

    def benchmark_function(fn, *args, warmup=10, iterations=100):
        """Benchmark a function with proper GPU synchronization"""
        # Warmup
        for _ in range(warmup):
            fn(*args)
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            fn(*args)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        return (end_time - start_time) / iterations


    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=['size'],
            x_vals=[2**i for i in range(8, 14)],
            line_arg='provider',
            line_vals=['triton-none', 'pytorch-none', 'triton-relu', 'pytorch-relu', 'triton-silu', 'pytorch-silu'],
            line_names=['Triton (no activation)', 'PyTorch (no activation)', 'Triton + ReLU', 'PyTorch + ReLU', 'Triton + SiLU', 'PyTorch + SiLU'],
            styles=[('blue', '-'), ('red', '--'), ('green', '-'), ('orange', '--'), ('purple', '-'), ('brown', '--')],
            ylabel='Runtime (ms) (Lower = better)',
            plot_name='fused-linear-performance', # This is the FILENAME
            args={}
        )
    )
    def benchmark_linear_layers(size, provider):
        """Benchmark different linear layer implementations"""
        # This function remains the same
        batch_size = 1024
        in_features = size
        out_features = size
        device = 'cuda'
        dtype = torch.float16
        x = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        
        # Dummy benchmark function for demonstration
        def benchmark_function(func):
            # In a real scenario, this would use triton.testing.do_bench
            # For this example, let's just return a dummy value
            return 0.1 

        if provider.startswith('triton'):
            activation = provider.split('-')[1]
            # model = TritonFusedLinear(in_features, out_features, bias=True, activation=activation).to(device).to(dtype)
            def run_triton():
                # return model(x)
                pass
            return benchmark_function(run_triton) * 1000
        else:  # pytorch
            activation = provider.split('-')[1]
            # model = PyTorchLinear(in_features, out_features, bias=True, activation=activation).to(device).to(dtype)
            def run_pytorch():
                # return model(x)
                pass
            return benchmark_function(run_pytorch) * 1000


    benchmark_linear_layers.run(show_plots=False)
