import torch
import triton
import triton.language as tl
import torch.nn as nn

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def fused_ffn_kernel(
    X, Y, W1, W2, W3,
    N, EMBED_DIM, HIDDEN_DIM,
    stride_x_n, stride_x_emb,
    stride_w1_emb, stride_w1_hid,
    stride_w2_emb, stride_w2_hid,
    stride_w3_hid, stride_w3_emb,
    stride_y_n, stride_y_emb,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_EMB: tl.constexpr,
    BLOCK_SIZE_HID: tl.constexpr
):
    pid_n = tl.program_id(0)
    
    # Each program processes a BLOCK_SIZE_N chunk of rows from the input tensor X.
    # The computation is parallelized across the hidden dimension.
    
    # Offsets for the input rows
    offs_x_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Pointers to the input block
    x_ptrs = X + offs_x_n[:, None] * stride_x_n + tl.arange(0, BLOCK_SIZE_EMB)[None, :] * stride_x_emb
    
    # Accumulator for the final output
    acc_out = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_EMB), dtype=tl.float32)

    # Loop over the hidden dimension in blocks
    for hid_start in range(0, HIDDEN_DIM, BLOCK_SIZE_HID):
        offs_hid = hid_start + tl.arange(0, BLOCK_SIZE_HID)
        
        # --- First layer: Gate and Up projections ---
        # Pointers to weight matrices for the current hidden block
        w1_ptrs = W1 + tl.arange(0, BLOCK_SIZE_EMB)[:, None] * stride_w1_emb + offs_hid[None, :] * stride_w1_hid
        w2_ptrs = W2 + tl.arange(0, BLOCK_SIZE_EMB)[:, None] * stride_w2_emb + offs_hid[None, :] * stride_w2_hid
        
        # Accumulators for the first two matmuls
        acc1 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_HID), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_HID), dtype=tl.float32)
        
        # Loop over the embedding dimension in blocks
        for emb_start in range(0, EMBED_DIM, BLOCK_SIZE_EMB):
            # Load a block of input X
            x_block = tl.load(x_ptrs, mask=(tl.arange(0, BLOCK_SIZE_EMB)[None, :] < EMBED_DIM - emb_start), other=0.0)
            
            # Load blocks of weights W1 and W2
            w1_block = tl.load(w1_ptrs, mask=(tl.arange(0, BLOCK_SIZE_EMB)[:, None] < EMBED_DIM - emb_start), other=0.0)
            w2_block = tl.load(w2_ptrs, mask=(tl.arange(0, BLOCK_SIZE_EMB)[:, None] < EMBED_DIM - emb_start), other=0.0)
            
            # Perform matmuls
            acc1 += tl.dot(x_block, w1_block)
            acc2 += tl.dot(x_block, w2_block)
            
            # Advance pointers for the next block in the embedding dimension
            x_ptrs += BLOCK_SIZE_EMB * stride_x_emb
            w1_ptrs += BLOCK_SIZE_EMB * stride_w1_emb
            w2_ptrs += BLOCK_SIZE_EMB * stride_w2_emb

        # Reset input pointers for the next hidden block loop
        x_ptrs -= EMBED_DIM * stride_x_emb

        # --- Activation and element-wise product ---
        intermediate = silu(acc1) * acc2
        intermediate = intermediate.to(tl.float16)

        # --- Second layer: Down projection ---
        # Pointers to the weight matrix W3 for the current hidden block
        w3_ptrs = W3 + offs_hid[:, None] * stride_w3_hid + tl.arange(0, BLOCK_SIZE_EMB)[None, :] * stride_w3_emb
        
        # Loop over the embedding dimension for the output
        for emb_start in range(0, EMBED_DIM, BLOCK_SIZE_EMB):
            w3_block = tl.load(w3_ptrs, mask=(tl.arange(0, BLOCK_SIZE_EMB)[None, :] < EMBED_DIM - emb_start), other=0.0)
            
            # Perform matmul with the intermediate result
            acc_out += tl.dot(intermediate, w3_block)
            
            # Advance pointers
            w3_ptrs += BLOCK_SIZE_EMB * stride_w3_emb

    # Store the final result
    offs_y_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_ptrs = Y + offs_y_n[:, None] * stride_y_n + tl.arange(0, BLOCK_SIZE_EMB)[None, :] * stride_y_emb
    y_mask = (offs_y_n[:, None] < N)
    tl.store(y_ptrs, acc_out.to(tl.float16), mask=y_mask)


def fully_fused_ffn(x, w1, w2, w3):
    N, EMBED_DIM = x.shape
    HIDDEN_DIM, _ = w1.shape
    
    Y = torch.empty_like(x)
    
    # Ensure weights are transposed correctly for consumption by the kernel
    w1 = w1.t().contiguous()
    w2 = w2.t().contiguous()
    w3 = w3.t().contiguous()

    grid = (triton.cdiv(N, 16),)
    
    fused_ffn_kernel[grid](
        X=x, Y=Y, W1=w1, W2=w2, W3=w3,
        N=N, EMBED_DIM=EMBED_DIM, HIDDEN_DIM=HIDDEN_DIM,
        stride_x_n=x.stride(0), stride_x_emb=x.stride(1),
        stride_w1_emb=w1.stride(0), stride_w1_hid=w1.stride(1),
        stride_w2_emb=w2.stride(0), stride_w2_hid=w2.stride(1),
        stride_w3_hid=w3.stride(0), stride_w3_emb=w3.stride(1),
        stride_y_n=Y.stride(0), stride_y_emb=Y.stride(1),
        BLOCK_SIZE_N=16,
        BLOCK_SIZE_EMB=64,
        BLOCK_SIZE_HID=64
    )
    return Y

if __name__ == "__main__":
    import time

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        exit()

    class FFN(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int):
            super().__init__()
            self.linear_layer1 = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=False)
            self.linear_layerP = nn.Linear(in_features=in_dim, out_features=hidden_dim, bias=False)
            self.silu = nn.SiLU()
            self.linear_layer2 = nn.Linear(in_features=hidden_dim, out_features=in_dim, bias=False)

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

    print(f"Testing FFN with embed_dim={embed_dim}, hidden_dim={hidden_dim}, batch_size={batch_size}")

    ffn_torch = FFN(embed_dim, hidden_dim).to(DEVICE).half()
    input_tensor = torch.randn(batch_size, embed_dim, device=DEVICE, dtype=torch.float16)

    # --- Verification ---
    print("\n=== Verification ===")
    with torch.no_grad():
        output_torch = ffn_torch(input_tensor)
        output_triton = fully_fused_ffn(
            input_tensor,
            ffn_torch.linear_layer1.weight,
            ffn_torch.linear_layerP.weight,
            ffn_torch.linear_layer2.weight
        )

    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    mean_diff = torch.mean(torch.abs(output_torch - output_triton)).item()
    rel_diff = mean_diff / torch.mean(torch.abs(output_torch)).item()

    print(f"Output shapes - PyTorch: {output_torch.shape}, Triton: {output_triton.shape}")
    print(f"Max absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Relative difference: {rel_diff:.6f}")

    if rel_diff < 1e-2:
        print("✓ Verification PASSED")
    else:
        print("✗ Verification FAILED")

    # --- Benchmarking ---
    print("\n=== Benchmarking ===")
    num_warmup = 50
    num_iter = 100

    # Warmup
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = ffn_torch(input_tensor)
            _ = fully_fused_ffn(
                input_tensor,
                ffn_torch.linear_layer1.weight,
                ffn_torch.linear_layerP.weight,
                ffn_torch.linear_layer2.weight
            )
        torch.cuda.synchronize()

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iter):
        with torch.no_grad():
            _ = ffn_torch(input_tensor)
    torch.cuda.synchronize()
    torch_time = (time.time() - start_time) / num_iter * 1000

    # Benchmark Triton
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iter):
        _ = fully_fused_ffn(
            input_tensor,
            ffn_torch.linear_layer1.weight,
            ffn_torch.linear_layerP.weight,
            ffn_torch.linear_layer2.weight
        )
    torch.cuda.synchronize()
    triton_time = (time.time() - start_time) / num_iter * 1000

    print(f"PyTorch time: {torch_time:.3f} ms")
    print(f"Triton time (fully fused): {triton_time:.3f} ms")
    print(f"Speedup: {torch_time/triton_time:.2f}x")