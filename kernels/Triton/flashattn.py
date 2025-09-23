import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i,
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the current q position
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q)
    else:
        # Only used for non-causal attention
        lo, hi = 0, SEQ_LEN

    
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # loop over k, v and update accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # let the compiler know that start_kv is a multiple of block_size_kv so that compiler can do optimization -> according to official triton docs
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)

        # -- compute qk ----
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)

        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1))
            QK_block -= m_ij[:, None]
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]

        # Compute the exponential of each dot product, so now we are computing exp(qk_ij - m_ij)
        P_block = tl.math.exp(QK_block)
        # Compute the sum by rows of the attention scores
        l_ij = tl.sum(P_block, 1)

        # This is the correction factor for the previous l_i
        alpha = tl.math.exp(m_i - m_ij)
        # Apply the correction factor to the previous l_i and add the new l_ij
        l_i = l_i * alpha + l_ij

        V_block = tl.load(V_block_ptr)
        P_block = P_block.to(tl.float16)
        V_block = V_block.to(tl.float16)
        # This computes the following: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None]
        O_block = tl.dot(P_block, V_block, O_block)

        m_i = m_ij

        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV))
    return O_block, l_i, m_i


# @triton.autotune(
#     [
#         triton.Config(
#             {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
#             num_stages=num_stages,
#             num_warps=num_warps,
#         )
#         for BLOCK_SIZE_Q in [64, 128]
#         for BLOCK_SIZE_KV in [32, 64]
#         for num_stages in ([3, 4, 7])
#         for num_warps in [2, 4]
#     ],
#     key=["SEQ_LEN", "HEAD_DIM"],
# )
"""
BEST CONFIG:
BLOCK_SIZE_Q : 128
BLOCK_SIZE_KV: 64
num_warps: 4
num_ctas: 1
num_stages: 3
"""
@triton.jit
def _attn_fwd(
    Q,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale,
    M,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O,  # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch,
    stride_Q_head,
    stride_Q_seq,
    stride_Q_dim,
    stride_K_batch,
    stride_K_head,
    stride_K_seq,
    stride_K_dim,
    stride_V_batch,
    stride_V_head,
    stride_V_seq,
    stride_V_dim,d
    stride_O_batch,
    stride_O_head,
    stride_O_seq,
    stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM)

    #pid along the seq len dimension
    block_index_q = tl.program_id(0)

    #pid for batch_dim * n_heads [[[head1] , [head2] ..] ... ]
    index_batch_head = tl.program_id(1)
    
    #which batch am I in
    index_batch = index_batch_head // NUM_HEADS
    # which head am I in
    index_head = index_batch_head % NUM_HEADS

    #offset to reach the particular qkv value from batch and head
    qvk_offset = (
        index_batch.to(tl.int64) * stride_Q_batch
        + index_head.to(tl.int64) * stride_Q_head
    )


    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, #where the block starts
        shape=(SEQ_LEN, HEAD_DIM), #shape of the block
        strides=(stride_Q_seq, stride_Q_dim), #strides of the block
        offsets=(block_index_q * BLOCK_SIZE_Q, 0), # starting offset of the block
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM), #shape of block 
        order=(1, 0), #column major mem order -> why ? because we traverse along the head_dim which is columns
    )


    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_V_seq, stride_V_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_KV, HEAD_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, SEQ_LEN), #transpose the matrix (head_dim , seq_len)
        strides=(stride_K_dim,stride_K_seq,),  
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_KV),
        order=(0, 1), #row major as we are traversing along the row axis
    )

    O_block_ptr = tl.make_block_ptr(
        base=O + qvk_offset,
        shape=(SEQ_LEN, HEAD_DIM),
        strides=(stride_O_seq, stride_O_dim),
        offsets=(block_index_q * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    offs_kv = tl.arange(0, BLOCK_SIZE_KV)

    #running maxium (total size -> block_size_q as each row will have one)
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")

    #running sum for each row
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0

    #output acc
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)

    #load the O block so that it can be in SRAM hence writing speed would be high
    Q_block = tl.load(Q_block_ptr)

    #Stage 3 -> casual
    if STAGE == 1 or STAGE == 3:
        #attn for diagonal 
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )

    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i,
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    # epilogue
    m_i += tl.math.log(
        l_i
    )  # This is needed to compute the logsumexp for the backwards pass
    O_block = O_block / l_i[:, None]
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))


def launch_attn(
    Q , K , V,
    causal,
    softmax_scale
):
    HEAD_DIM_Q, HEAD_DIM_K = Q.shape[-1], K.shape[-1]
    HEAD_DIM_V = V.shape[-1]

    BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape

    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V

    O = torch.empty_like(Q)
    stage = 3 if causal else 1

    grid = lambda args: (
        triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]),
        BATCH_SIZE * NUM_HEADS,
        1,
    )

    # M is the logsumexp for the backward pass, one for each query
    M = torch.empty(
        (BATCH_SIZE, NUM_HEADS, SEQ_LEN), device=Q.device, dtype=torch.float32
    )
    BLOCK_SIZE_Q =128
    BLOCK_SIZE_KV= 64
    num_warps= 4
    num_ctas= 1
    num_stages= 3
    _attn_fwd[grid](
        Q=Q,
        K=K,
        V=V,
        softmax_scale=softmax_scale,
        M=M,
        O=O,
        stride_Q_batch=Q.stride(0),
        stride_Q_head=Q.stride(1),
        stride_Q_seq=Q.stride(2),
        stride_Q_dim=Q.stride(3),
        stride_K_batch=K.stride(0),
        stride_K_head=K.stride(1),
        stride_K_seq=K.stride(2),
        stride_K_dim=K.stride(3),
        stride_V_batch=V.stride(0),
        stride_V_head=V.stride(1),
        stride_V_seq=V.stride(2),
        stride_V_dim=V.stride(3),
        stride_O_batch=O.stride(0),
        stride_O_head=O.stride(1),
        stride_O_seq=O.stride(2),
        stride_O_dim=O.stride(3),
        BATCH_SIZE=Q.shape[0],
        NUM_HEADS=Q.shape[1],
        SEQ_LEN=Q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        STAGE=stage,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_KV=BLOCK_SIZE_KV,
        num_warps=num_warps,
        num_ctas=num_ctas,
        num_stages=num_stages
    )

    return O


def pytorch_attention(q, k, v, causal, softmax_scale):
    """
    Standard PyTorch implementation of scaled dot-product attention
    for verification.
    """
    # (B, H, S, D) x (B, H, D, S) -> (B, H, S, S)
    S = (q @ k.transpose(-2, -1)) * softmax_scale
    if causal:
        # Create a causal mask
        mask = torch.triu(
            torch.ones(S.shape[-2], S.shape[-1], device=q.device), diagonal=1
        )
        S = S.masked_fill(mask.bool(), float("-inf"))

    # Apply softmax to get attention weights
    P = torch.softmax(S, dim=-1)

    # (B, H, S, S) x (B, H, S, D) -> (B, H, S, D)
    return P @ v


def run_benchmark_and_verify():
    """
    Main function to run verification and benchmarks.
    """
    # Test configurations
    BATCH, N_HEADS, SEQ_LEN, D_HEAD = 4, 16, 1024, 64
    softmax_scale = 1.0 / (D_HEAD**0.5)

    # Generate random tensors
    q = torch.randn(
        (BATCH, N_HEADS, SEQ_LEN, D_HEAD),
        dtype=torch.float16,
        device="cuda",
        requires_grad=False,
    )
    k = torch.randn(
        (BATCH, N_HEADS, SEQ_LEN, D_HEAD),
        dtype=torch.float16,
        device="cuda",
        requires_grad=False,
    )
    v = torch.randn(
        (BATCH, N_HEADS, SEQ_LEN, D_HEAD),
        dtype=torch.float16,
        device="cuda",
        requires_grad=False,
    )

    print("--- Verification ---")
    # --- Test non-causal attention ---
    triton_output_non_causal = launch_attn(
        q, k, v, causal=False, softmax_scale=softmax_scale
    )
    pytorch_output_non_causal = pytorch_attention(
        q, k, v, causal=False, softmax_scale=softmax_scale
    )

    is_correct_non_causal = torch.allclose(
        triton_output_non_causal, pytorch_output_non_causal, atol=1e-2, rtol=0
    )
    print(f"Non-Causal Attention Correct: {is_correct_non_causal}")

    # --- Test causal attention ---
    triton_output_causal = launch_attn(
        q, k, v, causal=True, softmax_scale=softmax_scale
    )
    pytorch_output_causal = pytorch_attention(
        q, k, v, causal=True, softmax_scale=softmax_scale
    )

    is_correct_causal = torch.allclose(
        triton_output_causal, pytorch_output_causal, atol=1e-2, rtol=0
    )
    print(f"Causal Attention Correct: {is_correct_causal}")

    print("\n--- Benchmark (ms) ---")

    # Use triton.testing.do_bench for accurate measurements
    quantiles = [0.2, 0.5, 0.8]

    # Benchmark non-causal
    ms_triton_non_causal, _, _ = triton.testing.do_bench(
        lambda: launch_attn(q, k, v, causal=False, softmax_scale=softmax_scale),
        quantiles=quantiles,
    )
    ms_pytorch_non_causal, _, _ = triton.testing.do_bench(
        lambda: pytorch_attention(q, k, v, causal=False, softmax_scale=softmax_scale),
        quantiles=quantiles,
    )

    # Benchmark causal
    ms_triton_causal, _, _ = triton.testing.do_bench(
        lambda: launch_attn(q, k, v, causal=True, softmax_scale=softmax_scale),
        quantiles=quantiles,
    )
    ms_pytorch_causal, _, _ = triton.testing.do_bench(
        lambda: pytorch_attention(q, k, v, causal=True, softmax_scale=softmax_scale),
        quantiles=quantiles,
    )

    print(f"{'Mode':<20} | {'Triton':<10} | {'PyTorch':<10}")
    print("-" * 45)
    print(
        f"{'Non-Causal Attention':<20} | {ms_triton_non_causal:10.3f} | {ms_pytorch_non_causal:10.3f}"
    )
    print(
        f"{'Causal Attention':<20} | {ms_triton_causal:10.3f} | {ms_pytorch_causal:10.3f}"
    )


if __name__ == "__main__":
    run_benchmark_and_verify()
