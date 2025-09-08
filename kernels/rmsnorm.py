import triton
import triton.language as tl

import torch

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
    
    cols_offset = tl.arange(0 , BLOCK_SIZE)

    input_ptrs = input_matrix + row_start + cols_offset

    mask = cols_offset < N
    
    row = tl.load(input_ptrs , mask=mask , other=0.0)
    weight = tl.load(weight_matrix + cols_offset , mask=mask , other=1.0)

    _rms = tl.rsqrt(((tl.sum(row * row))/N) + eps)

    out_row = (row * _rms) * weight

    output_ptrs = output_matrix + row_start + cols_offset
    tl.store(output_ptrs , out_row , mask=mask)


def _rmsnorm_bwd(
    input_matrix,
    weight_matrix,
    grad_matrix,
    grad_x_matrix,
    grad_w_acc_matrix,
    M , N , grad_matrix_stride,
    eps,
    BLOCKS_SIZE : tl.constexpr,
):

    #base indexing
    row_index = tl.program_id(0)
    cols_offset = tl.arange(0,BLOCKS_SIZE)

    #ptrs for all 
    input_ptrs = input_matrix + row_index * N + cols_offset
    grad_ptrs = grad_matrix + row_index * N + cols_offset
    weight_ptrs = weight_matrix + cols_offset

    grad_w_ptrs = grad_w_acc_matrix + row_index * N + cols_offset
    grad_x_ptrs = grad_x_matrix + row_index * N + cols_offset

    mask = cols_offset < N

    #load
    input_row = tl.load(input_ptrs , mask=mask , other=0.0)
    grad_row = tl.load(grad_ptrs , mask=mask , other=0.0)
    weight = tl.load(weight_ptrs , mask=mask  ,other=0.0)


    #recompute RMS
    _rms = tl.sqrt((tl.sum(input_row * input_row) / N) +eps)
    

    grad_weight_partial = (input_row / _rms) * grad_row

    tl.store(grad_w_ptrs,grad_weight_partial,mask=mask)

    _first_term = (grad_row/ _rms) * weight

    _second_term_num = tl.sum(input_row * weight * grad_row)
    _second_term_denom = N + _rms * _rms * _rms

    grad_x = _first_term - (_second_term_num / _second_term_denom)

    tl.store(grad_x_ptrs , grad_x , mask=mask)



