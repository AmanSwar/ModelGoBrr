from typing import Any
import triton
import triton.language as tl

import torch
from torch.autograd import Function

from .config_tool import tuner

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

@triton.jit
def _rmsnorm_bwd(
    input_matrix,
    weight_matrix,
    grad_matrix,
    grad_x_matrix,
    grad_w_acc_matrix,
    M , N ,
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


class RMSNormTritonFunction(Function):

    @staticmethod
    def forward(ctx , x : torch.Tensor , weight , eps=1e-6):
        assert x.is_contiguous(), "Input must be contiguous"
        assert weight.is_contiguous(), "Weight must be contiguous"
        assert x.shape[-1] == weight.shape[0], "Feature dimension mismatch"

        # get all dims
        *batch_dims, N = x.shape

        M = x.numel() // N

        x_2d_view = x.view(M , N)
        y = torch.empty_like(x_2d_view)

        BLOCK_SIZE , num_warps = tuner(N)
        grid = (M,)

        _rmsnorm_fwd[grid](
            input_matrix=x_2d_view,
            output_matrix=y,
            weight_matrix=weight,
            M=M,
            N=N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps
        )

        ctx.save_for_backward(x_2d_view , weight)
        ctx.eps = eps
        ctx.N = N
        ctx.original_shape = x.shape

        return y.view(x.shape)

    @staticmethod
    def backward(ctx, grad_outputs):

        x_2d_view , weight = ctx.saved_tensors
        eps = ctx.eps
        N = ctx.N
        original_shape = ctx.original_shape

        grad_outputs = grad_outputs.contiguous().view(x_2d_view.shape)

        M = x_2d_view.shape[0]

        grad_x = torch.zeros_like(x_2d_view)
        grad_weight = torch.zeros_like(weight)

        grid = (M,)
        BLOCK_SIZE , num_warps = tuner(N)

        _rmsnorm_bwd[grid](
            input_matrix=x_2d_view,
            weight_matrix=weight,
            grad_matrix=grad_outputs,
            grad_x_matrix=grad_x,
            grad_w_acc_matrix=grad_weight,
            M=M,
            N=N,
            eps=eps,
            BLOCKS_SIZE= BLOCK_SIZE,
            num_warps=num_warps
        )


        return grad_x , grad_weight
    

def _rmsnorm(x , weight , eps=1e-6):
    return RMSNormTriton.apply(x  ,weight , eps)


class RMSNormTriton(torch.nn.Module):

    def __init__(self , embed_dim , eps=1e-6):

        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self,  x):
        return _rmsnorm(x , self.weight , self.eps)
    

