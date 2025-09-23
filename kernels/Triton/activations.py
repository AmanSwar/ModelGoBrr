import triton
import triton.language as tl


@triton.jit
def relu(x):
    return tl.where(x > 0, x, 0)

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def swiglu(
    x,
    weight1_ptr,
    weight2_ptr
):
    y = x * weight2_ptr
    swish_y = y * tl.sigmoid(y)
    return weight1_ptr * row * swish_y
    