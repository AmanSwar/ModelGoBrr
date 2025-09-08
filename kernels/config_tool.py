import triton

def tuner(
    N : int
):
    """
    returns the optimal block size
    Inspired from Unsloth github
    https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/utils.py

    Args:
        N (int): cols
    Returns:
        Tuple(BLOCK_SIZE , num_warps)
    """
    BLOCK_SIZE : int = triton.next_power_of_2(N)

    num_warps : int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    return BLOCK_SIZE, num_warps
