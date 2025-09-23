import torch
from dataclasses import dataclass

@dataclass
class QwenConfig_bfloat16:
    vocab_size : int = 151936
    context_length : int = 32768
    embed_dim : int = 1024
    n_heads : int = 16
    n_layers : int = 28
    head_dim : int = 128
    n_kv_heads : int = 8  
    hidden_dim : int = 3072
    qk_norm : bool = True
    rope_base : float = 1e6
    dtype : torch.dtype = torch.bfloat16


@dataclass
class QwenConfig_float16:
    vocab_size: int = 151936
    context_length: int = 32768
    embed_dim: int = 1024
    n_heads: int = 16
    n_layers: int = 28
    head_dim: int = 128
    n_kv_heads: int = 8
    hidden_dim: int = 3072
    qk_norm: bool = True
    rope_base: float = 1e6
    dtype: torch.dtype = torch.float16
