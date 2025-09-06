
from dataclasses import dataclass

@dataclass
class QwenConfig:
    vocab_size : int = 151936
    context_length : int = 32000
    embed_dim : int = 1024
    n_heads : int = 16
    n_layers : int = 28
    head_dim : int = 128
    n_kv_grps : int = 2 # since qwen3 0.6B have 16/8 split of Q and KV heads -> 16 / 8 -> grps of 2
    hidden_dim : int = 3072
    qk_norm : bool = True
    rope_base : float = 1e6
    dtype : torch.dtype = torch.bfloat16
    