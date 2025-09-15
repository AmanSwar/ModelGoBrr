from dataclasses import dataclass


@dataclass
class StableDiffConfig:
    clip_emebd : int = 49408
    model_dim : int  = 768
    seq_len : int = 77
    n_heads : int = 12