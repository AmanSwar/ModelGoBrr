import torch
import torch.nn as nn

from qwen3.config import QwenConfig


def compute_rope_params(
    head_dim,
    theta_base,
    context_length,
    dtype= torch.float32
):
    assert head_dim % 2 == 0 , "head dim must be divisible by 2"

    inv_freq = 1 / (theta_base ** (torch.arange(0 , head_dim  , 2 , dtype=dtype)[: head_dim//2]/float() / head_dim))

    position = torch.arange(context_length , dtype=dtype)
    angles = position[: , None] * inv_freq[None, :]

    angles = torch.cat([angles , angles] , dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos , sin



def apply_rope(
        x : torch.Tensor, 
        cos : torch.Tensor, 
        sin : torch.Tensor
    ):
    batch_size , num_heads , seq_len , head_dim = x.shape

    assert head_dim % 2 == 0 , "Head dim is not divisible by 2"

    x1 = x[... , :head_dim // 2]
    x2 = x[... , head_dim// 2 : ]

    cos = cos[:seq_len , :].unsqueeze(0).unsqueeze(0) #-> (1 , 1 , seq_len , head_dim)
    sin = sin[:seq_len , :].unsqueeze(0).unsqueeze(0) #-> (1 , 1 , seq_len , head_dim)

    rotated = torch.cat((-x2 , x1) , dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)





class GQA(nn.Module):
    """ Grouped Query Attentio"""  

    def __init__(
        self,
        d_in : int,
        num_heads : int, 
        num_kv_grps : int,
        head_dim : int | None = None,
        qk_norm : bool = True,
        dtype = None
    ):
        super().__init__()

        assert num_heads % num_kv_grps == 0 , "Num heads is not divisible by num kv grps"

        self.num_heads = num_heads
        self.num_kv_grps = num_kv_grps
        self.grp_size = num_heads // num_kv_grps

        if head_dim is None:
            assert d_in % num_heads == 0 , "in dimension must be divisible by number of heads"
            head_dim = d_in // num_heads

        self.head_dim : int = head_dim
        self.d_out = self.head_dim * self.num_heads

        self.Wq = nn.Linear(in_features=d_in , out_features= self.d_out , bias=False , dtype=dtype)
        self.Wk = nn.Linear(in_features=d_in , out_features= self.num_kv_grps * self.head_dim , bias=False , dtype=dtype)
        self.Wv = nn.Linear(in_features=d_in , out_features= self.num_kv_grps * self.head_dim , bias=False , dtype=dtype)

        self.out_projection = nn.Linear(in_features=self.d_out , out_features=d_in , bias=False,dtype=dtype)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim , eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim , eps=1e-6)

        else:
            self.q_norm = self.k_norm = None

    def forward(self , x , mask , cos , sin):

        bs , seq_len , _ = x.shape
        
        Q : torch.Tensor = self.Wq(x)
        K : torch.Tensor = self.Wk(x)
        V : torch.Tensor = self.Wv(x)

        Q = Q.view(bs , seq_len , self.num_heads ,self.head_dim).transpose(1,2)
        K = K.view(bs , seq_len , self.num_kv_grps ,self.head_dim).transpose(1,2)
        V = V.view(bs , seq_len , self.num_kv_grps ,self.head_dim).transpose(1,2)

        if self.q_norm:
            Q = self.q_norm(Q)

        if self.k_norm:
            K = self.k_norm(K)

        Q = apply_rope(Q ,cos , sin)
        K = apply_rope(K , cos , sin)

        K = K.repeat_interleave(self.grp_size , dim=1)
        V = V.repeat_interleave(self.grp_size , dim=1)

        scores = Q @ K.transpose(2,3)
        scores = scores.masked_fill(mask , -torch.inf).to(x.dtype)

        scores = torch.softmax(scores / (self.head_dim**0.5) , dim=-1)
        
        attn_out = (scores @ V).transpose(1,2).reshape(bs , seq_len , self.d_out)

        return self.out_projection(attn_out)
    


class FFN(nn.Module):
    
    def __init__(self , in_dim : int ,  hidden_dim : int):
        super().__init__()

        self.linear_layer1 = nn.Linear(in_features=in_dim , out_features=hidden_dim , bias=False)
        self.linear_layerP = nn.Linear(in_features=in_dim , out_features=hidden_dim , bias=False)
        self.silu = nn.SiLU()
        self.linear_layer2 =  nn.Linear(in_features=hidden_dim , out_features=in_dim , bias=False)

    def forward(self, x):
        x_l = self.linear_layer1(x)
        x_p = self.linear_layerP(x)
        x = self.silu(x_l)
        x = x * x_p
        x = self.linear_layer2(x)
        return x


class Transformer(nn.Module):
    
    def __init__(
        self,
        cfg : QwenConfig
    ):
        super().__init__()

        self.attn = GQA(
            d_in = cfg.embed_dim,
            num_heads= cfg.n_heads,
            head_dim=cfg.head_dim,
            num_kv_grps=cfg.n_kv_grps,
            qk_norm=cfg.qk_norm,
            dtype=cfg.dtype
        )

        self.rms_norm1 = nn.RMSNorm(cfg.embed_dim ,eps=1e-6)
        self.rms_norm2 = nn.RMSNorm(cfg.embed_dim ,eps=1e-6)
        
        self.ffn = FFN(cfg.embed_dim , cfg.hidden_dim)

    def forward(self , x , mask , cos , sin):
        x_res = x
        x = self.rms_norm1(x)
        x = x.to(torch.bfloat16)
        assert x.dtype == torch.bfloat16 , "input not in bfloat16"
        x = self.attn(x , mask , cos , sin)

        x = x + x_res

        x_res = x
        x = self.rms_norm2(x)
        x = self.ffn(x)
        x = x + x_res

        return x
    


class Qwen3(nn.Module):
    
    def __init__(self , cfg : QwenConfig):

        super().__init__()

        self.tok_embed = nn.Embedding(cfg.vocab_size , cfg.embed_dim , dtype=cfg.dtype)

        self.transformer_blocs = nn.ModuleList(
            [Transformer(cfg=cfg) for _ in range(cfg.n_layers)]
        )

        self.final_rmsnorm = nn.RMSNorm(cfg.embed_dim)
        
        self.out_head = nn.Linear(cfg.embed_dim , cfg.vocab_size , bias=False , dtype=cfg.dtype)

        if cfg.head_dim is None:
            head_dim = cfg.embed_dim // cfg.n_head

        else:
            head_dim = cfg.head_dim

        cos , sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg.rope_base,
            context_length=cfg.context_length
        )

        self.register_buffer("cos" , cos , persistent=False)
        self.register_buffer("sin" , sin , persistent=False)

        self.cfg = cfg

    def forward(self, x):

        token_embed : torch.Tensor = self.tok_embed(x)

        x = token_embed
        num_tokens = x.shape[1]

        mask = torch.triu(torch.ones(num_tokens , num_tokens , device=token_embed.device , dtype=torch.bool) ,diagonal=1)

        assert x.dtype == torch.bfloat16 , "input not in bfloat16"
        for block in self.transformer_blocs:
            x = block(x , mask , self.cos , self.sin)

        x = self.final_rmsnorm(x)

        logits = self.out_head(x.to(self.cfg.dtype))
        return logits
    

    
