import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttentionCQKV(nn.Module):

    def __init__(
        self,
        n_heads ,
        embed_dim,
        in_proj_bias=False,
        out_proj_bias=False
    ):
        """
        Implementation of self attention but with common QKV weight
        """
        super().__init__()

        self.in_proj = nn.Linear(embed_dim , embed_dim * 3 , bias=in_proj_bias)

        self.out_proj = nn.Linear(embed_dim , embed_dim , bias=out_proj_bias)

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

    def forward(
        self,
        x : torch.Tensor,
        casual_mask=False
    ):
        bs , seq_len , embed_dim = x.shape
        # x = x.reshape(bs , seq_len , self.n_heads , self.head_dim)

        #we are going to create just one single matrix and split into 3
        q , k , v = self.in_proj(x).chunk(3 , dim=-1)

        q: torch.Tensor = q.view(bs , seq_len , self.n_heads , self.head_dim).transpose(1,2)
        k : torch.Tensor = k.view(bs , seq_len , self.n_heads , self.head_dim).transpose(1,2)
        v : torch.Tensor = v.view(bs , seq_len , self.n_heads , self.head_dim).transpose(1,2)

        score = q @ k.transpose(-1,-2)

        if casual_mask:

            mask = torch.ones_like(score , dtype=torch.bool).triu(1)
            score.masked_fill_(mask , -torch.inf)

        score /= math.sqrt(self.head_dim)

        score = torch.nn.functional.softmax(score , dim=-1)

        attn_out = (score @ v).transpose(1,2).reshape(bs , seq_len , embed_dim)

        out = self.out_proj(attn_out)

        return out


class CrossAttention(nn.Module):

    def __init__(
        self, n_heads, embed_dim, cross_dim, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=in_proj_bias)
        self.k_proj = nn.Linear(cross_dim, embed_dim, bias=in_proj_bias)
        self.v_proj = nn.Linear(cross_dim, embed_dim, bias=in_proj_bias)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_proj_bias)

        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

    def forward(self, x: torch.Tensor ,y : torch.Tensor):

        # x -> latent: bs , seq_len , qdim
        # y -> context : bs , seqL-eln , dim = (bs , 77 , 768)

        input_shape = x.shape
        bs , seq_len , embed_dim = input_shape

        interim_shape = (bs , -1 , self.n_heads , self.head_dim)

        q: torch.Tensor = self.q_proj(x)
        k: torch.Tensor = self.k_proj(y)
        v: torch.Tensor = self.v_proj(y)

        q = q.view(interim_shape).transpose(1,2)
        k = k.view(interim_shape).transpose(1,2)
        v = v.view(interim_shape).transpose(1,2)


        score = q @ k.transpose(-1,-2)
        score /= math.sqrt(self.head_dim)

        score = torch.nn.functional.softmax(score , dim=-1)

        attn_out = (score @ v).transpose(1,2).contiguous()

        attn_out = attn_out.view(input_shape)

        out = self.out_proj(attn_out)
        return out
    

    