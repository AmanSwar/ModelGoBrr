import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SelfAttentionCQKV
from ..config import StableDiffConfig



class ClipEmbedding(nn.Module):

    def __init__(
        self,
        n_vocab: int,
        n_emebd : int,
        n_token : int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab , n_emebd)

        self.position_embedding = nn.Parameter(torch.zeros((n_token , n_emebd)))


    def forward(
        self,
        token
    ):
        x = self.token_embedding(token)
        x += self.position_embedding

        return x


class CLIPLayer(nn.Module):

    def __init__(
        self,
        n_head : int,
        n_embed : int
    ):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(n_embed)

        self.attn = SelfAttentionCQKV(n_head , n_embed)

        self.layernorm2 = nn.LayerNorm(n_embed)

        self.linear1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear2 = nn.Linear(4 * n_embed , n_embed)

    def forward(self , x):
        x1 = self.layernorm1(x)

        x2 = self.attn(x)

        x = x1 + x2

        residue = x

        x = self.layernorm2(x)

        x = self.linear1(x)
        x = x * torch.sigmoid(1.702  *x)

        x = self.linear2(x)

        x += residue

        return x


class CLIP(nn.Module):

    def __init__(self , config : StableDiffConfig):
        super().__init__()
        self.embedding = ClipEmbedding(config.clip_emebd , config.model_dim , config.seq_len)

        self.layers = nn.ModuleList(
            [
                CLIPLayer(config.n_heads, config.model_dim) for i in range(12)
            ]
        )

        self.layernorm = nn.LayerNorm(config.model_dim)


    def forward(
        self,
        tokens : torch.LongTensor
    ) -> torch.FloatTensor:
        
        tokens = tokens.type(torch.long)

        state = self.embedding(tokens)

        for layer in self.layers:

            state = layer(state)

        output = self.layernorm(state)


        return output
    


