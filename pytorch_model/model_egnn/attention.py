import torch
import torch.nn as nn
from .helpers_egnn_model import exists
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum

class Attention(nn.Module):
    """

    """

    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, mask=None):
        h = self.heads

        q = self.to_q(x)
        kv = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, *kv))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b n -> b () () n')
            dots.masked_fill_(~mask, mask_value)

        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class GlobalLinearAttention(nn.Module):

    def __init__(self, *, dim, heads=8, dim_head=64):
        super().__init__()
        self.norm_seq = nn.LayerNorm(dim)
        self.norm_queries = nn.LayerNorm(dim)
        self.attn1 = Attention(dim, heads, dim_head)
        self.attn2 = Attention(dim, heads, dim_head)

        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, queries, mask=None):
        res_x, res_queries = x, queries
        x, queries = self.norm_seq(x), self.norm_queries(queries)

        induced = self.attn1(queries, x, mask=mask)
        out = self.attn2(x, induced)

        x = out + res_x
        queries = induced + res_queries

        x = self.ff(x) + x
        return x, queries