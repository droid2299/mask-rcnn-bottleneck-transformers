from torch import nn, einsum
from postional_embeddings import RelPosEmb, AbsPosEmb


class Attention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            fmap_size,
            heads=4,
            dim_head=128,
            rel_pos_emb=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(b, heads, h * w, -1), (q, k, v))

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb(q)

        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.view(b, heads * dim_head, h, w)
        return out
