import torch
from torch import nn, einsum


def pair(x):
    if isinstance(x, tuple):
        return x
    else:
        return (x, x)


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    sizes = list(t.size())
    sizes[dim] = k
    return t.expand(*sizes)


def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim=3)
    flat_x = x.view(b, h, l * 2)
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.view(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = logits.view(b, heads * h, w, 2 * w - 1)
    logits = rel_to_abs(logits)
    logits = logits.view(b, heads, h, w, w)
    logits = expand_dim(logits, dim=3, k=h)
    return logits


class AbsPosEmb(nn.Module):
    def __init__(
            self,
            fmap_size,
            dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = self.height.unsqueeze(1) + self.width.unsqueeze(0)
        emb = emb.view(-1, emb.size(-1))
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits


class RelPosEmb(nn.Module):
    def __init__(
            self,
            fmap_size,
            dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = q.view(b, h, w, w, dim)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rel_logits_w.view(b, h, h * w, w * w)

        q = q.permute(0, 1, 3, 2, 4)
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rel_logits_h.view(b, h, w * w, h * w)
        return rel_logits_w + rel_logits_h
