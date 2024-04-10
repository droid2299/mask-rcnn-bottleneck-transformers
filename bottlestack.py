from torch import nn
from postional_embeddings import pair
from attention import Attention


class BottleBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            fmap_size,
            dim_out,
            proj_factor,
            downsample,
            heads=4,
            dim_head=128,
            rel_pos_emb=False,
            activation=nn.ReLU()
    ):
        super().__init__()

        # shortcut

        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv2d(dim, dim_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head

        self.net = nn.Sequential(
            nn.Conv2d(dim, attn_dim_in, 1, bias=False),
            nn.BatchNorm2d(attn_dim_in),
            activation,
            Attention(
                dim=attn_dim_in,
                fmap_size=fmap_size,
                heads=heads,
                dim_head=dim_head,
                rel_pos_emb=rel_pos_emb
            ),
            nn.AvgPool2d((2, 2)) if downsample else nn.Identity(),
            nn.BatchNorm2d(attn_dim_out),
            activation,
            nn.Conv2d(attn_dim_out, dim_out, 1, bias=False),
            nn.BatchNorm2d(dim_out)
        )

        nn.init.zeros_(self.net[-1].weight)

        self.activation = activation

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.net(x)
        x = x + shortcut
        return self.activation(x)


class BottleStack(nn.Module):
    def __init__(
            self,
            *,
            dim,
            fmap_size,
            dim_out=2048,
            proj_factor=4,
            num_layers=3,
            heads=4,
            dim_head=128,
            downsample=True,
            rel_pos_emb=False,
            activation=nn.ReLU()
    ):
        super().__init__()
        fmap_size = pair(fmap_size)

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample

            fmap_divisor = (2 if downsample and not is_first else 1)
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(BottleBlock(
                dim=dim,
                fmap_size=layer_fmap_size,
                dim_out=dim_out,
                proj_factor=proj_factor,
                heads=heads,
                dim_head=dim_head,
                downsample=layer_downsample,
                rel_pos_emb=rel_pos_emb,
                activation=activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert h == self.fmap_size[0] and w == self.fmap_size[
            1], f'height and width ({h} {w}) of feature map must match the fmap_size given at init {self.fmap_size}'
        feature_maps = {f"p{i}": x for i in range(2, 7)}
        return feature_maps
