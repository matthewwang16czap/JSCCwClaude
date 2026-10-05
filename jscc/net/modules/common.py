"""Small building blocks shared by every backbone."""

import math

import torch
import torch.nn as nn


def init_weights(m):
    """trunc-normal(0.02) for Linear, unit/zero for LayerNorm (ViT/Swin convention)."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def sincos_2d(h, w, dim, device=None, dtype=torch.float32):
    """(h*w, dim) fixed 2D sinusoidal embedding in raster order.

    Fixed rather than learned wherever a grid can change size (native-resolution
    evaluation), so the same weights work at any H, W."""
    if dim % 4:
        raise ValueError(f"sincos_2d needs dim divisible by 4, got {dim}")
    q = dim // 4
    omega = torch.exp(-math.log(10000.0)
                      * torch.arange(q, device=device, dtype=torch.float32) / q)
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                          torch.arange(w, device=device, dtype=torch.float32),
                          indexing="ij")
    out = []
    for c in (y.flatten(), x.flatten()):
        a = c[:, None] * omega[None, :]
        out += [torch.sin(a), torch.cos(a)]
    return torch.cat(out, dim=1).to(dtype)


def tile_image(x, t):
    """(B, C, H, W) -> (B*T, C, t, t), plus the shape that undoes it."""
    B, C, H, W = x.shape
    if H % t or W % t:
        raise ValueError(f"{H}x{W} is not a whole number of {t}x{t} tiles")
    th, tw = H // t, W // t
    x = x.reshape(B, C, th, t, tw, t).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(B * th * tw, C, t, t), (B, th, tw)


def untile_image(x, shape):
    """Exact inverse of tile_image."""
    B, th, tw = shape
    C, t = x.shape[1], x.shape[-1]
    x = x.reshape(B, th, tw, C, t, t).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, C, th * t, tw * t)


def block_grid(g, H, W, bs):
    """(B, H*W, C) raster grid -> (B*T, bs*bs, C) blocks, T = (H/bs)*(W/bs)."""
    B, N, C = g.shape
    if N != H * W or H % bs or W % bs:
        raise ValueError(f"grid {H}x{W} is not a whole number of {bs}x{bs} blocks")
    th, tw = H // bs, W // bs
    g = g.reshape(B, th, bs, tw, bs, C).permute(0, 1, 3, 2, 4, 5)
    return g.reshape(B * th * tw, bs * bs, C)


def unblock_grid(g, H, W, bs):
    """Exact inverse of block_grid."""
    BT, _, C = g.shape
    th, tw = H // bs, W // bs
    g = g.reshape(BT // (th * tw), th, tw, bs, bs, C).permute(0, 1, 3, 2, 4, 5)
    return g.reshape(-1, H * W, C)


def unpatchify(x, h, w, p, c=3):
    """(B, h*w, c*p*p) patch logits -> (B, c, h*p, w*p)."""
    B = x.shape[0]
    x = x.reshape(B, h, w, c, p, p).permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, c, h * p, w * p)


def to_image(logits):
    """Image logits -> [0, 1], the SwinJSCC output convention."""
    return 0.5 * (torch.tanh(logits) + 1.0)


class Refine(nn.Module):
    """Optional residual conv tail on the image logits (vit, adatok).

    A linear patch head leaves block edges; this is the usual cheap cure. The
    last conv is zero-initialised, so the tail starts as the identity. Off by
    default (--refine-ch 0) so the ViT arms stay plain transformers."""

    def __init__(self, ch, layers=2):
        super().__init__()
        self.inp = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[nn.Sequential(nn.GELU(), nn.Conv2d(ch, ch, 3, 1, 1))
                                    for _ in range(layers)])
        self.out = nn.Conv2d(ch, 3, 3, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return x + self.out(self.body(self.inp(x)))
