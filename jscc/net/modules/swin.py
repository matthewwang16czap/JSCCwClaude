"""SwinJSCC stages, unconditioned: no SNR and no rate input anywhere.

Attribute names follow the previous tree (and the published SwinJSCC code), so
earlier baseline and hybrid checkpoints load with every tensor matched.
"""

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import init_weights, to_image


def window_partition(x, ws):
    """(B, H, W, C) -> (B*nW, ws*ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)


def window_reverse(win, ws, H, W):
    """(B*nW, ws*ws, C) -> (B, H, W, C)"""
    C = win.shape[-1]
    B = win.shape[0] // (H * W // ws // ws)
    x = win.view(B, H // ws, W // ws, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)


@functools.lru_cache(maxsize=32)
def shift_mask(H, W, ws, shift, device):
    """Cyclic-shift attention mask (nW, ws*ws, ws*ws), cached by geometry."""
    img = torch.zeros(1, H, W, 1, device=device)
    spans = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
    for i, h in enumerate(spans):
        for j, w in enumerate(spans):
            img[:, h, w, :] = i * len(spans) + j
    ids = window_partition(img, ws).squeeze(-1)
    m = ids.unsqueeze(1) - ids.unsqueeze(2)
    return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)


def relative_position_index(ws):
    coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws),
                                        indexing="ij")).flatten(1)
    rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
    rel[:, :, 0] += ws - 1
    rel[:, :, 1] += ws - 1
    rel[:, :, 0] *= 2 * ws - 1
    return rel.sum(-1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.area = window_size * window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.register_buffer("relative_position_index",
                             relative_position_index(window_size))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        if N != self.area:
            raise ValueError(f"window of {N} tokens, bias table built for {self.area}: "
                             f"the feature map is smaller than the window")
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(N, N, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0)).view(-1, self.num_heads, N, N)
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.0):
        super().__init__()
        self.window_size, self.shift_size = window_size, shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, H, W):
        B, L, C = x.shape
        if L != H * W:
            raise ValueError(f"feature length {L} != H*W = {H * W}")
        ws = min(self.window_size, H, W)
        shift = self.shift_size if ws > self.shift_size else 0
        h = self.norm1(x).view(B, H, W, C)
        if shift:
            h = torch.roll(h, (-shift, -shift), dims=(1, 2))
        mask = shift_mask(H, W, ws, shift, h.device) if shift else None
        h = window_reverse(self.attn(window_partition(h, ws), mask), ws, H, W)
        if shift:
            h = torch.roll(h, (shift, shift), dims=(1, 2))
        x = x + h.reshape(B, L, C)
        return x + self.mlp(self.norm2(x))


class SwinStage(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4.0,
                 downsample=None, upsample=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, num_heads, window_size,
                      0 if i % 2 == 0 else window_size // 2, mlp_ratio)
            for i in range(depth)])
        self.downsample = downsample
        self.upsample = upsample
        self.keep, self.kept = False, None   # keep the output before resampling

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        if self.keep:
            self.kept = (x, H, W)
        resample = self.downsample if self.downsample is not None else self.upsample
        if resample is not None:
            x, H, W = resample(x, H, W)
        return x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        _, _, H, W = x.shape
        p = self.patch_size
        if H % p or W % p:
            raise ValueError(f"{H}x{W} is not a multiple of patch_size {p}")
        return self.norm(self.proj(x).flatten(2).transpose(1, 2)), H // p, W // p


class PatchUnembed(nn.Module):
    def __init__(self, patch_size, out_chans, embed_dim):
        super().__init__()
        self.patch_size, self.out_chans = patch_size, out_chans
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_chans)

    def forward(self, x, H, W):
        B = x.shape[0]
        p, c = self.patch_size, self.out_chans
        x = self.proj(self.norm(x)).view(B, H, W, c, p, p)
        return x.permute(0, 3, 1, 4, 2, 5).reshape(B, c, H * p, W * p), H * p, W * p


class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        if H % 2 or W % 2:
            raise ValueError(f"{H}x{W} is not even and cannot be merged")
        x = x.view(B, H, W, C)
        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2],
                       x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        return self.reduction(self.norm(x.view(B, -1, 4 * C))), H // 2, W // 2


class PatchReverseMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B = x.shape[0]
        x = self.increment(self.norm(x)).view(B, H, W, -1).permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, 2)
        return x.flatten(2).transpose(1, 2), H * 2, W * 2


class SwinEncoder(nn.Module):
    """image -> (B, H/16 * W/16, embed_dims[-1]) grid, final LayerNorm."""

    def __init__(self, in_chans, embed_dims, depths, num_heads, patch_size=2,
                 window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dims[0])
        last = len(depths) - 1
        self.layers = nn.ModuleList([
            SwinStage(embed_dims[i], depths[i], num_heads[i], window_size, mlp_ratio,
                      downsample=None if i == last
                      else PatchMerging(embed_dims[i], embed_dims[i + 1]))
            for i in range(len(depths))])
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.apply(init_weights)

    def forward(self, x):
        x, H, W = self.patch_embed(x)
        for layer in self.layers:
            x, H, W = layer(x, H, W)
        return self.norm(x), H, W


class SwinDecoder(nn.Module):
    """(B, H*W, embed_dims[0]) grid -> image in [0, 1]."""

    def __init__(self, out_chans, embed_dims, depths, num_heads, patch_size=2,
                 window_size=8, mlp_ratio=4.0):
        super().__init__()
        last = len(depths) - 1
        self.layers = nn.ModuleList([
            SwinStage(embed_dims[i], depths[i], num_heads[i], window_size, mlp_ratio,
                      upsample=None if i == last
                      else PatchReverseMerging(embed_dims[i], embed_dims[i + 1]))
            for i in range(len(depths))])
        self.patch_unembed = PatchUnembed(patch_size, out_chans, embed_dims[-1])
        self.apply(init_weights)

    def forward(self, x, H, W):
        if H * W != x.shape[1]:
            raise ValueError(f"feature length {x.shape[1]} != H*W = {H * W}")
        for layer in self.layers:
            x, H, W = layer(x, H, W)
        x, _, _ = self.patch_unembed(x, H, W)
        return to_image(x)

    def tap_grid(self):
        """Hidden state of the first stage (the latent grid), (B, C, H, W)."""
        x, H, W = self.layers[0].kept
        return x.transpose(1, 2).reshape(x.shape[0], -1, H, W)
