"""Pre-norm transformer blocks with global attention (SDPA), optional budget
modulation and key padding. Used by the hybrid trunk, the ViT and AdaTok."""

import torch.nn as nn
import torch.nn.functional as F

from .rate_mod import BlockMod


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, key_padding=None):
        """key_padding: (B, N) bool, True = may be attended to."""
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        mask = None if key_padding is None else key_padding[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, rate_mod="none", anchors=8, rank=16):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.mod = BlockMod(dim, hidden, anchors, rate_mod, rank) \
            if rate_mod != "none" else None

    def forward(self, x, b=None, key_padding=None):
        m = self.mod if b is not None else None
        h = self.norm1(x)
        if m is not None and m.film1 is not None:
            h = m.film1(h, b)
        x = x + self.attn(h, key_padding)
        h = self.norm2(x)
        if m is not None and m.film2 is not None:
            h = m.film2(h, b)
        a = self.fc1(h)
        if m is not None and m.lora_fc1 is not None:
            a = a + m.lora_fc1(h, b)
        a = self.act(a)
        o = self.fc2(a)
        if m is not None and m.lora_fc2 is not None:
            o = o + m.lora_fc2(a, b)
        return x + o


class Trunk(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio=4.0, rate_mod="none",
                 anchors=8, rank=16):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio, rate_mod, anchors, rank)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.tap, self.tapped = None, None   # keep the hidden state after block `tap`

    def zero_init_residuals(self):
        """Zero the last linear of every residual branch: each block is then the
        identity at step 0, so the trunk starts as a per-position map
        (LayerNorm) and attention can only add to it."""
        for blk in self.blocks:
            for lin in (blk.attn.proj, blk.fc2):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)

    def forward(self, x, b=None, key_padding=None):
        for i, blk in enumerate(self.blocks):
            x = blk(x, b, key_padding)
            if i == self.tap:
                self.tapped = x
        return self.norm(x)
