"""Depth-cascade rate adaptation: a stack of small modules between the backbone
encoder and decoder, one exit per rate.

    prefix scheme (everything before)    one ordered codeword z_0; rate = how much
                                         of its head is sent; ONE decoder path
    cascade (this file)                  z_0 -> e_1 -> z_1 -> e_2 -> z_2 ... ; the
                                         code at level k has fewer channels than
                                         level k-1; rate k sends z_k and the
                                         decoder runs d_k, ..., d_1 (its own
                                         modules) before the shared backbone

    encoder   backbone -> z_0 (the top rate, e.g. 192 channels) -> e_1 -> z_1 ->
              e_2 -> z_2 -> ... Level k's module sees the CLEAN code of level
              k-1 (the transmitter holds it), never a noisy one.
    channel   z_k, all of it, at the rate of level k.
    decoder   received z_k -> d_k -> d_{k-1} -> ... -> d_1 -> an estimate of z_0
              -> the unchanged backbone decoder. Each rate has its own d-chain.

Every stage is   out = skip(in) + body(in)
    skip   channel truncation (encoder) / zero padding (decoder), optionally a
           learned linear map INITIALISED to that truncation (`proj`, default),
           or nothing (`none`)
    body   in_proj -> `depth` blocks (mlp | attn | swin) -> out_proj, the last
           one ZERO-initialised
so at step 0 every level is exactly the prefix scheme's truncation, whatever the
body: the cascade starts as the nested code and training can only add to it
(tools/smoke.py checks this bit for bit against a prefix model). A stage with
depth 0 and skip `trunc` IS truncation: the prefix scheme inside this framework.

The three body kinds have the same parameter count per block (8 w^2): `mlp` mixes
channels only; `attn` adds global attention over the positions of a tile / image
(fixed sinusoid positions); `swin` adds shifted-window attention (relative
positions, any image size).

The modules run in fp32 (autocast off): their outputs face the power constraint
and, for tokens, the DC estimate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.common import init_weights, sincos_2d
from .modules.swin import SwinBlock
from .modules.transformer import Block as AttnBlock
from .tokens import RunningDC, post_symbols

KINDS = ("mlp", "attn", "swin")
SKIPS = ("proj", "trunc", "none")


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.alpha, None


def grad_scale(x, alpha):
    """Identity forward; the gradient is multiplied by alpha (0 = detach)."""
    return x if alpha == 1.0 else _GradScale.apply(x, float(alpha))


# -- blocks: (B, L, w), grid (H, W) -> (B, L, w), residual ---------------------------
class MlpBlock(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, int(dim * ratio))
        self.fc2 = nn.Linear(int(dim * ratio), dim)

    def forward(self, x, hw):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class GlobalBlock(nn.Module):
    """Global self-attention + MLP (ratio 2), the transformer block of the trunks."""

    def __init__(self, dim, heads):
        super().__init__()
        self.blk = AttnBlock(dim, heads, mlp_ratio=2.0)

    def forward(self, x, hw):
        return self.blk(x)


class WindowBlock(nn.Module):
    def __init__(self, dim, heads, window, shift):
        super().__init__()
        self.blk = SwinBlock(dim, heads, window, shift, mlp_ratio=2.0)

    def forward(self, x, hw):
        return self.blk(x, hw[0], hw[1])


def make_block(kind, dim, index, window):
    if kind == "mlp":
        return MlpBlock(dim)
    if kind == "attn":
        return GlobalBlock(dim, dim // 32)
    if kind == "swin":
        return WindowBlock(dim, dim // 16, window, 0 if index % 2 == 0 else window // 2)
    raise ValueError(f"module kind must be one of {KINDS}, got {kind!r}")


class Body(nn.Module):
    """in_proj -> blocks -> out_proj (zero-init): the correction a stage adds to its skip."""

    def __init__(self, c_in, c_out, kind, depth, width, window, pos_scale=0.25):
        super().__init__()
        self.kind, self.width, self.pos_scale = kind, int(width), float(pos_scale)
        self.inp = nn.Linear(c_in, width)
        self.blocks = nn.ModuleList([make_block(kind, width, i, window) for i in range(depth)])
        self.out = nn.Linear(width, c_out)
        self.blocks.apply(init_weights)
        nn.init.trunc_normal_(self.inp.weight, std=c_in ** -0.5)
        nn.init.zeros_(self.inp.bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z, hw):
        h = self.inp(z)
        if self.kind == "attn":
            h = h + self.pos_scale * sincos_2d(hw[0], hw[1], self.width, h.device, h.dtype)
        for blk in self.blocks:
            h = blk(h, hw)
        return self.out(h)


class Stage(nn.Module):
    """c_in -> c_out channels per position: compress (c_out < c_in, encoder) or
    expand (c_out > c_in, decoder)."""

    def __init__(self, c_in, c_out, kind, depth, width, skip, window):
        super().__init__()
        if skip not in SKIPS:
            raise ValueError(f"skip must be one of {SKIPS}, got {skip!r}")
        if depth == 0 and skip == "none":
            raise ValueError("a stage with depth 0 needs a skip (trunc or proj)")
        self.c_in, self.c_out, self.mode = int(c_in), int(c_out), skip
        self.skip = nn.Parameter(torch.eye(c_out, c_in)) if skip == "proj" else None
        self.body = Body(c_in, c_out, kind, depth, width, window) if depth > 0 else None

    def skip_path(self, z):
        if self.mode == "proj":
            return F.linear(z, self.skip)
        if self.mode == "trunc":
            return z[..., :self.c_out] if self.c_out < self.c_in \
                else F.pad(z, (0, self.c_out - self.c_in))
        return None

    def forward(self, z, hw):
        s = self.skip_path(z)
        if self.body is None:
            return s
        g = self.body(z, hw)
        return g if s is None else s + g


class TokenPost(nn.Module):
    """Gain, DC removal and prenorm of one level's tokens (net/tokens.py), fp32.
    Each level has its own: its code differs from the other levels'."""

    def __init__(self, n_tokens, width):
        super().__init__()
        self.gain = nn.Parameter(torch.zeros(n_tokens))
        self.dc = RunningDC(n_tokens, width)

    def forward(self, tok):
        return post_symbols(tok, self.gain, self.dc)

    @torch.no_grad()
    def seed(self, gain_full, dc_full):
        """Start from the head of the prefix scheme's tables, so that the level
        sends exactly what the prefix model sends at this budget (up to the
        scalar the channel's power normalisation removes)."""
        g = (gain_full.float() - gain_full.float().mean()).clamp(-2.0, 2.0)[:self.gain.numel()]
        self.gain.copy_(g)
        self.dc.mean.copy_(dc_full[:self.gain.numel()].float() * torch.exp(-g.mean()))


class _Chain(nn.Module):
    def __init__(self, dims, kinds, depths, width, skip, window, expand):
        super().__init__()
        self.dims = list(dims)
        pairs = [(dims[k], dims[k - 1]) if expand else (dims[k - 1], dims[k])
                 for k in range(1, len(dims))]
        self.stages = nn.ModuleList([Stage(a, b, kinds[k], depths[k], width, skip, window)
                                     for k, (a, b) in enumerate(pairs)])


class EncoderCascade(_Chain):
    """Levels 0..K-1 (descending rate). `advance` computes and caches the codes
    up to a level; the gradient of deeper exits into the shared shallower code
    is scaled by `alpha` (1 = joint training, 0 = greedy, each level trained on
    a detached input)."""

    def __init__(self, dims, kinds, depths, width, skip, window, alpha=1.0,
                 token_sizes=None, sym2=None):
        super().__init__(dims, kinds, depths, width, skip, window, expand=False)
        self.alpha = float(alpha)
        self.posts = nn.ModuleList([TokenPost(n, sym2) for n in token_sizes[1:]]) \
            if token_sizes is not None else None

    def advance(self, enc, level):
        codes = enc["codes"]
        with torch.autocast(device_type=codes[0].device.type, enabled=False):
            for k in range(len(codes), level + 1):
                codes.append(self.stages[k - 1](grad_scale(codes[k - 1].float(), self.alpha),
                                                enc["hw"]))
        return codes[level]

    @torch.no_grad()
    def seed(self, gain_full, dc_full):
        for post in self.posts:
            post.seed(gain_full, dc_full)


class DecoderCascade(_Chain):
    """Received code of level k -> an estimate of z_0, through the decoder-side
    modules d_k ... d_1 (its own, per rate)."""

    def __init__(self, dims, kinds, depths, width, skip, window):
        super().__init__(dims, kinds, depths, width, skip, window, expand=True)

    def forward(self, z, level, hw):
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            for k in range(level, 0, -1):
                z = self.stages[k - 1](z, hw)
        return z


def build(cfg):
    """(EncoderCascade, DecoderCascade) for a Config with cascade = True."""
    kw = dict(dims=cfg.code_dims, kinds=cfg.mod_kinds, depths=cfg.mod_depths,
              width=cfg.mod_width, skip=cfg.mod_skip, window=cfg.window)
    sizes = [u for u in cfg.cbr_units[::-1]] if cfg.token else None
    enc = EncoderCascade(alpha=cfg.cascade_grad, token_sizes=sizes, sym2=2 * cfg.sym
                         if cfg.token else None, **kw)
    return enc, DecoderCascade(**kw)
