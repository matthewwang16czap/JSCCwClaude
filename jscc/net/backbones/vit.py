"""Backbone `vit`: a plain ViT with the same phase-token interface as `hybrid`.

    encoder  per tile: patch embedding + fixed 2D sinusoid -> pre-norm
             ViT blocks (global attention within the tile) -> one symbol head
             per phase, applied to the patches in transmission order
    channel  the first l tokens of every tile
    decoder  per tile: the received tokens are embedded (per-phase linear +
             learned phase identity) and FOLDED onto their patch positions ->
             budget-modulated ViT blocks -> linear patch head -> tiles
             (+ optional conv refine over the whole image) -> image

The default patch 8 gives 1024 positions per 256 px tile and 4-symbol tokens,
so the hybrid's six phases carry over (every predefined CBR on a phase
boundary). With --patch-size 16 --sym-per-token 16 the token geometry equals the
hybrid's exactly and the two arms differ only in the transform.

Why tiles on both sides: training sees one 256 px tile per crop, so the
transformer only ever learns positions and attention inside one tile. The first
version decoded native-resolution images as one joint grid (Kodak: 6 tiles, a
6x longer sequence, unseen positions) and lost ~14 dB on Kodak while leading on
the 256 px validation (wave 1). Tile-by-tile decoding is exactly the training
condition; the price is that tiles are reconstructed independently.
"""

import torch
import torch.nn as nn

from ..modules.common import (Refine, init_weights, sincos_2d, tile_image, to_image, unpatchify,
                              untile_image)
from ..modules.rate_mod import BudgetMixer
from ..modules.transformer import Trunk
from ..tokens import RunningDC, fold_phases, position_order, post_symbols, token_valid


class ViTEncoder(nn.Module):
    def __init__(self, tile, patch, dim, depth, heads, n_tokens, sym, order, zero_init):
        super().__init__()
        self.tile, self.patch, self.dim = int(tile), int(patch), int(dim)
        self.side = self.tile // self.patch
        self.N = self.side ** 2
        self.P = n_tokens // self.N
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.trunk = Trunk(dim, depth, heads)
        self.heads = nn.ModuleList([nn.Linear(dim, 2 * sym) for _ in range(self.P)])
        self.gain = nn.Parameter(torch.zeros(n_tokens))
        self.dc = RunningDC(n_tokens, 2 * sym)
        self.register_buffer("order", position_order(order, self.side, self.side),
                             persistent=False)
        self.apply(init_weights)
        if zero_init:
            self.trunk.zero_init_residuals()

    def forward(self, x):
        xt, shape = tile_image(x, self.tile)
        f = self.patch_embed(xt).flatten(2).transpose(1, 2)
        f = f + sincos_2d(self.side, self.side, self.dim, f.device, f.dtype).unsqueeze(0)
        f = self.trunk(f)[:, self.order]
        with torch.autocast(device_type=f.device.type, enabled=False):
            f = f.float()
            sym = torch.cat([h(f) for h in self.heads], 1)
            sym = post_symbols(sym, self.gain, self.dc)
        return sym, shape


class ViTDecoder(nn.Module):
    def __init__(self, tile, patch, dim, depth, heads, n_tokens, sym, order, zero_init,
                 rate_mod, anchors, rank, l_min, refine_ch=0):
        super().__init__()
        self.tile, self.patch, self.dim = int(tile), int(patch), int(dim)
        self.side = self.tile // self.patch
        self.N = self.side ** 2
        self.P = n_tokens // self.N
        self.M = n_tokens
        self.embeds = nn.ModuleList([nn.Linear(2 * sym, dim) for _ in range(self.P)])
        self.phase = nn.Parameter(torch.zeros(self.P, dim))
        self.mixer = BudgetMixer(l_min, n_tokens, anchors) if rate_mod != "none" else None
        self.trunk = Trunk(dim, depth, heads, rate_mod=rate_mod, anchors=anchors, rank=rank)
        self.head = nn.Linear(dim, 3 * patch * patch)
        self.refine = Refine(refine_ch) if refine_ch else None
        o = position_order(order, self.side, self.side)
        self.register_buffer("inv", torch.argsort(o), persistent=False)
        self.apply(init_weights)
        nn.init.trunc_normal_(self.phase, std=0.02)
        if zero_init:
            self.trunk.zero_init_residuals()

    def _src(self, y, lengths):
        N = self.N
        if not torch.is_tensor(lengths):
            l, parts = int(lengths), []
            for p in range(self.P):
                if p * N >= l:
                    break
                parts.append(self.embeds[p](y[:, p * N:min((p + 1) * N, l)]) + self.phase[p])
            return torch.cat(parts, 1)
        v = token_valid(lengths, self.M).unsqueeze(-1)
        emb = torch.cat([self.embeds[p](y[:, p * N:(p + 1) * N]) + self.phase[p]
                         for p in range(self.P)], 1)
        return emb * v.to(emb.dtype)

    def forward(self, y, lengths, shape):
        """Every tile is decoded on its own, exactly as in training."""
        self._shape = shape
        s, D, rows = self.side, self.dim, y.shape[0]
        grid = fold_phases(self._src(y, lengths), self.P, self.N, self.inv)
        grid = grid + sincos_2d(s, s, D, grid.device, grid.dtype).unsqueeze(0)
        b = self.mixer(lengths, rows, grid.device, grid.dtype) if self.mixer is not None else None
        img = untile_image(unpatchify(self.head(self.trunk(grid, b)), s, s, self.patch), shape)
        if self.refine is not None:
            img = self.refine(img)
        return to_image(img)

    def tap_grid(self):
        """Trunk hidden state at the tap, tiles joined, (B, D, H/p, W/p)."""
        t, (B, th, tw), s = self.trunk.tapped, self._shape, self.side
        g = t.reshape(B, th, tw, s, s, t.shape[-1]).permute(0, 5, 1, 3, 2, 4)
        return g.reshape(B, t.shape[-1], th * s, tw * s)


def build(cfg):
    common = dict(tile=cfg.tile, patch=cfg.patch, dim=cfg.token_dim, depth=cfg.depth,
                  heads=cfg.heads, n_tokens=cfg.n_tokens, sym=cfg.sym,
                  order=cfg.phase_order, zero_init=cfg.zero_init)
    return (ViTEncoder(**common),
            ViTDecoder(rate_mod=cfg.rate_mod, anchors=cfg.rate_anchors, rank=cfg.rate_rank,
                       l_min=cfg.l_min, refine_ch=cfg.refine_ch, **common))
