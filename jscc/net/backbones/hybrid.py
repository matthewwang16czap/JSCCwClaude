"""Backbone `hybrid`: Swin + attention (the previous tree's best AdaJSCC).

    encoder  Swin stages on the WHOLE image -> 16x16 latent blocks
             -> grid_in + pos_scale * fixed 2D sinusoid
             -> P phase copies of the block, one learned identity per token
             -> joint self-attention over [grid ; tokens], zero-init residuals
             -> one symbol head per phase -> gain / DC / prenorm (fp32)
    channel  the first l tokens of every block
    decoder  per-phase embedding + learned token position; the grid stream is
             the FOLD of the received tokens onto their grid positions (plus a
             grid identity and the sinusoid); joint attention, FiLM keyed
             on l; the grid half -> Swin stages on the whole image

Token t = p*N + i carries phase p of grid position order[i]. Every token is
tied to a position and the grid half of the decoder is seeded with content,
the two changes that ended the previous tree's collapse. With `order = spread`
a partially sent phase refines the whole block uniformly; `raster` (top rows
first) is the previous tree's order, kept to evaluate its checkpoints.

Previous best (wave C): sym 16 (P = 6), dim 256, depth 4, heads 8, FiLM,
pos_scale 0.25, zero-init, latent-space blocks, uniform prefix sampling, 2000
warm-up steps. Those are the defaults (configs/config.py).
"""

import torch
import torch.nn as nn

from ..modules.common import block_grid, init_weights, sincos_2d, unblock_grid
from ..modules.rate_mod import BudgetMixer
from ..modules.swin import SwinDecoder, SwinEncoder
from ..modules.transformer import Trunk
from ..tokens import RunningDC, fold_phases, position_order, post_symbols, token_valid


class HybridEncoder(nn.Module):
    def __init__(self, swin_kwargs, block, n_tokens, sym, dim, depth, heads,
                 pos_scale, zero_init, order):
        super().__init__()
        self.grid = SwinEncoder(**swin_kwargs)
        self.block, self.dim, self.pos_scale = int(block), int(dim), float(pos_scale)
        self.N = self.block ** 2
        self.P = n_tokens // self.N
        self.grid_in = nn.Linear(swin_kwargs["embed_dims"][-1], dim)
        self.query = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.trunk = Trunk(dim, depth, heads)
        self.heads = nn.ModuleList([nn.Linear(dim, 2 * sym) for _ in range(self.P)])
        self.gain = nn.Parameter(torch.zeros(n_tokens))
        self.dc = RunningDC(n_tokens, 2 * sym)
        self.register_buffer("order", position_order(order, self.block, self.block),
                             persistent=False)
        self.apply(init_weights)
        nn.init.trunc_normal_(self.query, std=0.02)
        if zero_init:
            self.trunk.zero_init_residuals()

    def forward(self, x):
        grid, H, W = self.grid(x)
        g = block_grid(grid, H, W, self.block)
        g = self.grid_in(g) + self.pos_scale * sincos_2d(
            self.block, self.block, self.dim, g.device, g.dtype).unsqueeze(0)
        lat = g[:, self.order].repeat(1, self.P, 1) + self.query.to(g.dtype)
        tok = self.trunk(torch.cat([g, lat], 1))[:, self.N:]
        with torch.autocast(device_type=tok.device.type, enabled=False):
            t, N = tok.float(), self.N
            sym = torch.cat([h(t[:, p * N:(p + 1) * N]) for p, h in enumerate(self.heads)], 1)
            sym = post_symbols(sym, self.gain, self.dc)
        return sym, (H, W)


class HybridDecoder(nn.Module):
    def __init__(self, swin_kwargs, block, n_tokens, sym, dim, depth, heads,
                 pos_scale, zero_init, order, rate_mod, anchors, l_min):
        super().__init__()
        self.block, self.dim, self.pos_scale = int(block), int(dim), float(pos_scale)
        self.N = self.block ** 2
        self.P = n_tokens // self.N
        self.M = n_tokens
        self.embeds = nn.ModuleList([nn.Linear(2 * sym, dim) for _ in range(self.P)])
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.grid_query = nn.Parameter(torch.zeros(1, 1, dim))
        self.grid_out = nn.Linear(dim, swin_kwargs["embed_dims"][0])
        self.mixer = BudgetMixer(l_min, n_tokens, anchors) if rate_mod != "none" else None
        self.trunk = Trunk(dim, depth, heads, rate_mod=rate_mod, anchors=anchors)
        self.grid = SwinDecoder(**swin_kwargs)
        o = position_order(order, self.block, self.block)
        self.register_buffer("inv", torch.argsort(o), persistent=False)
        self.apply(init_weights)
        for prm in (self.pos, self.grid_query):
            nn.init.trunc_normal_(prm, std=0.02)
        if zero_init:
            self.trunk.zero_init_residuals()

    def _embed(self, y):
        L, N = y.shape[1], self.N
        parts = []
        for p in range(self.P):
            if p * N >= L:
                break
            parts.append(self.embeds[p](y[:, p * N:min((p + 1) * N, L)]))
        return torch.cat(parts, 1)

    def forward(self, y, lengths, shape):
        """y: (B*T, M, 2*sym) with zeros after the prefix; lengths: int, or a
        (B*T,) tensor for per-sample budgets (key-padded path)."""
        H, W = self._hw = shape
        BT, N = y.shape[0], self.N
        b = self.mixer(lengths, BT, y.device, y.dtype) if self.mixer is not None else None
        if not torch.is_tensor(lengths):
            l = int(lengths)
            tok = self._embed(y[:, :l]) + self.pos[:, :l]
            kp = None
        else:
            v = token_valid(lengths, self.M)
            tok = (self._embed(y) + self.pos) * v.unsqueeze(-1).to(y.dtype)
            kp = torch.cat([torch.ones(BT, N, dtype=torch.bool, device=y.device), v], 1)
        gq = fold_phases(tok, self.P, N, self.inv) + self.grid_query.to(tok.dtype) \
            + self.pos_scale * sincos_2d(self.block, self.block, self.dim,
                                         tok.device, tok.dtype).unsqueeze(0)
        seq = self.trunk(torch.cat([gq, tok], 1), b, key_padding=kp)
        grid = unblock_grid(self.grid_out(seq[:, :N]), H, W, self.block)
        return self.grid(grid, H, W)

    def tap_grid(self):
        """Grid half of the token-trunk hidden state at the tap, (B, D, H, W)."""
        H, W = self._hw
        g = unblock_grid(self.trunk.tapped[:, :self.N], H, W, self.block)
        return g.transpose(1, 2).reshape(g.shape[0], -1, H, W)


def build(cfg):
    common = dict(block=cfg.grid_side, n_tokens=cfg.n_tokens, sym=cfg.sym,
                  dim=cfg.token_dim, depth=cfg.depth, heads=cfg.heads,
                  pos_scale=cfg.pos_scale, zero_init=cfg.zero_init, order=cfg.phase_order)
    enc = HybridEncoder(cfg.swin_encoder_kwargs, **common)
    dec = HybridDecoder(cfg.swin_decoder_kwargs, rate_mod=cfg.rate_mod,
                        anchors=cfg.rate_anchors, l_min=cfg.l_min, **common)
    return enc, dec
