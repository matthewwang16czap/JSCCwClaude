"""Backbone `adatok`: AdaTok's tokenizer (TiTok-S) as an analog JSCC codec.

    encoder  per tile: [16x16 patch tokens + learned position ;
             M learned latent tokens] -> pre-norm ViT blocks -> the latent half
             -> one shared symbol head -> gain / DC / prenorm (fp32)
    channel  the first l latent tokens (nested tail masking = prefix)
    decoder  [N mask tokens + learned patch position ;
             embedded received tokens + learned latent position] -> ViT blocks
             with multi-head LoRA keyed on l (AdaTok's MH-LoRA) -> the patch
             half -> linear patch head -> tiles -> (optional conv refine)

AdaTok uses TiTok-S (8 blocks, width 512, 8 heads, 16x16 patches) with LoRA
rank 16 and 8 budget heads. The default here is one size up, TiTok-B (12 blocks,
width 768, 12 heads), to fill a 24 GB card at batch 16; --token-dim 512
--depth 8 --heads 8 is the paper's size. Departures, all forced by the channel
or by continuous rates:
no vector quantiser (the tokens are analog channel symbols), DC removal and
prenorm before the power constraint, and the budget heads are hat-interpolated
between log-spaced anchors instead of argmax buckets, so any prefix length is
decodable. Latent tokens are 1D and tile-bound, so tiles are coded and decoded
independently (seams at native resolution are part of this design).

--zero-init is refused for this backbone: with identity blocks the latent
tokens would be image-independent at step 0, the DC estimate would remove all
of them, and the encoder would transmit amplified rounding noise.
"""

import torch
import torch.nn as nn

from ..modules.common import Refine, init_weights, tile_image, to_image, unpatchify, untile_image
from ..modules.rate_mod import BudgetMixer
from ..modules.transformer import Trunk
from ..tokens import RunningDC, post_symbols, token_valid


class AdaTokEncoder(nn.Module):
    def __init__(self, tile, patch, dim, depth, heads, n_tokens, sym):
        super().__init__()
        self.tile = int(tile)
        self.N = (self.tile // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, self.N, dim))
        self.latent = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.trunk = Trunk(dim, depth, heads)
        self.head = nn.Linear(dim, 2 * sym)
        self.gain = nn.Parameter(torch.zeros(n_tokens))
        self.dc = RunningDC(n_tokens, 2 * sym)
        self.apply(init_weights)
        for prm in (self.pos, self.latent):
            nn.init.trunc_normal_(prm, std=0.02)

    def forward(self, x):
        xt, shape = tile_image(x, self.tile)
        f = self.patch_embed(xt).flatten(2).transpose(1, 2) + self.pos
        seq = torch.cat([f, self.latent.expand(f.shape[0], -1, -1).to(f.dtype)], 1)
        z = self.trunk(seq)[:, self.N:]
        with torch.autocast(device_type=z.device.type, enabled=False):
            sym = post_symbols(self.head(z.float()), self.gain, self.dc)
        return sym, shape


class AdaTokDecoder(nn.Module):
    def __init__(self, tile, patch, dim, depth, heads, n_tokens, sym, rate_mod, anchors,
                 rank, l_min, refine_ch=0):
        super().__init__()
        self.patch = int(patch)
        self.side = int(tile) // self.patch
        self.N = self.side ** 2
        self.M = n_tokens
        self.embed = nn.Linear(2 * sym, dim)
        self.latent_pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, self.N, dim))
        self.mixer = BudgetMixer(l_min, n_tokens, anchors) if rate_mod != "none" else None
        self.trunk = Trunk(dim, depth, heads, rate_mod=rate_mod, anchors=anchors, rank=rank)
        self.head = nn.Linear(dim, 3 * self.patch * self.patch)
        self.refine = Refine(refine_ch) if refine_ch else None
        self.apply(init_weights)
        for prm in (self.latent_pos, self.mask_token, self.pos):
            nn.init.trunc_normal_(prm, std=0.02)

    def forward(self, y, lengths, shape):
        BT, self._shape = y.shape[0], shape
        b = self.mixer(lengths, BT, y.device, y.dtype) if self.mixer is not None else None
        if not torch.is_tensor(lengths):
            l = int(lengths)
            tok = self.embed(y[:, :l]) + self.latent_pos[:, :l]
            kp = None
        else:
            v = token_valid(lengths, self.M)
            tok = (self.embed(y) + self.latent_pos) * v.unsqueeze(-1).to(y.dtype)
            kp = torch.cat([torch.ones(BT, self.N, dtype=torch.bool, device=y.device), v], 1)
        q = (self.mask_token + self.pos).expand(BT, -1, -1).to(tok.dtype)
        seq = self.trunk(torch.cat([q, tok], 1), b, key_padding=kp)
        img = unpatchify(self.head(seq[:, :self.N]), self.side, self.side, self.patch)
        img = untile_image(img, shape)
        if self.refine is not None:
            img = self.refine(img)
        return to_image(img)

    def tap_grid(self):
        """Patch half of the trunk hidden state at the tap, tiles joined."""
        t, (B, th, tw), s = self.trunk.tapped[:, :self.N], self._shape, self.side
        g = t.reshape(B, th, tw, s, s, t.shape[-1]).permute(0, 5, 1, 3, 2, 4)
        return g.reshape(B, t.shape[-1], th * s, tw * s)


def build(cfg):
    common = dict(tile=cfg.tile, patch=cfg.patch, dim=cfg.token_dim, depth=cfg.depth,
                  heads=cfg.heads, n_tokens=cfg.n_tokens, sym=cfg.sym)
    return (AdaTokEncoder(**common),
            AdaTokDecoder(rate_mod=cfg.rate_mod, anchors=cfg.rate_anchors, rank=cfg.rate_rank,
                          l_min=cfg.l_min, refine_ch=cfg.refine_ch, **common))
