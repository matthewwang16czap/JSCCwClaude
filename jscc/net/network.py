"""JSCC: encoder -> rate -> channel -> decoder, one class for all models.

    swin    SwinJSCC, feature transmission: real channels of a latent grid (the baseline)
    vit     plain ViT, token transmission: phase tokens tied to patch positions
    hybrid  Swin + attention phase tokens (prefix scheme only)

A "unit" is what the rate controls: real channels k per grid position for the
baseline (CBR = k/1536), tokens l per tile for the token models
(CBR = l * sym / (3 * tile^2)). No network input depends on the SNR or on the
channel state; the only thing a decoder is told is how many units arrived.

Two rate schemes (--cascade selects the second):

    prefix   one ordered codeword; rate = how much of its head is sent. One decoder
             path for every rate (token decoders also get a FiLM keyed on l).
    cascade  a stack of small modules between backbone and codec head, one exit per
             predefined CBR (net/cascade.py). Rate k sends the code of level k and
             the receiver runs the decoder modules of that level; units must be one
             of cfg.cbr_units. The number of symbols that arrive still identifies
             the level, so nothing is signalled.

Per-sample budgets (the hook for a future allocation policy): pass a (B,)
integer tensor as `units` to send()/decode() of a prefix token model. SNR
profiles (tokens sent in slots of different SNR, docs/PROBLEM.md): pass a
(B, n_tokens) tensor as `snr` to send() of a prefix token model. Each image
then transmits its own prefix; decoders use key padding (hybrid) or masked
folding (vit). Only uniform budgets are trained and evaluated in this tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import per_image_mse, psnr
from . import cascade
from .backbones import hybrid, swin_linear, vit
from .channel import Channel
from .loss import Objective
from .tokens import post_symbols, token_valid

BUILDERS = {"hybrid": hybrid.build, "vit": vit.build}


class JSCC(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.kind = cfg.backbone
        self.token = cfg.token
        self.cascade = cfg.cascade
        self.channel = Channel(cfg.channel_type, cfg.channel_backend, cfg.equalizer)
        if self.token:
            self.encoder, self.decoder = BUILDERS[self.kind](cfg)
        else:
            self.encoder, self.decoder, self.enc_adapter, self.dec_adapter = \
                swin_linear.build(cfg)
        if self.cascade:
            self.levels = list(cfg.levels)
            self.enc_cascade, self.dec_cascade = cascade.build(cfg)
        tap_dim = cfg.token_dim if self.token else cfg.swin_decoder_kwargs["embed_dims"][0]
        self.objective = Objective(cfg, tap_dim)
        if self.objective.wants_taps:   # decoder hidden state for the alignment term
            if self.token:
                self.decoder.trunk.tap = max(0, len(self.decoder.trunk.blocks) // 2 - 1)
            else:
                self.decoder.layers[0].keep = True

    # -- probes: train one half of a pretrained model ---------------------------
    def halves(self):
        """The modules on each side of the channel (adapters and cascade modules
        go with their side; the objective's own heads stay trainable)."""
        if self.token:
            halves = {"encoder": [self.encoder], "decoder": [self.decoder]}
        else:
            halves = {"encoder": [self.encoder, self.enc_adapter],
                      "decoder": [self.decoder, self.dec_adapter]}
        if self.cascade:
            halves["encoder"].append(self.enc_cascade)
            halves["decoder"].append(self.dec_cascade)
        return halves

    def freeze(self, part):
        """Stop gradients into one half; returns the number of frozen tensors."""
        mods = self.halves()[part]
        n = 0
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(False)
                n += 1
        self._frozen = mods
        return n

    def train(self, mode=True):
        super().train(mode)
        for m in getattr(self, "_frozen", []):
            m.eval()    # a frozen half behaves exactly as at test time
        return self

    @torch.no_grad()
    def seed_levels(self):
        """After a PREFIX token checkpoint was loaded into a cascade model: start
        every level's gain and DC tables from the head of the prefix model's, so
        that at step 0 each level sends what the prefix model sends at that budget."""
        if self.cascade and self.token:
            self.enc_cascade.seed(self.encoder.gain, self.encoder.dc.mean)

    # -- rate -----------------------------------------------------------------
    def units(self, cbr):
        return self.cfg.units_for_cbr(cbr, exact=False)

    def cbr_of(self, units):
        return self.cfg.cbr_for_units(units)

    def level_of(self, units):
        """Cascade: the level (0 = top rate) whose code carries `units`."""
        if torch.is_tensor(units):
            raise ValueError("a cascade sends one level per batch; per-image budgets belong to "
                             "the prefix scheme")
        if int(units) not in self.levels:
            raise ValueError(f"{units} units is not a level of this cascade {self.levels}")
        return self.levels.index(int(units))

    def sample_units(self):
        """Training budgets for one step, ascending. Host RNG only (no sync)."""
        c = self.cfg
        if c.fixed_cbr is not None:
            return [c.units_for_cbr(c.fixed_cbr)]
        if c.rate_sampling == "grid":
            pick = torch.randperm(len(c.cbrs))[:c.rates_per_step].sort().values
            return [c.cbr_units[i] for i in pick.tolist()]
        if c.rate_sampling == "sandwich":
            mid = (torch.randperm(len(c.cbrs) - 2)[:c.rates_per_step - 2] + 1).tolist()
            return [c.cbr_units[i] for i in sorted([0, len(c.cbrs) - 1] + mid)]
        lo, hi = float(c.cbrs[0]), float(c.cbrs[-1])
        draws = torch.rand(c.rates_per_step).tolist()
        tops = torch.rand(c.rates_per_step).tolist() if c.top_prob else [1.0] * len(draws)
        return sorted(c.cbr_units[-1] if t < c.top_prob else self.units(lo + (hi - lo) * u)
                      for u, t in zip(draws, tops))

    # -- encode / send / decode --------------------------------------------------
    def encode(self, x):
        B = x.shape[0]
        if self.token:
            if self.cascade:
                z, shape = self.encoder.code(x)
                return {"B": B, "codes": [z], "shape": shape,
                        "hw": (self.encoder.side, self.encoder.side)}
            sym, shape = self.encoder(x)
            return {"B": B, "sym": sym, "shape": shape}
        feat, H, W = self.encoder(x)
        full = self.enc_adapter(feat)
        enc = {"B": B, "full": full, "shape": (H, W)}
        if self.cascade:
            enc.update(codes=[full], hw=(H, W))
        return enc

    def codeword(self, enc, units):
        """What the transmitter puts on the channel for `units`, before the
        channel's power normalisation: tokens (rows, n, 2 sym) or channels (B, N, k)."""
        if not self.cascade:
            return enc["sym"][:, :units] if self.token else enc["full"][..., :units]
        j = self.level_of(units)
        z = self.enc_cascade.advance(enc, j)
        if not self.token:
            return z
        with torch.autocast(device_type=z.device.type, enabled=False):
            tok = self.encoder.layout(z.float())
            if j == 0:
                return post_symbols(tok, self.encoder.gain, self.encoder.dc)
            return self.enc_cascade.posts[j - 1](tok)

    @staticmethod
    def _rows(v, rows, batch, device, dtype):
        """scalar, (B,) per image, or (rows,) -> (rows,). Token rows are tiles."""
        t = (v if torch.is_tensor(v) else torch.tensor(v)).to(device=device, dtype=dtype)
        t = t.reshape(-1)
        if t.numel() == 1:
            return t.expand(rows)
        if t.numel() == batch and rows != batch:
            return t.repeat_interleave(rows // batch)
        if t.numel() != rows:
            raise ValueError(f"{t.numel()} values for {rows} rows ({batch} images)")
        return t

    @staticmethod
    def _profile(snr, rows, batch, length, device):
        """A per-token SNR profile (B or rows, >= length) -> (rows, length). The
        tiles of an image share its profile: token t of every tile goes out in
        the same chunk (net/channel.py, piecewise_snr)."""
        p = snr.to(device=device, dtype=torch.float32)
        if p.shape[0] == batch and rows != batch:
            p = p.repeat_interleave(rows // batch, 0)
        if p.shape[0] != rows or p.shape[1] < length:
            raise ValueError(f"an SNR profile of shape {tuple(snr.shape)} for {rows} rows of "
                             f"{length} tokens ({batch} images)")
        return p[:, :length]

    def send(self, enc, units, snr):
        profile = torch.is_tensor(snr) and snr.dim() == 2 and snr.shape[1] > 1   # (B, 1): per row
        if profile and (self.cascade or not self.token):
            raise ValueError("an SNR profile (piecewise SNR along the code) needs a token "
                             "model under the prefix scheme")
        if self.cascade:
            cw = self.codeword(enc, units)
            return self.channel(cw, self._rows(snr, cw.shape[0], enc["B"], cw.device,
                                               torch.float32))
        if not self.token:
            if torch.is_tensor(units):
                raise ValueError("the linear baseline takes one channel width per batch")
            z = enc["full"][..., :units]
            return self.channel(z, self._rows(snr, z.shape[0], enc["B"], z.device, torch.float32))
        sym = enc["sym"]
        rows, M = sym.shape[:2]
        snr = self._profile(snr, rows, enc["B"], M, sym.device) if profile else \
            self._rows(snr, rows, enc["B"], sym.device, torch.float32)
        if torch.is_tensor(units):
            u = self._rows(units, rows, enc["B"], sym.device, torch.long)
            mask = token_valid(u, M).unsqueeze(-1).expand(-1, -1, sym.shape[-1] // 2)
            return self.channel(sym, snr, mask=mask)
        y = self.channel(sym[:, :units], snr[:, :units] if profile else snr)
        return F.pad(y, (0, 0, 0, M - units))

    def _expand(self, enc, rx, units):
        """Cascade: the received code of a level -> the estimate of z_0 (full width)
        through that level's decoder modules."""
        if self.token:
            rx = self.decoder.inv_layout(rx)
        return self.dec_cascade(rx, self.level_of(units), enc["hw"])

    def _swin_input(self, enc, rx, units):
        """Baseline decoder input for one received code."""
        return self.dec_adapter(self._expand(enc, rx, units) if self.cascade else rx)

    def decode(self, enc, rx, units):
        if self.cascade and self.token:
            zhat = self._expand(enc, rx, units)
            grid = self.decoder.embed_code(zhat, int(units) // self.decoder.N)
            return self.decoder.finish(grid, int(units), enc["shape"])
        if self.token:
            if torch.is_tensor(units):
                units = self._rows(units, rx.shape[0], enc["B"], rx.device, torch.long)
            return self.decoder(rx, units, enc["shape"])
        H, W = enc["shape"]
        return self.decoder(self._swin_input(enc, rx, units), H, W)

    # -- forward ------------------------------------------------------------------
    def forward(self, x, snr, cbr=None, units=None, want_metrics=True):
        """Training: budgets from sample_units() (or `units`). Evaluation: the
        given `cbr` (default: the top predefined CBR). Returns (recon at the
        last budget, loss, metrics); metrics["cbr"] is the CBR actually sent."""
        if units is None:
            if cbr is not None:
                units = [self.units(cbr)]
            elif self.training:
                units = self.sample_units()
            else:
                units = [self.cfg.cbr_units[-1]]
        enc = self.encode(x)
        rxs = [self.send(enc, u, snr) for u in units]
        taps = [] if self.objective.wants_taps else None
        if not self.token and len(units) > 1:
            H, W = enc["shape"]
            stacked = torch.cat([self._swin_input(enc, r, u) for r, u in zip(rxs, units)], 0)
            recons = list(self.decoder(stacked, H, W).chunk(len(units), 0))
            if taps is not None:
                taps = list(self.decoder.tap_grid().chunk(len(units), 0))
        else:
            recons = []
            for r, u in zip(rxs, units):
                recons.append(self.decode(enc, r, u))
                if taps is not None:
                    taps.append(self.decoder.tap_grid())
        loss, metrics = self.objective(recons, x, units, taps, want_metrics)
        if want_metrics:
            metrics["cbr"] = [self.cbr_of(u) for u in units]
        return recons[-1], loss, metrics

    @torch.no_grad()
    def reconstruct(self, x, snr, cbr):
        """Evaluation: (reconstruction, CBR actually sent), no loss."""
        u = self.units(cbr)
        enc = self.encode(x)
        return self.decode(enc, self.send(enc, u, snr), u), self.cbr_of(u)

    @torch.no_grad()
    def diagnose(self, x, snr, cbr):
        """Does the decoder use what arrives? PSNR as received, with the
        received symbols zeroed, and swapped between images: a decoder that
        ignores the channel shows psnr == zeroed == swapped. `common` is the
        share of the transmitted power that all images of the batch have in
        common: about 1/batch when healthy, more when power goes into a
        carrier the DC estimate misses."""
        was = self.training
        self.eval()
        try:
            enc = self.encode(x)
            u = self.units(cbr)
            rx = self.send(enc, u, snr)
            variants = {"psnr": rx, "zeroed": torch.zeros_like(rx)}
            if x.shape[0] > 1:
                variants["swapped"] = rx.roll(rx.shape[0] // x.shape[0], 0)
            out = {k: float(psnr(per_image_mse(self.decode(enc, v, u).float().clamp(0, 1),
                                               x.float())).mean())
                   for k, v in variants.items()}
            sent = self.codeword(enc, u).float()
            out["common"] = float(sent.mean(0).pow(2).mean() / sent.pow(2).mean().clamp_min(1e-12))
            out["cbr"] = self.cbr_of(u)
        finally:
            self.train(was)
        return out
