"""Complex baseband channel: Sionna 2 blocks (the standard, default) and a
pure-PyTorch reference with the same statistics (used by the CPU tests).

Conventions (tools/check_channel.py measures them for both backends):
  * input (R, N, C) real, C even, packed [real | imag] -> C/2 complex symbols
    per position; R rows are images (baseline) or tiles (token models)
  * power is normalised PER ROW (per codeword) to unit mean power per
    TRANSMITTED complex symbol; `mask` marks the transmitted symbols and the
    untransmitted ones leave the channel as exact zeros
  * no = 10^(-SNR/10) is the noise variance per complex symbol (no/2 per real
    dimension), Sionna's convention; one SNR per row
  * awgn:     sionna.phy.channel.AWGN
  * rayleigh: sionna.phy.channel.FlatFadingChannel with one antenna at each end,
    applied symbol by symbol: an i.i.d. h ~ CN(0, 1) per complex symbol plus
    AWGN. The receiver knows h and equalises
        mmse (default)  y_eq = h* y / (|h|^2 + no)   noise variance <= 1/4
        zf              y_eq = y / h                 noise no/|h|^2, infinite mean
    The NETWORKS never see h, the equaliser gain or the SNR.

Randomness: Sionna draws from its own generators (sionna.phy.config), which
torch.manual_seed does not reach. seed_channel() seeds them; main.py and the
evaluation call it wherever torch is seeded.
"""

import torch
import torch.nn as nn


def seed_channel(seed):
    """Seed Sionna's generators (Sionna also reseeds torch's default ones)."""
    from sionna.phy import config
    config.seed = int(seed)


class Channel(nn.Module):
    KINDS = ("awgn", "rayleigh", "none")

    def __init__(self, kind="awgn", backend="sionna", equalizer="mmse"):
        super().__init__()
        if kind not in self.KINDS:
            raise ValueError(f"channel must be one of {self.KINDS}, got {kind!r}")
        if backend not in ("sionna", "torch"):
            raise ValueError(f"channel backend must be sionna|torch, got {backend!r}")
        if equalizer not in ("mmse", "zf"):
            raise ValueError(f"equalizer must be mmse|zf, got {equalizer!r}")
        self.kind, self.backend, self.equalizer = kind, backend, equalizer
        self._blocks = {}   # device -> Sionna block, kept out of the module registry

    @staticmethod
    def normalize(xc, mask=None, eps=1e-12):
        """Unit mean power per transmitted complex symbol, per row."""
        p = xc.real.pow(2) + xc.imag.pow(2)
        if mask is None:
            power = p.mean(dim=(1, 2), keepdim=True)
        else:
            m = mask.to(p.dtype)
            power = (p * m).sum(dim=(1, 2), keepdim=True) \
                / m.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        return xc * torch.rsqrt(power + eps)

    @staticmethod
    def noise_variance(snr_db, rows, device):
        s = snr_db if torch.is_tensor(snr_db) else torch.tensor(float(snr_db))
        s = s.to(device=device, dtype=torch.float32).reshape(-1)
        if s.numel() == 1:
            s = s.expand(rows)
        if s.numel() != rows:
            raise ValueError(f"{s.numel()} SNR values for {rows} rows")
        return torch.pow(10.0, -s / 10.0).view(rows, 1, 1)

    def forward(self, x, snr_db, mask=None):
        x = x.float()
        R, _, C = x.shape
        if C % 2:
            raise ValueError(f"need an even number of real channels, got {C}")
        C2 = C // 2
        xc = self.normalize(torch.complex(x[..., :C2], x[..., C2:]), mask)
        if self.kind == "none":
            yc = xc
        else:
            no = self.noise_variance(snr_db, R, x.device)
            if self.kind == "awgn":
                yc = self._awgn(xc, no)
            else:
                y, h = self._fading(xc, no)
                yc = self._equalize(y, h, no)
        if mask is not None:
            yc = yc * mask.to(yc.real.dtype)
        return torch.cat([yc.real, yc.imag], dim=-1)

    def _block(self, device):
        key = str(device)
        if key not in self._blocks:
            from sionna.phy.channel import AWGN, FlatFadingChannel
            self._blocks[key] = AWGN(device=key) if self.kind == "awgn" else \
                FlatFadingChannel(num_tx_ant=1, num_rx_ant=1, return_channel=True, device=key)
        return self._blocks[key]

    def _awgn(self, xc, no):
        if self.backend == "sionna":
            return self._block(xc.device)(xc, no)
        return xc + self._cn(xc, torch.sqrt(no / 2.0))

    def _fading(self, xc, no):
        """y = h x + n, one i.i.d. h ~ CN(0, 1) per complex symbol."""
        if self.backend == "sionna":
            R, N, C2 = xc.shape
            no_flat = no.expand(R, N, C2).reshape(-1, 1)
            y, h = self._block(xc.device)(xc.reshape(-1, 1), no_flat)
            return y.reshape(R, N, C2), h.reshape(R, N, C2)
        h = self._cn(xc, 0.5 ** 0.5)
        return h * xc + self._cn(xc, torch.sqrt(no / 2.0)), h

    def _equalize(self, y, h, no):
        if self.equalizer == "zf":
            return y / torch.polar(h.abs().clamp_min(1e-8), h.angle())
        return h.conj() * y / (h.real.pow(2) + h.imag.pow(2) + no)

    @staticmethod
    def _cn(like, scale):
        return torch.complex(torch.randn_like(like.real), torch.randn_like(like.real)) * scale
