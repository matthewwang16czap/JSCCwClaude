"""The token interface shared by the three token backbones.

A token is `sym` complex channel symbols, packed [real | imag]. An encoder
emits M tokens per tile (or per latent block) in TRANSMISSION ORDER: a prefix of
l tokens is the codeword at CBR = l * sym / (3 * tile^2). The receiver counts
the symbols that arrive, so it knows l without side information.

Everything that faces the power-constrained channel is fp32 and lives here:

  gain     one learned log-gain per token, zero-mean and clamped (only ratios
           matter under the per-codeword power constraint)
  DC       per-token running mean of the symbols, subtracted in train AND eval.
           An image-independent component costs transmit power and carries
           nothing; without this the carrier eats the power budget (the
           previous tree's first collapse).
  prenorm  unit power per sample; the channel then renormalises the prefix
"""

import torch
import torch.nn as nn


def spread_order(h, w):
    """Grid positions ordered so that EVERY PREFIX covers the grid uniformly.

    Bit-reversed Morton order: the first quarter is the stride-2 sub-lattice,
    the first sixteenth the stride-4 one, and so on. Used within a phase, a
    partially sent phase refines scattered positions of the whole block rather
    than its top rows (raster order)."""
    bits = max(1, (max(h, w) - 1).bit_length())
    keys = []
    for y in range(h):
        for x in range(w):
            m = 0
            for b in range(bits):
                m |= ((x >> b) & 1) << (2 * b)
                m |= ((y >> b) & 1) << (2 * b + 1)
            keys.append((int(format(m, f"0{2 * bits}b")[::-1], 2), y * w + x))
    return torch.tensor([i for _, i in sorted(keys)], dtype=torch.long)


def position_order(kind, h, w):
    if kind == "raster":
        return torch.arange(h * w)
    if kind == "spread":
        return spread_order(h, w)
    raise ValueError(f"phase order must be spread|raster, got {kind!r}")


def token_valid(lengths, n_tokens):
    """(B,) prefix lengths -> (B, M) bool, True where the token was sent."""
    idx = torch.arange(n_tokens, device=lengths.device)
    return idx.unsqueeze(0) < lengths.reshape(-1, 1)


class RunningDC(nn.Module):
    """Per-token running mean, subtracted in train AND eval, updated in train.

    Gradients flow through the subtraction; the buffer never receives any."""

    def __init__(self, n_tokens, width, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer("mean", torch.zeros(n_tokens, width))

    def forward(self, sym):
        if self.training:
            with torch.no_grad():
                self.mean.lerp_(sym.detach().float().mean(0), self.momentum)
        return sym - self.mean.to(sym.dtype)


def post_symbols(sym, gain, dc):
    """gain -> DC removal -> prenorm, all fp32."""
    if gain is not None:
        g = gain.float()
        g = (g - g.mean()).clamp(-2.0, 2.0)
        sym = sym * torch.exp(g).view(1, -1, 1)
    sym = dc(sym)
    power = sym.pow(2).mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return sym * torch.rsqrt(power * 2.0)


def fold_phases(src, phases, n, inv):
    """(B, L, D), L <= phases*n, token t = p*n + i at grid position order[i]
    -> (B, n, D) in raster order, summed over phases. Missing tokens add 0.
    `inv` is argsort(order)."""
    B, L, D = src.shape
    if L < phases * n:
        src = torch.cat([src, src.new_zeros(B, phases * n - L, D)], 1)
    return src.view(B, phases, n, D)[:, :, inv].sum(1)

