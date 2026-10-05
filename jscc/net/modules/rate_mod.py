"""Budget-conditioned decoder modulation (FiLM), made continuous.

The decoder is told only HOW MANY tokens arrived (the receiver counts them), so
this is rate conditioning, not channel conditioning. K anchors are log-spaced
over the budget range; a budget between two anchors mixes their modulations
with hat weights (at most two active, summing to one), so the rate axis stays
smooth instead of showing K plateaus. Everything is zero-initialised: at step 0
the modulation is the identity.

    film  per-anchor scale and shift of the normalised activations (cheap; the
          best validated setting for the hybrid and the ViT)
"""

import math

import torch
import torch.nn as nn


class Budget:
    """Mixing weights for one decode: (rows, anchors)."""

    __slots__ = ("w",)

    def __init__(self, w):
        self.w = w


class BudgetMixer(nn.Module):
    def __init__(self, lo, hi, anchors=8):
        super().__init__()
        lo, hi = float(lo), float(hi)
        if anchors < 2 or not 0.0 < lo < hi:
            raise ValueError(f"need >= 2 anchors over a positive range, got "
                             f"{anchors} over [{lo}, {hi}]")
        a = torch.exp(torch.linspace(math.log(lo), math.log(hi), anchors))
        self.register_buffer("anchors", a)
        self.k = anchors
        self._host = a.tolist()

    def _pairs(self, budget):
        a = self._host
        k = min(max(float(budget), a[0]), a[-1])
        j = next(i for i in range(1, len(a)) if a[i] >= k)
        t = (k - a[j - 1]) / max(a[j] - a[j - 1], 1e-9)
        return [(i, w) for i, w in ((j - 1, 1.0 - t), (j, t)) if w > 0.0]

    def forward(self, lengths, batch, device, dtype):
        """lengths: an int (whole batch) or a (batch,) tensor."""
        if not torch.is_tensor(lengths):
            w = torch.zeros(self.k)
            for i, v in self._pairs(lengths):
                w[i] = v
            return Budget(w.to(device=device, dtype=dtype).expand(batch, -1))
        a = self.anchors.float()
        k = lengths.to(device=device, dtype=torch.float32).reshape(-1)
        k = k.clamp(a[0], a[-1]).unsqueeze(-1)
        idx = (a.unsqueeze(0) >= k).float().argmax(-1).clamp_min(1)
        lo, hi = a[idx - 1], a[idx]
        t = ((k.squeeze(-1) - lo) / (hi - lo).clamp_min(1e-9)).clamp(0.0, 1.0)
        w = torch.zeros(k.shape[0], self.k, device=device)
        w.scatter_(1, (idx - 1).unsqueeze(1), (1.0 - t).unsqueeze(1))
        w.scatter_add_(1, idx.unsqueeze(1), t.unsqueeze(1))
        return Budget(w.to(dtype))


class FiLM(nn.Module):
    def __init__(self, dim, anchors):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(anchors, dim))
        self.beta = nn.Parameter(torch.zeros(anchors, dim))

    def forward(self, x, b):
        w = b.w.to(self.gamma.dtype)
        return x * (1.0 + (w @ self.gamma).unsqueeze(1)) + (w @ self.beta).unsqueeze(1)


class BlockMod(nn.Module):
    """What one transformer block gets: a FiLM before attention and before the MLP."""

    def __init__(self, dim, anchors, mode="film"):
        super().__init__()
        if mode != "film":
            raise ValueError(f"rate_mod must be none|film, got {mode!r}")
        self.film1 = FiLM(dim, anchors)
        self.film2 = FiLM(dim, anchors)
