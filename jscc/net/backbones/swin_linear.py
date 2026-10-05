"""Baseline: SwinJSCC encoder/decoder + the `linear` rate adapter.

Bit-identical in its math to the previous tree's `--adapters linear` path, so
earlier baseline numbers stay comparable and its checkpoints load: one
projection each side (default PyTorch init, as before), the first k REAL
columns of the 192-wide latent are sent, and the channel pairs them into k/2
complex symbols as [first k/2 | next k/2]. No SNR and no rate conditioning.
CBR = k / 1536 at stride 16.
"""

import torch.nn as nn
import torch.nn.functional as F

from ..modules.swin import SwinDecoder, SwinEncoder


class LinearRateProjection(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        return self.proj(x)


class LinearRateExpand(nn.Module):
    """Zero-pad the k received columns back to the full width, then project."""

    def __init__(self, width):
        super().__init__()
        self.width = width
        self.proj = nn.Linear(width, width)

    def forward(self, y):
        pad = self.width - y.shape[-1]
        return self.proj(F.pad(y, (0, pad)) if pad > 0 else y)


def build(cfg):
    return (SwinEncoder(**cfg.swin_encoder_kwargs),
            SwinDecoder(**cfg.swin_decoder_kwargs),
            LinearRateProjection(cfg.width),
            LinearRateExpand(cfg.width))
