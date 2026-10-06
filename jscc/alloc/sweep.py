"""Per-image quality-vs-budget curves of a frozen codec, with common random numbers.

For every image the codec encodes ONCE; one noise realisation is drawn for the
FULL codeword, and every candidate budget transmits a prefix of that same
codeword through the prefix of that same noise. Candidates therefore differ
only in how much of the code arrives, never in their luck with the channel,
so per-image differences between budgets are measured almost noise-free.

The channel reproduces net/channel.py exactly (per-row unit power over the
transmitted symbols, no = 10^(-SNR/10) per complex symbol, MMSE or ZF
equalisation of per-symbol Rayleigh fading); only the noise source differs.
Rayleigh is supported for the token models, whose complex pairing lives inside
a token; the Swin baseline pairs channel c with c + k/2, which moves with the
budget k, so its fading cannot be shared across budgets (AWGN can: every real
dimension keeps its own noise sample).
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from engine import autocast
from net.channel import Channel
from utils.metrics import MetricSuite

IMAGE_FEATURES = ("luma_mean", "luma_std", "colourfulness",
                  "grad_s1", "lap_s1", "grad_s2", "lap_s2", "grad_s4", "lap_s4",
                  "block_std_p10", "block_std_p50", "block_std_p90", "flat_fraction",
                  "high_freq_ratio")
LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)


def candidate_units(cfg, n):
    """n budgets evenly spread over the trained range, plus the predefined ones."""
    lo, hi = cfg.cbr_units[0], cfg.cbr_units[-1]
    grid = np.linspace(lo, hi, max(int(n), 2))
    vals = np.round(grid) if cfg.token else np.round(grid / 2) * 2
    return sorted(set(int(v) for v in vals) | set(int(u) for u in cfg.cbr_units))


def codeword(model, enc):
    """The full-budget codeword: (rows, M, 2*sym) tokens, or (B, N, width) Swin."""
    return enc["sym"] if model.token else enc["full"]


def draw_noise(model, enc, kind, generator):
    z = codeword(model, enc)
    n = torch.randn(z.shape, generator=generator, device=z.device, dtype=torch.float32)
    h = None
    if kind == "rayleigh":
        if not model.token:
            raise ValueError("common-random-number Rayleigh needs a token model (the Swin "
                             "baseline's complex pairing changes with the budget); use AWGN")
        shape = z.shape[:-1] + (z.shape[-1] // 2,)
        h = torch.complex(torch.randn(shape, generator=generator, device=z.device),
                          torch.randn(shape, generator=generator, device=z.device)) * 0.5 ** 0.5
    return n, h


def send(model, enc, units, snr_rows=None, noise=None, fading=None, kind="none",
         equalizer="mmse"):
    """model.send with the noise supplied: the first `units` of the codeword
    through the first `units` of `noise` (and `fading`)."""
    z = codeword(model, enc).float()
    if model.token:
        x = z[:, :units]
        n = None if noise is None else noise[:, :units]
        h = None if fading is None else fading[:, :units]
    else:
        x = z[..., :units]
        n = None if noise is None else noise[..., :units]
        h = None
    c2 = x.shape[-1] // 2
    xc = Channel.normalize(torch.complex(x[..., :c2], x[..., c2:]))
    if kind == "none":
        yc = xc
    else:
        no = torch.pow(10.0, -snr_rows.float() / 10.0).view(-1, 1, 1)
        nc = torch.complex(n[..., :c2], n[..., c2:]) * torch.sqrt(no / 2.0)
        if kind == "awgn":
            yc = xc + nc
        elif kind == "rayleigh":
            y = h * xc + nc
            if equalizer == "zf":
                yc = y / torch.polar(h.abs().clamp_min(1e-8), h.angle())
            else:
                yc = h.conj() * y / (h.real.pow(2) + h.imag.pow(2) + no)
        else:
            raise ValueError(f"channel {kind!r}")
    y = torch.cat([yc.real, yc.imag], dim=-1)
    if model.token:
        y = F.pad(y, (0, 0, 0, z.shape[1] - units))
    return y


@torch.no_grad()
def image_features(x):
    """Content statistics the transmitter has for free: (B, 14), see IMAGE_FEATURES."""
    x = x.float()
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    rg, yb = r - g, 0.5 * (r + g) - b
    col = (rg.std((1, 2)).pow(2) + yb.std((1, 2)).pow(2)).sqrt() \
        + 0.3 * (rg.mean((1, 2)).pow(2) + yb.mean((1, 2)).pow(2)).sqrt()
    feats = [y.mean((1, 2)), y.std((1, 2)), col]
    yy = y[:, None]
    lap = LAPLACE.to(x.device)
    for s in (1, 2, 4):
        z = F.avg_pool2d(yy, s) if s > 1 else yy
        gx = (z[..., :, 1:] - z[..., :, :-1]).abs().mean((1, 2, 3))
        gy = (z[..., 1:, :] - z[..., :-1, :]).abs().mean((1, 2, 3))
        feats += [gx + gy, F.conv2d(z, lap, padding=1).abs().mean((1, 2, 3))]
    blocks = F.unfold(yy, 8, stride=8).std(1)               # (B, blocks)
    qs = torch.quantile(blocks, torch.tensor([0.1, 0.5, 0.9], device=x.device), dim=1)
    feats += [qs[0], qs[1], qs[2], (blocks < 0.02).float().mean(1)]
    blur = F.avg_pool2d(yy, 5, stride=1, padding=2, count_include_pad=False)
    total = (yy - yy.mean((2, 3), keepdim=True)).pow(2).mean((1, 2, 3)).clamp_min(1e-8)
    feats.append((yy - blur).pow(2).mean((1, 2, 3)) / total)
    return torch.stack(feats, 1)


@torch.no_grad()
def codeword_features(model, enc, batch, groups=12):
    """Energy profile of the full codeword along the transmission order:
    (B, 2*groups) = each segment's share of the energy, and the coefficient of
    variation of the energy inside it (how unevenly the content loads it)."""
    z = codeword(model, enc).float().pow(2)
    if model.token:
        rows, M, C = z.shape
        e = z.mean(-1).reshape(batch, rows // batch, M)     # (B, tiles, tokens)
    else:
        e = z                                                # (B, positions, channels)
    L = e.shape[-1]
    edges = np.linspace(0, L, groups + 1).round().astype(int)
    share, cv = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = e[..., a:max(b, a + 1)].reshape(batch, -1)
        m = seg.mean(1)
        share.append(m)
        cv.append(seg.std(1, unbiased=False) / m.clamp_min(1e-12))
    share = torch.stack(share, 1)
    return torch.cat([share / share.sum(1, keepdim=True).clamp_min(1e-12), torch.stack(cv, 1)], 1)


def _scores(suite, metrics, out, x, keys):
    vals = suite(out, x, keys)
    return np.stack([vals[m].numpy() if m in vals else np.full(x.shape[0], np.nan)
                     for m in metrics], 1)


@torch.no_grad()
def sweep(model, cfg, loader, units, metrics, draws=2, snrs=None, random_snrs=0,
          snr_range=(-2.0, 22.0), seed=0, log=print):
    """Curves for every image the loader yields.

    SNRs: a fixed grid `snrs` (every image, every value), or `random_snrs`
    values per image drawn uniformly from `snr_range`. Returns a dict of numpy
    arrays: scores (N, S, D, K, M) under the channel, clean (N, K, M) through a
    noise-free channel (what the transmitter can simulate), snr (N, S),
    feat_img (N, 14), feat_code (N, 24), pixels (N,)."""
    model.eval()
    if getattr(model, "cascade", False):
        raise ValueError("the allocation sweep reads prefixes of one codeword with common noise; a "
                         "cascade model has one code per level (the per-image choice among its "
                         "levels is future work)")
    dev, kind, eq = cfg.device, cfg.channel_type, cfg.equalizer
    if kind == "none":
        raise ValueError("a noise-free channel has no SNR to sweep; train with awgn or rayleigh")
    suite = MetricSuite(metrics, cfg.teacher, cfg.teacher_weights)
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    host = np.random.default_rng(int(seed))
    S = len(snrs) if not random_snrs else int(random_snrs)
    K, M = len(units), len(metrics)
    total = len(loader.dataset)
    acc = {k: [] for k in ("scores", "clean", "snr", "feat_img", "feat_code", "pixels")}
    seen, t0 = 0, time.perf_counter()
    for x in loader:
        x = x.to(dev, non_blocking=True)
        B = x.shape[0]
        keys = list(range(seen, seen + B))
        with autocast(cfg, dev):
            enc = model.encode(x)
        rows = codeword(model, enc).shape[0]
        per = rows // B
        clean = np.zeros((B, K, M), dtype=np.float32)
        for k, u in enumerate(units):
            y = send(model, enc, u, kind="none")
            with autocast(cfg, dev):
                out = model.decode(enc, y, u)
            clean[:, k] = _scores(suite, metrics, out, x, keys)
        if random_snrs:
            s_img = host.uniform(snr_range[0], snr_range[1], size=(B, S))
        else:
            s_img = np.tile(np.asarray(snrs, dtype=np.float64), (B, 1))
        sc = np.zeros((B, S, draws, K, M), dtype=np.float32)
        for s in range(S):
            snr_rows = torch.tensor(s_img[:, s], device=dev, dtype=torch.float32)
            snr_rows = snr_rows.repeat_interleave(per)
            for d in range(draws):
                noise, fading = draw_noise(model, enc, kind, gen)
                for k, u in enumerate(units):
                    y = send(model, enc, u, snr_rows, noise, fading, kind, eq)
                    with autocast(cfg, dev):
                        out = model.decode(enc, y, u)
                    sc[:, s, d, k] = _scores(suite, metrics, out, x, keys)
        acc["scores"].append(sc)
        acc["clean"].append(clean)
        acc["snr"].append(s_img.astype(np.float32))
        acc["feat_img"].append(image_features(x).cpu().numpy())
        acc["feat_code"].append(codeword_features(model, enc, B).cpu().numpy())
        acc["pixels"].append(np.full(B, x.shape[-2] * x.shape[-1], dtype=np.float64))
        seen += B
        el = time.perf_counter() - t0
        log(f"  {seen}/{total} images | {el / 60:.1f} min elapsed, "
            f"~{el / seen * (total - seen) / 60:.1f} min left")
    return {k: np.concatenate(v, 0) for k, v in acc.items()}
