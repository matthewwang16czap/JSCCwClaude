"""One training epoch, and the evaluation grid."""

import contextlib
import time
from collections import defaultdict

import torch

from net.channel import piecewise_snr, seed_channel
from utils.common import amp_dtype


def reseed(cfg, seed):
    """torch, and with the Sionna backend Sionna's own generators."""
    torch.manual_seed(seed)
    if cfg.channel_backend == "sionna" and cfg.channel_type != "none":
        seed_channel(seed)


def autocast(cfg, device):
    dt = amp_dtype(cfg)
    return torch.autocast(device_type=device.type, dtype=dt, enabled=dt is not None)


def sample_snr(cfg, batch, device):
    """One SNR per image, uniform over --snr-range. With --snr-chunks K > 1 a
    per-token profile (batch, n_tokens) instead: K chunks at independent SNRs, cut
    at K - 1 random token positions (tokens sent in slots of a block-fading
    channel); a prefix shorter than a cut sees only the chunks before it."""
    lo, hi = cfg.train_snr_range

    def draw(*shape):
        if hi > lo:
            return torch.empty(shape, device=device).uniform_(lo, hi)
        return torch.full(shape, lo, device=device)

    if cfg.snr_chunks <= 1:
        return draw(batch)
    cuts = torch.randint(1, cfg.n_tokens, (batch, cfg.snr_chunks - 1), device=device)
    return piecewise_snr(draw(batch, cfg.snr_chunks), cuts.sort(1).values, cfg.n_tokens)


def train_one_epoch(epoch, step, net, model, loader, optimizer, scheduler, cfg, log=None,
                    ema=None):
    """Returns (optimiser steps so far, steps skipped in this epoch).

    A non-finite gradient norm discards the WHOLE accumulation window. Under
    DDP the norm is computed on all-reduced gradients, so every rank takes the
    same decision without an extra collective. Losses and metrics are read
    back from the GPU only at print steps."""
    net.train()
    ddp = hasattr(net, "no_sync")
    params = [p for p in model.parameters() if p.requires_grad]
    accum = cfg.accum_steps
    max_norm = cfg.grad_clip if cfg.grad_clip > 0 else float("inf")
    optimizer.zero_grad(set_to_none=True)
    skipped, seen, t0 = 0, 0, time.perf_counter()
    for i, x in enumerate(loader):
        x = x.to(cfg.device, non_blocking=True)
        snr = sample_snr(cfg, x.shape[0], x.device)
        boundary = (i + 1) % accum == 0
        verbose = log is not None and boundary and (step + 1) % cfg.print_step == 0
        with (net.no_sync() if ddp and not boundary else contextlib.nullcontext()):
            with autocast(cfg, x.device):
                _, loss, metrics = net(x, snr, want_metrics=verbose)
            (loss / accum).backward()
        seen += x.shape[0]
        if not boundary:
            continue
        norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        if torch.isfinite(norm):
            optimizer.step()
            if ema is not None:
                ema.update(model)
        else:
            skipped += 1
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1
        if verbose:
            with autocast(cfg, x.device):
                d = model.diagnose(x, cfg.diag_snr, cfg.diag_cbr)
            rate = seen / (time.perf_counter() - t0)
            cbr = " ".join(f"{c:.4f}" for c in metrics["cbr"])
            ps = " ".join(f"{p:.2f}" for p in metrics["psnr"])
            ps += "".join(f" {k} {metrics[k]:.4f}" for k in ("sem", "align", "lpips") if k in metrics)
            log.info(f"ep {epoch + 1} step {step} lr {scheduler.get_last_lr()[0]:.2e} "
                     f"{rate:.1f} img/s | loss {float(loss):.5f} cbr {cbr} psnr {ps} | "
                     f"diag cbr {d['cbr']:.4f} {cfg.diag_snr:g} dB: psnr {d['psnr']:.2f} "
                     f"zeroed {d['zeroed']:.2f} swapped {d.get('swapped', float('nan')):.2f} "
                     f"common {d['common']:.3f} | skipped {skipped}")
            seen, t0 = 0, time.perf_counter()
    optimizer.zero_grad(set_to_none=True)   # a trailing partial window is dropped
    return step, skipped


@torch.no_grad()
def evaluate(model, loader, cfg, snrs, cbrs, suite, repeats=1, per_image=False):
    """Every (SNR, CBR) cell over the whole loader, scored by `suite`
    (utils/metrics.py).

    Cell (i, j) reseeds torch and Sionna with eval_seed + 1000 i + j, so every
    checkpoint, of any configuration and any training --seed, sees the same
    channel draws (common random numbers). `cbr` in the output is the CBR
    actually transmitted; NaN values (a metric undefined for an image) are left
    out of the means. Afterwards the caller's random stream continues from a
    fresh point drawn beforehand (it must not restart from the same state after
    every validation)."""
    resume = int(torch.randint(0, 2 ** 31 - 1, (1,)))
    model.eval()
    results, records = [], []
    for i, snr in enumerate(snrs):
        for j, cbr in enumerate(cbrs):
            reseed(cfg, cfg.eval_seed + 1000 * i + j)
            sums, counts, n, sent = defaultdict(float), defaultdict(int), 0, None
            t0 = time.perf_counter()
            for r in range(repeats):
                seen = 0
                for k, x in enumerate(loader):
                    x = x.to(cfg.device, non_blocking=True)
                    with autocast(cfg, x.device):
                        out, sent = model.reconstruct(x, float(snr), cbr)
                    b = x.shape[0]
                    vals = suite(out, x, keys=range(seen, seen + b))
                    seen, n = seen + b, n + b
                    for key, v in vals.items():
                        valid = ~torch.isnan(v)
                        sums[key] += float(v[valid].sum())
                        counts[key] += int(valid.sum())
                    if per_image and b == 1:
                        records.append({"snr": float(snr), "cbr_nominal": str(cbr), "cbr": sent,
                                        "repeat": r, "image": k,
                                        **{key: float(v[0]) for key, v in vals.items()}})
            cell = {"snr": float(snr), "cbr_nominal": str(cbr), "cbr": sent}
            cell.update({key: sums[key] / counts[key] if counts[key] else float("nan")
                         for key in sums})
            cell["sec_per_image"] = (time.perf_counter() - t0) / max(n, 1)
            results.append(cell)
    reseed(cfg, resume)
    return results, records
