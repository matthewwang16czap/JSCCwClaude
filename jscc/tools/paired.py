"""Paired per-image comparison of runs, from their logs/test_per_image.json.

    python tools/paired.py history/<control> history/<variant A> [history/<variant B> ...]
    python tools/paired.py <control> <variant> --metric lpips_vgg --snrs 1 4
    python tools/paired.py <control> <specialist> --cbrs 1/8     # only the rows that matter

Every variant is compared with the FIRST run (the control). The test records the
same image under the same channel draw in every run (the evaluation reseeds per
(SNR, CBR) cell), so the per-image difference removes most of the image-to-image
spread. Per CBR the table gives the mean difference variant - control over
images (each image averaged over the SNRs and channel draws first), a 95%
bootstrap interval that resamples IMAGES, and the share of images the variant
wins. The interval covers the images only, not run-to-run training noise: a gap
that the interval clears is a gap on THESE images for THESE weights. Kodak has
24 images; repeat seeds before treating a difference below ~0.1 dB as real.
"""

import argparse
import json
import os
from fractions import Fraction

import numpy as np

HIGHER_IS_BETTER = {"psnr", "ssim", "msssim", "cls_top1", "cls_prob", "det_f1", "obj_psnr",
                    "bg_psnr", "dino_sim"}


def load(run, metric, snrs):
    path = os.path.join(run, "logs", "test_per_image.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} does not exist (the test runs after training)")
    with open(path) as f:
        records = json.load(f)
    cells = {}
    for r in records:
        if metric not in r:
            raise SystemExit(f"no {metric!r} in {path}")
        if snrs is not None and r["snr"] not in snrs:
            continue
        cells.setdefault((r["cbr_nominal"], r["image"]), []).append(r[metric])
    # one value per (cbr, image): the mean over SNRs and channel draws; NaN when undefined
    return {k: float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan")
            for k, v in cells.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="control first, then the variants")
    ap.add_argument("--metric", default="psnr")
    ap.add_argument("--snrs", nargs="+", type=float, default=None, help="only these SNRs (dB)")
    ap.add_argument("--cbrs", nargs="+", default=None,
                    help="print only these CBRs, e.g. 1/8 for a 1/8 specialist ('all' then "
                         "averages over these only)")
    ap.add_argument("--boot", type=int, default=4000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if len(a.runs) < 2:
        raise SystemExit("need a control and at least one variant")
    rng = np.random.default_rng(a.seed)
    data = [load(r, a.metric, a.snrs) for r in a.runs]
    keys = sorted(set.intersection(*[set(d) for d in data]))
    if not keys or any(set(d) != set(data[0]) for d in data):
        raise SystemExit("the runs scored different (CBR, image) cells: same test set, protocol "
                         "and --eval-cbrs are needed for a paired comparison")
    cbrs = sorted({c for c, _ in keys}, key=Fraction)
    if a.cbrs:
        want = {Fraction(c) for c in a.cbrs}
        cbrs = [c for c in cbrs if Fraction(c) in want]
        if not cbrs:
            raise SystemExit(f"none of --cbrs {a.cbrs} was scored")
        keys = [k for k in keys if k[0] in cbrs]
    sign = 1.0 if a.metric in HIGHER_IS_BETTER else -1.0
    sign_note = "" if sign > 0 else " (lower is better: positive = the variant is better)"
    print(f"{a.metric.upper()}, variant minus control{sign_note}; mean over SNRs and channel draws, "
          f"95% bootstrap CI over images")
    print(f"control: {os.path.basename(a.runs[0].rstrip('/'))}")
    for run, d in zip(a.runs[1:], data[1:]):
        print(f"\nvariant: {os.path.basename(run.rstrip('/'))}")
        print(f"{'cbr':>6s} {'control':>9s} {'variant':>9s} {'diff':>8s} {'95% CI':>20s} {'wins':>6s}")
        for c in cbrs + ["all"]:
            sel = [k for k in keys if (c == "all" or k[0] == c)]
            x0 = np.array([data[0][k] for k in sel])
            x1 = np.array([d[k] for k in sel])
            if c == "all":      # per-image mean over CBRs, then the same statistics
                imgs = sorted({k[1] for k in sel})
                x0 = np.array([np.nanmean([data[0][k] for k in sel if k[1] == i]) for i in imgs])
                x1 = np.array([np.nanmean([d[k] for k in sel if k[1] == i]) for i in imgs])
            ok = ~(np.isnan(x0) | np.isnan(x1))
            x0, x1 = x0[ok], x1[ok]
            diff = sign * (x1 - x0)
            if len(diff) == 0:
                continue
            boots = diff[rng.integers(0, len(diff), (a.boot, len(diff)))].mean(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"{c:>6s} {x0.mean():9.3f} {x1.mean():9.3f} {diff.mean():+8.3f} "
                  f"{f'[{lo:+.3f}, {hi:+.3f}]':>20s} {100 * np.mean(diff > 0):5.0f}%"
                  f"{'  n=' + str(len(diff)) if c == cbrs[0] else ''}")


if __name__ == "__main__":
    main()
