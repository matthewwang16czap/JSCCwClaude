"""Per-CBR PSNR (or --metric) across runs, mean over SNRs, as differences from the first.

    python tools/compare_runs.py history/<run A> history/<run B> [--window 5] [--upto 1600]
    python tools/compare_runs.py history/<run A> history/<run B> --test

Default: the mean over the last W validations of each run, with the sd across
that window. --test: the final test (logs/test.json). Single validations of
identical-seed runs differed by ~0.7 dB in the previous tree; repeat seeds
before trusting a gap smaller than the spread printed here.
"""

import argparse
import glob
import json
import os
import re
import statistics


def load_valid(run, upto):
    out = []
    for f in sorted(glob.glob(os.path.join(run, "logs", "valid_EP*.json"))):
        ep = int(re.search(r"EP(\d+)", f).group(1))
        if upto is None or ep <= upto:
            with open(f) as fh:
                out.append((ep, json.load(fh)))
    if not out:
        raise SystemExit(f"no validation files under {run}/logs")
    return out


def per_cbr(results, key="psnr"):
    by = {}
    for r in results:
        if key not in r:
            raise SystemExit(f"no {key!r} in these results")
        by.setdefault(r["cbr_nominal"], []).append(r[key])
    by = {k: [v for v in vs if v == v] for k, vs in by.items()}   # drop NaN
    return {k: sum(v) / len(v) if v else float("nan") for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--upto", type=int, default=None, help="ignore validations after this epoch")
    ap.add_argument("--test", action="store_true", help="compare logs/test.json instead")
    ap.add_argument("--metric", default="psnr",
                    help="any key of the result files: psnr, ssim, msssim, lpips, lpips_vgg, "
                         "dists, cls_top1, cls_prob, det_f1, obj_psnr, bg_psnr, obj_lpips, dino_sim")
    a = ap.parse_args()
    table, cbrs = [], None
    for run in a.runs:
        if a.test:
            path = os.path.join(run, "logs", "test.json")
            if not os.path.exists(path):
                raise SystemExit(f"{path} does not exist yet (the test runs after training)")
            with open(path) as fh:
                rows, span = [per_cbr(json.load(fh), a.metric)], "test"
        else:
            vals = load_valid(run, a.upto)[-a.window:]
            rows, span = [per_cbr(r, a.metric) for _, r in vals], f"{vals[0][0]}-{vals[-1][0]}"
        cbrs = cbrs or list(rows[0])
        stats = {c: (statistics.mean(r[c] for r in rows),
                     statistics.stdev(r[c] for r in rows) if len(rows) > 1 else 0.0)
                 for c in cbrs}
        table.append((os.path.basename(run.rstrip("/")), span, stats))
    what = "final test" if a.test else f"last {a.window} validations (sd across them)"
    print(f"{a.metric.upper()}, mean over SNRs, {what}; rows after the first are differences")
    print(f"{'run':56s} {'epochs':>11s} " + " ".join(f"{c:>14s}" for c in cbrs))
    base = table[0][2]
    for name, span, stats in table:
        cells = []
        for c in cbrs:
            m, s = stats[c]
            v = f"{m:7.3f}" if stats is base else f"{m - base[c][0]:+7.3f}"
            cells.append(v + (f"+-{s:4.2f}" if not a.test else ""))
        print(f"{name[:56]:56s} {span:>11s} " + " ".join(f"{x:>14s}" for x in cells))


if __name__ == "__main__":
    main()
