"""Fit the utility model of alloc/utility.py to tools/mixed_snr.py runs. CPU, seconds.

    python tools/utility_fit.py --mixed results/p2_full_snrc4.json --out results/utility_snrc4.json

The mixed-SNR run gives, per CBR, every image's PSNR under constant, one-chunk-at-a-time
and random chunk-SNR profiles. The fit (alloc/utility.py) estimates one noise response
(kappa), per prefix length the share of noise sensitivity each chunk carries (rho), and
per image its noise-free MSE and its sensitivity; the random profiles are held out and
scored. Printed per CBR: rho, the fit and held-out errors (predicted - actual PSNR), and
the same held-out profiles predicted by the best symmetric rule of tools/mixed_snr.py (the
noise rule on each image's own constant-SNR curve), for comparison.

With --chunks phase runs covering 1/48 ... 1/8 (one level per number of phases) it also
writes the utility file of tools/schedule_sim.py. Older 2-chunk grid runs fit too (no
held-out profiles, no utility file).
"""

import argparse
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import Utility, fit  # noqa: E402


def load_levels(path):
    """Levels of a mixed_snr JSON (format 2, or the first 2-chunk grid format)."""
    with open(path) as f:
        d = json.load(f)
    if d.get("format") == 2:
        levels = []
        for lv in d["levels"]:
            levels.append({"key": lv["cbr"], "cbr": lv["cbr"], "units": lv["units"],
                           "profiles": lv["profiles"], "kinds": lv["kinds"], "psnr": lv["psnr"],
                           "chunks": len(lv["cuts"]) + 1})
        return d, levels
    psnr = np.asarray(d["psnr"])                                  # (N, C, P)
    kinds = ["constant" if len(set(p)) == 1 else "mixed" for p in d["profiles"]]
    levels = [{"key": c, "cbr": c, "units": u, "profiles": d["profiles"], "kinds": kinds,
               "psnr": psnr[:, ci].tolist(), "chunks": len(d["cuts"][ci]) + 1}
              for ci, (c, u) in enumerate(zip(d["cbrs"], d["units"]))]
    d.setdefault("chunking", "equal")
    d.setdefault("grid", sorted({p[0] for p in d["profiles"] if len(set(p)) == 1}))
    return d, levels


def symmetric_rule_errors(lv, grid):
    """Held-out profiles predicted by the noise rule on each image's constant curve."""
    Y = np.asarray(lv["psnr"], dtype=np.float64)
    prof = np.asarray(lv["profiles"], dtype=np.float64)
    kinds = np.array(lv["kinds"])
    const = [int(np.flatnonzero((kinds == "constant") & (prof[:, 0] == s))[0]) for s in grid]
    curve = Y[:, const]                                           # (N, S)
    errs = []
    for j in np.flatnonzero(kinds == "random"):
        s_eff = -10 * np.log10(np.mean(10 ** (-prof[j] / 10)))
        pred = np.array([np.interp(s_eff, grid, curve[i]) for i in range(len(Y))])
        errs.append(pred - Y[:, j])
    if not errs:
        return None
    e = np.concatenate(errs)
    return {"bias": float(e.mean()), "mae": float(np.abs(e).mean()),
            "p95": float(np.percentile(np.abs(e), 95))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mixed", nargs="+", required=True, help="tools/mixed_snr.py JSON file(s)")
    ap.add_argument("--out", default=None, help="the utility file (needs --chunks phase runs)")
    ap.add_argument("--d-empty", type=float, default=0.1,
                    help="MSE of an image of which nothing arrived (0.1 = 10 dB)")
    a = ap.parse_args()
    meta, levels, images = None, [], None
    for path in a.mixed:
        d, lv = load_levels(path)
        if images is not None and d["images"] != images:
            raise SystemExit("the mixed-SNR runs scored different images")
        images, meta = d["images"], meta or d
        levels += lv
    kappa, params, report = fit(levels)
    grid = [float(s) for s in meta["grid"]]
    print(f"{len(images)} images, {len(levels)} levels; noise response phi(n) = n / (1 + n / "
          f"{kappa:.3f}); fit rms {report['fit_rms']:.3f} dB")
    print(f"{'CBR':>6s} {'chunks':>6s}  {'rho (share of the noise sensitivity per chunk)':48s} "
          f"{'fit MAE/p95':>13s} {'held-out MAE/p95':>17s} {'noise rule MAE/p95':>19s}")
    for lv in levels:
        rep = report["levels"][lv["key"]]
        fit_s = f"{rep['fit']['mae']:.3f}/{rep['fit']['p95']:.3f}"
        ho = rep.get("held_out")
        ho_s = f"{ho['mae']:.3f}/{ho['p95']:.3f}" if ho else "-"
        rule = symmetric_rule_errors(lv, grid) if ho else None
        rule_s = f"{rule['mae']:.3f}/{rule['p95']:.3f}" if rule else "-"
        rho = " ".join(f"{v:.3f}" for v in rep["rho"])
        print(f"{lv['cbr']:>6s} {lv['chunks']:6d}  {rho:48s} {fit_s:>13s} {ho_s:>17s} {rule_s:>19s}")
        if ho and rule:
            rep["noise_rule_held_out"] = rule
    if a.out:
        per = meta.get("per_phase")
        if meta.get("chunking") != "phase" or not per:
            raise SystemExit("--out needs --chunks phase runs (one chunk per phase)")
        by_L = {lv["units"] // per: lv for lv in levels if lv["units"] % per == 0}
        P = max(by_L)
        if sorted(by_L) != list(range(1, P + 1)) or any(by_L[L]["chunks"] != L for L in by_L):
            raise SystemExit(f"--out needs one level per number of phases 1..{P}, chunked by "
                             f"phase; have {sorted(by_L)}")
        dsrc = np.stack([params[by_L[L]["key"]][0] for L in range(1, P + 1)], 1)
        csens = np.stack([params[by_L[L]["key"]][1] for L in range(1, P + 1)], 1)
        rho = [params[by_L[L]["key"]][2] for L in range(1, P + 1)]
        util = Utility(kappa, rho, dsrc, csens, images, a.d_empty,
                       meta={"sources": a.mixed, "run_name": meta.get("run_name"),
                             "checkpoint": meta.get("checkpoint"),
                             "cbr_per_phase": str(Fraction(by_L[1]["cbr"])),
                             "report": report})
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        util.save(a.out)
        print(f"wrote {a.out}: {len(images)} images x {P} prefix lengths")


if __name__ == "__main__":
    main()
