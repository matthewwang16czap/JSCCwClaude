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

--calib S1 S2 ... --belief-out FILE: what a transmitter can know without the mixed-SNR
measurements of the image it sends. The pooled kappa and rho come from OTHER images
(--folds cross-fitting: images split by index, each fold calibrated with the parameters
fitted on the rest), and each image's own Dsrc and C per prefix length from its
constant-SNR decodes at the --calib SNRs only (2 SNRs x P lengths = 2P decodes at the
transmitter, which has the image and the codec). Printed: that belief's errors on every
profile it did not use, next to the full fit's held-out errors. FILE is a --belief for
tools/schedule_sim.py.

    python tools/utility_fit.py --mixed results/p2_full_snrc4.json --calib 1 13 \\
        --belief-out results/belief_snrc4.json

--nested: one master sensitivity profile for every prefix length (alloc/utility.nested_rho),
which makes each user's state two numbers (phases held, accumulated weighted noise) for the
dynamic program of alloc/whittle.py; printed: its held-out errors against the free fit.
"""

import argparse
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import Utility, fit, noise, phi  # noqa: E402


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


def abstraction_errors(levels, params, kappa, grid):
    """Link abstractions compared on the held-out profiles of every level with >= 2 chunks:
    each profile mapped to ONE effective SNR, then read off the image's own constant-SNR
    curve (the way EESM / MIESM serve digital system simulation). The chunks are averaged by
    rule -- mean dB; noise variance; capacity (MIESM-like, Gaussian input); exponential (EESM,
    beta fitted per level on the one-at-a-time profiles); MMSE of a Gaussian of variance kappa,
    n / (1 + n / kappa), the noise response of alloc/utility.py -- with equal weights or with
    the fitted sensitivity weights rho. MMSE with rho is the additive model read through the
    measured curve. Returns {(rule, weights): array of errors (predicted - actual PSNR)}."""
    def s_eff(prof, w, rule, beta=1.0):
        lin = 10.0 ** (prof / 10.0)
        n = 1.0 / lin
        if rule == "mean dB":
            return prof @ w
        if rule == "noise":
            return -10.0 * np.log10(n @ w)
        if rule == "capacity":
            return 10.0 * np.log10(np.maximum(2.0 ** (np.log2(1.0 + lin) @ w) - 1.0, 1e-12))
        if rule == "EESM":
            return 10.0 * np.log10(np.maximum(-beta * np.log(np.maximum(np.exp(-lin / beta) @ w,
                                                                          1e-300)), 1e-12))
        m = (n / (1.0 + n / kappa)) @ w                       # MMSE
        return -10.0 * np.log10(np.maximum(m / np.maximum(1.0 - m / kappa, 1e-12), 1e-12))

    out = {}
    for lv in levels:
        prof = np.asarray(lv["profiles"], dtype=np.float64)
        kinds = np.array(lv["kinds"])
        if prof.shape[1] < 2 or not (kinds == "random").any():
            continue
        Y = np.asarray(lv["psnr"], dtype=np.float64)
        const = [int(np.flatnonzero((kinds == "constant") & np.all(prof == s, 1))[0]) for s in grid]
        curve = Y[:, const]
        k = prof.shape[1]
        rho = np.asarray(params[lv["key"]][2], dtype=np.float64)
        for wname, w in (("equal", np.full(k, 1.0 / k)), ("rho", rho / rho.sum())):
            def err(rule, beta, sel):
                e = [np.interp(s_eff(prof[j], w, rule, beta), grid, curve[i]) - Y[i, j]
                     for j in np.flatnonzero(sel) for i in range(len(Y))]
                return np.array(e)
            beta = 1.0
            if (kinds == "oat").any():                         # EESM's one parameter
                betas = np.logspace(-1, 2, 31)
                beta = min(betas, key=lambda b: np.abs(err("EESM", b, kinds == "oat")).mean())
            for rule in ("mean dB", "noise", "capacity", "EESM", "MMSE"):
                out.setdefault((rule, wname), []).append(err(rule, beta, kinds == "random"))
    return {key: np.concatenate(v) for key, v in out.items()}


def phase_levels(meta, levels):
    """{L: level} for --chunks phase runs with one level per number of phases 1..P."""
    per = meta.get("per_phase")
    if meta.get("chunking") != "phase" or not per:
        raise SystemExit("needs --chunks phase runs (one chunk per phase)")
    by_L = {lv["units"] // per: lv for lv in levels if lv["units"] % per == 0}
    P = max(by_L)
    if sorted(by_L) != list(range(1, P + 1)) or any(by_L[L]["chunks"] != L for L in by_L):
        raise SystemExit(f"needs one level per number of phases 1..{P}, chunked by phase; "
                         f"have {sorted(by_L)}")
    return by_L, P


def calibrate(by_L, P, images, calib, folds=2, nested=False):
    """Cross-fitted transmitter belief (see the module doc). Returns (Utility, report)."""
    N = len(images)
    fold = np.arange(N) % folds
    kappa = np.zeros(N)
    rho = [np.zeros((N, L)) for L in range(1, P + 1)]
    dsrc, csens = np.zeros((N, P)), np.zeros((N, P))
    for f in range(folds):
        rest, mine = fold != f, np.flatnonzero(fold == f)
        sub = [dict(by_L[L], key=str(L), psnr=np.asarray(by_L[L]["psnr"])[rest].tolist())
               for L in range(1, P + 1)]
        k_f, params, _ = fit(sub, nested=nested)
        kappa[mine] = k_f
        for L in range(1, P + 1):
            lv = by_L[L]
            rho[L - 1][mine] = params[str(L)][2]
            prof = np.asarray(lv["profiles"], dtype=np.float64)
            kinds = np.array(lv["kinds"])
            cols = [j for j in range(len(prof)) if kinds[j] == "constant"
                    and any(abs(prof[j, 0] - c) < 1e-9 for c in calib)]
            if len(cols) < 2:
                raise SystemExit(f"--calib needs >= 2 constant SNRs that the run measured; "
                                 f"{len(cols)} at {L} phases")
            x = phi(noise(prof[cols, 0]), k_f)
            A = np.stack([np.ones_like(x), x], 1)
            for i in mine:
                D = 10.0 ** (-np.asarray(lv["psnr"][i], dtype=np.float64)[cols] / 10.0)
                sol = np.linalg.lstsq(A / D[:, None], np.ones_like(D), rcond=None)[0]
                dsrc[i, L - 1], csens[i, L - 1] = max(sol[0], 1e-6), max(sol[1], 0.0)
    belief = Utility(kappa, rho, dsrc, csens, images)
    report = {}
    for L in range(1, P + 1):
        lv = by_L[L]
        prof = np.asarray(lv["profiles"], dtype=np.float64)
        kinds = np.array(lv["kinds"])
        used = np.array([kinds[j] == "constant" and any(abs(prof[j, 0] - c) < 1e-9 for c in calib)
                         for j in range(len(prof))])
        Y = np.asarray(lv["psnr"], dtype=np.float64)
        rep = {}
        for name, m in (("constant", (kinds == "constant") & ~used), ("one_at_a_time", kinds == "oat"),
                        ("random", kinds == "random"), ("all_unused", ~used)):
            if m.any():
                e = np.array([[belief.psnr(i, prof[j].tolist()) - Y[i, j] for j in np.flatnonzero(m)]
                              for i in range(N)])
                rep[name] = {"bias": float(e.mean()), "mae": float(np.abs(e).mean()),
                             "p95": float(np.percentile(np.abs(e), 95)), "n": int(e.size)}
        report[str(lv["cbr"])] = rep
    return belief, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mixed", nargs="+", required=True, help="tools/mixed_snr.py JSON file(s)")
    ap.add_argument("--out", default=None, help="the utility file (needs --chunks phase runs)")
    ap.add_argument("--d-empty", type=float, default=0.1,
                    help="MSE of an image of which nothing arrived (0.1 = 10 dB)")
    ap.add_argument("--calib", nargs="+", type=float, default=None,
                    help="SNRs (dB) of the constant decodes a transmitter calibrates from")
    ap.add_argument("--folds", type=int, default=2, help="--calib: cross-fitting folds")
    ap.add_argument("--belief-out", default=None, help="--calib: the belief file to write")
    ap.add_argument("--compare-abstractions", action="store_true",
                    help="held-out errors of effective-SNR link abstractions (mean dB, noise, "
                         "capacity = MIESM-like, EESM, MMSE), equal or sensitivity weights, next "
                         "to the additive model")
    ap.add_argument("--nested", action="store_true",
                    help="one master sensitivity profile for every prefix length (rho_L = "
                         "master[:L] / sum): each user's state becomes two numbers, for the "
                         "dynamic program of alloc/whittle.py; prints its cost against the free "
                         "fit and writes the nested model")
    a = ap.parse_args()
    meta, levels, images = None, [], None
    for path in a.mixed:
        d, lv = load_levels(path)
        if images is not None and d["images"] != images:
            raise SystemExit("the mixed-SNR runs scored different images")
        images, meta = d["images"], meta or d
        levels += lv
    kappa, params, report = fit(levels)
    if a.nested:
        free = report
        kappa, params, report = fit(levels, nested=True)
        print(f"nested model: master profile {' '.join(f'{v:.3f}' for v in report['master'])}; "
              f"fit rms {report['fit_rms']:.3f} dB (free fit {free['fit_rms']:.3f}); held-out "
              f"MAE / p95 per CBR, nested against free:")
        for lv in levels:
            fh, nh = free["levels"][lv["key"]].get("held_out"), report["levels"][lv["key"]].get("held_out")
            if fh and nh:
                print(f"  {lv['cbr']:>6s}  nested {nh['mae']:.3f} / {nh['p95']:.3f}   free "
                      f"{fh['mae']:.3f} / {fh['p95']:.3f}   (+{nh['mae'] - fh['mae']:.3f} dB MAE)")
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
    if a.compare_abstractions:
        ab = abstraction_errors(levels, params, kappa, grid)
        held = [np.asarray(report["levels"][lv["key"]]["held_out"]["mae"]) for lv in levels
                if "held_out" in report["levels"][lv["key"]] and lv["chunks"] >= 2]
        print("link abstractions on the held-out profiles (levels with >= 2 chunks): one "
              "effective SNR read off the image's own constant curve; MAE / p95 (dB)")
        print(f"  {'averaging rule':28s} {'equal weights':>15s} {'sensitivity weights':>21s}")
        names = {"mean dB": "mean dB", "noise": "noise variance", "capacity": "capacity (MIESM-like)",
                 "EESM": "exponential (EESM)", "MMSE": "MMSE, kappa " + format(kappa, ".2f")}
        for rule in ("mean dB", "noise", "capacity", "EESM", "MMSE"):
            cells = []
            for wname in ("equal", "rho"):
                e = ab.get((rule, wname))
                cells.append("-" if e is None else f"{np.abs(e).mean():.3f} / "
                             f"{np.percentile(np.abs(e), 95):.3f}")
            print(f"  {names[rule]:28s} {cells[0]:>15s} {cells[1]:>21s}")
        if held:
            print("  additive model (alloc/utility.py), held-out MAE by level: "
                  + " ".join(f"{float(h):.3f}" for h in held))
    if a.calib:
        by_L, P = phase_levels(meta, levels)
        belief, crep = calibrate(by_L, P, images, a.calib, a.folds, a.nested)
        print(f"transmitter belief: kappa and rho cross-fitted over {a.folds} folds of images, "
              f"each image's Dsrc and C from its constant decodes at {a.calib} dB "
              f"({2 * P if len(a.calib) == 2 else len(a.calib) * P} decodes); errors (belief - "
              f"actual PSNR, MAE/p95) on the profiles it did not use")
        print(f"{'CBR':>6s} {'other constants':>16s} {'one at a time':>14s} {'random':>12s} "
              f"{'all unused':>12s} {'full fit, held-out':>19s}")
        for lv in levels:
            r = crep.get(str(lv["cbr"]), {})
            cell = lambda k: f"{r[k]['mae']:.3f}/{r[k]['p95']:.3f}" if k in r else "-"  # noqa: E731
            ho = report["levels"][lv["key"]].get("held_out")
            ho_s = f"{ho['mae']:.3f}/{ho['p95']:.3f}" if ho else "-"
            print(f"{lv['cbr']:>6s} {cell('constant'):>16s} {cell('one_at_a_time'):>14s} "
                  f"{cell('random'):>12s} {cell('all_unused'):>12s} {ho_s:>19s}")
        if a.belief_out:
            belief.d_empty = a.d_empty
            belief.meta = {"sources": a.mixed, "calib_snrs": a.calib, "folds": a.folds,
                           "run_name": meta.get("run_name"), "report": crep}
            os.makedirs(os.path.dirname(os.path.abspath(a.belief_out)), exist_ok=True)
            belief.save(a.belief_out)
            print(f"wrote {a.belief_out}")
    if a.out:
        by_L, P = phase_levels(meta, levels)
        dsrc = np.stack([params[by_L[L]["key"]][0] for L in range(1, P + 1)], 1)
        csens = np.stack([params[by_L[L]["key"]][1] for L in range(1, P + 1)], 1)
        rho = [params[by_L[L]["key"]][2] for L in range(1, P + 1)]
        util = Utility(kappa, rho, dsrc, csens, images, a.d_empty,
                       meta={"sources": a.mixed, "run_name": meta.get("run_name"),
                             "checkpoint": meta.get("checkpoint"),
                             "cbr_per_phase": str(Fraction(by_L[1]["cbr"])),
                             "nested": report.get("master"), "report": report})
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        util.save(a.out)
        print(f"wrote {a.out}: {len(images)} images x {P} prefix lengths")


if __name__ == "__main__":
    main()
