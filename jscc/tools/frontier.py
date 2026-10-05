"""Story B's decisive test: does a BUDGET-SCHEDULED perceptual weight beat CONSTANT ones?

    python tools/frontier.py \\
        --const curves/vit_mse_kodak.npz curves/vit_lp02c_kodak.npz curves/vit_lp_kodak.npz \\
        --test curves/vit_lp05b_kodak.npz --out results/frontier.json --plot results/frontier

Inputs are curve files from tools/alloc_sweep.py: every model scored on the
same images, SNRs and budgets. Sweeps run with the same --seed use the SAME
channel noise, so model-vs-model differences are paired image by image.

At every budget the constant-weight models (weight 0 = the MSE control) give
points (PSNR, perceptual metric): the perception-distortion trade-off that
constant weights reach there. The scheduled model is placed against it on the
HELD-OUT metrics (default DISTS and LPIPS-VGG; the training metric LPIPS-Alex
is printed for reference and never used in the verdict). The unknown curve
between the sampled weights is bracketed:

  chord   straight lines between neighbouring constant weights. Always
          reachable: decoding a share of the images with each of the two
          models lands exactly on it.
  bound   the neighbouring chords extended, and the segment's own left end.
          A concave trade-off curve cannot rise above them. With only 2
          constant weights the bound is just that end point, so only a model
          that dominates a constant one can be ABOVE.

Classes per budget (gaps in dB-equivalent: the perceptual difference at equal
PSNR divided by the local exchange rate along the chord, i.e. roughly the PSNR
the schedule saves at equal perceptual quality):

  ABOVE   above the bound by >= --min-db, 95% CI excluding 0: beats every
          constant weight, whatever the curve does between the sampled ones
  chord+  above the chord by >= --min-db (CI excluding 0) but not the bound:
          beats mixing the sampled models; a constant weight in between decides
  on      inside the bracket
  BELOW   below the chord (or dominated) by >= --min-db, CI excluding 0: a mix
          of constant-weight models does better

Verdict for the EFFICIENCY version of story B, on the held-out metrics:
  LIVES   ABOVE on both metrics at >= 2 budgets, BELOW nowhere
  DEAD    ABOVE or chord+ nowhere: the schedule only picks operating points
          that constant weights already reach (the PREFERENCE version remains)
  MIXED   ABOVE/chord+ somewhere and BELOW somewhere
  OPEN    otherwise: add the constant weight printed under the verdict

CIs are over images (paired bootstrap); they do NOT include run-to-run training
noise. If a verdict rests on gaps within ~2x --min-db, repeat the scheduled run
with another --seed before believing it.
"""

import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.core import SIGN  # noqa: E402
from alloc.io import grid_snrs, load_curves  # noqa: E402

PERC_TERMS = ("sem", "al", "imp", "lp")
INF = float("inf")


# -- which training objective produced a checkpoint ----------------------------------------

def run_dir(checkpoint):
    """history/<run>/models/last.pt -> <run>."""
    parts = os.path.normpath(str(checkpoint)).split(os.sep)
    if len(parts) >= 3 and parts[-2] == "models":
        return parts[-3]
    return parts[-2] if len(parts) >= 2 else parts[-1]


def parse_run(name):
    """The perceptual part of a run name (configs/config.py run_name):
    '..._lp0.5-const_p1' -> {'weights': {'lp': 0.5}, 'schedule': 'constant', 'gamma': 1.0};
    no perceptual part -> weights {} and schedule 'none' (pure pixel loss)."""
    for tok in str(name).split("_"):
        weights, sched, gamma, ok = {}, "budget", 1.0, True
        for item in tok.split("-"):
            m = re.fullmatch(r"(sem|al|imp|lp)(\d+(?:\.\d+)?)", item)
            if m:
                weights[m.group(1)] = float(m.group(2))
            elif item == "const":
                sched = "constant"
            elif re.fullmatch(r"g\d+(?:\.\d+)?", item):
                gamma = float(item[1:])
            else:
                ok = False
                break
        if ok and weights:
            return {"weights": weights, "schedule": sched, "gamma": gamma}
    return {"weights": {}, "schedule": "none", "gamma": 1.0}


def scalar_weight(info):
    ws = set(info["weights"].values())
    if not ws:
        return 0.0
    if len(ws) > 1:
        raise SystemExit(f"terms with different weights {info['weights']}: pass the weight "
                         f"explicitly (W=path for --const, --schedule for --test)")
    return ws.pop()


def schedule_factor(u, lo, hi, gamma):
    """w(u) exactly as net/loss.py: 1 at the smallest budget, 0 at the full one."""
    t = (hi - float(u)) / max(hi - lo, 1e-9)
    return min(1.0, max(0.0, t)) ** gamma


# -- the bracket ---------------------------------------------------------------------------

def hull(points):
    """Constant-weight points [(X, Y)] (X = fidelity, Y = signed perceptual, both
    higher-better) -> the upper-right frontier, X ascending: Pareto-efficient
    points on their upper concave hull (time-sharing reaches every chord)."""
    pts = sorted((float(x), float(y)) for x, y in points)
    eff, best = [], -INF
    for x, y in sorted(pts, key=lambda p: (-p[0], -p[1])):
        if y > best + 1e-15:
            eff.append((x, y))
            best = y
    eff.sort()
    H = []
    for p in eff:
        while len(H) >= 2:
            a, b = H[-2], H[-1]
            if (b[1] - a[1]) * (p[0] - a[0]) <= (p[1] - a[1]) * (b[0] - a[0]):
                H.pop()             # b on or under the chord a -> p
            else:
                break
        H.append(p)
    return H


def bracket(H, x):
    """(lower, upper, slope, extrapolated) of the constant-weight curve at X = x.

    lower  the chord inside the sampled range; left of it (less fidelity than
           the most perceptual model) that model's Y (it dominates anything
           below it there); right of it -inf (no constant model is that faithful)
    upper  the neighbouring chords extended (concavity) and the segment's left
           vertex (the curve does not rise with PSNR); outside the range the
           nearest chord extended
    slope  the local chord's slope (perceptual per dB, negative): the exchange
           rate that converts gaps to dB-equivalent"""
    n = len(H)
    xs = [p[0] for p in H]
    ys = [p[1] for p in H]

    def line(i, j, v):
        return ys[i] + (ys[j] - ys[i]) * (v - xs[i]) / (xs[j] - xs[i])

    def slope(i, j):
        return (ys[j] - ys[i]) / (xs[j] - xs[i])

    if n == 1:
        return (ys[0] if x <= xs[0] else -INF), INF, float("nan"), x != xs[0]
    if x < xs[0]:
        return ys[0], line(0, 1, x), slope(0, 1), True
    if x > xs[-1]:
        return -INF, line(n - 2, n - 1, x), slope(n - 2, n - 1), True
    k = min(int(np.searchsorted(xs, x, side="right")) - 1, n - 2)
    # the curve cannot rise above its left vertex (a model with more PSNR AND
    # better perception than another constant-weight model would contradict that
    # model's own optimum), nor above the neighbouring chords extended (concavity)
    ups = [ys[k]]
    if k >= 1:
        ups.append(line(k - 1, k, x))
    if k + 2 <= n - 1:
        ups.append(line(k + 1, k + 2, x))
    return line(k, k + 1, x), min(ups), slope(k, k + 1), False


def gaps(const_xy, test_xy):
    """dB-equivalent gaps of the test point to the lower and upper curve (positive
    = the test model is better), and whether it lies outside the sampled range."""
    H = hull(const_xy)
    lo, up, s, extra = bracket(H, test_xy[0])
    if not np.isfinite(s) or abs(s) < 1e-12:
        return float("nan"), float("nan"), extra
    rate = abs(s)
    return (test_xy[1] - lo) / rate, (test_xy[1] - up) / rate, extra


# -- statistics ----------------------------------------------------------------------------

def ci(samples, level=0.95):
    a = np.asarray(samples, dtype=np.float64)
    a = a[~np.isnan(a)]
    if len(a) < 2:
        return float("nan"), float("nan")
    a = np.clip(a, -1e9, 1e9)
    lo, hi = np.quantile(a, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def classify(gl, gl_ci, gu, gu_ci, extra, min_db):
    if np.isfinite(gu) and gu >= min_db and gu_ci[0] > 0:
        return "ABOVE"
    if np.isfinite(gl) and gl <= -min_db and gl_ci[1] < 0:
        return "BELOW"
    if not extra and np.isfinite(gl) and gl >= min_db and gl_ci[0] > 0:
        return "chord+"
    return "on"


def fmt(v, width=7):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return f"{'n/a':>{width}}"
    if not np.isfinite(v):
        return f"{('+inf' if v > 0 else '-inf'):>{width}}"
    return f"{v:+{width}.3f}"


def fmt_ci(c):
    if any(math.isnan(v) for v in c):
        return f"{'':>17}"

    def one(v):
        return ("+inf" if v > 0 else "-inf") if abs(v) >= 1e9 else f"{v:+.3f}"
    return f"[{one(c[0])}, {one(c[1])}]".rjust(17)


# -- loading -------------------------------------------------------------------------------

def split_spec(spec):
    """'LABEL=path' or 'path'."""
    if "=" in spec and not os.path.exists(spec):
        label, path = spec.split("=", 1)
        return label, path
    return None, spec


def load_models(const_specs, test_specs, schedule):
    models = []
    for spec in const_specs:
        label, path = split_spec(spec)
        d = load_curves(path)
        info = parse_run(run_dir(d["meta"].get("checkpoint", "")))
        if label is None:
            if info["schedule"] == "budget":
                raise SystemExit(f"{path}: its checkpoint trained a BUDGET schedule; a --const "
                                 f"entry must be a constant-weight (or MSE-only) model")
            w = scalar_weight(info)
        else:
            try:
                w = float(label)
            except ValueError:
                raise SystemExit(f"--const {spec!r}: the label must be the weight, e.g. 0.2=path")
        models.append({"role": "const", "weight": w, "label": f"w={w:g}", "path": path,
                       "data": d, "info": info})
    for spec in test_specs:
        label, path = split_spec(spec)
        d = load_curves(path)
        info = parse_run(run_dir(d["meta"].get("checkpoint", "")))
        if schedule is not None:
            lam, gamma = float(schedule[0]), float(schedule[1])
        else:
            if info["schedule"] != "budget":
                raise SystemExit(f"{path}: no budget schedule in its checkpoint name "
                                 f"({run_dir(d['meta'].get('checkpoint', ''))!r}); pass "
                                 f"--schedule WEIGHT GAMMA")
            lam, gamma = scalar_weight(info), info["gamma"]
        name = label or os.path.splitext(os.path.basename(path))[0]
        models.append({"role": "test", "weight": lam, "gamma": gamma, "label": name,
                       "path": path, "data": d, "info": info})
    return models


def check_same(models):
    """Every file must score the same images, budgets, SNRs and draws; same seed
    means the same channel noise (paired to the draw)."""
    ref = models[0]["data"]
    shared = True
    for m in models[1:]:
        d = m["data"]
        if d["names"] != ref["names"]:
            raise SystemExit(f"{m['path']}: different images from {models[0]['path']} "
                             f"(sweep every model with the same --split/--testset/--max-images)")
        if list(np.asarray(d["units"]).tolist()) != list(np.asarray(ref["units"]).tolist()):
            raise SystemExit(f"{m['path']}: different budgets from {models[0]['path']}")
        if grid_snrs(d) != grid_snrs(ref) or d["scores"].shape[1:3] != ref["scores"].shape[1:3]:
            raise SystemExit(f"{m['path']}: different SNR grid or number of draws")
        if d["meta"].get("seed") != ref["meta"].get("seed"):
            shared = False
    if grid_snrs(ref) is None:
        raise SystemExit("these curves draw random SNRs (training curves): sweep the test split")
    return shared


# -- the analysis --------------------------------------------------------------------------

def analyse(models, metrics_y, x_metric, n_boot, seed, min_db):
    ref = models[0]["data"]
    units = np.asarray(ref["units"], dtype=np.float64)
    pre = ref["meta"].get("predefined_units") or [units[0], units[-1]]
    lo_u, hi_u = float(pre[0]), float(pre[-1])
    names = list(ref["metrics"])
    for m in [x_metric] + list(metrics_y):
        if m not in names:
            raise SystemExit(f"metric {m!r} is not in the curve files ({names})")
    jx = names.index(x_metric)
    const = [m for m in models if m["role"] == "const"]
    tests = [m for m in models if m["role"] == "test"]
    N = const[0]["data"]["scores"].shape[0]
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, N, size=(n_boot, N))
    for m in models:
        sc = m["data"]["scores"].astype(np.float64)
        m["v"] = np.nanmean(sc, axis=(1, 2))                 # (N, K, M): SNR- and draw-mean
        m["mean"] = m["v"].mean(0)                             # (K, M)
        m["boot"] = m["v"][boots].mean(1)                      # (B, K, M), paired over models
        m["snr"] = np.nanmean(sc, axis=2).mean(0)              # (S, K, M)
    S = const[0]["snr"].shape[0]
    out = {}
    for t in tests:
        weff = [t["weight"] * schedule_factor(u, lo_u, hi_u, t["gamma"]) for u in units]
        res = {"weight": t["weight"], "gamma": t["gamma"], "w_eff": weff, "metrics": {},
               "matched": []}
        for my in metrics_y:
            jy, sg = names.index(my), SIGN[my]

            def xy(arr, k):
                return float(arr[k, jx]), float(sg * arr[k, jy])

            rows = []
            for k in range(len(units)):
                cxy = [xy(c["mean"], k) for c in const]
                gl, gu, extra = gaps(cxy, xy(t["mean"], k))
                bl, bu = [], []
                for b in range(n_boot):
                    a_, b_, _ = gaps([xy(c["boot"][b], k) for c in const], xy(t["boot"][b], k))
                    bl.append(a_)
                    bu.append(b_)
                gl_ci, gu_ci = ci(bl), ci(bu)
                per_snr = [list(gaps([xy(c["snr"][s], k) for c in const], xy(t["snr"][s], k))[:2])
                           for s in range(S)]
                H = hull(cxy)
                rows.append({"units": float(units[k]), "cbr": float(ref["cbr"][k]),
                             "w_eff": weff[k], "x_test": xy(t["mean"], k)[0],
                             "x_range": [min(p[0] for p in cxy), max(p[0] for p in cxy)],
                             "gap_chord": gl, "gap_chord_ci": list(gl_ci),
                             "gap_bound": gu, "gap_bound_ci": list(gu_ci),
                             "extrapolated": bool(extra),
                             "hull_weights": [c["weight"] for c, p in zip(const, cxy) if p in H],
                             "class": classify(gl, gl_ci, gu, gu_ci, extra, min_db),
                             "per_snr": per_snr})
            res["metrics"][my] = rows
        # budgets where the schedule's local weight equals a constant one: a paired
        # comparison with the SAME local objective (only the other budgets differ)
        for k, w in enumerate(weff):
            for c in const:
                if abs(w - c["weight"]) < 1e-6:
                    row = {"units": float(units[k]), "cbr": float(ref["cbr"][k]),
                           "weight": c["weight"], "delta": {}, "ci": {}}
                    for m in [x_metric] + list(metrics_y):
                        j = names.index(m)
                        row["delta"][m] = float(t["mean"][k, j] - c["mean"][k, j])
                        row["ci"][m] = list(ci(t["boot"][:, k, j] - c["boot"][:, k, j]))
                    res["matched"].append(row)
        out[t["label"]] = res
    return out


def verdict(res, held_out, min_db, const_weights):
    cls = {m: [r["class"] for r in res["metrics"][m]] for m in held_out}
    K = len(next(iter(cls.values())))
    above_all = sum(all(cls[m][k] == "ABOVE" for m in held_out) for k in range(K))
    any_below = any(c == "BELOW" for m in held_out for c in cls[m])
    any_gain = any(c in ("ABOVE", "chord+") for m in held_out for c in cls[m])
    if above_all >= 2 and not any_below:
        v = "LIVES"
    elif not any_gain:
        v = "DEAD"
    elif any_below:
        v = "MIXED"
    else:
        v = "OPEN"
    # where the bracket is too wide: the constant weights around the schedule's
    # local weight at the chord+ budgets
    suggest = set()
    for m in held_out:
        for r in res["metrics"][m]:
            if r["class"] == "chord+":
                below = [w for w in const_weights if w <= r["w_eff"] + 1e-9]
                above = [w for w in const_weights if w >= r["w_eff"] - 1e-9]
                if below and above and max(below) != min(above):
                    suggest.add(round(0.5 * (max(below) + min(above)), 3))
    return v, sorted(suggest)


# -- output --------------------------------------------------------------------------------

def print_report(models, out, held_out, ref_metric, x_metric, min_db, shared, snrs):
    const = [m for m in models if m["role"] == "const"]
    print("constant weights: " + ", ".join(f"{c['weight']:g} ({c['path']})" for c in const))
    d0 = const[0]["data"]
    print(f"{len(d0['names'])} images x SNRs {snrs} x {d0['scores'].shape[2]} draw(s); "
          f"channel noise {'SHARED (same seed): paired to the draw' if shared else 'NOT shared (different seeds): still paired by image, noisier'}")
    if len(const) < 3:
        print("NOTE: only 2 constant weights -> the upper bound is loose: ABOVE needs a model "
              "that beats a constant one on BOTH axes; chord+ / on / BELOW still mean what "
              "they say")
    for name, res in out.items():
        print(f"\n=== {name}: weight {res['weight']:g} x w(u), gamma {res['gamma']:g} ===")
        print(f"gaps in dB-equivalent at equal {x_metric} (+ = the schedule is better); "
              f"classes need |gap| >= {min_db} dB and a 95% CI excluding 0")
        for m in held_out + ([ref_metric] if ref_metric else []):
            tag = "held out" if m in held_out else "TRAINING METRIC: reference only"
            print(f"\n {m} ({tag})")
            print(f"  {'CBR':>7} {'w_eff':>6} {x_metric + ' test':>10} {'constant range':>17}  "
                  f"{'vs chord':>8} {'95% CI':>17}  {'vs bound':>8} {'95% CI':>17}  class")
            for r in res["metrics"][m]:
                rng = f"{r['x_range'][0]:.3f}-{r['x_range'][1]:.3f}"
                flag = " (outside)" if r["extrapolated"] else ""
                print(f"  {r['cbr']:7.4f} {r['w_eff']:6.3f} {r['x_test']:10.3f} {rng:>17}  "
                      f"{fmt(r['gap_chord'], 8)} {fmt_ci(r['gap_chord_ci'])}  "
                      f"{fmt(r['gap_bound'], 8)} {fmt_ci(r['gap_bound_ci'])}  {r['class']}{flag}")
        if res["matched"]:
            print(f"\n same local weight as a constant model (only the OTHER budgets' "
                  f"training differs): test - constant, 95% CI ({x_metric} higher is "
                  f"better; " + ", ".join(f"{m} {'higher' if SIGN[m] > 0 else 'lower'}"
                                          for m in held_out) + " is better)")
            for row in res["matched"]:
                cells = "  ".join(f"{m} {row['delta'][m]:+.4f} {fmt_ci(row['ci'][m]).strip()}"
                                  for m in [x_metric] + held_out)
                print(f"  CBR {row['cbr']:.4f} vs w={row['weight']:g}: {cells}")
        print(f"\n VERDICT (efficiency version of story B, on {', '.join(held_out)}): "
              f"{res['verdict']}")
        explain = {
            "LIVES": "the schedule beats every constant weight at >= 2 budgets and loses "
                     "nowhere: a Pareto gain, the strong claim",
            "DEAD": "nowhere better than constant weights (or a mix of two): the schedule only "
                    "chooses operating points that constant weights reach; what remains is the "
                    "preference version (fidelity at large budgets, perception at small ones)",
            "MIXED": "better at some budgets, worse at others: a trade, not a Pareto gain; "
                     "report both sides",
            "OPEN": "better than mixing the sampled models somewhere, but the bracket is too "
                    "wide to say more"}
        print(f"   {explain[res['verdict']]}")
        if res["suggest"]:
            print(f"   to narrow the bracket: train constant weight(s) "
                  f"{', '.join(f'{w:g}' for w in res['suggest'])} and rerun")


def plot(models, out, held_out, x_metric, prefix):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(no matplotlib: plots skipped; pip install matplotlib)")
        return
    const = [m for m in models if m["role"] == "const"]
    tests = [m for m in models if m["role"] == "test"]
    names = list(const[0]["data"]["metrics"])
    jx = names.index(x_metric)
    units = np.asarray(const[0]["data"]["units"], dtype=np.float64)
    cbr = np.asarray(const[0]["data"]["cbr"], dtype=np.float64)
    K = len(units)
    cols = 4
    rows = int(math.ceil(K / cols))
    for my in held_out:
        jy, sg = names.index(my), SIGN[my]
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.3 * rows), squeeze=False)
        for k in range(K):
            ax = axes[k // cols][k % cols]
            pts = [(float(c["v"][:, k, jx].mean()), float(c["v"][:, k, jy].mean())) for c in const]
            H = hull([(x, sg * y) for x, y in pts])
            ax.plot([p[0] for p in H], [sg * p[1] for p in H], "-", color="0.4", lw=1)
            for c, (x, y) in zip(const, pts):
                ax.plot(x, y, "o", color="0.2", ms=4)
                ax.annotate(f"{c['weight']:g}", (x, y), textcoords="offset points",
                            xytext=(4, 3), fontsize=7)
            for kk in range(len(H) - 1):            # the upper bound actually used
                seg = np.linspace(H[kk][0], H[kk + 1][0], 40)
                ax.plot(seg, [sg * bracket(H, x)[1] for x in seg], ":", color="0.5", lw=1)
            for t, colour in zip(tests, ("C3", "C0", "C2", "C1")):
                x, y = float(t["v"][:, k, jx].mean()), float(t["v"][:, k, jy].mean())
                ax.plot(x, y, "*", color=colour, ms=9, label=t["label"])
            ax.set_title(f"CBR {cbr[k]:.4f}", fontsize=9)
            ax.tick_params(labelsize=7)
            if k % cols == 0:
                ax.set_ylabel(my, fontsize=8)
            if k // cols == rows - 1:
                ax.set_xlabel(x_metric, fontsize=8)
        for k in range(K, rows * cols):
            axes[k // cols][k % cols].axis("off")
        axes[0][0].legend(fontsize=7)
        better = "lower" if sg < 0 else "higher"
        fig.suptitle(f"{my} ({better} is better) vs {x_metric}: constant weights (dots, labelled "
                     f"by weight; chord solid, bound dotted) and the schedule (stars)", fontsize=10)
        fig.tight_layout()
        path = f"{prefix}_{my}.png"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--const", nargs="+", required=True,
                   help="curve files of constant-weight models (the MSE control = weight 0); "
                        "'W=path' overrides the weight read from the checkpoint name")
    p.add_argument("--test", nargs="+", required=True,
                   help="curve files of budget-scheduled models ('NAME=path' to label)")
    p.add_argument("--schedule", nargs=2, type=float, default=None, metavar=("WEIGHT", "GAMMA"),
                   help="the test models' schedule (default: read from the checkpoint name)")
    p.add_argument("--held-out", nargs="+", default=["dists", "lpips_vgg"],
                   help="perceptual metrics the verdict uses (never the training metric)")
    p.add_argument("--train-metric", default="lpips",
                   help="printed for reference only ('' to hide)")
    p.add_argument("--x-metric", default="psnr")
    p.add_argument("--min-db", type=float, default=0.05,
                   help="smallest gap (dB-equivalent) that counts")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="json with every number")
    p.add_argument("--plot", default=None, help="prefix for one png per held-out metric")
    a = p.parse_args()

    models = load_models(a.const, a.test, a.schedule)
    const = [m for m in models if m["role"] == "const"]
    if len(const) < 2:
        raise SystemExit("need at least two constant-weight models (e.g. the MSE control and one)")
    if len({c["weight"] for c in const}) != len(const):
        raise SystemExit("two --const models have the same weight")
    shared = check_same(models)
    snrs = grid_snrs(const[0]["data"])
    ref_metric = a.train_metric if a.train_metric and a.train_metric in const[0]["data"]["metrics"] \
        and a.train_metric not in a.held_out else None
    metrics_y = list(a.held_out) + ([ref_metric] if ref_metric else [])
    out = analyse(models, metrics_y, a.x_metric, a.boot, a.seed, a.min_db)
    weights = sorted(c["weight"] for c in const)
    for res in out.values():
        res["verdict"], res["suggest"] = verdict(res, list(a.held_out), a.min_db, weights)
    print_report(models, out, list(a.held_out), ref_metric, a.x_metric, a.min_db, shared, snrs)
    if a.plot:
        plot(models, out, list(a.held_out), a.x_metric, a.plot)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"const": [{"weight": c["weight"], "path": c["path"]} for c in const],
                       "tests": out, "shared_noise": shared, "snrs": snrs, "min_db": a.min_db,
                       "held_out": a.held_out}, f, indent=1, default=float)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
