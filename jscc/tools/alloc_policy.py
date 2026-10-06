"""Per-image budget policy: the oracle, then the simple predictor-plus-argmax policy.

    # 1. How much could ANY per-image policy gain? (no training; test curves)
    python tools/alloc_policy.py oracle --curves curves/vit_kodak.npz --objective lpips

    # 2. Train the curve predictor on training curves (random SNRs)
    python tools/alloc_policy.py train --curves curves/vit_div2k_train.npz \\
        --out policies/vit_w1.pt

    # 3. The policy against uniform allocation, the noise-free heuristic and the oracle
    python tools/alloc_policy.py eval --curves curves/vit_kodak.npz \\
        --predictor policies/vit_w1.pt --objective lpips

Every method is compared with UNIFORM allocation at the same average CBR (see
alloc/core.py): the target average budget is met exactly, in expectation.

    uniform      every image gets the target budget (the fixed-rate system)
    sim          the transmitter decodes each image noise-free at every budget
                 and allocates on those curves (no learning, K local decodes)
    policy       the predictor's curves (3 local decodes + statistics + SNR)
    oracle       the measured curves themselves, selected on half of the noise
                 draws and scored on the other half (the ceiling)
    oracle_psnr  the oracle for PSNR: does the perceptual objective allocate
                 differently from the fidelity one?

--objective is what the allocation maximises ('lpips', 'dists', 'psnr', or a
mix such as 'lpips:0.7,psnr:0.3'); its first metric is the PRIMARY one, whose
gain, CI and equal-quality bandwidth saving are reported. Judge the result on
the metrics the objective did not use (the others printed alongside).
"""

import argparse
import json
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.core import (SIGN, bootstrap_ci, compare, describe, metric_scales,  # noqa: E402
                        objective, parse_objective, spearman)
from alloc.io import grid_snrs, load_curves, subset  # noqa: E402

GATES = {"psnr": 0.1, "obj_psnr": 0.1, "bg_psnr": 0.1, "msssim": 0.002, "ssim": 0.002,
         "lpips": 0.005, "lpips_vgg": 0.005, "dists": 0.005, "obj_lpips": 0.005}


# -- helpers ------------------------------------------------------------------------------

def nanmean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def targets_of(d, given):
    per_unit = float(d["cbr"][0]) / float(d["units"][0])
    if given:
        cbrs = [Fraction(c) for c in given]
    else:
        pre = d["meta"].get("predefined_cbrs") if isinstance(d["meta"], dict) else None
        if not pre or len(pre) < 3:
            raise SystemExit("pass --targets (e.g. 1/24 1/16 1/12)")
        cbrs = [Fraction(c) for c in pre[1:-1]]          # the endpoints have no freedom
    out = []
    for c in cbrs:
        t = float(c) / per_unit
        if not (d["units"][0] <= t <= d["units"][-1]):
            raise SystemExit(f"target CBR {c} is outside the swept budgets")
        out.append((str(c), t))
    return out, per_unit


def need_grid(d):
    snrs = grid_snrs(d)
    if snrs is None:
        raise SystemExit("these curves draw a random SNR per image (training curves); the "
                         "oracle and eval need a fixed grid: sweep the test split with --snrs")
    return snrs


def run(d, selectors_at, targets, scales, primary, folds):
    """compare() for every SNR column. selectors_at(s) -> {name: selector}."""
    cost = d["pixels"] / d["pixels"].mean()
    units = np.asarray(d["units"], dtype=np.float64)
    ts = [t for _, t in targets]
    return [compare(d["scores"][:, s].astype(np.float64), cost, units, d["metrics"],
                    selectors_at(s), ts, scales, primary, folds)
            for s in range(d["scores"].shape[1])]


def summarise(per_snr, methods, metrics, targets, per_unit, primary, seed):
    """SNR-mean per-image values -> means, paired CIs vs uniform, savings."""
    jp = metrics.index(primary)
    table = {}
    for label, t in targets:
        uni = np.mean([r["uniform"][t] for r in per_snr], axis=0)            # (N, M)
        row = {"uniform": {"cbr": t * per_unit, "means": uni.mean(0).tolist()}}
        for m in methods:
            vals = np.mean([r[m][t]["values"] for r in per_snr], axis=0)
            diff = vals - uni
            lo, hi = bootstrap_ci(diff[:, jp], seed=seed)
            row[m] = {"cbr": float(np.mean([r[m][t]["avg"] for r in per_snr])) * per_unit,
                      "means": vals.mean(0).tolist(), "delta": diff.mean(0).tolist(),
                      "ci": [lo, hi],
                      "saving": nanmean([r[m][t]["saving"] for r in per_snr]),
                      "reached": nanmean([r[m][t]["reached"] for r in per_snr]),
                      "units": np.mean([r[m][t]["units"] for r in per_snr], axis=0).tolist()}
        table[label] = row
    return table


def per_snr_deltas(per_snr, methods, metrics, targets, primary):
    jp = metrics.index(primary)
    out = {}
    for label, t in targets:
        out[label] = {m: [float((r[m][t]["values"][:, jp] - r["uniform"][t][:, jp]).mean())
                          for r in per_snr] for m in methods}
    return out


def print_tables(table, deltas, snrs, methods, metrics, primary, crossfit):
    print(f"\nSNR-mean over {snrs} dB; delta = method - uniform at the same average CBR; "
          f"95% bootstrap CI over images{'' if crossfit else ' (ORACLE NOT CROSS-FITTED: 1 draw)'}")
    head = "  ".join(f"{m:>9}" for m in metrics)
    for label, row in table.items():
        print(f"\n CBR {label}   {'method':<12} {'avg CBR':>8}  {head}   "
              f"{'delta ' + primary:>14} {'95% CI':>20} {'saving':>7}")
        if label == next(iter(table)):
            print(f" {'':9} ({primary}: {'higher' if SIGN[primary] > 0 else 'lower'} is better; "
                  f"saving = bandwidth cut at which the method matches uniform's {primary})")
        u = row["uniform"]
        print(f" {'':9} {'uniform':<12} {u['cbr']:8.5f}  " +
              "  ".join(f"{v:9.4f}" for v in u["means"]))
        for m in methods:
            c = row[m]
            ci = f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
            sv = f"{100 * c['saving']:6.1f}%" if np.isfinite(c["saving"]) else "    n/a"
            sv += "*" if c.get("reached", 1.0) < 1.0 - 1e-9 else " "
            print(f" {'':9} {m:<12} {c['cbr']:8.5f}  " +
                  "  ".join(f"{v:9.4f}" for v in c["means"]) +
                  f"   {c['delta'][metrics.index(primary)]:+14.4f} {ci:>20} {sv:>7}")
    if any(row[m].get("reached", 1.0) < 1.0 - 1e-9 for row in table.values() for m in methods):
        print(f"\n * at some SNR/fold the method never reached uniform's {primary} inside the "
              f"swept range; charged the full range there (saving <= 0), not skipped")
    print(f"\n delta {primary} per SNR ({', '.join(f'{s:g}' for s in snrs)} dB):")
    for label, row in deltas.items():
        for m, vals in row.items():
            print(f"   CBR {label:6} {m:<12} " + "  ".join(f"{v:+.4f}" for v in vals))


def allocation_difference(per_snr, a, b, targets, step):
    """How differently two methods spend: share of images whose budgets differ
    by at least half a candidate step, and the rank correlation of budgets."""
    out = {}
    for label, t in targets:
        fr, rho = [], []
        for r in per_snr:
            ua, ub = r[a][t]["units"], r[b][t]["units"]
            fr.append(float((np.abs(ua - ub) >= 0.5 * step).mean()))
            rho.append(spearman(ua, ub))
        out[label] = {"differ": float(np.mean(fr)), "spearman": float(np.nanmean(rho))}
    return out


def verdicts(table, method, metrics, primary, gate, gate_saving):
    jp, sign = metrics.index(primary), SIGN[primary]
    res = {}
    for label, row in table.items():
        c = row[method]
        gain = sign * c["delta"][jp]
        lo, hi = (sign * c["ci"][0], sign * c["ci"][1])
        sure = min(lo, hi) > 0
        # the gain must be real (paired CI over images excludes 0) in every case; the
        # size can then show as quality or as bandwidth. A saving alone is not
        # enough: on a flat, noisy curve a quality difference far inside the noise
        # moves the equal-quality crossing a long way.
        big = gain >= gate or (np.isfinite(c["saving"]) and c["saving"] >= gate_saving)
        passed = sure and big
        res[label] = {"gain": gain, "significant": bool(sure), "saving": c["saving"],
                      "pass": bool(passed)}
    return res


def dump(obj, path):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=1, default=float)
        print(f"\nwrote {path}")


# -- subcommands ----------------------------------------------------------------------------

def cmd_oracle(a):
    d = load_curves(a.curves[0])
    metrics = d["metrics"]
    spec = parse_objective(a.objective, metrics)
    primary = spec[0][0]
    snrs = need_grid(d)
    targets, per_unit = targets_of(d, a.targets)
    scales = metric_scales(np.nanmean(d["scores"], axis=(1, 2)))
    q_sim = objective(d["clean"].astype(np.float64), metrics, spec, scales)
    selectors = {"sim": q_sim, "oracle": ("oracle", spec)}
    if spec != [("psnr", 1.0)] and "psnr" in metrics:
        selectors["oracle_psnr"] = ("oracle", [("psnr", 1.0)])
    per_snr = run(d, lambda s: selectors, targets, scales, primary, a.folds)
    methods = list(selectors)
    table = summarise(per_snr, methods, metrics, targets, per_unit, primary, a.seed)
    deltas = per_snr_deltas(per_snr, methods, metrics, targets, primary)
    print(f"curves {a.curves[0]}: {len(d['names'])} images, {d['scores'].shape[2]} draw(s), "
          f"budgets {[int(u) for u in d['units']]}\nobjective {describe(spec)} "
          f"(primary: {primary})")
    print_tables(table, deltas, snrs, methods, metrics, primary, per_snr[0]["crossfit"])
    out = {"curves": a.curves[0], "objective": describe(spec), "primary": primary,
           "snrs": snrs, "table": table, "per_snr": deltas}
    step = float(np.min(np.diff(np.asarray(d["units"], dtype=np.float64))))
    if "oracle_psnr" in selectors:
        diff = allocation_difference(per_snr, "oracle", "oracle_psnr", targets, step)
        out["vs_psnr_allocation"] = diff
        print(f"\n does {describe(spec)} allocate differently from PSNR? "
              f"(oracle vs oracle_psnr, SNR-mean)")
        for label, v in diff.items():
            print(f"   CBR {label:6} budgets differ for {100 * v['differ']:5.1f}% of images, "
                  f"rank correlation {v['spearman']:+.3f}")
    gate = a.gate if a.gate is not None else GATES.get(primary, 0.01)
    ver = verdicts(table, "oracle", metrics, primary, gate, a.gate_saving)
    out["gate"] = {"threshold": gate, "saving_threshold": a.gate_saving, "oracle": ver}
    print(f"\n GATE (oracle): primary gain with its 95% CI excluding 0, AND either gain >= "
          f"{gate} or equal-quality saving >= {100 * a.gate_saving:.0f}%")
    for label, v in ver.items():
        print(f"   CBR {label:6} gain {v['gain']:+.4f}{' (CI excludes 0)' if v['significant'] else ''}"
              f", saving {100 * v['saving']:.1f}% -> {'PASS' if v['pass'] else 'fail'}")
    if any(v["pass"] for v in ver.values()):
        print("   -> a per-image policy has room here: train the predictor (step 3)")
    else:
        print("   -> even a perfect per-image policy gains too little: stop, or change the "
              "objective / codec")
    dump(out, a.out)


def cmd_train(a):
    from alloc import predictor as P
    d = load_curves(*a.curves)
    avail = d["metrics"]
    metrics = [m for m in (a.metrics or ["psnr", "msssim", "lpips", "lpips_vgg", "dists"])
               if m in avail]
    if not metrics:
        raise SystemExit(f"none of the requested metrics are in the curves ({avail})")
    if a.valid_curves:
        tr, va = d, load_curves(*a.valid_curves)
    else:
        rng = np.random.default_rng(a.seed)
        n = len(d["names"])
        perm = rng.permutation(n)
        cut = max(1, int(round(a.holdout * n)))
        tr, va = subset(d, np.sort(perm[cut:])), subset(d, np.sort(perm[:cut]))
    print(f"training curves: {len(tr['names'])} images x {tr['snr'].shape[1]} SNR(s); "
          f"validation: {len(va['names'])} images; metrics {metrics}; features {a.features}")
    ck = P.fit(tr, va, metrics, a.features, a.n_clean, a.hidden, a.depth, a.dropout, a.lr,
               a.weight_decay, a.epochs, a.batch, a.patience, a.seed, a.device)
    ck["curves"] = list(a.curves)
    rel = ck["valid_mse"] / max(ck["valid_mse_mean_curve"], 1e-12)
    print(f"best valid mse {ck['valid_mse']:.5f}; predicting the mean training curve for "
          f"every image would score {ck['valid_mse_mean_curve']:.5f} (ratio {rel:.2f}; "
          f"lower is better, 1.0 = learned nothing per image)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    import torch
    torch.save(ck, a.out)
    print(f"wrote {a.out}")


def cmd_eval(a):
    from alloc import predictor as P
    d = load_curves(a.curves[0])
    net, ck = P.load(a.predictor)
    metrics = d["metrics"]
    spec = parse_objective(a.objective, ck["metrics"])
    primary = spec[0][0]
    snrs = need_grid(d)
    targets, per_unit = targets_of(d, a.targets)
    # the predictor's scales, so policy, heuristic and oracle weigh a mixed objective alike
    scales = metric_scales(np.nanmean(d["scores"], axis=(1, 2)))
    for m, s in ck["scales"].items():
        scales[metrics.index(m)] = s
    pred = P.predict(net, ck, d, a.snr_offset)                       # (N, S, K, metrics)
    q_pol = P.predicted_objective(pred, ck, spec)                   # (N, S, K)
    q_sim = objective(d["clean"].astype(np.float64), metrics, spec, scales)

    def selectors(s):
        return {"sim": q_sim, "policy": q_pol[:, s], "oracle": ("oracle", spec)}

    per_snr = run(d, selectors, targets, scales, primary, a.folds)
    methods = ["sim", "policy", "oracle"]
    table = summarise(per_snr, methods, metrics, targets, per_unit, primary, a.seed)
    deltas = per_snr_deltas(per_snr, methods, metrics, targets, primary)
    # how well the curves themselves are predicted (signed gain over budget 0 / spread)
    v = np.nanmean(d["scores"], axis=2)                              # (N, S, K, M)
    rmse = {}
    for j, m in enumerate(ck["metrics"]):
        jm = metrics.index(m)
        true = SIGN[m] * (v[..., jm] - v[..., :1, jm]) / ck["scales"][m]
        rmse[m] = float(np.sqrt(np.nanmean((pred[..., j] - true) ** 2)))
    print(f"curves {a.curves[0]}: {len(d['names'])} images; predictor {a.predictor}"
          f"{f' (SNR fed {a.snr_offset:+g} dB off)' if a.snr_offset else ''}\n"
          f"objective {describe(spec)} (primary: {primary}); curve RMSE in spread units: " +
          " ".join(f"{m} {e:.3f}" for m, e in rmse.items()))
    print_tables(table, deltas, snrs, methods, metrics, primary, per_snr[0]["crossfit"])
    jp, sign = metrics.index(primary), SIGN[primary]
    print(f"\n share of the oracle's {primary} gain captured (SNR-mean):")
    captured = {}
    for label, row in table.items():
        o = sign * row["oracle"]["delta"][jp]
        captured[label] = {m: (sign * row[m]["delta"][jp] / o if o > 0 else float("nan"))
                           for m in ("sim", "policy")}
        print(f"   CBR {label:6} " + "  ".join(f"{m} {100 * captured[label][m]:6.1f}%"
                                             for m in ("sim", "policy")))
    dump({"curves": a.curves[0], "predictor": a.predictor, "objective": describe(spec),
          "primary": primary, "snrs": snrs, "snr_offset": a.snr_offset, "rmse": rmse,
          "table": table, "per_snr": deltas, "captured": captured}, a.out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--curves", nargs="+", required=True)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--out", default=None)

    def scoring(s):
        s.add_argument("--objective", default="lpips",
                       help="'lpips', 'dists', 'psnr', or a mix like 'lpips:0.7,psnr:0.3'")
        s.add_argument("--targets", nargs="+", default=None,
                       help="average CBRs to compare at (default: the interior predefined ones)")
        s.add_argument("--folds", type=int, default=2, help="cross-fitting folds over draws")

    o = sub.add_parser("oracle", help="the per-image allocation ceiling (no training)")
    common(o)
    scoring(o)
    o.add_argument("--gate", type=float, default=None,
                   help="minimum primary gain (default: 0.1 dB PSNR, 0.005 LPIPS/DISTS)")
    o.add_argument("--gate-saving", type=float, default=0.05,
                   help="or a minimum equal-quality bandwidth saving (the gain's CI "
                        "must still exclude 0)")
    t = sub.add_parser("train", help="fit the curve predictor on training curves")
    common(t)
    t.add_argument("--valid-curves", nargs="+", default=None,
                   help="early-stopping curves (default: hold out --holdout of the images)")
    t.add_argument("--holdout", type=float, default=0.1)
    t.add_argument("--metrics", nargs="+", default=None)
    t.add_argument("--features", nargs="+", default=["img", "code", "clean"],
                   choices=["img", "code", "clean"])
    t.add_argument("--n-clean", type=int, default=3,
                   help="noise-free decodes the transmitter runs (at budgets spread evenly)")
    t.add_argument("--hidden", type=int, default=256)
    t.add_argument("--depth", type=int, default=3)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--epochs", type=int, default=400)
    t.add_argument("--patience", type=int, default=40)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--device", default="cpu")
    e = sub.add_parser("eval", help="the trained policy vs uniform, sim and the oracle")
    common(e)
    scoring(e)
    e.add_argument("--predictor", required=True)
    e.add_argument("--snr-offset", type=float, default=0.0,
                   help="feed the predictor SNR + offset (robustness to feedback error)")
    a = p.parse_args()
    if a.cmd == "train" and not a.out:
        raise SystemExit("train needs --out <predictor.pt>")
    {"oracle": cmd_oracle, "train": cmd_train, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
