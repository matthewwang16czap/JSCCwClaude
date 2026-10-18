"""The latency budget at equal quality (docs/PROBLEM.md, section 3.7): how short a frame each
policy needs for a given quality. tools/schedule_sim.py at several frame lengths T (K users,
the same arguments otherwise), then each policy's quality against T, inverted linearly between
the lengths run. CPU.

    python tools/latency.py --utility U --belief B --slots 12 16 20 24 28 32 36 \\
        --policies pf slope_adv --ref slope_adv --at 16 24
    python tools/latency.py --utility U --belief B --target 30 --policies pf padc online \\
        --ref online --at 24 --levels 50 60 70 80

Reported: the curves (mean PSNR, or with --target the % of users at the target, per T); for
each --at T0 the reference policy's quality there and the frame length every other policy needs
to match it, with the extra slots and the share of that frame the reference saves, 1 - T0 /
T_needed (a policy that needs 26.4 slots where the reference needs 24 is matched with 9% fewer
slots), each with a 95% bootstrap interval over frames; with --levels, the length each policy
needs for each quality level (dB; with --target, % of users at it). A length outside the slots
run is shown as > (or <=) the nearest one.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import load_utility  # noqa: E402
from tools.schedule_sim import NEEDS_TARGET, POLICIES, simulate  # noqa: E402


def needed(Ts, vals, level):
    """The frame length at which a curve (vals at lengths Ts, linear in between) first reaches
    level: (T, "") inside the grid, (Ts[0], "<=") if it is there at the shortest already,
    (Ts[-1], ">") if it never gets there."""
    for j, v in enumerate(vals):
        if v >= level:
            if j == 0:
                return float(Ts[0]), "<="
            a = vals[j - 1]
            return float(Ts[j - 1] + (level - a) / (v - a) * (Ts[j] - Ts[j - 1])), ""
    return float(Ts[-1]), ">"


def boot_needed(Ts, frames, level_fn, boot, seed):
    """Bootstrap over frames (resampled per length, the same frames for every policy): the
    distribution of each policy's needed length. frames[p][j]: (F,) per-frame quality at Ts[j];
    level_fn(curves) -> the level to reach. Returns {p: (B,) lengths (NaN: outside the grid)}."""
    rng = np.random.default_rng(seed)
    pols = list(frames)
    out = {p: np.full(boot, np.nan) for p in pols}
    for b in range(boot):
        idx = [rng.integers(len(frames[pols[0]][j]), size=len(frames[pols[0]][j]))
               for j in range(len(Ts))]
        curves = {p: [float(frames[p][j][idx[j]].mean()) for j in range(len(Ts))] for p in pols}
        level = level_fn(curves)
        for p in pols:
            T, flag = needed(Ts, curves[p], level)
            if not flag:
                out[p][b] = T
    return out


def ci(x):
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--utility", required=True, help="tools/utility_fit.py output")
    ap.add_argument("--belief", default=None, help="the transmitter's curves (default: the truth)")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--slots", nargs="+", type=int, default=[12, 16, 20, 24, 28, 32, 36],
                    help="frame lengths to run (increasing)")
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--mean-snr", nargs=2, type=float, default=[0.0, 15.0])
    ap.add_argument("--fading", default="rayleigh", choices=["rayleigh", "none"])
    ap.add_argument("--snr-clip", nargs=2, type=float, default=[-2.0, 22.0])
    ap.add_argument("--policies", nargs="+", default=None, choices=POLICIES)
    ap.add_argument("--target", type=float, default=None, help="PSNR target (dB)")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--plan-offset", type=float, default=0.0)
    ap.add_argument("--ref", default=None, help="the policy matched (default: the last listed)")
    ap.add_argument("--at", nargs="+", type=int, default=None,
                    help="frame lengths at which the reference is matched (default: 24 if run)")
    ap.add_argument("--levels", nargs="*", type=float, default=[],
                    help="quality levels to reach (mean PSNR dB; with --target, % of users)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.policies is None:
        a.policies = ["pf", "padc", "online"] if a.target is not None else ["pf", "slope_adv"]
    needs = sorted(set(a.policies) & set(NEEDS_TARGET))
    if needs and a.target is None:
        ap.error(f"{', '.join(needs)} need --target")
    Ts = sorted(set(a.slots))
    if len(Ts) < 2:
        ap.error("--slots: at least two frame lengths")
    ref = a.ref or a.policies[-1]
    if ref not in a.policies:
        ap.error("--ref must be one of --policies")
    at = a.at if a.at is not None else ([24] if 24 in Ts else [Ts[len(Ts) // 2]])
    if any(t not in Ts for t in at):
        ap.error("--at must be among --slots")
    util = load_utility(a.utility)
    belief = load_utility(a.belief) if a.belief else None
    key = "mean_psnr" if a.target is None else "satisfied"
    scale, unit = (1.0, " dB") if a.target is None else (100.0, "%")
    curves = {p: [] for p in a.policies}
    frames = {p: [] for p in a.policies}
    for T in Ts:
        summ, _, _ = simulate(util, a.users, T, a.frames, a.mean_snr, a.fading, a.snr_clip,
                              a.policies, a.seed, target=a.target, margin=a.margin,
                              offset=a.plan_offset, belief_util=belief, keep_frames=True)
        for p in a.policies:
            curves[p].append(float(summ[p][key]))
            frames[p].append(np.asarray(summ[p]["frames"], dtype=np.float64))
    what = "mean PSNR (dB)" if a.target is None else f"users at {a.target:g} dB (%)"
    print(f"{a.users} users, {a.frames} frames per length, mean SNR U{a.mean_snr} dB, fading "
          f"{a.fading}; utility {a.utility}{f', transmitter curves {a.belief}' if a.belief else ''}")
    print(f"{what} by frame length T (slots; T / K phases per user on average)")
    print(f"{'T':>4s} {'T/K':>5s}  " + " ".join(f"{p:>11s}" for p in a.policies))
    for j, T in enumerate(Ts):
        print(f"{T:4d} {T / a.users:5.2f}  " + " ".join(f"{scale * curves[p][j]:11.2f}"
                                                     for p in a.policies))
    result = {"args": vars(a), "slots": Ts, "curves": curves, "match": {}, "levels": {}}
    for T0 in at:
        j0 = Ts.index(T0)
        level = curves[ref][j0]
        print(f"\nthe frame length each policy needs to match {ref} at T = {T0} "
              f"({scale * level:.2f}{unit}), 95% bootstrap intervals over frames:")
        dist = boot_needed(Ts, frames, lambda c, j0=j0: c[ref][j0], a.boot, a.seed)
        rows = {}
        for p in a.policies:
            if p == ref:
                continue
            T, flag = needed(Ts, curves[p], level)
            lo, hi = ci(dist[p])
            save = 1.0 - T0 / T if not flag else float("nan")
            slo, shi = (1.0 - T0 / lo, 1.0 - T0 / hi) if np.isfinite(lo) else (float("nan"),) * 2
            rows[p] = {"needed": T, "flag": flag, "ci": [lo, hi], "saved": save, "saved_ci": [slo, shi]}
            if flag:
                print(f"  {p:12s} needs {flag} {T:g} slots (outside the lengths run)")
            else:
                print(f"  {p:12s} needs {T:6.2f} slots [{lo:.2f}, {hi:.2f}]: {T - T0:+.2f} slots; "
                      f"{ref} needs {100 * save:.1f}% fewer [{100 * slo:.1f}, {100 * shi:.1f}]")
        result["match"][str(T0)] = {"level": level, "rows": rows}
    if a.levels:
        print("\nthe frame length each policy needs to reach each level (95% bootstrap intervals):")
        print(f"{'level':>8s}  " + " ".join(f"{p:>24s}" for p in a.policies))
        for lv in a.levels:
            target_level = lv / scale
            dist = boot_needed(Ts, frames, lambda c, x=target_level: x, a.boot, a.seed)
            cells, row = [], {}
            for p in a.policies:
                T, flag = needed(Ts, curves[p], target_level)
                lo, hi = ci(dist[p])
                row[p] = {"needed": T, "flag": flag, "ci": [lo, hi]}
                cells.append(f"{flag + format(T, '.0f'):>24s}" if flag
                             else f"{T:9.2f} [{lo:6.2f}, {hi:6.2f}]")
            result["levels"][str(lv)] = row
            print(f"{format(lv, 'g') + ('%' if a.target is not None else ' dB'):>8s}  " + " ".join(cells))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
