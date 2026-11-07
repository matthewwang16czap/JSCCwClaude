"""How long each scheduler takes to decide, on one CPU thread (docs/PAPER_OUTLINE.md, IV,
complexity). CPU.

    python tools/sched_timing.py --utility results/utility_snrc4_nested.json \\
        --belief results/belief_snrc4_nested.json --users 8 16 32 64 128 --frames 10

The frames of tools/rmab_scale.py (the same seed and arguments give the same frames): K users,
T slots, a slot of B = K / --users-per-unit resource units, a user at most one unit per slot,
exact reports. Only the transmitter's own work is timed, on its calibrated curves (--belief;
default: the truth), as it would run before and during a frame:

    clasp    the proposed scheduler, tools/rmab_scale.py's lagrange (mean PSNR) or, with
             --target, lagrange_soft. Before the first slot: every user's program
             (alloc/whittle.Arm on a grid of --ny x --ng), the per-slot prices that minimise the
             Lagrangian bound (--price-steps subgradient steps) and the programs priced at
             them. In every slot: for each user with phases left, its gain at those prices and
             its Whittle index (the tie rule), then the top B. With --target the programs are
             the calibration-aware ones, whose spread (soft_spread, from --utility against
             --belief) is measured once, offline, and not timed.
    quality  the quality-based allocation of the JSCC designs [10]-[12], tools/schedule_sim.py's
             fixed_slope: before the first slot, the phases of every user from its single-SNR
             curve at its mean SNR by the equal-slope rule over the frame's B T units; in every
             slot, PF among the users below their quota (mean PSNR)
    greedy   myopic quality, tools/schedule_sim.py's greedy: in every slot, the users with the
             largest predicted PSNR gain from this slot, given what they hold (mean PSNR)
    padc     per-image rate selection (PADC, Zhang et al., IEEE TWC 2023), tools/schedule_sim.py's
             padc: before the first slot, each user's shortest prefix whose predicted quality at
             its mean SNR reaches the target, shortest first while the units last; in every
             slot, PF among the users below their length (with --target)
    pf       proportional fair, tools/rmab_scale.py's pf: in every slot, rate over served-rate
             average, top B

Every slot's leftover units go to PF over the users with phases left, as in tools/rmab_scale.py.
Reported per K: each part's median over the frames and its 90th percentile, in milliseconds,
for the planning before the frame and the decision per slot (the frame's decisions over its T
slots); each policy's quality on the same frames, scored on --utility (mean PSNR per user, or
the share of users at the target), as a check that the timed policies are the ones of the
other tools; and the CPU. The threads of numpy's BLAS are limited to one before numpy loads.
"""

import argparse
import json
import os
import platform
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                 # one thread: set before numpy loads

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from alloc.utility import load_utility  # noqa: E402
from tools.rmab_scale import _score, draw, run, top  # noqa: E402
from tools.schedule_sim import need, rate, targets  # noqa: E402

MEAN = ("clasp", "quality", "greedy", "pf")
TARGET = ("clasp", "padc", "pf")
PARTS = {"clasp": ("programs", "prices", "priced")}


def cpu_name():
    """The CPU's model name (Linux), or what platform reports."""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def options(a):
    """The options tools/rmab_scale.run reads, for exact reports and no preemption."""
    return SimpleNamespace(target=a.target, margin=0.0, cqi_naive=False, cqi_corr=1.0,
                           snr_clip=list(a.snr_clip), preempt=0.0, tie_band=0.5, resolve=6,
                           price_steps=a.price_steps, price_scale="price", kink_grad=False,
                           slots=a.slots, mean_snr=list(a.mean_snr))


def clasp_frame(o, belief, imgs, means, snr, units, B, grid, soft):
    """CLASP on one frame: its schedule and the seconds of each part (programs, prices, priced
    before the frame; decide over the frame's slots)."""
    from alloc.whittle import Arm, lagrangian_bound
    T = snr.shape[1]
    clip = tuple(o.snr_clip)
    t0 = time.perf_counter()
    arms = [Arm(belief, int(i), float(m), T, target=o.target, p_lost=0.0, clip=clip,
                cqi_corr=1.0, soft=soft, **grid) for i, m in zip(imgs, means)]
    t1 = time.perf_counter()
    _, prices = lagrangian_bound(arms, T, 0.0, per_slot=o.price_steps, units=B,
                                 scale=o.price_scale)
    t2 = time.perf_counter()
    W = [a.priced(prices) for a in arms]
    t3 = time.perf_counter()
    out, _ = run(o, belief, belief, imgs, means, snr, units, None, "lagrange", B, arms, W, prices)
    t4 = time.perf_counter()
    return out, {"programs": t1 - t0, "prices": t2 - t1, "priced": t3 - t2,
                 "plan": t3 - t0, "decide": t4 - t3}


def baseline_frame(policy, belief, imgs, means, snr, units, goal):
    """quality, greedy, padc or pf on one frame: the schedule and the seconds of the planning
    before the frame and of the decisions over its slots."""
    K, T = snr.shape
    P = belief.P
    hist = [[] for _ in range(K)]
    served = np.zeros(K)
    t0 = time.perf_counter()
    quota = None
    if policy == "quality":                            # equal-slope quotas at the mean SNRs
        quota = targets(belief, imgs, means, int(np.sum(units)), "slope")
    elif policy == "padc":                             # shortest prefix to the target, shortest
        L = np.array([need(belief, int(i), [], float(m), goal) for i, m in zip(imgs, means)],
                     dtype=float)                      # first while the units last
        quota = np.zeros(K, dtype=int)
        left = int(np.sum(units))
        for k in np.argsort(L, kind="stable"):
            if np.isfinite(L[k]) and L[k] <= left:
                quota[k], left = int(L[k]), left - int(L[k])
    t1 = time.perf_counter()
    for t in range(T):
        Bt = int(units[t])
        if Bt == 0:
            continue
        has = np.array([len(h) < P for h in hist])
        if not has.any():
            break
        g = snr[:, t]
        r = rate(g)
        pf = r / (served + 1e-3 * r.mean())
        claim = has.copy()
        if policy == "greedy":                         # the predicted gain of this slot
            score = np.full(K, -np.inf)
            for k in np.flatnonzero(claim):
                i = int(imgs[k])
                score[k] = belief.psnr(i, hist[k] + [float(g[k])]) - belief.psnr(i, hist[k])
        else:
            score = pf
            if quota is not None:
                claim &= np.array([len(h) < q for h, q in zip(hist, quota)])
        chosen = top(score, claim, Bt)
        if len(chosen) < Bt:                           # leftover units: PF
            rest = has.copy()
            rest[chosen] = False
            chosen = np.concatenate([chosen, top(pf, rest, Bt - len(chosen))])
        served *= (1.0 - 1.0 / T)
        for k in chosen:
            hist[k].append(float(g[k]))
            served[k] += r[k] / T
    t2 = time.perf_counter()
    return hist, {"plan": t1 - t0, "decide": t2 - t1}


def stats(x):
    """Median and 90th percentile, in ms."""
    x = 1000.0 * np.asarray(x, dtype=np.float64)
    return float(np.median(x)), float(np.percentile(x, 90))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--utility", required=True, help="a --nested utility (tools/utility_fit.py)")
    ap.add_argument("--belief", default=None, help="the transmitter's curves (default: the truth)")
    ap.add_argument("--users", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--users-per-unit", type=int, default=8,
                    help="K / B: users per resource unit of a slot")
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--frames", type=int, default=10, help="frames timed per K")
    ap.add_argument("--mean-snr", nargs=2, type=float, default=[0.0, 15.0])
    ap.add_argument("--snr-clip", nargs=2, type=float, default=[-2.0, 22.0])
    ap.add_argument("--target", type=float, default=None,
                    help="PSNR target (dB): CLASP calibration-aware, against PADC and PF")
    ap.add_argument("--soft-sigma", default="auto",
                    help="the calibration-aware programs' spread: auto (measured) or dB")
    ap.add_argument("--policies", nargs="+", default=None,
                    help="default: clasp quality greedy pf (mean), clasp padc pf (--target)")
    ap.add_argument("--price-steps", type=int, default=10, help="subgradient steps on the prices")
    ap.add_argument("--ny", type=int, default=128, help="the programs' Y grid points")
    ap.add_argument("--ng", type=int, default=16, help="the programs' fading bins")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON of the per-K timings")
    a = ap.parse_args()
    allowed = TARGET if a.target is not None else MEAN
    pols = list(a.policies or allowed)
    bad = [p for p in pols if p not in allowed]
    if bad:
        ap.error(f"{' '.join(bad)}: not a policy for this objective (choose from "
                 f"{' '.join(allowed)})")
    if any(K % a.users_per_unit for K in a.users):
        ap.error("every --users value must be a multiple of --users-per-unit")
    if a.frames < 1 or a.ny < 8 or a.ng < 2 or a.price_steps < 0:
        ap.error("--frames >= 1, --ny >= 8, --ng >= 2, --price-steps >= 0")
    if a.soft_sigma != "auto":
        try:
            a.soft_sigma = float(a.soft_sigma)
        except ValueError:
            ap.error("--soft-sigma is 'auto' or a number (dB)")
    util = load_utility(a.utility)
    if getattr(util, "kind", "") == "digital":
        ap.error("the programs need a token-prefix utility")
    belief = load_utility(a.belief) if a.belief else util
    o = options(a)
    grid = dict(ny=a.ny, ng=a.ng)
    soft = None
    if a.target is not None and "clasp" in pols:
        from alloc.whittle import soft_spread
        soft = soft_spread(util, belief, a.soft_sigma, mean_snr=tuple(a.mean_snr),
                           clip=tuple(a.snr_clip))
    threads = os.environ.get("OMP_NUM_THREADS")
    cpu = cpu_name()
    what = "mean PSNR" if a.target is None else f"the share of users at {a.target:g} dB"
    print(f"CPU: {cpu}; numpy BLAS threads {threads}; {a.slots} slots, {a.users_per_unit} users "
          f"per unit, mean SNR U{a.mean_snr} dB, Rayleigh, exact reports; {what}; transmitter "
          f"curves {a.belief or 'the true ones'}; programs: Y grid {a.ny}, {a.ng} fading bins, "
          f"{a.price_steps} subgradient steps on the prices")
    if soft is not None:
        print("clasp is calibration-aware (lagrange_soft): the curves' error per prefix length, "
              "sigma " + " ".join(f"{v:.3f}" for v in soft[0]) + " dB (measured once, not timed)")
    table = {}
    for K in a.users:
        B = K // a.users_per_unit
        frames = draw(util, K, B, a.slots, a.frames, a.mean_snr, a.snr_clip, 0.0, 1.0, a.seed)
        rec = {p: {"plan": [], "decide": [], "score": []} for p in pols}
        if "clasp" in pols:
            rec["clasp"].update({q: [] for q in PARTS["clasp"]})
        t_k = time.perf_counter()
        for imgs, means, snr, units, _ in frames:
            for p in pols:
                if p == "clasp":
                    out, t = clasp_frame(o, belief, imgs, means, snr, units, B, grid, soft)
                else:
                    out, t = baseline_frame(p, belief, imgs, means, snr, units, a.target)
                for q, v in t.items():
                    rec[p][q].append(v)
                rec[p]["score"].append(_score(util, imgs, out, a.target))
        wall = time.perf_counter() - t_k
        print(f"\nK {K} users, B {B} units per slot, {a.frames} frames ({wall:.1f} s in all)")
        print(f"  {'policy':8s} {'plan per frame, ms':>22s} {'decide per slot, ms':>22s} "
              f"{'quality':>9s}   (median [90th percentile] over frames)")
        row = {"B": B, "frames": a.frames, "wall_s": wall}
        for p in pols:
            pm, p90 = stats(rec[p]["plan"])
            per_slot = np.asarray(rec[p]["decide"]) / a.slots
            dm, d90 = stats(per_slot)
            sc = float(np.mean(rec[p]["score"]))
            q = f"{sc:8.3f} dB" if a.target is None else f"{100 * sc:7.1f} %"
            print(f"  {p:8s} {pm:12.2f} [{p90:8.2f}] {dm:12.3f} [{d90:8.3f}] {q:>9s}")
            row[p] = {"plan_ms": [pm, p90], "decide_per_slot_ms": [dm, d90], "score": sc,
                      "plan_s": rec[p]["plan"], "decide_s": rec[p]["decide"]}
            if p == "clasp":
                parts = {q: stats(rec[p][q]) for q in PARTS["clasp"]}
                print("           clasp's planning: " + ", ".join(
                    f"{q} {v[0]:.2f} [{v[1]:.2f}]" for q, v in parts.items()) + " ms")
                row[p]["parts_ms"] = parts
        table[K] = row
    print(f"\nsummary, median ms (planning per frame / decision per slot), {cpu}")
    print(f"  {'K':>5s} {'B':>3s} " + " ".join(f"{p:>21s}" for p in pols))
    for K, row in table.items():
        print(f"  {K:5d} {row['B']:3d} " + " ".join(
            f"{row[p]['plan_ms'][0]:10.2f} / {row[p]['decide_per_slot_ms'][0]:8.3f}" for p in pols))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "cpu": cpu, "blas_threads": threads,
                       "per_K": {str(k): v for k, v in table.items()}}, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
