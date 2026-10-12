"""How much better than our schedulers any scheduler could do: the case for (or against) a
learned, e.g. reinforcement-learning, policy. CPU; about 1-3 s per frame and worker.

    python tools/headroom.py --utility results/utility_snrc4.json \\
        --belief results/belief_snrc4.json --base slope_adv --frames 200 --workers 16
    python tools/headroom.py --utility results/utility_snrc4.json \\
        --belief results/belief_snrc4.json --target 30 --base replan_adv --frames 200

The frames are tools/schedule_sim.py's (the same --seed, --users, --slots, --mean-snr,
--fading, --fade-corr, --preempt, --cqi-error give the same images, mean SNRs, fades, lost
slots and SNR reports). Per frame, three schedules:

    base      the --base policy of tools/schedule_sim.py, as it is
    rollout   one step of policy improvement on the base, causal (it knows what the base
              knows: the current slot's reported SNRs, never a future one). At every slot
              each candidate user (the --candidates the advantage index ranks highest, and
              the base's own choice) is served in turn, the rest of the frame is played out
              by the base policy over --futures sampled futures (fading from the mean SNRs,
              conditioned on the current slot with --fade-corr; lost slots with --preempt;
              the same futures for every candidate), scored on the transmitter's curves; the
              candidate best in expectation is served, if its paired gain over the base's
              choice exceeds --z standard errors (else the base's). In expectation never worse than the
              base (rollout, Bertsekas): its gain is a lower bound on what a better causal
              policy -- a trained reinforcement-learning one included -- can add with the
              same information and the same quality model
    oracle    clairvoyant local search: knows every slot's true SNR (and which are lost) in
              advance; from the base's and the rollout's schedules it moves single slots to
              other users (or leaves them unused) and swaps slots between users while the
              frame's objective improves, and keeps the better result. A lower bound on what
              perfect foresight could add

The objective is the mean PSNR, or with --target the share of users at or above it (rollout
and oracle break ties by the shortfall below the target). Scored on --utility. Reported per
schedule, and rollout - base and oracle - base paired by frame with 95% bootstrap CIs over
frames. If rollout - base is small, the base is close to the best causal policy for this
model and a learned policy has little to gain; if it is large, there is room to learn.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import load_utility  # noqa: E402
from tools.schedule_sim import (ADV_TIMING, TARGET_POLICIES, MEAN_POLICIES, bootstrap,  # noqa: E402
                                draw_frames, fades, planned_slots, rate, run_frame)

SHORTFALL = 0.002   # target objective: per dB below the target, against 1 for reaching it


def objective(util, imgs, hist, target):
    q = np.array([util.psnr(i, h) for i, h in zip(imgs, hist)])
    if target is None:
        return float(q.mean())
    return float(np.mean(q >= target) + SHORTFALL * np.mean(np.minimum(q - target, 0.0)))


def score(util, imgs, hist, target):
    """What is reported: mean PSNR, or the share at the target."""
    q = np.array([util.psnr(i, h) for i, h in zip(imgs, hist)])
    return float(q.mean()) if target is None else float(np.mean(q >= target))


def futures(rng, means, rep_t, t, T, fading, corr, clip, preempt, n):
    """n sampled continuations of slots t+1 .. T-1 (dB, users x slots) and their
    availability; with corr > 0 conditioned on the current slot's gain (its phase uniform)."""
    K, left = len(means), T - t - 1
    out = []
    for _ in range(n):
        if left <= 0:
            out.append((np.zeros((K, 0)), None if preempt <= 0 else np.zeros(0, dtype=bool)))
            continue
        if corr > 0 and fading == "rayleigh":
            g = 10.0 ** ((np.asarray(rep_t) - means) / 10.0)
            h = np.sqrt(g) * np.exp(2j * np.pi * rng.random(K))
            cols = []
            for _ in range(left):
                w = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) / np.sqrt(2)
                h = corr * h + np.sqrt(1.0 - corr ** 2) * w
                cols.append(10 * np.log10(np.maximum(np.abs(h) ** 2, 1e-12)))
            fade = np.stack(cols, 1)
        else:
            fade = fades(rng, K, left, fading)
        snr = np.clip(means[:, None] + fade, clip[0], clip[1])
        av = rng.random(left) >= preempt if preempt > 0 else None
        out.append((snr, av))
    return out


def rollout_frame(o, util, belief, imgs, means, snr, avail, seen, rng, log=None):
    """The rollout schedule of one frame: delivered SNR lists (true), as run_frame; `log`
    receives (slot, user) for every served slot."""
    K, T = snr.shape
    P = util.P
    rep = snr if seen is None else seen
    hist = [[] for _ in range(K)]
    out = [[] for _ in range(K)]
    served = np.zeros(K)
    avg = belief.average() if o.target is not None else None
    kw = dict(alpha=0.0, target=o.target, margin=o.margin, offset=o.plan_offset, avg=avg,
              corr=o.fade_corr if o.plan == "conditional" else 0.0, p_avail=1.0 - o.preempt,
              t_plan=o.t_plan)
    for t in range(T):
        if avail is not None and not avail[t]:
            continue
        left = [k for k in range(K) if len(hist[k]) < P]
        if not left:
            break
        log = []
        run_frame(belief, belief, imgs, means, rep, o.base, avail=None, start=(t, hist, served),
                  stop_after=1, log=log, **kw)
        own = log[0][1] if log else None
        g = rep[:, t]
        adv = {k: belief.psnr(imgs[k], hist[k] + [float(g[k])])
               - belief.psnr(imgs[k], hist[k] + [float(means[k] + o.plan_offset)]) for k in left}
        ranked = sorted(left, key=lambda k: -adv[k])
        cand = ranked if o.candidates <= 0 else ranked[:o.candidates]
        if own is not None and own not in cand:
            cand = cand + [own]
        if len(cand) == 1:
            best = cand[0]
        else:
            fut = futures(rng, means, g, t, T, o.fading, o.fade_corr, o.snr_clip, o.preempt,
                          o.futures)
            val = {}
            for k in cand:
                h2 = [list(h) for h in hist]
                h2[k].append(float(g[k]))
                s2 = (1 - 1.0 / T) * served
                s2[k] += rate(g[k]) / T
                v = []
                for fs, fa in fut:
                    sim = np.concatenate([rep[:, :t + 1], fs], 1)
                    av = None if fa is None else np.concatenate([np.ones(t + 1, dtype=bool), fa])
                    fin = run_frame(belief, belief, imgs, means, sim, o.base, avail=av,
                                    start=(t + 1, h2, s2), **kw)
                    v.append(objective(belief, imgs, fin, o.target))
                val[k] = np.array(v)
            best = own if own is not None else max(cand, key=lambda k: val[k].mean())
            if own is not None:                  # leave the base only for a clear gain: the
                gain = 0.0                       # paired (same futures) mean above z std errors
                for k in cand:
                    d = val[k] - val[own]
                    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                    if d.mean() > max(o.z * se, 1e-12) and d.mean() > gain:
                        best, gain = k, d.mean()
        hist[best].append(float(g[best]))
        out[best].append(float(snr[best, t]))
        if log is not None:
            log.append((t, best))
        served = (1 - 1.0 / T) * served
        served[best] += rate(g[best]) / T
    return out


def oracle_frame(o, util, imgs, snr, avail, log, max_passes=30):
    """Clairvoyant local search from the base's schedule (`log`: its (slot, user) pairs)."""
    K, T = snr.shape
    P = util.P
    owner = np.full(T, -1)
    for t, k in log:
        owner[t] = k
    usable = [t for t in range(T) if avail is None or avail[t]]

    def hist_of(k):
        return [float(snr[k, t]) for t in range(T) if owner[t] == k]

    def val(k):
        q = util.psnr(imgs[k], hist_of(k))
        if o.target is None:
            return q
        return float(q >= o.target) + SHORTFALL * min(q - o.target, 0.0)

    cur = np.array([val(k) for k in range(K)])
    count = np.array([int(np.sum(owner == k)) for k in range(K)])
    for _ in range(max_passes):
        better = False
        for t in usable:                                   # move slot t to user b (or nobody)
            a = owner[t]
            for b in [-1] + list(range(K)):
                if b == a or (b >= 0 and count[b] >= P):
                    continue
                owner[t] = b
                va = val(a) if a >= 0 else 0.0
                vb = val(b) if b >= 0 else 0.0
                old = (cur[a] if a >= 0 else 0.0) + (cur[b] if b >= 0 else 0.0)
                if va + vb > old + 1e-9:
                    if a >= 0:
                        cur[a], count[a] = va, count[a] - 1
                    if b >= 0:
                        cur[b], count[b] = vb, count[b] + 1
                    a, better = b, True
                else:
                    owner[t] = a
        for i, t1 in enumerate(usable):                    # swap slots between two users
            for t2 in usable[i + 1:]:
                a, b = owner[t1], owner[t2]
                if a == b or a < 0 or b < 0:
                    continue
                owner[t1], owner[t2] = b, a
                va, vb = val(a), val(b)
                if va + vb > cur[a] + cur[b] + 1e-9:
                    cur[a], cur[b], better = va, vb, True
                else:
                    owner[t1], owner[t2] = a, b
        if not better:
            break
    return [hist_of(k) for k in range(K)]


_G = {}


def _work(args):
    f, (imgs, means, snr, belief, avail, seen) = args
    o, util = _G["o"], _G["util"]
    rng = np.random.default_rng([o.seed, 2, f])
    avg = belief.average() if o.target is not None else None
    log = []
    base = run_frame(util, belief, imgs, means, snr, o.base, 0.0, o.target, o.margin,
                     o.plan_offset, avg, o.fade_corr if o.plan == "conditional" else 0.0, False,
                     avail, 1.0 - o.preempt, o.t_plan, seen, log=log)
    rlog = []
    roll = rollout_frame(o, util, belief, imgs, means, snr, avail, seen, rng, rlog)
    orc = None
    if o.oracle:                         # local search from both schedules, the better one
        orc = max((oracle_frame(o, util, imgs, snr, avail, lg) for lg in (log, rlog)),
                  key=lambda h: objective(util, imgs, h, o.target))
    res = {"base": score(util, imgs, base, o.target), "rollout": score(util, imgs, roll, o.target)}
    if orc is not None:
        res["oracle"] = score(util, imgs, orc, o.target)
    res["phases"] = {"base": float(np.mean([len(h) for h in base])),
                     "rollout": float(np.mean([len(h) for h in roll]))}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--utility", required=True)
    ap.add_argument("--belief", default=None, help="the transmitter's curves (default: the truth)")
    ap.add_argument("--base", default=None, help="default slope_adv, or replan_adv with --target")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--plan-offset", type=float, default=0.0)
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--mean-snr", nargs=2, type=float, default=[0.0, 15.0])
    ap.add_argument("--fading", default="rayleigh", choices=["rayleigh", "none"])
    ap.add_argument("--fade-corr", type=float, default=0.0)
    ap.add_argument("--plan", default="mean", choices=["mean", "conditional"])
    ap.add_argument("--preempt", type=float, default=0.0)
    ap.add_argument("--preempt-quantile", type=float, default=None)
    ap.add_argument("--cqi-error", type=float, default=0.0)
    ap.add_argument("--snr-clip", nargs=2, type=float, default=[-2.0, 22.0])
    ap.add_argument("--futures", type=int, default=8, help="rollout: sampled futures per decision")
    ap.add_argument("--candidates", type=int, default=3,
                    help="rollout: users tried per slot (the index's top ones + the base's "
                         "choice); 0 = every user with phases left")
    ap.add_argument("--z", type=float, default=1.0,
                    help="rollout: leave the base's choice only if the paired gain over the "
                         "sampled futures exceeds z standard errors (0: any gain)")
    ap.add_argument("--no-oracle", dest="oracle", action="store_false")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    o = ap.parse_args()
    o.base = o.base or ("slope_adv" if o.target is None else "replan_adv")
    pols = TARGET_POLICIES if o.target is not None else MEAN_POLICIES
    if o.base not in pols:
        ap.error(f"--base {o.base} is not a {'target' if o.target is not None else 'mean'} policy")
    util = load_utility(o.utility)
    if getattr(util, "kind", "") == "digital":
        ap.error("the headroom study is for token-prefix utilities")
    belief = load_utility(o.belief) if o.belief else None
    o.t_plan = planned_slots(o.slots, o.preempt, o.preempt_quantile)
    frames = list(draw_frames(util, o.users, o.slots, o.frames, o.mean_snr, o.fading, o.snr_clip,
                              o.seed, False, 0.0, belief, o.fade_corr, o.preempt, o.cqi_error))
    _G.update(o=o, util=util)
    t0 = time.time()
    if o.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(o.workers) as pool:
            res = pool.map(_work, list(enumerate(frames)), chunksize=1)
    else:
        res = [_work(x) for x in enumerate(frames)]
    print(f"{o.users} users, {o.slots} slots, {o.frames} frames, mean SNR U{o.mean_snr} dB, "
          f"fading {o.fading}"
          f"{f' (slot correlation {o.fade_corr:g}, plan {o.plan})' if o.fade_corr else ''}"
          f"{f', {100 * o.preempt:g}% of slots preempted' if o.preempt else ''}"
          f"{f', CQI error {o.cqi_error:g} dB' if o.cqi_error else ''}"
          f"{f', transmitter curves {o.belief}' if o.belief else ''}; utility {o.utility}")
    what = "mean PSNR (dB)" if o.target is None else f"share at {o.target:g} dB"
    print(f"base {o.base}{' (index timing)' if o.base in ADV_TIMING else ''}; rollout: "
          f"{o.futures} futures, {o.candidates or 'all'} candidates; {time.time() - t0:.0f} s")
    keys = ["base", "rollout"] + (["oracle"] if o.oracle else [])
    scale = 1.0 if o.target is None else 100.0
    unit = " dB" if o.target is None else " pts"
    for k in keys:
        v = np.array([r[k] for r in res])
        extra = f"   phases/user {np.mean([r['phases'][k] for r in res]):.2f}" if k in res[0]["phases"] else ""
        print(f"  {k:8s} {what}: {scale * v.mean():8.3f}{'%' if o.target is not None else ''}{extra}")
    for k in keys[1:]:
        d = np.array([r[k] - r["base"] for r in res])
        lo, hi = bootstrap(d, seed=o.seed)
        print(f"  {k} - base: {scale * d.mean():+.3f}{unit} [{scale * lo:+.3f}, {scale * hi:+.3f}]  "
              f"(frames better / worse: {int(np.sum(d > 1e-9))} / {int(np.sum(d < -1e-9))})")


if __name__ == "__main__":
    main()
