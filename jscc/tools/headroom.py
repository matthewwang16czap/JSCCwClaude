"""How much better than our schedulers any scheduler could do, and the exact optimum of the
frame with perfect foresight: the case for (or against) a learned, e.g. reinforcement-
learning, policy. CPU; about 3-4 s per frame and worker (scipy for the optimum).

    python tools/headroom.py --utility results/utility_snrc4.json \\
        --belief results/belief_snrc4.json --base slope_adv --frames 200 --workers 16
    python tools/headroom.py --utility results/utility_snrc4.json \\
        --belief results/belief_snrc4.json --target 30 --base replan_adv --frames 200

The frames are tools/schedule_sim.py's (the same --seed, --users, --slots, --mean-snr,
--fading, --fade-corr, --preempt, --cqi-error give the same images, mean SNRs, fades, lost
slots and SNR reports). Per frame:

    base      the --base policy of tools/schedule_sim.py, as it is
    rollout   one step of policy improvement on the base, causal (it knows what the base
              knows: the current slot's reported SNRs, never a future one). At every slot
              each candidate user (the --candidates the advantage index ranks highest, and
              the base's own choice) is served in turn, the rest of the frame is played out
              by the base policy over --futures sampled futures (fading from the mean SNRs,
              conditioned on the current slot with --fade-corr; lost slots with --preempt;
              the same futures for every candidate), scored on the transmitter's curves; the
              candidate best in expectation is served, if its paired gain over the base's
              choice exceeds --z standard errors (else the base's). In expectation never
              worse than the base (rollout, Bertsekas): its gain is a lower bound on what a
              better causal policy -- a trained reinforcement-learning one included -- can
              add with the same information and the same quality model
    oracle    clairvoyant local search: knows every slot's true SNR (and which are lost) in
              advance; from the base's and the rollout's schedules it moves single slots to
              other users (or leaves them unused) and swaps slots between users while the
              frame's objective improves, and keeps the better result
    optimum, bound
              the frame's problem with every SNR known -- assign the T slots to the K users,
              at most P each, to maximise the sum of the users' predicted quality -- solved
              by column generation on its configuration LP (each user one set of slots, each
              slot to at most one user; every set of up to P slots is priced, so the LP's
              value is an upper bound on every schedule, causal or not) and an integer
              program over the generated columns (the best schedule found). bound = optimum
              certifies it. Prefix order (the phases go out in time order), and with any
              phase order (each user's phases mapped to its slots by sensitivity)
    base, phases mapped by sensitivity
              the base's own slots with each user's most sensitive phase in its best slot:
              what choosing the phase per slot could add when the slot count is known
    slot counts
              (slope_adv) how many slots each user is planned at its mean SNR: the greedy
              equal-slope rule against the exact dynamic program over users

The objective is the mean PSNR, or with --target the share of users at or above it (rollout
and oracle break ties by the shortfall below the target; the optimum maximises the count).
Scored on --utility. Paired by frame against the base with 95% bootstrap CIs over frames.
If rollout - base is small, the base is close to the best causal policy for this model and a
learned policy has little to gain; optimum - base is what even perfect foresight would add.
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
        step = []
        run_frame(belief, belief, imgs, means, rep, o.base, avail=None, start=(t, hist, served),
                  stop_after=1, log=step, **kw)
        own = step[0][1] if step else None
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


def by_sensitivity(util, i, snrs):
    """The same slots with the phases re-mapped: the most noise-sensitive phase in the best
    slot, and so on down (the best mapping for a fixed set of slots: rearrangement)."""
    L = len(snrs)
    if L < 2:
        return [float(x) for x in snrs]
    r = util.rho[L - 1] if util.rho[L - 1].ndim == 1 else util.rho[L - 1][i]
    out = np.empty(L)
    out[np.argsort(-r, kind="stable")] = np.sort(np.asarray(snrs, dtype=np.float64))[::-1]
    return out.tolist()


def oracle_frame(o, util, imgs, snr, avail, log, free=False, max_passes=30):
    """Clairvoyant local search from a schedule (`log`: its (slot, user) pairs). free: the
    phases of each user's slots mapped by sensitivity instead of sent in prefix order.
    Returns the histories (in phase order) and the slot sets [(user, slots)]."""
    K, T = snr.shape
    P = util.P
    owner = np.full(T, -1)
    for t, k in log:
        owner[t] = k
    usable = [t for t in range(T) if avail is None or avail[t]]

    def hist_of(k):
        h = [float(snr[k, t]) for t in range(T) if owner[t] == k]
        return by_sensitivity(util, imgs[k], h) if free else h

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
    sets = [(k, tuple(int(t) for t in np.flatnonzero(owner == k))) for k in range(K)]
    return [hist_of(k) for k in range(K)], sets


_CB = {}


def combos(T, L):
    """Every set of L of T slots, as sorted index rows (C(T, L), L); cached."""
    if (T, L) not in _CB:
        from itertools import combinations
        _CB[T, L] = np.array(list(combinations(range(T), L)), dtype=np.intp).reshape(-1, L)
    return _CB[T, L]


def n_sets(T, P):
    from math import comb
    return sum(comb(T, L) for L in range(1, P + 1))


def subset_values(util, imgs, snr, avail, target, free):
    """Every user's value of every set of up to P slots, with perfect foresight: its PSNR
    (or 1 if it reaches the target) when its phases go out in those slots in time order,
    or, with free=True, mapped by sensitivity. v0 (K,): nothing; V[L - 1] (K, C(T, L))."""
    K, T = snr.shape
    v0 = np.array([util.psnr(i, []) for i in imgs])
    if target is not None:
        v0 = (v0 >= target).astype(float)
    V = []
    for L in range(1, util.P + 1):
        cb = combos(T, L)
        out = np.empty((K, len(cb)))
        for k, i in enumerate(imgs):
            kappa = util.kappa if np.ndim(util.kappa) == 0 else util.kappa[i]
            n = 10.0 ** (-np.asarray(snr[k], dtype=np.float64) / 10.0)
            F = (n / (1.0 + n / kappa))[cb]                  # (C, L): time order = phase order
            r = util.rho[L - 1] if util.rho[L - 1].ndim == 1 else util.rho[L - 1][i]
            if free:                                         # the best slot to the most
                F, r = np.sort(F, 1), np.sort(r)[::-1]       # sensitive phase
            q = -10.0 * np.log10(np.maximum(util.dsrc[i, L - 1] + util.csens[i, L - 1] * (F @ r),
                                            1e-12))
            out[k] = q if target is None else (q >= target)
        if avail is not None:
            out[:, ~np.asarray(avail)[cb].all(1)] = -np.inf  # a lost slot carries nothing
        V.append(out)
    return v0, V


def set_value(util, i, snrs, target, free):
    q = util.psnr(i, by_sensitivity(util, i, snrs) if free else list(snrs))
    return q if target is None else float(q >= target)


def clairvoyant(util, imgs, snr, avail, target, free=False, start=(), max_iter=300,
                ip_time=20.0, enrich=4):
    """The best schedule with perfect foresight, bracketed. The configuration LP (each user
    one set of slots, each slot to at most one user, relaxed to fractions) is solved by
    column generation; its value, the Lagrangian dual of "one user per slot", bounds EVERY
    schedule from above. The integer program over the generated columns gives the best
    schedule found. Returns (lower, upper, histories) in the objective summed over users;
    lower == upper certifies the optimum. start: (user, slots) sets that seed the columns
    (heuristic schedules); enrich: each user's best sets per length added before the integer
    program."""
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    K, T = snr.shape
    v0, V = subset_values(util, imgs, snr, avail, target, free)
    cbs = [combos(T, L) for L in range(1, util.P + 1)]
    cols, seen = [], set()

    def add(k, S, v):
        if (k, S) in seen or not np.isfinite(v):
            return False
        seen.add((k, S))
        cols.append((k, S, float(v)))
        return True

    for k in range(K):
        add(k, (), v0[k])
    for k, S in start:
        if S and (avail is None or all(avail[t] for t in S)):
            add(k, tuple(S), set_value(util, imgs[k], [snr[k, t] for t in S], target, free))

    def master():
        n = len(cols)
        A_eq, A_ub = np.zeros((K, n)), np.zeros((T, n))
        for j, (k, S, _) in enumerate(cols):
            A_eq[k, j] = 1.0
            if S:
                A_ub[list(S), j] = 1.0
        return -np.array([v for _, _, v in cols]), A_eq, A_ub

    upper = np.inf
    for _ in range(max_iter):
        c, A_eq, A_ub = master()
        res = linprog(c, A_ub=A_ub, b_ub=np.ones(T), A_eq=A_eq, b_eq=np.ones(K),
                      bounds=(0, None), method="highs")
        if res.status != 0:
            break
        lam = np.maximum(-res.ineqlin.marginals, 0.0)      # the price of each slot
        mu = -res.eqlin.marginals
        best, arg = v0.copy(), [None] * K                  # pricing: each user's best set
        for Li, cb in enumerate(cbs):
            sc = V[Li] - lam[cb].sum(1)[None, :]
            j = sc.argmax(1)
            m = sc[np.arange(K), j]
            for k in np.flatnonzero(m > best):
                best[k], arg[k] = m[k], (Li, int(j[k]))
        upper = min(upper, float(lam.sum() + best.sum()))  # weak duality: valid at any price
        added = 0
        for k in range(K):
            if arg[k] is not None and best[k] - mu[k] > 1e-9:
                Li, j = arg[k]
                added += add(k, tuple(int(t) for t in cbs[Li][j]), V[Li][k, j])
        if not added or upper + res.fun < 1e-7:
            break
    if res.status == 0:                  # more columns for the integer program: each user's
        for Li, cb in enumerate(cbs):    # few best sets of every length at the final prices
            sc = V[Li] - lam[cb].sum(1)[None, :]
            m = min(enrich, sc.shape[1])
            for k in range(K):
                for j in np.argpartition(-sc[k], m - 1)[:m]:
                    add(k, tuple(int(t) for t in cb[j]), V[Li][k, j])
    if target is not None:               # a count: no schedule exceeds the bound's floor
        upper = float(np.floor(upper + 1e-6))
    c, A_eq, A_ub = master()
    res = milp(c, constraints=[LinearConstraint(A_eq, 1, 1), LinearConstraint(A_ub, -np.inf, 1)],
               integrality=np.ones(len(cols)), bounds=Bounds(0, 1),
               options={"time_limit": ip_time})
    hist = [[] for _ in range(K)]
    if res.x is None:
        return -np.inf, upper, hist
    lower = 0.0
    for j in np.flatnonzero(res.x > 0.5):
        k, S, v = cols[j]
        lower += v
        h = [float(snr[k, t]) for t in S]
        hist[k] = by_sensitivity(util, imgs[k], h) if free else h
    return lower, upper, hist


def exact_counts(util, imgs, means, T):
    """How many slots each user gets, planned at the mean SNRs, solved exactly: a dynamic
    program over users (the separable allocation max sum_k q_k(L_k), sum L_k = min(T, K P),
    L_k <= P). Returns the lengths and the curves q (K, P + 1)."""
    K, P = len(imgs), util.P
    q = np.array([[util.psnr(i, [m] * L) for L in range(P + 1)] for i, m in zip(imgs, means)])
    n = min(T, K * P)
    best, choice = np.zeros(1), []
    for k in range(K):                                     # best[m]: users 0..k with m slots
        new = np.full(min(n, (k + 1) * P) + 1, -np.inf)
        ch = np.zeros(len(new), dtype=int)
        for m in range(len(new)):
            for L in range(max(0, m - len(best) + 1), min(P, m) + 1):
                v = best[m - L] + q[k, L]
                if v > new[m]:
                    new[m], ch[m] = v, L
        best = new
        choice.append(ch)
    L, m = np.zeros(K, dtype=int), n
    for k in reversed(range(K)):
        L[k] = choice[k][m]
        m -= L[k]
    return L, q


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
    K = len(imgs)
    res = {"base": score(util, imgs, base, o.target), "rollout": score(util, imgs, roll, o.target),
           "phases": {"base": float(np.mean([len(h) for h in base])),
                      "rollout": float(np.mean([len(h) for h in roll]))}}
    res["base_remapped"] = score(util, imgs, [by_sensitivity(util, i, h) for i, h in
                                              zip(imgs, base)], o.target)
    sets = []
    if o.oracle:                         # local search from both schedules, the better one
        found = [oracle_frame(o, util, imgs, snr, avail, lg) for lg in (log, rlog)]
        orc, sets = max(found, key=lambda x: objective(util, imgs, x[0], o.target))
        res["oracle"] = score(util, imgs, orc, o.target)
        sets = [s for _, ss in found for s in ss]
    for lg in (log, rlog):
        sets += [(k, tuple(t for t, kk in lg if kk == k)) for k in range(K)]
    if o.bound and f < o.bound_frames:
        for tag, free in (("", False), ("_any", True))[:2 if o.any_order else 1]:
            lo, up, h = clairvoyant(util, imgs, snr, avail, o.target, free, sets)
            res["bound" + tag] = up / K
            if np.isfinite(lo):
                res["optimum" + tag] = lo / K
    if o.target is None and o.base in ("fixed_slope", "slope_adv"):
        from tools.schedule_sim import targets
        g = targets(belief, imgs, means, o.slots if o.t_plan is None else o.t_plan, "slope")
        L, q = exact_counts(belief, imgs, means, o.slots if o.t_plan is None else o.t_plan)
        k = np.arange(K)
        res["counts_loss"] = float((q[k, L].sum() - q[k, g].sum()) / K)
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
    ap.add_argument("--no-bound", dest="bound", action="store_false",
                    help="skip the clairvoyant optimum and its bound (needs scipy)")
    ap.add_argument("--bound-frames", type=int, default=None,
                    help="the optimum and bound on the first N frames only (default: all)")
    ap.add_argument("--no-any-order", dest="any_order", action="store_false",
                    help="skip the clairvoyant optimum with phases mapped by sensitivity")
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
    o.bound_frames = o.frames if o.bound_frames is None else o.bound_frames
    if o.bound:
        import importlib.util
        if importlib.util.find_spec("scipy") is None:
            print("scipy not found (pip install scipy): no clairvoyant optimum / bound")
            o.bound = False
    if o.bound and n_sets(o.slots, util.P) > 2_000_000:
        print(f"{n_sets(o.slots, util.P):,} slot sets per user: too many to enumerate, no "
              f"clairvoyant optimum / bound")
        o.bound = False
    if o.bound:
        for L in range(1, util.P + 1):                    # built once, shared by the workers
            combos(o.slots, L)
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
    scale = 1.0 if o.target is None else 100.0
    unit = " dB" if o.target is None else " pts"
    label = {"base": "base (our rule)", "rollout": "rollout (causal)",
             "oracle": "oracle (clairvoyant local search)",
             "optimum": "clairvoyant optimum found", "bound": "clairvoyant upper bound",
             "base_remapped": "base, phases mapped by sensitivity",
             "optimum_any": "clairvoyant optimum, any phase order",
             "bound_any": "clairvoyant bound, any phase order"}
    order = ["base", "rollout", "oracle", "optimum", "bound", "base_remapped", "optimum_any",
             "bound_any"]
    for k in order:
        rows = [r for r in res if k in r]
        if not rows:
            continue
        v = np.array([r[k] for r in rows])
        ph = f"   phases/user {np.mean([r['phases'][k] for r in rows]):.2f}" if k in rows[0]["phases"] else ""
        n = "" if len(rows) == len(res) else f"   ({len(rows)} frames)"
        print(f"  {label[k]:38s} {what}: {scale * v.mean():8.3f}{'%' if o.target is not None else ''}{ph}{n}")
    print("  paired by frame, minus the base (95% CI over frames; frames better / worse):")
    for k in order[1:]:
        rows = [r for r in res if k in r]
        if not rows:
            continue
        d = np.array([r[k] - r["base"] for r in rows])
        lo, hi = bootstrap(d, seed=o.seed)
        print(f"    {label[k]:38s} {scale * d.mean():+8.3f}{unit} [{scale * lo:+.3f}, {scale * hi:+.3f}]"
              f"  ({int(np.sum(d > 1e-9))} / {int(np.sum(d < -1e-9))})")
    rows = [r for r in res if "bound" in r and "optimum" in r]
    if rows:
        gap = np.array([r["bound"] - r["optimum"] for r in rows])
        print(f"  clairvoyant problem certified: bound - optimum found {scale * gap.mean():.3f}{unit} "
              f"on average (max {scale * gap.max():.3f}); our rule reaches "
              f"{scale * np.mean([r['base'] for r in rows]):.3f} of at most "
              f"{scale * np.mean([r['bound'] for r in rows]):.3f}")
    rows = [r for r in res if "counts_loss" in r]
    if rows:
        cl = np.array([r["counts_loss"] for r in rows])
        print(f"  slot counts at the mean SNRs: the greedy rule ({o.base}'s quotas) equals the exact "
              f"dynamic program in {100 * np.mean(cl < 1e-9):.1f}% of frames; predicted loss "
              f"{cl.mean():.4f} dB on average (max {cl.max():.4f})")


if __name__ == "__main__":
    main()
