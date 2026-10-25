"""Index policies as the cell grows: the restless-bandit theory's regime (docs/PROBLEM.md,
section 3; alloc/whittle.py). CPU.

    python tools/rmab_scale.py --utility results/utility_snrc4_nested.json \\
        --users 8 16 32 64 128 --users-per-unit 8 --frames 100 --workers 16

The theory of index policies for restless bandits is asymptotic in the number of arms: with
N arms and a fixed share of them served per period, Lagrangian / LP-priority index
policies lose O(sqrt(N)) in total against the Lagrangian bound, O(1/sqrt(N)) per arm,
without needing indexability (finite horizon: Hu and Frazier 2017; Brown and Smith 2020;
Zhang and Frazier 2021, O(1) under non-degeneracy; Gast, Gaujal and Yan 2023). The main
frame of this project (K = 8 users, one phase per slot) is far from that regime, which is
why its bound is 0.26 dB above our rule. Here a slot carries B resource units (e.g. B
subband groups of a TTI), each carrying one phase of one user, a user at most one unit per
slot; K users, T slots, K / B = --users-per-unit fixed, so a user gets B T / K phases on
average at every K. Per K: each policy's mean PSNR per user (or share at --target), the
Lagrangian bound per user, the gap (bound - policy) per user with a 95% CI over frames,
and the gap times sqrt(K) (flat if the gap falls like 1 / sqrt(K)).

Policies (every free unit of a slot is used: units no rule claims go to PF over the users
with phases left, as in tools/schedule_sim.py):
    pf          proportional fair: rate over served-rate average, top B
    adv         our rule (schedule_sim slope_adv): equal-slope quotas for the B T (1 - p)
                units at the mean SNRs, the advantage index among the users below their
                quota, top B (mean objective only)
    whittle     the Whittle index of each user's program, top B
    lagrange    the Lagrangian (LP-priority) index: each user's gain from this slot in its own
                program priced at the bound's optimal prices lambda*_t (Hao, Xu and Tong's
                Lagrangian priority policy; Hu and Frazier; Brown and Smith), ties (users who
                would take a later unit as well) broken by the Whittle index, top B (their
                theorem breaks ties at random by the relaxation's own policy instead)
    id_guided   the CMU blog's ID policy (Hong et al., Math. Oper. Res. 2025): a user WANTS the
                slot if taking it is optimal in its program at lambda*_t (its ideal action);
                when more users want it than there are units, the ones first in a fixed order
                drawn once per frame (the arms shuffled at time 0, as for heterogeneous arms):
                persistency, no ranking by value
    rand_guided the same ideal actions, conflicts broken in a fresh random order every slot
                (the blog's failing baseline: no persistency)
    pi          a PI-type ratio index (Ayesta, Erausquin and Jacko's potential improvement):
                g_k(s) / E[(g_k(S') - g_k(s))^+], g_k(s) the predicted PSNR gain of user k's
                next phase at this slot's SNR s, S' a slot from k's own fading (32
                equiprobable quantiles at its mean SNR; with --cqi-corr and an aware
                transmitter, their aware SNRs); infinite at the top of the range, ties by the
                largest gain; top B (mean objective only)

The relaxation's tie rule and price solver (docs/PAPER.md owed item 9: at 30 dB the
Lagrangian index stays ~1.5 points per user below the bound at every K):
    lagrange_relax  the Lagrangian index in Zhang and Frazier's fluid-priority order: users
                whose gain at lambda*_t is above the programs' precision first (by gain), then
                the indifferent ones (|gain| within it: they would take a later unit as well)
                in a random order, which activates every indifferent state with the same
                probability, as the relaxation's own mix does (alloc/whittle.kink_demand),
                then the rest by gain. lagrange breaks equal rounded gains by the Whittle
                index instead (every result before 2026-10-22)
    lagrange_lpu  lagrange_relax with LP-update: every --resolve slots lambda* is re-solved
                from the users' current states over the slots left (Gast, Gaujal and Yan;
                Brown and Smith); the slot's own fading counts as unknown in the re-solve.
                About 4 times lagrange's CPU at --resolve 6; not in the default --policies
    --tie-band b    a gain within b times the smallest grid price of 0 is indifferent
                    (default 0.5: the programs' interpolation error)
    --price-scale supply   the subgradient step divided by the supply (default: price, the
                    earlier step, which grows with B)
    --kink-grad     the subgradient taken at the relaxation's mix of the indifferent states
    --ny, --ng      the programs' Y grid and fading bins (default 128, 16)
Per K the output also gives, at lambda* (the bound's programs), the demand that is sure (gain
above the precision) and the indifferent mass, as shares of the supply: the tie rule matters
when the sure demand falls short of the supply and the indifferent mass fills it.

--belief: the policies decide on the transmitter's calibrated curves, the users are scored and
the bound is computed on --utility, so the gap to the bound includes the cost of the curves'
errors (none for the mean, first order at a target, where a user is stopped at its predicted
threshold); without --belief the policies know the true curves. --margin m: the policies'
programs aim at target + m dB, a hedge against those errors (it makes things worse: COMMANDS 23).
whittle_soft, lagrange_soft (with --target): the same indices on calibration-aware programs,
whose terminal reward is the probability that the true PSNR reaches the target when the curves
are off by N(bias, sigma^2) dB (alloc/whittle.soft_spread): --soft-sigma auto (default) measures
sigma and bias per prefix length, pooled over the images, from --utility against --belief (what
a transmitter learns by cross-validating its calibration), or takes a number (dB, no bias). They
are priced on their own relaxation; the bound and the score keep the hard target.

--delay (with --target): also when each user's decoded image reaches the target, per policy
the share of users there by slots T/4, T/2, 3T/4 and T (a CDF over all users) and the mean slot
with misses counted at T + 1 (docs/PROBLEM.md section 3, item 7).

Channel: i.i.d. Rayleigh per user and slot (the programs' model); each unit lost to other
traffic with probability --preempt. --cqi-corr r: stale reports (alloc/csi.py); the
transmitter knows r (adv: expected distortion; whittle, lagrange and the bound: programs
that decide on the report) unless --cqi-naive (all decide on the reports as if exact; the
bound still knows r). Images drawn from the utility's pool with replacement.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import load_utility  # noqa: E402
from tools.schedule_sim import PI_Q, aware_snr, bootstrap, rate, targets, tie_keys  # noqa: E402

POLICIES = ("pf", "adv", "pi", "whittle", "lagrange", "lagrange_relax", "lagrange_lpu",
            "id_guided", "rand_guided")
DEFAULT = tuple(p for p in POLICIES if p != "lagrange_lpu")
POLICIES = POLICIES + ("whittle_soft", "lagrange_soft")  # calibration-aware, on request
MEAN_ONLY = ("adv", "pi")
TARGET_ONLY = ("whittle_soft", "lagrange_soft")
PRICED = ("lagrange", "lagrange_relax", "lagrange_lpu", "id_guided", "rand_guided")  # at lambda*
_G = {}


def draw(util, K, B, T, frames, mean_snr, clip, preempt, cqi_corr, seed):
    """The frames for K users: (images, mean SNRs, true SNRs (K, T), units per slot (T,),
    reports (K, T) or None)."""
    rng = np.random.default_rng([seed, K])
    arng = np.random.default_rng([seed, K, 4])
    out = []
    for _ in range(frames):
        imgs = rng.integers(len(util.names), size=K)
        means = rng.uniform(mean_snr[0], mean_snr[1], K)
        fade = 10.0 * np.log10(rng.exponential(1.0, (K, T)))
        snr = np.clip(means[:, None] + fade, clip[0], clip[1])
        units = ((rng.random((T, B)) >= preempt).sum(1) if preempt > 0
                 else np.full(T, B, dtype=int))
        seen = None
        if cqi_corr < 1.0:
            from alloc.csi import outdated
            seen = means[:, None] + outdated(arng, fade, cqi_corr)
        out.append((imgs, means, snr, units, seen))
    return out


def top(score, claim, n, tie=None):
    """Indices of the n largest scores among claim; equal scores ordered by `tie` (larger
    first), then by index."""
    idx = np.flatnonzero(claim)
    if len(idx) == 0 or n <= 0:
        return idx[:0]
    keys = (-score[idx],) if tie is None else (-tie[idx], -score[idx])
    return idx[np.lexsort(keys)[:n]]


def run(o, util, belief, imgs, means, snr, units, seen, policy, B, arms=None, W_star=None,
        prices=None, order=None, rng=None):
    """One frame under one policy: the delivered (true) SNR lists of the users and, with a
    target, the slot (1-based) each user's decoded image first reaches it (inf: never).
    `order`: the frame's fixed user order (id_guided); `rng`: the random orders of
    rand_guided. A *_soft policy is the same index on the calibration-aware programs `arms`."""
    if policy.endswith("_soft"):
        policy = policy[:-len("_soft")]
    K, T = snr.shape
    P = util.P
    goal = o.target
    aware = seen is not None and not o.cqi_naive
    if seen is None:
        dec, typ = snr, np.asarray(means, dtype=np.float64)
    elif aware:
        dec = aware_snr(util, belief, imgs, means, seen, o.cqi_corr, o.snr_clip)
        typ = aware_snr(util, belief, imgs, means, np.asarray(means)[:, None], o.cqi_corr,
                        o.snr_clip)[:, 0]
    else:
        dec, typ = np.clip(seen, *o.snr_clip), np.asarray(means, dtype=np.float64)
    hist = [[] for _ in range(K)]
    out = [[] for _ in range(K)]
    reach = np.full(K, np.inf)
    served = np.zeros(K)
    quota = None
    if policy == "adv":                        # quotas for the units expected, not the realised
        quota = targets(belief, imgs, typ, int(round(T * B * (1.0 - o.preempt))), "slope")
    pis = pi_slots(o, util, belief, imgs, means) if policy == "pi" else None
    tol = 0.5 * float(arms[0].lams[1]) if arms is not None else 0.0    # the programs' precision
    band = o.tie_band * float(arms[0].lams[1]) if arms is not None else 0.0
    for t in range(T):
        Bt = int(units[t])
        if Bt == 0:
            continue
        has = np.array([len(h) < P for h in hist])
        if not has.any():
            break
        if policy == "lagrange_lpu" and t > 0 and t % o.resolve == 0:   # LP-update
            from alloc.whittle import lagrangian_bound
            _, prices = lagrangian_bound(
                arms, T, o.preempt, per_slot=o.price_steps, units=B, scale=o.price_scale,
                kink=band if o.kink_grad else None,
                start=(t, [arms[k].state(hist[k]) for k in range(K)]))
            W_star = [a.priced(prices) for a in arms]
        g = dec[:, t]
        r = rate(g)
        pf = r / (served + 1e-3 * r.mean())
        claim = has.copy()
        score = np.full(K, -np.inf)
        tie = None
        if policy == "pf":
            score = pf
        elif policy == "pi":                              # potential improvement
            score, tie = pi_index(belief, imgs, hist, g, pis, claim)
        elif policy == "adv":
            claim &= np.array([len(h) < q for h, q in zip(hist, quota)])
            tie = pf                                      # exact ties (useless reports): PF
            for k in np.flatnonzero(claim):
                score[k] = (belief.psnr(imgs[k], hist[k] + [float(g[k])])
                            - belief.psnr(imgs[k], hist[k] + [float(typ[k])]))
        else:
            wi = np.full(K, -np.inf)
            for k in np.flatnonzero(claim):
                n, Y = arms[k].state(hist[k])
                x = float(seen[k, t]) if arms[k].kern is not None else float(g[k])
                wi[k] = arms[k].index(t, n, Y, x)
                if policy in PRICED:
                    score[k] = arms[k].gain_at(W_star[k], prices, t, n, Y, x)
            if policy == "whittle":
                score = wi
            elif policy in ("lagrange_relax", "lagrange_lpu"):   # sure > indifferent > the
                fin = np.isfinite(score)                  # rest; the indifferent at random
                score = np.where(fin & (np.abs(score) <= band), 0.0,
                                 np.where(fin, score, -np.inf))
                tie = tie_keys(snr[:, t], t)
            elif policy == "lagrange":     # the Lagrangian gain, ties by the Whittle index
                score, tie = np.where(np.isfinite(score), np.round(score / tol), -np.inf), wi
            else:                          # guided: the users whose ideal action is to take it,
                claim &= np.isfinite(score) & (score >= -tol)     # in an order, not by value
                perm = order if policy == "id_guided" else rng.permutation(K)
                rank = np.empty(K)
                rank[perm] = np.arange(K)
                score = -rank
            if goal is not None:           # no gain for anyone: leftover units (PF below)
                claim &= wi > 1e-12
        chosen = top(score, claim, Bt, tie)
        if len(chosen) < Bt:                                  # leftover units: PF
            rest = has.copy()
            rest[chosen] = False
            chosen = np.concatenate([chosen, top(pf, rest, Bt - len(chosen))])
        served *= (1.0 - 1.0 / T)
        for k in chosen:
            hist[k].append(float(g[k]))
            out[k].append(float(snr[k, t]))
            served[k] += r[k] / T
            if goal is not None and not np.isfinite(reach[k]) and util.psnr(int(imgs[k]), out[k]) >= goal:
                reach[k] = t + 1
    return out, reach


def pi_slots(o, util, belief, imgs, means):
    """The PI index's typical slots: per user the SNRs (K, 32) its decisions would see on 32
    equiprobable fades at its mean SNR (with stale reports and an aware transmitter, the
    aware SNRs of reports at those fades)."""
    means = np.asarray(means, dtype=np.float64)
    rep = means[:, None] + PI_Q[None, :]
    if o.cqi_corr < 1.0 and not o.cqi_naive:
        return aware_snr(util, belief, imgs, means, rep, o.cqi_corr, o.snr_clip)
    return np.clip(rep, *o.snr_clip)


def pi_index(belief, imgs, hist, g, pis, claim):
    """(score, tie) of the PI-type ratio rule: the gain of this slot over the expected
    excess of a typical slot's gain above it (inf where no typical slot beats it), ties by
    the gain itself."""
    K = len(hist)
    score, gain = np.full(K, -np.inf), np.full(K, -np.inf)
    for k in np.flatnonzero(claim):
        now = belief.psnr(int(imgs[k]), hist[k])
        gain[k] = belief.psnr(int(imgs[k]), hist[k] + [float(g[k])]) - now
        alt = np.array([belief.psnr(int(imgs[k]), hist[k] + [float(x)]) for x in pis[k]]) - now
        ex = float(np.maximum(alt - gain[k], 0.0).mean())
        score[k] = gain[k] / ex if ex > 1e-12 else np.inf
    return score, gain


def _score(util, imgs, hist, target):
    q = np.array([util.psnr(int(i), h) for i, h in zip(imgs, hist)])
    return float(q.mean()) if target is None else float(np.mean(q >= target))


def _work(args):
    f, (imgs, means, snr, units, seen) = args
    o, util, belief, K, B = _G["o"], _G["util"], _G["belief"], _G["K"], _G["B"]
    from alloc.whittle import Arm, lagrangian_bound
    T = o.slots
    p_user = o.preempt ** B                    # a user's slot is lost only if every unit is
    r = o.cqi_corr if seen is not None else 1.0
    clip = tuple(o.snr_clip)
    res = {}
    t0 = time.time()
    grid = dict(ny=o.ny, ng=o.ng)
    arms = [Arm(util, int(i), float(m), T, target=o.target, p_lost=p_user, clip=clip,
                cqi_corr=r, **grid) for i, m in zip(imgs, means)]
    band = o.tie_band * float(arms[0].lams[1])
    opt = dict(per_slot=o.price_steps, units=B, scale=o.price_scale,
               kink=band if o.kink_grad else None)
    info = {}
    ub, prices = lagrangian_bound(arms, T, o.preempt, info=info, **opt)
    res["bound"] = ub / K
    res["sure_share"], res["mid_share"] = info["sure_share"], info["mid_share"]
    pol_arms = arms
    if belief is not util or (seen is not None and o.cqi_naive) or o.margin:
        aim = None if o.target is None else o.target + o.margin   # the policies aim higher
        pol_arms = [Arm(belief, int(i), float(m), T, target=aim, p_lost=p_user, clip=clip,
                        cqi_corr=1.0 if o.cqi_naive else r, **grid) for i, m in zip(imgs, means)]
    W_star = None
    if any(p in o.policies for p in PRICED):
        if pol_arms is not arms:
            _, prices = lagrangian_bound(pol_arms, T, o.preempt, **opt)
        W_star = [a.priced(prices) for a in pol_arms]
    soft_arms = W_soft = sprices = None
    if o.target is not None and any(p in o.policies for p in TARGET_ONLY):
        soft_arms = [Arm(belief, int(i), float(m), T, target=o.target + o.margin, p_lost=p_user,
                         clip=clip, cqi_corr=1.0 if o.cqi_naive else r, soft=_G["soft"], **grid)
                     for i, m in zip(imgs, means)]
        if "lagrange_soft" in o.policies:
            _, sprices = lagrangian_bound(soft_arms, T, o.preempt, **opt)
            W_soft = [a.priced(sprices) for a in soft_arms]
    for p in o.policies:
        if (p in MEAN_ONLY and o.target is not None) or (p in TARGET_ONLY and o.target is None):
            continue
        soft = p in TARGET_ONLY
        h, reach = run(o, util, belief, imgs, means, snr, units, seen, p, B,
                       soft_arms if soft else pol_arms, W_soft if soft else W_star,
                       sprices if soft else prices,
                       order=np.random.default_rng([o.seed, K, f, 7]).permutation(K),
                       rng=np.random.default_rng([o.seed, K, f, 8]))
        res[p] = _score(util, imgs, h, o.target)
        res[p + "_reach"] = reach.tolist()
        res[p + "_phases"] = float(np.mean([len(x) for x in h]))
    if f < o.index_frames:
        res["nonindexable"] = float(np.mean([a.indexability() for a in arms[:o.index_arms]]))
    res["seconds"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--utility", required=True, help="a --nested utility (tools/utility_fit.py)")
    ap.add_argument("--belief", default=None, help="the transmitter's curves (default: the truth)")
    ap.add_argument("--users", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    ap.add_argument("--users-per-unit", type=int, default=8,
                    help="K / B: users per resource unit of a slot (8: the main frame at K = 8)")
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--frames", type=int, default=100, help="frames per K")
    ap.add_argument("--mean-snr", nargs=2, type=float, default=[0.0, 15.0])
    ap.add_argument("--snr-clip", nargs=2, type=float, default=[-2.0, 22.0])
    ap.add_argument("--preempt", type=float, default=0.0,
                    help="each unit is taken by other traffic with this probability")
    ap.add_argument("--cqi-corr", type=float, default=1.0,
                    help="stale reports: correlation of each report with the slot (alloc/csi.py)")
    ap.add_argument("--cqi-naive", action="store_true",
                    help="--cqi-corr < 1: the policies take the reports as exact")
    ap.add_argument("--target", type=float, default=None,
                    help="score the share of users at this PSNR (adv is not run)")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="with --target: the policies' programs aim at target + margin dB (the "
                         "bound and the score stay at the target)")
    ap.add_argument("--soft-sigma", default="auto",
                    help="whittle_soft / lagrange_soft: the curves' error, 'auto' (measured per "
                         "prefix length from --utility against --belief) or a number in dB")
    ap.add_argument("--policies", nargs="+", default=list(DEFAULT), choices=POLICIES)
    ap.add_argument("--delay", action="store_true",
                    help="with --target: when each user reaches it (a CDF by slot, mean slot)")
    ap.add_argument("--price-steps", type=int, default=10,
                    help="subgradient steps on a price per slot for the bound (0: one price)")
    ap.add_argument("--tie-band", type=float, default=0.5,
                    help="a gain within this many smallest grid prices of 0 is indifferent")
    ap.add_argument("--price-scale", default="price", choices=["price", "supply"],
                    help="the subgradient step: the earlier one, or divided by the supply")
    ap.add_argument("--kink-grad", action="store_true",
                    help="the subgradient at the relaxation's mix of the indifferent states")
    ap.add_argument("--resolve", type=int, default=6,
                    help="lagrange_lpu: re-solve lambda* every N slots")
    ap.add_argument("--ny", type=int, default=128, help="the programs' Y grid points")
    ap.add_argument("--ng", type=int, default=16, help="the programs' fading bins")
    ap.add_argument("--index-frames", type=int, default=1,
                    help="frames whose programs are checked for indexability")
    ap.add_argument("--index-arms", type=int, default=8, help="programs checked per such frame")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON of the per-K results")
    o = ap.parse_args()
    if any(K % o.users_per_unit for K in o.users):
        ap.error("every --users value must be a multiple of --users-per-unit")
    if not 0.0 <= o.cqi_corr <= 1.0:
        ap.error("--cqi-corr is a correlation in [0, 1]")
    if not 0.0 <= o.preempt < 1.0:
        ap.error("--preempt is in [0, 1)")
    if o.delay and o.target is None:
        ap.error("--delay needs --target")
    if o.margin and o.target is None:
        ap.error("--margin needs --target")
    if o.soft_sigma != "auto":
        try:
            o.soft_sigma = float(o.soft_sigma)
        except ValueError:
            ap.error("--soft-sigma is 'auto' or a number (dB)")
        if o.soft_sigma < 0:
            ap.error("--soft-sigma >= 0")
    if o.resolve < 1 or o.ny < 8 or o.ng < 2 or o.tie_band < 0:
        ap.error("--resolve >= 1, --ny >= 8, --ng >= 2, --tie-band >= 0")
    util = load_utility(o.utility)
    if getattr(util, "kind", "") == "digital":
        ap.error("the programs need a token-prefix utility")
    belief = load_utility(o.belief) if o.belief else util
    from alloc.whittle import master_profile
    dev = max(master_profile(util, i)[1] for i in range(len(util.names)))
    print(f"{o.slots} slots, {o.users_per_unit} users per unit, mean SNR U{o.mean_snr} dB, "
          f"Rayleigh{f', {100 * o.preempt:g}% of units preempted' if o.preempt else ''}"
          f"{f', stale reports (correlation {o.cqi_corr:g}, transmitter ' + ('naive' if o.cqi_naive else 'aware') + ')' if o.cqi_corr < 1 else ''}"
          f"; {'mean PSNR' if o.target is None else f'share at {o.target:g} dB'}"
          f"{f' (policies aim at {o.target + o.margin:g})' if o.margin else ''}"
          f"; transmitter curves {o.belief or 'the true ones'}; utility "
          f"{o.utility}{f' (not nested: largest rho deviation {dev:.3f}; the programs use its nested approximation)' if dev > 1e-6 else ''}")
    print(f"programs: Y grid {o.ny}, {o.ng} fading bins; prices: {o.price_steps} subgradient steps, "
          f"step scaled by {o.price_scale}{', at the kink' if o.kink_grad else ''}"
          f"{f'; lagrange_lpu re-solves every {o.resolve} slots' if 'lagrange_lpu' in o.policies else ''}"
          f"; indifference band {o.tie_band:g} x the smallest grid price")
    soft = None
    if o.target is not None and any(p in o.policies for p in TARGET_ONLY):
        from alloc.whittle import soft_spread
        soft = soft_spread(util, belief, o.soft_sigma, mean_snr=tuple(o.mean_snr),
                           clip=tuple(o.snr_clip))
        print("calibration-aware programs (*_soft): the curves' error per prefix length, sigma "
              + " ".join(f"{v:.3f}" for v in soft[0]) + " dB, bias (truth - belief) "
              + " ".join(f"{v:+.3f}" for v in soft[1]) + f" dB ({'measured' if o.soft_sigma == 'auto' else 'given'})")
    elif any(p in o.policies for p in TARGET_ONLY):
        print("note: whittle_soft / lagrange_soft need --target; skipped")
    scale, unit = (1.0, "dB") if o.target is None else (100.0, "pts")
    pols = [p for p in o.policies if not (p in MEAN_ONLY and o.target is not None)
            and not (p in TARGET_ONLY and o.target is None)]
    table = {}
    for K in o.users:
        B = K // o.users_per_unit
        frames = draw(util, K, B, o.slots, o.frames, o.mean_snr, o.snr_clip, o.preempt,
                      o.cqi_corr, o.seed)
        _G.update(o=o, util=util, belief=belief, K=K, B=B, soft=soft)
        t0 = time.time()
        if o.workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(o.workers) as pool:
                res = pool.map(_work, list(enumerate(frames)), chunksize=1)
        else:
            res = [_work(x) for x in enumerate(frames)]
        bound = np.array([r["bound"] for r in res])
        print(f"\nK {K} users, B {B} units per slot, {o.frames} frames ({time.time() - t0:.0f} s; "
              f"{np.mean([r['seconds'] for r in res]):.1f} s per frame and worker)")
        print(f"  {'Lagrangian bound (per user)':30s} {scale * bound.mean():8.3f}")
        sure = np.mean([r["sure_share"] for r in res])
        mid = np.mean([r["mid_share"] for r in res])
        print(f"  at lambda*: sure demand {100 * sure:.1f}% of the supply, indifferent mass "
              f"{100 * mid:.1f}% (within {o.tie_band:g} x the smallest grid price of a zero gain)")
        print(f"  {'policy':12s} {'value':>8s} {'phases':>6s}   {'gap to the bound [95% CI]':>32s}"
              f"   {'gap x sqrt(K)':>13s}")
        row = {"B": B, "bound": float(bound.mean()), "sure_share": float(sure),
               "mid_share": float(mid)}
        for p in pols:
            v = np.array([r[p] for r in res])
            d = bound - v
            lo, hi = bootstrap(d, seed=o.seed)
            print(f"  {p:12s} {scale * v.mean():8.3f} {np.mean([r[p + '_phases'] for r in res]):6.2f}"
                  f"   {scale * d.mean():+10.3f} {unit} [{scale * lo:+.3f}, {scale * hi:+.3f}]"
                  f"   {scale * d.mean() * np.sqrt(K):13.3f}")
            row[p] = float(v.mean())
            row[p + "_gap"] = float(d.mean())
            row[p + "_gap_ci"] = [float(lo), float(hi)]
        for a, b in (("lagrange", "whittle"), ("whittle", "adv"), ("lagrange", "adv"), ("adv", "pf"),
                     ("whittle", "pf"), ("id_guided", "rand_guided"), ("lagrange", "id_guided"),
                     ("adv", "pi"), ("pi", "pf"), ("lagrange", "pi"),
                     ("lagrange_relax", "lagrange"), ("lagrange_lpu", "lagrange_relax"),
                     ("lagrange_lpu", "lagrange"), ("lagrange_soft", "lagrange"),
                     ("whittle_soft", "whittle"), ("lagrange_soft", "whittle_soft")):
            if a in pols and b in pols:
                d = np.array([r[a] - r[b] for r in res])
                lo, hi = bootstrap(d, seed=o.seed)
                print(f"  {a} - {b}: {scale * d.mean():+.3f} {unit} [{scale * lo:+.3f}, {scale * hi:+.3f}]")
        if o.delay:
            report_delay(o, res, pols, row)
        ni = [r["nonindexable"] for r in res if "nonindexable" in r]
        if ni:
            print(f"  indexability: the decision flips back as the price rises at "
                  f"{100 * np.mean(ni):.2f}% of grid states ({len(ni)} frame(s), "
                  f"{min(o.index_arms, K)} programs each; 0 = indexable on the grid)")
            row["nonindexable"] = float(np.mean(ni))
        table[K] = row
    print("\nsummary: gap to the Lagrangian bound per user (" + unit + ")")
    print(f"  {'K':>5s} {'B':>3s} " + " ".join(f"{p:>10s}" for p in pols))
    for K, row in table.items():
        print(f"  {K:5d} {row['B']:3d} " + " ".join(f"{scale * row[p + '_gap']:10.3f}" for p in pols))
    if o.out:
        os.makedirs(os.path.dirname(os.path.abspath(o.out)), exist_ok=True)
        with open(o.out, "w") as f:
            json.dump({"args": vars(o), "per_K": {str(k): v for k, v in table.items()}}, f, indent=1)
        print(f"wrote {o.out}")


def report_delay(o, res, pols, row):
    """--delay: per policy the share of users at the target by T/4, T/2, 3T/4 and T, and the
    mean slot each reaches it with misses at T + 1, with paired CIs over frames."""
    T = o.slots
    cut = sorted({max(1, int(round(T * f))) for f in (0.25, 0.5, 0.75, 1.0)})
    print(f"  delay: users at {o.target:g} dB by slot {', '.join(map(str, cut))} (% of all "
          f"users), mean slot reached (misses at T + 1 = {T + 1})")
    per = {p: np.array([r[p + "_reach"] for r in res], dtype=np.float64) for p in pols}
    for p in pols:
        tt = per[p]
        by = [float(np.mean(tt <= c)) for c in cut]
        slot = np.minimum(tt, T + 1).mean(1)              # per frame
        row[p + "_by"] = by
        row[p + "_slot"] = float(slot.mean())
        print(f"    {p:12s} " + " ".join(f"{100 * b:5.1f}%" for b in by) + f"   {slot.mean():6.2f}")
    for a, b in (("lagrange", "whittle"), ("whittle", "pf"), ("lagrange", "pf"),
                 ("id_guided", "rand_guided"), ("lagrange", "id_guided"),
                 ("lagrange_relax", "lagrange"), ("lagrange_lpu", "lagrange"),
                 ("lagrange_soft", "lagrange"), ("whittle_soft", "whittle")):
        if a in pols and b in pols:
            d = np.minimum(per[a], T + 1).mean(1) - np.minimum(per[b], T + 1).mean(1)
            lo, hi = bootstrap(d, seed=o.seed)
            both = np.isfinite(per[a]) & np.isfinite(per[b])     # the same users: no selection
            same = [float((per[a][f][both[f]] - per[b][f][both[f]]).mean())
                    for f in range(len(res)) if both[f].any()]
            line = f"    mean slot {a} - {b}: {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]"
            if len(same) > 1:
                slo, shi = bootstrap(same, seed=o.seed)
                line += (f"; on the users both bring there {np.mean(same):+.2f} "
                         f"[{slo:+.2f}, {shi:+.2f}] ({int(both.sum())} users)")
                row[f"same_{a}_{b}"] = float(np.mean(same))
            print(line)


if __name__ == "__main__":
    main()
