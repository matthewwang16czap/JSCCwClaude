"""Opportunistic scheduling of encode-once token prefixes over a block-fading downlink
(docs/PROBLEM.md), simulated on the utility model of alloc/utility.py. CPU, about a minute.

    python tools/schedule_sim.py --utility results/utility_snrc4.json --users 8 --slots 24 \\
        --frames 2000 --mean-snr 0 15 --out results/sched_main.json
    python tools/schedule_sim.py --utility results/utility_snrc4.json --target 30

A frame: K users, each owed one image drawn from the utility file's pool, and T slots. A
slot carries one PHASE of one user's prefix (CBR 1/48 per phase at 256 px tiles), so a user
holds at most P phases (1/8). User k's SNR in slot t is its mean SNR plus a Rayleigh
block-fading draw (10 log10 |h|^2, |h|^2 ~ Exp(1)), clipped to --snr-clip. The base station
knows every user's SNR in the current slot (CQI), never a future one. Phases go out in
prefix order; at the deadline every user decodes what it holds.

Two objectives. Without --target: the mean PSNR at the deadline. Policies:

    rr           round robin over users with phases left (channel- and content-blind)
    maxsnr       the largest current SNR
    pf           proportional fair: log2(1 + snr) over the user's served-rate average
    fixed_uni    T / K phases per user, fixed (what one model per CBR allows), PF timing
    fixed_eq     PADC-style: before the frame, every user's number of phases from its
                 predicted constant-SNR curve at its MEAN SNR, by PADC's equal-quality rule
                 over the T slots; delivered with PF timing among users below their target
    fixed_slope  the same targets by the equal-slope rule (mean quality)
    adaptive_slope  fixed_slope re-planned every slot: equal-slope targets for the slots left,
                 from what every user already holds and future phases at its mean SNR; PF
                 timing among users below their current target
    slope_adv    fixed_slope's targets, delivered by the ADVANTAGE index: serve the user whose
                 predicted PSNR gain from this slot exceeds its gain from a slot at its mean
                 SNR by the most. Position-aware (which phase goes next, how noise-sensitive it
                 is) and quality-aware (a user near its noise-free quality gains little from a
                 good slot and takes the bad ones), where PF sees only the rate
    greedy       the largest predicted PSNR gain of this slot, given what the user already
                 holds and the slot's SNR (myopic marginal utility; --alpha > 0 weighs it by
                 the user's current PSNR^-alpha for alpha-fairness)
    greedy_rel   that gain times its ratio to the gain at the user's mean SNR (a pure ratio
                 starves users that hold nothing yet; --alpha does not apply)

With --target Q (dB): the share of users whose image reaches Q by the deadline (a quality of
service). Every policy aims at Q + --margin and predicts future phases at the mean SNR +
--plan-offset; a slot that no rule of the policy claims goes to PF over all users with
phases left, so every policy can use the whole frame. Policies:

    rr, pf       as above, blind to the target
    pf_len       content-blind link adaptation: every user's length from the POOL-AVERAGE
                 curve at its mean SNR (no per-image knowledge), PF timing
    padc         PADC (TWC 2023) per user: the shortest prefix whose predicted constant-SNR
                 quality at the mean SNR reaches the target, fixed before the frame; users
                 admitted shortest first while the lengths fit in T (the rest get leftovers);
                 PF timing
    padc_adv     padc's lengths and admission, the advantage index for timing
    stop         online stopping: a user is served (PF timing) until the model, from the
                 SNRs its phases actually had, predicts it at the target
    stop_adv     stop with the advantage index for timing
    online_pf    online's admission and stopping with PF timing (the online policy for a
                 digital utility, whose step curves give the advantage index nothing)
    plan, plan_adv  padc / padc_adv with the spare slots planned before the frame as for
                 fixed codes (fill: left-out users first, at the longest run that fits)
    replan_adv   plan_adv re-planned every slot from what each user holds, stopping each
                 user at the target: the planner of fixed codes plus a prefix code's
                 online length
    online       stop_adv plus admission every slot: the users with the smallest remaining
                 need (phases at the mean SNR) that fit in the slots left are served first;
                 spare slots open to all. Content-aware sizing, channel-aware stopping and
                 timing: what only an encode-once prefix code with a mixed-SNR utility allows
    whittle      (both objectives) the Whittle index of the frame as a restless bandit
                 (alloc/whittle.py): each user's own dynamic program over (slot, phases
                 held, accumulated weighted noise) under a slot price; serve the user whose
                 indifference price for this slot is highest (with --target, a slot nobody
                 gains from goes to PF). Exact for a --nested utility, an approximation
                 otherwise
    lagrange     (both objectives) the Lagrangian (LP-priority) index: the same programs
                 priced at the per-slot prices lambda*_t that minimise the frame's Lagrangian
                 bound (alloc/whittle.lagrangian_bound); serve the user whose program gains
                 most from taking this slot at that price (Hao, Xu and Tong's Lagrangian
                 priority policy, IEEE TAC; Hu and Frazier; Brown and Smith; needs no
                 indexability), ties by the Whittle index; with --target only users
                 with a positive Whittle index claim the slot, the rest goes to PF (as
                 tools/rmab_scale.py, which it reproduces with one unit per slot)

All policies see the same images, mean SNRs and fades (common random numbers). Reported per
policy: mean PSNR at the deadline, the 5th percentile over users, the worst user of a frame on
average (mean objective) or the satisfied share (target), the share of served slots below 0
dB, phases per user, the "lift" (served slot SNR minus the user's mean SNR, dB), and the
difference to pf (and to padc with a target) with a 95% bootstrap interval over frames.
Ablations: --same-image (no content diversity: only the channel part), --fading none (no
fading: only the content and mean-SNR part), --fade-corr r (slot gains correlated in time,
Gauss-Markov; --plan conditional predicts future phases from the current slot instead of
the mean), --length-bonus with --fixed-codes (one model per CBR: every length better by
the nesting penalty, and a fixed-length policy cannot send more than the code; the
baseline of DeepJSCC-MIMO / SwinJSCC-style per-rate models: the slots that shortest-first
admission leaves are planned before the frame, left-out users first; a code not fully
received decodes to nothing), --preempt p (each slot taken by other traffic with
probability p, unknown in advance: the case for codes whose length is not fixed when
encoded), --pred-noise (the transmitter's curves are off: log-normal errors on each
image's noise-free MSE and sensitivity), --belief FILE (the transmitter decides on its own
calibrated curves, tools/utility_fit.py --calib; users are scored on --utility).
--delay adds the timing of the deliveries (tools/latency.py and tools/stream_sim.py go further):
the slot of each user's first decodable phase, its PSNR averaged over the frame's slots (as
decoded after each slot), and with --target the share of users at the target by every slot (a
CDF over all users), the slot each user reaches it, and that slot compared on the users two
policies both bring to the target.
--utility may also be a digital baseline (tools/digital_rd.py, alloc/digital.py: an image
codec's R-D points over the same slots with ideal link adaptation; --link, --gap-db,
--digital-mode, --digital-max-phases override the file). --cqi-error s: the reported slot
SNRs are off by N(0, s^2) dB; every policy decides on the reports (needs, stopping, timing),
a token prefix is decoded at the true SNR (the codec needs no SNR at either end), and a
digital slot is sent at the rate of its report minus --cqi-backoff (default: tuned per
policy and objective on other frames, the best outer-loop link adaptation could settle at) and
lost if the true SNR does not support that rate (the transmitter learns it: HARQ ACK). --harq
ir|chase: a lost block is retransmitted in the user's next slot and combined with what was
received (incremental redundancy / Chase combining, alloc/digital.py) up to --harq-max
transmissions, as NR does; the backoff is then tuned with HARQ on (and with --cqi-aware an
offset on the aware rate too). --dump N writes the delivery histories of the first N frames
for tools/mixed_snr.py --design file, which decodes them with the codec itself.

Stale reports (alloc/csi.py): --cqi-corr r gives every slot a report of its channel with
correlation r (h_rep = r h + sqrt(1 - r^2) e; r = J0(2 pi f_D tau) for Doppler f_D and report
age tau; 1 = fresh, 0 = useless), the true channel unchanged (the same frames at any r).
Every policy decides on the reports as if they were exact, unless --cqi-aware: then the
transmitter knows r and decides on the posterior of the slot's channel (a token prefix: the
SNR of its expected distortion; a digital link: the rate of largest expected goodput, no
backoff; whittle: its program decides on the report, alloc/whittle.Arm(cqi_corr=r)). A
digital link without --cqi-aware gets the tuned backoff, as under --cqi-error.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import load_utility  # noqa: E402

MEAN_POLICIES = ("rr", "maxsnr", "pf", "fixed_uni", "fixed_eq", "fixed_slope",
                 "adaptive_slope", "slope_adv", "greedy", "greedy_rel", "whittle", "lagrange")
TARGET_POLICIES = ("rr", "pf", "pf_len", "padc", "padc_adv", "plan", "plan_adv", "stop", "stop_adv",
                   "online_pf", "online", "replan_adv", "whittle", "lagrange")
POLICIES = tuple(dict.fromkeys(MEAN_POLICIES + TARGET_POLICIES))
NEEDS_TARGET = ("pf_len", "padc", "padc_adv", "plan", "plan_adv", "stop", "stop_adv", "online_pf",
                "online", "replan_adv")
PF_TIMING = ("pf", "fixed_uni", "fixed_eq", "fixed_slope", "adaptive_slope", "pf_len", "padc",
             "plan", "stop", "online_pf")
ADV_TIMING = ("slope_adv", "padc_adv", "plan_adv", "stop_adv", "online", "replan_adv")


def rate(snr_db):
    return np.log2(1.0 + 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0))


def targets(util, imgs, means, T, rule):
    """Phases per user decided before the frame from constant-SNR curves at the mean SNRs."""
    K, P = len(imgs), util.P
    if rule == "uni":                                        # equal shares, content-blind
        return np.minimum(T // K + (np.arange(K) < T % K), P)
    q = np.array([[util.psnr(i, [m] * L) for L in range(P + 1)] for i, m in zip(imgs, means)])
    L = np.zeros(K, dtype=int)
    for _ in range(T):
        ok = L < P
        if not ok.any():
            break
        if rule == "eq":                                     # raise the worst user (max-min)
            cur = np.where(ok, q[np.arange(K), L], np.inf)
            k = int(np.argmin(cur))
        else:                                                # largest gain (equal slope)
            gain = np.where(ok, q[np.arange(K), np.minimum(L + 1, P)] - q[np.arange(K), L], -np.inf)
            k = int(np.argmax(gain))
        L[k] += 1
    return L


def replan(belief, imgs, means, hist, slots_left):
    """Equal-slope targets for the slots left, given what every user holds; future
    phases predicted at the user's mean SNR."""
    K, P = len(imgs), belief.P
    L = np.array([len(h) for h in hist])
    q = lambda k, n: belief.psnr(imgs[k], hist[k] + [float(means[k])] * (n - L[k]))  # noqa: E731
    cur = np.array([q(k, L[k]) for k in range(K)])
    nxt = np.array([q(k, L[k] + 1) if L[k] < P else -np.inf for k in range(K)])
    tgt = L.copy()
    for _ in range(slots_left):
        gain = nxt - cur
        k = int(np.argmax(gain))
        if not np.isfinite(gain[k]):
            break
        tgt[k] += 1
        cur[k] = nxt[k]
        nxt[k] = q(k, tgt[k] + 1) if tgt[k] < P else -np.inf
    return tgt


def need(belief, i, hist, snr_plan, goal):
    """Further phases at snr_plan dB until the predicted PSNR reaches goal (0: already
    there); inf if the prefix runs out first."""
    for n in range(belief.P - len(hist) + 1):
        if belief.psnr(i, hist + [float(snr_plan)] * n) >= goal:
            return n
    return np.inf


def plan_snr(means, snr, t, offset=0.0, corr=0.0):
    """The SNR (dB) a policy assumes for a user's future phases at slot t: its mean, or
    with Gauss-Markov fading of field correlation corr (--plan conditional) the mean of
    E[|h|^2 | the current slot] over the slots left: corr^(2 tau) g_t + 1 - corr^(2 tau)."""
    base = np.asarray(means, dtype=np.float64) + offset
    T = snr.shape[1]
    if corr <= 0 or t >= T - 1:
        return base
    g = 10.0 ** ((snr[:, t] - np.asarray(means)) / 10.0)
    w = corr ** (2.0 * np.arange(1, T - t))
    return base + 10.0 * np.log10((w[None, :] * g[:, None] + (1.0 - w)[None, :]).mean(1))


FILL = "admit"
_ARMS = {}


def arms_for(belief, imgs, means, T, goal, offset, p_avail, clip, cqi_corr=1.0):
    """The users' dynamic programs for the whittle policy (alloc/whittle.py), one per user
    of the frame, cached for the frame (rollouts resume the same frame many times)."""
    from alloc.whittle import Arm
    key = (id(belief), tuple(int(i) for i in imgs), tuple(np.round(means, 9)), T, goal, offset,
           p_avail, tuple(clip), cqi_corr)
    if key not in _ARMS:
        if len(_ARMS) > 8:
            _ARMS.clear()
        _ARMS[key] = [Arm(belief, int(i), float(m) + offset, T, target=goal,
                          p_lost=1.0 - p_avail, clip=clip, cqi_corr=cqi_corr)
                      for i, m in zip(imgs, means)]
    return _ARMS[key]


_PRICED = {}
PRICE_STEPS = 10                                              # subgradient steps (rmab_scale's)


def priced_for(arms, T, p_lost):
    """The lagrange policy: the per-slot prices lambda*_t that minimise the frame's
    Lagrangian bound (one unit per slot) and every program priced at them, cached with the
    frame's programs."""
    from alloc.whittle import lagrangian_bound
    key = id(arms)
    hit = _PRICED.get(key)
    if hit is None or hit[0] is not arms:
        if len(_PRICED) > 8:
            _PRICED.clear()
        _, prices = lagrangian_bound(arms, T, p_lost, per_slot=PRICE_STEPS)
        hit = (arms, prices, [a.priced(prices) for a in arms])
        _PRICED[key] = hit
    return hit[1], hit[2]


_AWARE = {}


def aware_snr(util, belief, imgs, means, seen, r, clip):
    """--cqi-aware: the decision SNRs of a transmitter that knows its reports `seen` (dB,
    unclipped) have correlation r with the slots (alloc/csi.py). Cached by content: backoff
    tuning and every policy see the same frames again."""
    from alloc.csi import equivalent_snr, goodput_snr
    means = np.asarray(means, dtype=np.float64)
    rep = np.asarray(seen, dtype=np.float64) - means[:, None]
    digital = getattr(util, "kind", "") == "digital"
    kap = None if digital else (belief.kappa if np.ndim(belief.kappa) == 0
                                else belief.kappa[np.asarray(imgs, dtype=int)])
    what = (util.link, util.gap_db) if digital else np.asarray(kap, dtype=np.float64).tobytes()
    key = (what, rep.shape, rep.tobytes(), means.tobytes(), float(r), tuple(clip))
    out = _AWARE.get(key)
    if out is None:
        out = (goodput_snr(rep, means, util.efficiency, r, clip) if digital
               else equivalent_snr(rep, means, kap, r, clip))
        if len(_AWARE) >= 4000:
            _AWARE.clear()
        _AWARE[key] = out
    return out.copy()



def replan_target(belief, imgs, hist, plan, rest, ok, left, P):
    """replan_adv, every slot: the users whose remaining need (phases at their planned SNR,
    from what they hold) fits in the slots left are admitted shortest first; the slots
    still spare go, like fill's "admit", to left-out users one at a time, each planned for
    the longest run that fits, the one with the best predicted quality at it first.
    Returns the users planned to receive more phases."""
    K = len(hist)
    e = np.zeros(K, dtype=int)
    for k in np.argsort(rest, kind="stable"):
        if ok[k] and rest[k] <= left:
            e[k], left = int(rest[k]), left - int(rest[k])
    while left > 0:
        out = [k for k in range(K) if ok[k] and e[k] == 0 and len(hist[k]) < P]
        if not out:
            break
        k = max(out, key=lambda j: belief.psnr(imgs[j], hist[j] + [float(plan[j])]
                                                * min(P - len(hist[j]), left)))
        e[k] = min(P - len(hist[k]), left)
        left -= e[k]
    return e > 0


def fill(belief, imgs, plan, goal, tgt, spare, P, rule="admit", delta=1.0):
    """Fixed-length codes: spend the slots that shortest-first admission left over before
    the frame, since a fixed code cannot be extended later. "margin": one more phase at a
    time to the admitted user with the smallest predicted margin over the goal (at its
    planned SNR), then the users left out, each at the longest code that fits; "admit":
    the users left out first (the default: the strongest on the stand-in); "delta":
    margins up to delta dB first, then the users left out, then margins again."""
    tgt = tgt.copy()
    K = len(tgt)
    q = lambda k, n: belief.psnr(imgs[k], [float(plan[k])] * n)  # noqa: E731

    def lengthen(cap):
        nonlocal spare
        while spare > 0:
            m = [(q(k, tgt[k]) - goal, k) for k in range(K) if 0 < tgt[k] < P]
            m = [x for x in m if x[0] < cap]
            if not m:
                return
            tgt[min(m)[1]] += 1
            spare -= 1

    def admit():
        nonlocal spare
        while spare > 0:
            out = [k for k in range(K) if tgt[k] == 0]
            if not out:
                return
            n = min(P, spare)
            k = max(out, key=lambda j: q(j, n))
            tgt[k], spare = n, spare - n

    if rule == "admit":
        admit(); lengthen(np.inf)
    elif rule == "delta":
        lengthen(delta); admit(); lengthen(np.inf)
    else:
        lengthen(np.inf); admit()
    return tgt


FAILED = -100.0                                               # a lost digital block (no bits)


def harq_deliver(util, pend, k, d, s, harq=None):
    """A digital slot of user k, decided at SNR d (dB: its rate is efficiency(d)), true SNR s
    (dB): what the user's history records, d if a block is decoded in this slot (its bits),
    FAILED if none is. Without HARQ a block is decoded iff s supports its rate. harq = (mode,
    max_tx): a failed block waits in pend[k] and the user's next slot retransmits it (whatever
    that slot's d), combined with what was received (alloc/digital.py: "ir" adds the
    transmissions' mutual_info, "chase" their SNRs) until it is decoded or has had max_tx
    transmissions; then it is dropped. A block still waiting at the deadline is lost."""
    if harq is not None and pend[k] is not None:
        mode, max_tx = harq
        b = pend[k]
        b["tx"] += 1
        if mode == "ir":
            b["acc"] += float(util.mi(s))
            got = b["acc"]
        else:
            b["acc"] += 10.0 ** (s / 10.0)
            got = float(util.mi(10.0 * np.log10(b["acc"])))
        if got >= b["rate"] - 1e-12:
            pend[k] = None
            return b["d"]
        if b["tx"] >= max_tx:
            pend[k] = None
        return FAILED
    rate_d = float(util.efficiency(d))
    if rate_d <= float(util.efficiency(s)) + 1e-12:
        return d
    if harq is not None and harq[1] > 1:
        pend[k] = {"rate": rate_d, "d": d, "tx": 1,
                   "acc": float(util.mi(s)) if harq[0] == "ir" else 10.0 ** (s / 10.0)}
    return FAILED


def delay_times(util, imgs, hist, log, T, target=None, whole=False):
    """The timing of one frame's deliveries, per user: the slot (1-based) of its first
    decodable phase (T + 1: none), its PSNR averaged over the frame's T slots (as decoded
    after each slot; before any phase, the empty image) and, with a target, the slot at which
    its decoded PSNR first reaches it (inf: never). `log`: run_frame's (slot, user) pairs.
    whole: a code decodes only once complete (one model per CBR, --fixed-codes; run_frame
    returns nothing for an incomplete one)."""
    when = [[] for _ in hist]
    for t, k in log:
        when[k].append(t)
    first, tavg, reach = [], [], []
    for k, (i, h) in enumerate(zip(imgs, hist)):
        i, n = int(i), len(h)
        w = when[k][:n] if n == len(when[k]) else []     # an incomplete fixed code: nothing
        n = len(w)
        q = [util.psnr(i, h[:j]) for j in range(n + 1)]
        if whole and n:
            q = [q[0]] * n + [q[n]]
        qs = np.full(T, q[0])
        for j, t in enumerate(w):
            qs[t:] = q[j + 1]
        first.append((w[-1] if whole else w[0]) + 1 if n else T + 1)
        tavg.append(float(qs.mean()))
        if target is not None:
            hit = [j for j in range(1, n + 1) if q[j] >= target]
            reach.append(float(w[hit[0] - 1] + 1) if hit else np.inf)
    return first, tavg, reach


def run_frame(util, belief, imgs, means, snr, policy, alpha=0.0, target=None, margin=0.0,
              offset=0.0, avg=None, corr=0.0, fixed_codes=False, avail=None, p_avail=1.0,
              t_plan=None, seen=None, backoff=0.0, start=None, stop_after=None, log=None,
              clip=(-2.0, 22.0), aware=None, harq=None):
    """One frame under one policy: per-user delivered SNR lists. With a target (dB) the
    target policies size, stop and admit by it, and a slot that no policy rule claims goes
    to PF over every user with phases left (so every policy can use the whole frame).
    `seen`: the SNRs the transmitter is told (CQI with errors; default: the true `snr`).
    Every decision is made on them; a token prefix is decoded at the true SNR. A digital
    slot is sent at the rate of its reported SNR minus `backoff` dB and is lost when that
    rate exceeds what the true SNR supports (the transmitter learns which: HARQ ACK).
    For tools/headroom.py: `start` = (t0, histories, PF averages) resumes the policy at slot
    t0 from that state (the transmitter's own play-outs: no `seen`); `stop_after` n returns
    after n served slots; `log` (a list) receives (slot, user) for every served slot.
    `aware` = r: the reports are stale with correlation r and the transmitter knows it
    (--cqi-aware): it decides on aware_snr, the whittle policy on its reports. `harq` =
    (mode, max_tx): a digital link retransmits failed blocks (harq_deliver); a user's slot
    is then valued, by the policies that value slots, by the block waiting for it."""
    K, T = snr.shape
    P = util.P
    digital = getattr(util, "kind", "") == "digital"
    rep = snr if seen is None else np.clip(seen, clip[0], clip[1])   # what the transmitter knows
    pmeans = means                                            # the typical slot it plans with
    if aware is not None and seen is not None:                # ... and that its reports are stale:
        rep = aware_snr(util, belief, imgs, means, seen, aware, clip)  # a slot is worth its
        pmeans = aware_snr(util, belief, imgs, means,                 # posterior, a typical one
                           np.asarray(means, dtype=np.float64)[:, None], aware, clip)[:, 0]
    dec = rep - backoff if digital else rep                   # what it assumes a slot brings
    off = offset - (backoff if digital else 0.0)
    hist = [[] for _ in range(K)]                             # as the transmitter knows them
    out = [[] for _ in range(K)]                              # as delivered (what is decoded)
    pend = [None] * K                                         # HARQ: a block awaiting its user
    served = np.zeros(K)                                      # PF: served-rate average
    plan = plan_snr(pmeans, rep, 0, off, corr)               # SNR assumed for future phases
    T_plan = int(round(T * p_avail)) if t_plan is None else int(t_plan)  # slots planned for
    goal = None if target is None else target + margin
    tgt = None                                                # phases per user, fixed upfront
    if policy in ("fixed_uni", "fixed_eq", "fixed_slope", "slope_adv"):
        tgt = targets(belief, imgs, pmeans, T_plan,
                      {"fixed_uni": "uni", "fixed_eq": "eq"}.get(policy, "slope"))
    elif policy == "pf_len":                                  # content-blind link adaptation
        tgt = np.array([need(avg, 0, [], p, goal) for p in plan], dtype=float)
        tgt = np.where(np.isfinite(tgt), tgt, P).astype(int)
    elif policy in ("padc", "padc_adv", "plan", "plan_adv"):  # per image at the mean SNR
        L = np.array([need(belief, i, [], p, goal) for i, p in zip(imgs, plan)], dtype=float)
        tgt = np.zeros(K, dtype=int)
        left = T_plan
        for k in np.argsort(L, kind="stable"):                # shortest first, while it fits
            if np.isfinite(L[k]) and L[k] <= left:
                tgt[k], left = int(L[k]), left - int(L[k])
        if (fixed_codes or policy.startswith("plan")) and left > 0:   # plan the spare slots
            tgt = fill(belief, imgs, plan, goal, tgt, left, P, FILL)  # (a fixed code must)
    arms = (arms_for(belief, imgs, means, T, goal, offset, p_avail, clip,
                     1.0 if aware is None or seen is None else aware)
            if policy in ("whittle", "lagrange") else None)
    prices = Wp = None
    if policy == "lagrange":
        prices, Wp = priced_for(arms, T, 1.0 - p_avail)
    rest = None                                               # online: phases still needed
    if policy in ("stop", "stop_adv", "online_pf", "online", "replan_adv"):
        rest = np.array([need(belief, i, [], p, goal) for i, p in zip(imgs, plan)], dtype=float)
    t0, sent = 0, 0
    if start is not None:                                     # resume from a given state
        t0, hist0, served0 = start
        hist = [list(h) for h in hist0]
        out = [list(h) for h in hist0]
        served = np.array(served0, dtype=np.float64)
        if rest is not None:
            rest = np.array([need(belief, imgs[k], hist[k], plan[k], goal) for k in range(K)],
                            dtype=float)
    last = -1
    for t in range(t0, T):
        if avail is not None and not avail[t]:                # taken by other traffic
            continue
        has = np.array([len(h) < P for h in hist])
        if not has.any():
            break
        if corr > 0 and t > 0:                                # the channel state moved
            plan = plan_snr(pmeans, rep, t, off, corr)
            if rest is not None:
                rest = np.array([need(belief, imgs[k], hist[k], plan[k], goal) for k in range(K)],
                                dtype=float)
        ok = has.copy()
        if tgt is not None:
            ok &= np.array([len(h) < g for h, g in zip(hist, tgt)])
        if rest is not None:
            ok &= rest > 0                                    # not there yet
            if policy.startswith("online"):  # admit the smallest needs that fit in the slots left;
                adm = np.zeros(K, dtype=bool)                 # spare slots open to the others
                left = int(round((T - t) * p_avail))
                for k in np.argsort(rest, kind="stable"):
                    if ok[k] and rest[k] <= left:
                        adm[k], left = True, left - int(rest[k])
                if adm.any() and left < 1:
                    ok = adm
            elif policy == "replan_adv":  # padc + fill re-planned on what each user holds
                planned = replan_target(belief, imgs, hist, plan, rest, ok,
                                        int(round((T - t) * p_avail)), P)
                if planned.any():
                    ok = planned
        g = dec[:, t]
        if policy == "adaptive_slope":
            below = ok & (np.array([len(h) for h in hist])
                          < replan(belief, imgs, pmeans, hist, int(round((T - t) * p_avail))))
            if below.any():
                ok = below
        if arms is not None:                                  # the Whittle index (alloc/whittle.py)
            idx = np.full(K, -np.inf)
            lg = np.full(K, -np.inf)
            for j in np.flatnonzero(ok):
                n, Y = arms[j].state(hist[j])
                x = g[j] if arms[j].kern is None else seen[j, t]     # stale: the report itself
                idx[j] = arms[j].index(t, n, Y, float(x))
                if Wp is not None:                            # the Lagrangian index
                    lg[j] = arms[j].gain_at(Wp[j], prices, t, n, Y, float(x))
            if policy == "lagrange":
                cl = ok & np.isfinite(lg)
                if target is not None:                        # only users the slot helps
                    cl &= idx > 1e-12
                if cl.any():                                  # the largest gain at lambda*_t, in
                    q = np.where(cl, np.round(lg / (0.5 * float(arms[0].lams[1]))), -np.inf)
                    cand = np.flatnonzero(q >= q.max())       # the programs' precision; ties by
                    ok = np.zeros(K, dtype=bool)              # the Whittle index, then the first
                    ok[cand[int(np.argmax(idx[cand]))]] = True
                else:
                    ok = has                                  # nobody claims it: leftover (PF)
            elif np.isfinite(idx.max()) and (target is None or idx.max() > 1e-12):
                ok = idx >= idx.max()                         # served below (ties: first)
            else:
                ok = has                                      # nobody gains: leftover (PF)
        if not ok.any():
            if fixed_codes and tgt is not None:
                break                                         # a fixed-length code is complete
            if target is None and avail is None:
                break
            ok = has                                          # leftover slot (a prefix grows)
        if policy == "rr":
            order = [(last + 1 + j) % K for j in range(K)]
            k = next(j for j in order if ok[j])
        elif policy == "maxsnr":
            k = int(np.argmax(np.where(ok, g, -np.inf)))
        elif policy in PF_TIMING or policy in ("whittle", "lagrange"):   # ok is their choice
            r = rate(g)
            k = int(np.argmax(np.where(ok, r / (served + 1e-3 * r.mean()), -np.inf)))
        else:
            gv = g
            if any(b is not None for b in pend):              # HARQ: the slot carries the
                gv = g.copy()                                 # block waiting for the user
                for j, b in enumerate(pend):
                    if b is not None:
                        gv[j] = b["d"]
            gain = np.full(K, -np.inf)
            for j in np.flatnonzero(ok):
                now = belief.psnr(imgs[j], hist[j])
                gain[j] = belief.psnr(imgs[j], hist[j] + [float(gv[j])]) - now
                if policy in ADV_TIMING or policy == "greedy_rel":
                    ref = belief.psnr(imgs[j], hist[j] + [float(plan[j])]) - now
                    if policy in ADV_TIMING:                  # this slot against a typical one
                        gain[j] -= ref
                    else:                                     # tilted by the gain at the mean
                        gain[j] *= gain[j] / max(ref, 1e-9)
                elif alpha:
                    gain[j] *= max(now, 1e-3) ** (-alpha)
            tied = np.flatnonzero(gain >= gain.max() - 1e-12)
            if len(tied) > 1 and policy in ADV_TIMING:   # exact ties (useless reports: every
                r = rate(g)        # slot alike) go to PF, not to the first user, who would take
                k = int(tied[np.argmax((r / (served + 1e-3 * r.mean()))[tied])])  # every spare slot
            else:              # (greedy keeps the first: a digital step curve ties all the time)
                k = int(tied[0])
        if digital:                                           # the block arrives or is lost
            x = harq_deliver(util, pend, k, float(g[k]), float(snr[k, t]), harq)
            hist[k].append(x)
            out[k].append(x)
        else:
            hist[k].append(float(g[k]))
            out[k].append(float(snr[k, t]))
        if rest is not None:
            rest[k] = need(belief, imgs[k], hist[k], plan[k], goal)
        served = (1 - 1.0 / T) * served
        served[k] += rate(g[k]) / T
        last = k
        if log is not None:
            log.append((t, k))
        sent += 1
        if stop_after is not None and sent >= stop_after:
            break
    if fixed_codes and tgt is not None:                       # an incomplete fixed-length
        out = [h if len(h) >= n else [] for h, n in zip(out, tgt)]     # code cannot be decoded
    return out


def bootstrap(d, n=2000, seed=0):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fades(rng, users, slots, fading, corr=0.0):
    """Per-user slot gains in dB: i.i.d. Rayleigh (|h|^2 ~ Exp(1)), or with corr > 0 a
    Gauss-Markov h_t = corr h_(t-1) + sqrt(1 - corr^2) w_t (Jakes: corr = J0(2 pi f_D T))."""
    if fading == "none":
        return np.zeros((users, slots))
    if corr <= 0:
        return 10 * np.log10(rng.exponential(1.0, (users, slots)))
    w = (rng.standard_normal((users, slots)) + 1j * rng.standard_normal((users, slots))) / np.sqrt(2)
    h = np.empty_like(w)
    h[:, 0] = w[:, 0]
    for t in range(1, slots):
        h[:, t] = corr * h[:, t - 1] + np.sqrt(1.0 - corr ** 2) * w[:, t]
    return 10 * np.log10(np.maximum(np.abs(h) ** 2, 1e-12))


def goodput_backoff(util, sigma, mean_snr, fading, clip, n=200000, seed=12345, corr=1.0):
    """The MCS backoff (dB) that maximises a digital link's expected goodput when the
    reported SNR is off by N(0, sigma^2) dB (and, corr < 1, the report is stale: alloc/csi.py),
    over this simulation's slot SNRs (what outer-loop link adaptation settles at), and the
    share of blocks lost there."""
    if sigma <= 0 and corr >= 1.0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    m = rng.uniform(mean_snr[0], mean_snr[1], n)
    f = fades(rng, n, 1, fading)[:, 0]
    s = np.clip(m + f, clip[0], clip[1])
    if corr < 1.0 and fading != "none":
        from alloc.csi import outdated
        r = m + outdated(np.random.default_rng([seed, 4]), f, corr)
        r = np.clip(r + rng.normal(0.0, sigma, n) if sigma > 0 else r, clip[0], clip[1])
    else:
        r = np.clip(s + rng.normal(0.0, sigma, n), clip[0], clip[1])
    cap = util.efficiency(s)
    best = (-1.0, 0.0, 0.0)
    for b in np.arange(0.0, 3.0 * sigma + (12.0 if corr < 1.0 else 0.0) + 1e-9, 0.25):
        e = util.efficiency(r - b)
        ok = e <= cap + 1e-12
        good = float(np.mean(e * ok))
        if good > best[0]:
            best = (good, float(b), float(np.mean(~ok[e > 0])) if (e > 0).any() else 0.0)
    return best[1], best[2]


def planned_slots(slots, preempt=0.0, quantile=None):
    """--preempt-quantile: the slots a fixed-length plan assumes (that quantile of the
    binomial number of slots left to this traffic); None: plan for the expected share."""
    if not (preempt > 0 and quantile is not None):
        return None
    from math import comb
    cdf = 0.0
    for n in range(slots + 1):
        cdf += comb(slots, n) * (1 - preempt) ** n * preempt ** (slots - n)
        if cdf >= quantile:
            return n
    return slots


def draw_frames(util, users, slots, frames, mean_snr, fading, clip, seed=0, same_image=False,
                pred_noise=0.0, belief_util=None, fade_corr=0.0, preempt=0.0, cqi_error=0.0,
                cqi_corr=1.0):
    """The frames of a run: (images, mean SNRs, true slot SNRs, the transmitter's curves,
    available slots or None, reported SNRs or None) for each. The same arguments give the
    same frames to every policy, utility file and tool (common random numbers). Stale
    reports (cqi_corr < 1) are not clipped: run_frame clips them, or reads their posterior."""
    if users > len(util.names) and not same_image:
        raise ValueError(f"{users} users but only {len(util.names)} images in the pool")
    rng = np.random.default_rng(seed)
    crng = np.random.default_rng([seed, 1])              # CQI errors: the same frames at any sd
    arng = np.random.default_rng([seed, 4])              # stale reports: ... at any correlation
    if cqi_corr < 1.0:
        from alloc.csi import outdated
    for _ in range(frames):
        imgs = (np.full(users, rng.integers(len(util.names))) if same_image
                else rng.choice(len(util.names), users, replace=False))
        means = rng.uniform(mean_snr[0], mean_snr[1], users)
        fade = fades(rng, users, slots, fading, fade_corr)
        snr = np.clip(means[:, None] + fade, clip[0], clip[1])
        belief = (belief_util or util).perturbed(pred_noise, rng)
        avail = rng.random(slots) >= preempt if preempt > 0 else None
        if cqi_corr < 1.0 and fading != "none":
            seen = means[:, None] + outdated(arng, fade, cqi_corr)
            if cqi_error > 0:
                seen = seen + cqi_error * crng.standard_normal(snr.shape)
        else:
            seen = (np.clip(snr + cqi_error * crng.standard_normal(snr.shape), clip[0], clip[1])
                    if cqi_error > 0 else None)
        yield imgs, means, snr, belief, avail, seen


def tune_backoff(util, a, policy, sigma, belief, frames=300, corr=1.0, harq=None, aware=False):
    """A digital policy's MCS backoff under CQI errors of sd sigma (and stale reports of
    correlation corr), the best on a grid of 0.5 dB for its own objective on `frames` OTHER
    frames (seed + 1): where outer-loop link adaptation, which sees the ACKs of the blocks
    this policy schedules, would settle at best. harq: tuned with the retransmissions on,
    from -8 dB (a negative backoff sends above the report: with incremental redundancy a
    block that fails is finished by the next transmissions, so bold rates pay). aware (stale
    reports, --cqi-aware, with HARQ): an offset on the aware transmitter's rate instead,
    -12 ... +4 dB."""
    best = None
    lo = -8.0 if harq is not None else 0.0
    grid = (np.arange(-12.0, 4.01, 0.5) if aware
            else np.arange(lo, 2.0 * sigma + (10.0 if corr < 1.0 else 0.0) + 0.51, 0.5))
    for b in grid:
        s, _, _ = simulate(util, a.users, a.slots, frames, a.mean_snr, a.fading, a.snr_clip,
                           [policy], a.seed + 1, a.same_image, a.pred_noise, a.alpha, 0,
                           a.target, a.margin, a.plan_offset, belief, a.fade_corr, a.plan,
                           a.fixed_codes, a.preempt, a.preempt_quantile, sigma, float(b),
                           cqi_corr=corr, cqi_aware=aware, harq=harq)
        v = s[policy]["mean_psnr" if a.target is None else "satisfied"]
        if best is None or v > best[0] + 1e-12:
            best = (v, float(b))
    return best[1]


def simulate(util, users, slots, frames, mean_snr, fading, clip, policies, seed=0,
             same_image=False, pred_noise=0.0, alpha=0.0, dump=0, target=None, margin=0.0,
             offset=0.0, belief_util=None, fade_corr=0.0, plan="mean", fixed_codes=False,
             preempt=0.0, preempt_quantile=None, cqi_error=0.0, backoff=0.0, cqi_corr=1.0,
             cqi_aware=False, delay=False, keep_frames=False, harq=None):
    """belief_util: the transmitter's curves (default: the truth, `util`); policies decide
    on it, users are scored on `util`. cqi_error: sd (dB) of the error of the reported slot
    SNRs; backoff: a digital link's MCS backoff (dB), or one per policy {policy: dB};
    cqi_corr: stale reports (alloc/csi.py), cqi_aware: the transmitter knows they are.
    delay: the timing of the deliveries (delay_times); keep_frames: each policy's per-frame
    objective in its summary ("frames": frame mean PSNR, or share at the target); harq:
    (mode, max_tx), a digital link's retransmissions (harq_deliver)."""
    t_plan = planned_slots(slots, preempt, preempt_quantile)
    res = {p: {"psnr": [], "frame_mean": [], "frame_min": [], "deep": [], "served": [],
               "lift": [], "sat": [], "frame_sat": [], "failed": [], "first": [], "tavg": [],
               "reach": [], "frame_reach": []} for p in policies}
    dumped = []
    frame_iter = draw_frames(util, users, slots, frames, mean_snr, fading, clip, seed, same_image,
                             pred_noise, belief_util, fade_corr, preempt, cqi_error, cqi_corr)
    aware = cqi_corr if cqi_aware and cqi_corr < 1.0 and fading != "none" else None
    for f, (imgs, means, snr, belief, avail, seen) in enumerate(frame_iter):
        avg = belief.average() if target is not None else None
        for p in policies:
            lg = [] if delay else None
            hist = run_frame(util, belief, imgs, means, snr, p, alpha, target, margin, offset, avg,
                             fade_corr if plan == "conditional" else 0.0, fixed_codes,
                             avail, 1.0 - preempt, t_plan, seen,
                             backoff.get(p, 0.0) if isinstance(backoff, dict) else backoff,
                             log=lg, clip=clip, aware=aware, harq=harq)
            q = np.array([util.psnr(i, h) for i, h in zip(imgs, hist)])
            r = res[p]
            r["psnr"] += q.tolist()
            r["frame_mean"].append(float(q.mean()))
            r["frame_min"].append(float(q.min()))
            sent = [s for h in hist for s in h if s > FAILED]
            lost = sum(s <= FAILED for h in hist for s in h)
            r["deep"].append(float(np.mean(np.array(sent) < 0.0)) if sent else 0.0)
            r["failed"].append(lost / max(lost + len(sent), 1))
            r["served"].append(float(np.mean([len(h) for h in hist])))
            r["lift"] += [s - m for h, m in zip(hist, means) for s in h if s > FAILED]
            if target is not None:
                ok = q >= target
                r["sat"] += ok.tolist()
                r["frame_sat"].append(float(ok.mean()))
            if delay:
                fi, ta, re = delay_times(util, imgs, hist, lg, slots, target, fixed_codes)
                r["first"] += fi
                r["tavg"] += ta
                if target is not None:
                    r["reach"] += re
                    r["frame_reach"].append(float(np.mean(np.minimum(re, slots + 1))))
            if f < dump:
                for k, (i, h) in enumerate(zip(imgs, hist)):
                    dumped.append({"frame": f, "policy": p, "user": k,
                                   "image": util.names[int(i)], "mean_snr": float(means[k]),
                                   "snrs": h, "predicted_psnr": float(q[k])})
    base = "pf" if "pf" in policies else policies[0]
    key = "frame_mean" if target is None else "frame_sat"
    summary = {}
    for p in policies:
        r = res[p]
        d = np.array(r[key]) - np.array(res[base][key])
        lo, hi = bootstrap(d, seed=seed)
        summary[p] = {"mean_psnr": float(np.mean(r["psnr"])),
                      "p5_psnr": float(np.percentile(r["psnr"], 5)),
                      "worst_user": float(np.mean(r["frame_min"])),
                      "deep_fade_share": float(np.mean(r["deep"])),
                      "phases_per_user": float(np.mean(r["served"])),
                      "served_snr_lift": float(np.mean(r["lift"])) if r["lift"] else 0.0,
                      "failed_share": float(np.mean(r["failed"])),
                      f"vs_{base}": float(d.mean()), f"vs_{base}_ci": [lo, hi]}
        if target is not None:
            summary[p]["satisfied"] = float(np.mean(r["sat"]))
            if "padc" in policies:
                e = np.array(r[key]) - np.array(res["padc"][key])
                summary[p]["vs_padc"] = float(e.mean())
                summary[p]["vs_padc_ci"] = list(bootstrap(e, seed=seed))
        if keep_frames:
            summary[p]["frames"] = list(r[key])
        if delay:
            summary[p].update(delay_summary(res, p, base, policies, users, slots, target, seed))
    return summary, base, dumped


def delay_summary(res, p, base, policies, K, T, target, seed):
    """simulate --delay: the timing measures of policy p (against `base`, and padc if run)."""
    r = res[p]
    out = {"first_slot": float(np.mean(r["first"])), "time_avg_psnr": float(np.mean(r["tavg"]))}
    if target is None:
        return out
    tt = np.array(r["reach"], dtype=np.float64)
    got = tt[np.isfinite(tt)]
    out["at_target_by"] = [float(np.mean(tt <= t)) for t in range(1, T + 1)]
    out["reach_slot"] = float(got.mean()) if len(got) else None
    out["reach_p90"] = float(np.percentile(got, 90)) if len(got) else None
    out["reach_all"] = float(np.mean(np.minimum(tt, T + 1)))
    for ref in dict.fromkeys([base] + (["padc"] if "padc" in policies else [])):
        d = np.array(r["frame_reach"]) - np.array(res[ref]["frame_reach"])
        out[f"reach_all_vs_{ref}"] = float(d.mean())
        out[f"reach_all_vs_{ref}_ci"] = list(bootstrap(d, seed=seed))
        u = np.array(res[ref]["reach"], dtype=np.float64)          # the same users, both reach
        both = np.isfinite(tt) & np.isfinite(u)
        per_f = []
        for f in range(len(tt) // K):
            b = both[f * K:(f + 1) * K]
            if b.any():
                per_f.append(float((tt[f * K:(f + 1) * K][b] - u[f * K:(f + 1) * K][b]).mean()))
        out[f"same_users_vs_{ref}"] = float(np.mean(per_f)) if per_f else None
        out[f"same_users_vs_{ref}_ci"] = list(bootstrap(per_f, seed=seed)) if len(per_f) > 1 else None
        out[f"same_users_vs_{ref}_n"] = int(both.sum())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--utility", required=True, help="tools/utility_fit.py output")
    ap.add_argument("--users", type=int, default=8)
    ap.add_argument("--slots", type=int, default=24, help="slots per frame (one phase each)")
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--mean-snr", nargs=2, type=float, default=[0.0, 15.0],
                    help="users' mean SNRs, uniform in this range (dB)")
    ap.add_argument("--fading", default="rayleigh", choices=["rayleigh", "none"])
    ap.add_argument("--fade-corr", type=float, default=0.0,
                    help="slot-to-slot correlation of the Rayleigh gain (0: i.i.d. block fading)")
    ap.add_argument("--plan", default="mean", choices=["mean", "conditional"],
                    help="future phases at the mean SNR, or (with --fade-corr) at the mean "
                         "expected given the current slot")
    ap.add_argument("--length-bonus", nargs="+", type=float, default=None,
                    help="dB per prefix length 1..P added to the curves (truth and belief): "
                         "one model per CBR without the nesting penalty; one value = all")
    ap.add_argument("--fixed-codes", action="store_true",
                    help="one model per CBR: a policy with lengths fixed before the frame "
                         "cannot send a user more phases than its code (no leftover slots)")
    ap.add_argument("--preempt", type=float, default=0.0,
                    help="each slot is taken by other traffic with this probability, known "
                         "only when it starts; policies plan with the expected share")
    ap.add_argument("--preempt-quantile", type=float, default=None,
                    help="--preempt: plan lengths for this quantile of the slots left "
                         "(e.g. 0.1: hedged), instead of their expected number")
    ap.add_argument("--belief", default=None,
                    help="the transmitter's curves (tools/utility_fit.py --calib); default: the truth")
    ap.add_argument("--link", default=None, choices=["cqi", "shannon"],
                    help="digital utility files: link adaptation (default: the file's)")
    ap.add_argument("--gap-db", type=float, default=None, help="digital, shannon link: SNR gap")
    ap.add_argument("--digital-mode", default=None, choices=["step", "interp"],
                    help="digital: best complete file, or an ideal scalable codec")
    ap.add_argument("--digital-max-phases", type=int, default=None,
                    help="digital: slots one user may take (default: the file's, 12)")
    ap.add_argument("--cqi-error", nargs="+", type=float, default=[0.0],
                    help="sd (dB) of the error of the reported slot SNRs: every decision is "
                         "made on the reports, codes are decoded at the true SNR; several "
                         "values: one table each (same frames)")
    ap.add_argument("--cqi-corr", nargs="+", type=float, default=[1.0],
                    help="stale reports: correlation r of each slot's reported channel with its "
                         "true one (r = J0(2 pi f_D tau): Doppler f_D, report age tau; 1 = fresh, "
                         "0 = useless; alloc/csi.py); several values: one table each (same frames)")
    ap.add_argument("--cqi-aware", action="store_true",
                    help="with --cqi-corr < 1: the transmitter knows r and decides on the "
                         "posterior of each slot (token prefix: its expected distortion; digital: "
                         "the rate of largest expected goodput, no backoff; whittle: on the "
                         "report, through its program)")
    ap.add_argument("--harq", default=None, choices=["ir", "chase"],
                    help="digital: a lost block is retransmitted in the user's next slot and "
                         "combined (ir: incremental redundancy, chase: Chase combining)")
    ap.add_argument("--harq-max", type=int, default=4,
                    help="--harq: transmissions per block at most (NR: 4)")
    ap.add_argument("--cqi-backoff", default="auto",
                    help="digital: MCS backoff (dB) under --cqi-error; 'auto': tuned per policy "
                         "and objective on other frames (the best outer-loop link adaptation "
                         "could settle at)")
    ap.add_argument("--snr-clip", nargs=2, type=float, default=[-2.0, 22.0],
                    help="slot SNRs clipped to the codec's training range")
    ap.add_argument("--policies", nargs="+", default=None, choices=POLICIES,
                    help="default: the mean-quality set, or the target set with --target")
    ap.add_argument("--target", type=float, default=None,
                    help="PSNR target (dB): score the share of users that reach it")
    ap.add_argument("--margin", type=float, default=0.0,
                    help="target policies aim at target + margin (the model's error)")
    ap.add_argument("--plan-offset", type=float, default=0.0,
                    help="target policies predict future phases at the mean SNR + this (dB)")
    ap.add_argument("--same-image", action="store_true", help="every user the same image")
    ap.add_argument("--pred-noise", type=float, default=0.0,
                    help="sd of the log-normal error of the transmitter's curves")
    ap.add_argument("--alpha", type=float, default=0.0, help="greedy: alpha-fair weighting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", action="store_true",
                    help="also the timing of the deliveries: first phase, PSNR over the frame, "
                         "and with --target the slot each user reaches it (a CDF by slot)")
    ap.add_argument("--dump", type=int, default=0, help="histories of the first N frames")
    ap.add_argument("--dump-out", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.policies is None:
        a.policies = list(TARGET_POLICIES if a.target is not None else MEAN_POLICIES)
    needs = sorted(set(a.policies) & set(NEEDS_TARGET))
    if needs and a.target is None:
        ap.error(f"{', '.join(needs)} need --target")
    dig = {"link": a.link, "gap_db": a.gap_db, "mode": a.digital_mode, "P": a.digital_max_phases}
    util = load_utility(a.utility, **dig)
    belief = load_utility(a.belief, **dig) if a.belief else None
    if belief is not None and belief.names != util.names:
        ap.error("--belief must describe the same images, in the same order, as --utility")
    if not 0.0 <= a.preempt < 1.0:
        ap.error("--preempt is in [0, 1)")
    if not 0.0 <= a.fade_corr < 1.0:
        ap.error("--fade-corr is in [0, 1)")
    if min(a.cqi_error) < 0:
        ap.error("--cqi-error is an sd (dB), >= 0")
    if len(a.cqi_error) * len(a.cqi_corr) > 1 and (a.out or a.dump):
        ap.error("--out / --dump take one --cqi-error and one --cqi-corr value")
    if not all(0.0 <= r <= 1.0 for r in a.cqi_corr):
        ap.error("--cqi-corr is a correlation in [0, 1]")
    digital = getattr(util, "kind", "") == "digital"
    if a.harq and not digital:
        ap.error("--harq is for a digital utility (a token prefix has no blocks to lose)")
    if a.harq_max < 1:
        ap.error("--harq-max >= 1")
    harq = (a.harq, a.harq_max) if a.harq else None
    if a.length_bonus:
        if digital:
            ap.error("--length-bonus is for the token-prefix utility")
        bonus = a.length_bonus * util.P if len(a.length_bonus) == 1 else a.length_bonus
        util = util.shifted(bonus)
        belief = belief.shifted(bonus) if belief is not None else None
    for corr, sigma in [(c, s) for c in a.cqi_corr for s in a.cqi_error]:
        backoff, ref = 0.0, None
        stale = corr < 1.0 and a.fading != "none"
        aw = bool(a.cqi_aware and stale)                     # an aware rate: tuned only with HARQ
        if digital and (sigma > 0 or stale) and (not aw or harq is not None):
            if a.cqi_backoff == "auto":
                if not aw:
                    ref = goodput_backoff(util, sigma, a.mean_snr, a.fading, a.snr_clip, corr=corr)
                backoff = {p: tune_backoff(util, a, p, sigma, belief, corr=corr, harq=harq,
                                           aware=aw)
                           for p in a.policies}
            else:
                backoff = float(a.cqi_backoff)
        summary, base, dumped = simulate(util, a.users, a.slots, a.frames, a.mean_snr, a.fading,
                                         a.snr_clip, a.policies, a.seed, a.same_image,
                                         a.pred_noise, a.alpha, a.dump, a.target, a.margin,
                                         a.plan_offset, belief, a.fade_corr, a.plan,
                                         a.fixed_codes, a.preempt, a.preempt_quantile,
                                         sigma, backoff, corr, a.cqi_aware, delay=a.delay,
                                         harq=harq)
        report(a, util, digital, summary, base, sigma, backoff, ref, corr)
        if a.delay:
            report_delay(a, util, summary, base)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "summary": summary, "backoff": backoff,
                       "cqi_corr": a.cqi_corr[0], "cqi_error": a.cqi_error[0]}, f, indent=1)
        print(f"wrote {a.out}")
    if a.dump:
        path = a.dump_out or os.path.splitext(a.out or "sched")[0] + "_histories.json"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"chunking": "phase", "utility": a.utility, "target": a.target,
                       "entries": dumped}, f)
        print(f"wrote {path}: {len(dumped)} delivery histories")


def report(a, util, digital, summary, base, sigma, backoff, ref, corr=1.0):
    stale = corr < 1.0 and a.fading != "none"
    tx = "aware" if a.cqi_aware else "naive"
    print(f"{a.users} users, {a.slots} slots ({a.slots / a.users:.2f} phases per user, at most "
          f"{util.P}), {a.frames} frames, mean SNR U{a.mean_snr} dB, fading {a.fading}"
          f"{f' (slot correlation {a.fade_corr:g}, plan {a.plan})' if a.fade_corr else ''}"
          f"{f', length bonus {a.length_bonus} dB' if a.length_bonus else ''}"
          f"{', fixed-length codes' if a.fixed_codes else ''}"
          f"{f', {100 * a.preempt:g}% of slots preempted' if a.preempt else ''}"
          f"{f' (planned at their {a.preempt_quantile:g} quantile)' if a.preempt and a.preempt_quantile is not None else ''}"
          f"{', same image' if a.same_image else ''}"
          f"{f', transmitter curves {a.belief}' if a.belief else ''}"
          f"{f', digital: {util.link} link, {util.mode}, up to {util.P} slots' if digital else ''}"
          f"{f', HARQ {a.harq} (up to {a.harq_max} transmissions)' if digital and a.harq else ''}"
          f"{f', CQI error {sigma:g} dB' if sigma else ''}"
          f"{f', stale reports (correlation {corr:g}; transmitter {tx})' if stale else ''}"
          f"{f' (MCS backoff {backoff:g} dB)' if digital and (sigma or stale) and not isinstance(backoff, dict) else ''}"
          f"{f', prediction noise {a.pred_noise:g}' if a.pred_noise else ''}"
          f"{f', alpha {a.alpha:g}' if a.alpha else ''}; utility {a.utility}")
    if a.target is None:
        print(f"{'policy':15s} {'mean PSNR':>9s} {'p5 user':>8s} {'worst':>7s} {'<0 dB':>6s} "
              f"{'phases':>6s} {'lift':>5s}   {'vs ' + base + ' [95% CI]':>24s}")
        for p, s in summary.items():
            lo, hi = s[f"vs_{base}_ci"]
            print(f"{p:15s} {s['mean_psnr']:9.3f} {s['p5_psnr']:8.3f} {s['worst_user']:7.3f} "
                  f"{100 * s['deep_fade_share']:5.1f}% {s['phases_per_user']:6.2f} "
                  f"{s['served_snr_lift']:+5.1f}   {s[f'vs_{base}']:+8.3f} [{lo:+.3f}, {hi:+.3f}]")
    else:
        print(f"target {a.target:g} dB (policies aim at {a.target + a.margin:g}), future phases "
              f"planned at the mean SNR {a.plan_offset:+g} dB; satisfied = share of users at or "
              f"above the target at the deadline")
        vs = "padc" in summary
        print(f"{'policy':15s} {'satisfied':>9s} {'mean PSNR':>9s} {'<0 dB':>6s} {'phases':>6s} "
              f"{'lift':>5s}   {'vs ' + base + ' [95% CI]':>24s}"
              + (f"   {'vs padc [95% CI]':>24s}" if vs else ""))
        for p, s in summary.items():
            lo, hi = s[f"vs_{base}_ci"]
            line = (f"{p:15s} {100 * s['satisfied']:8.1f}% {s['mean_psnr']:9.3f} "
                    f"{100 * s['deep_fade_share']:5.1f}% {s['phases_per_user']:6.2f} "
                    f"{s['served_snr_lift']:+5.1f}   {100 * s[f'vs_{base}']:+7.1f}% "
                    f"[{100 * lo:+.1f}, {100 * hi:+.1f}]")
            if vs:
                lo, hi = s["vs_padc_ci"]
                line += f"   {100 * s['vs_padc']:+7.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            print(line)
    if digital and (sigma or stale):
        if isinstance(backoff, dict):
            what = ("offset on the aware rate" if a.cqi_aware and stale else "MCS backoff")
            print(f"{what} per policy, tuned on 300 other frames (0.5 dB grid): " + ", ".join(
                f"{p} {b:g} dB" for p, b in backoff.items())
                + (f"; goodput-optimal over all slots {ref[0]:g} dB ({100 * ref[1]:.1f}% of "
                   f"blocks lost there, one transmission)" if ref is not None else ""))
        print(("slots without a decoded block (a transmission that failed; HARQ retransmits "
               "it): " if a.harq else "blocks lost (rate above what the slot supported): ")
              + ", ".join(f"{p} {100 * s['failed_share']:.1f}%" for p, s in summary.items()))

def report_delay(a, util, summary, base):
    """The --delay table."""
    T = a.slots
    empty = util.psnr(0, []) if getattr(util, "kind", "") != "digital" else float("nan")
    print(f"timing: '1st' = the slot of a user's first decodable phase (mean; {T + 1} = never), "
          f"'PSNR over frame' = its PSNR averaged over the {T} slots, as decoded after each "
          f"(nothing yet = the empty image, {empty:.1f} dB)")
    if a.target is None:
        print(f"{'policy':15s} {'1st':>6s} {'PSNR over frame':>15s}")
        for p, s in summary.items():
            print(f"{p:15s} {s['first_slot']:6.2f} {s['time_avg_psnr']:15.3f}")
        return
    cut = sorted({max(1, int(round(T * f))) for f in (0.25, 0.5, 0.75, 1.0)})
    ref = "padc" if "padc" in summary else base
    print(f"target {a.target:g} dB: users at it by slot {', '.join(map(str, cut))} (% of all users); "
          f"the slot they reach it (those that do: mean, 90th percentile); the mean slot over all "
          f"users with misses at T + 1 = {T + 1}, against {ref}; and against {ref} on the users "
          f"both bring to it (the same users: no selection effect)")
    print(f"{'policy':15s} {'1st':>6s} {'PSNR over frame':>15s}   "
          + " ".join(f"{'by ' + str(c):>6s}" for c in cut)
          + f"   {'reach':>5s} {'p90':>4s}   {'all, vs ' + ref + ' [95% CI]':>30s}"
          f"   {'same users vs ' + ref + ' [95% CI]':>30s}")
    for p, s in summary.items():
        by = s["at_target_by"]
        ra, rp = s["reach_slot"], s["reach_p90"]
        line = (f"{p:15s} {s['first_slot']:6.2f} {s['time_avg_psnr']:15.3f}   "
                + " ".join(f"{100 * by[c - 1]:5.1f}%" for c in cut)
                + f"   {ra if ra is None else format(ra, '5.2f'):>5s} "
                  f"{rp if rp is None else format(rp, '4.0f'):>4s}   {s['reach_all']:6.2f}")
        if p != ref:
            lo, hi = s[f"reach_all_vs_{ref}_ci"]
            line += f" {s[f'reach_all_vs_{ref}']:+6.2f} [{lo:+.2f}, {hi:+.2f}]"
            m, ci = s[f"same_users_vs_{ref}"], s[f"same_users_vs_{ref}_ci"]
            line += (f"   {m:+6.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] on {s[f'same_users_vs_{ref}_n']}"
                     if m is not None and ci is not None else "   -")
        print(line)


if __name__ == "__main__":
    main()
