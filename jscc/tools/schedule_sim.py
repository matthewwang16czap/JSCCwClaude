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
    online       stop_adv plus admission every slot: the users with the smallest remaining
                 need (phases at the mean SNR) that fit in the slots left are served first;
                 spare slots open to all. Content-aware sizing, channel-aware stopping and
                 timing: what only an encode-once prefix code with a mixed-SNR utility allows

All policies see the same images, mean SNRs and fades (common random numbers). Reported per
policy: mean PSNR at the deadline, the 5th percentile over users, the worst user of a frame on
average (mean objective) or the satisfied share (target), the share of served slots below 0
dB, phases per user, the "lift" (served slot SNR minus the user's mean SNR, dB), and the
difference to pf (and to padc with a target) with a 95% bootstrap interval over frames.
Ablations: --same-image (no content diversity: only the channel part), --fading none (no
fading: only the content and mean-SNR part), --fade-corr r (slot gains correlated in time,
Gauss-Markov), --pred-noise (the transmitter's curves are off: log-normal errors on each
image's noise-free MSE and sensitivity), --belief FILE (the transmitter decides on its own
calibrated curves, tools/utility_fit.py --calib; users are scored on --utility).
--utility may also be a digital baseline (tools/digital_rd.py, alloc/digital.py: an image
codec's R-D points over the same slots with ideal link adaptation; --link, --gap-db,
--digital-mode, --digital-max-phases override the file). --dump N writes the
delivery histories of the first N frames for tools/mixed_snr.py --design file, which decodes
them with the codec itself.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import load_utility  # noqa: E402

MEAN_POLICIES = ("rr", "maxsnr", "pf", "fixed_uni", "fixed_eq", "fixed_slope",
                 "adaptive_slope", "slope_adv", "greedy", "greedy_rel")
TARGET_POLICIES = ("rr", "pf", "pf_len", "padc", "padc_adv", "stop", "stop_adv", "online_pf",
                   "online")
POLICIES = tuple(dict.fromkeys(MEAN_POLICIES + TARGET_POLICIES))
NEEDS_TARGET = ("pf_len", "padc", "padc_adv", "stop", "stop_adv", "online_pf", "online")
PF_TIMING = ("pf", "fixed_uni", "fixed_eq", "fixed_slope", "adaptive_slope", "pf_len", "padc",
             "stop", "online_pf")
ADV_TIMING = ("slope_adv", "padc_adv", "stop_adv", "online")


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


def run_frame(util, belief, imgs, means, snr, policy, alpha=0.0, target=None, margin=0.0,
              offset=0.0, avg=None):
    """One frame under one policy: per-user delivered SNR lists. With a target (dB) the
    target policies size, stop and admit by it, and a slot that no policy rule claims goes
    to PF over every user with phases left (so every policy can use the whole frame)."""
    K, T = snr.shape
    P = util.P
    hist = [[] for _ in range(K)]
    served = np.zeros(K)                                      # PF: served-rate average
    plan = np.asarray(means, dtype=np.float64) + offset      # SNR assumed for future phases
    goal = None if target is None else target + margin
    tgt = None                                                # phases per user, fixed upfront
    if policy in ("fixed_uni", "fixed_eq", "fixed_slope", "slope_adv"):
        tgt = targets(belief, imgs, means, T, {"fixed_uni": "uni", "fixed_eq": "eq"}.get(policy, "slope"))
    elif policy == "pf_len":                                  # content-blind link adaptation
        tgt = np.array([need(avg, 0, [], p, goal) for p in plan], dtype=float)
        tgt = np.where(np.isfinite(tgt), tgt, P).astype(int)
    elif policy in ("padc", "padc_adv"):                      # per image at the mean SNR
        L = np.array([need(belief, i, [], p, goal) for i, p in zip(imgs, plan)], dtype=float)
        tgt = np.zeros(K, dtype=int)
        left = T
        for k in np.argsort(L, kind="stable"):                # shortest first, while it fits
            if np.isfinite(L[k]) and L[k] <= left:
                tgt[k], left = int(L[k]), left - int(L[k])
    rest = None                                               # online: phases still needed
    if policy in ("stop", "stop_adv", "online_pf", "online"):
        rest = np.array([need(belief, i, [], p, goal) for i, p in zip(imgs, plan)], dtype=float)
    last = -1
    for t in range(T):
        has = np.array([len(h) < P for h in hist])
        if not has.any():
            break
        ok = has.copy()
        if tgt is not None:
            ok &= np.array([len(h) < g for h, g in zip(hist, tgt)])
        if rest is not None:
            ok &= rest > 0                                    # not there yet
            if policy.startswith("online"):  # admit the smallest needs that fit in the slots left;
                adm = np.zeros(K, dtype=bool)                 # spare slots open to the others
                left = T - t
                for k in np.argsort(rest, kind="stable"):
                    if ok[k] and rest[k] <= left:
                        adm[k], left = True, left - int(rest[k])
                if adm.any() and left < 1:
                    ok = adm
        g = snr[:, t]
        if policy == "adaptive_slope":
            below = ok & (np.array([len(h) for h in hist]) < replan(belief, imgs, means, hist, T - t))
            if below.any():
                ok = below
        if not ok.any():
            if target is None:
                break
            ok = has                                          # leftover slot
        if policy == "rr":
            order = [(last + 1 + j) % K for j in range(K)]
            k = next(j for j in order if ok[j])
        elif policy == "maxsnr":
            k = int(np.argmax(np.where(ok, g, -np.inf)))
        elif policy in PF_TIMING:
            r = rate(g)
            k = int(np.argmax(np.where(ok, r / (served + 1e-3 * r.mean()), -np.inf)))
        else:
            gain = np.full(K, -np.inf)
            for j in np.flatnonzero(ok):
                now = belief.psnr(imgs[j], hist[j])
                gain[j] = belief.psnr(imgs[j], hist[j] + [float(g[j])]) - now
                if policy in ADV_TIMING or policy == "greedy_rel":
                    ref = belief.psnr(imgs[j], hist[j] + [float(plan[j])]) - now
                    if policy in ADV_TIMING:                  # this slot against a typical one
                        gain[j] -= ref
                    else:                                     # tilted by the gain at the mean
                        gain[j] *= gain[j] / max(ref, 1e-9)
                elif alpha:
                    gain[j] *= max(now, 1e-3) ** (-alpha)
            k = int(np.argmax(gain))
        hist[k].append(float(g[k]))
        if rest is not None:
            rest[k] = need(belief, imgs[k], hist[k], plan[k], goal)
        served = (1 - 1.0 / T) * served
        served[k] += rate(g[k]) / T
        last = k
    return hist


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


def simulate(util, users, slots, frames, mean_snr, fading, clip, policies, seed=0,
             same_image=False, pred_noise=0.0, alpha=0.0, dump=0, target=None, margin=0.0,
             offset=0.0, belief_util=None, fade_corr=0.0):
    """belief_util: the transmitter's curves (default: the truth, `util`); policies decide
    on it, users are scored on `util`."""
    if users > len(util.names) and not same_image:
        raise ValueError(f"{users} users but only {len(util.names)} images in the pool")
    rng = np.random.default_rng(seed)
    res = {p: {"psnr": [], "frame_mean": [], "frame_min": [], "deep": [], "served": [],
               "lift": [], "sat": [], "frame_sat": []} for p in policies}
    dumped = []
    for f in range(frames):
        imgs = (np.full(users, rng.integers(len(util.names))) if same_image
                else rng.choice(len(util.names), users, replace=False))
        means = rng.uniform(mean_snr[0], mean_snr[1], users)
        fade = fades(rng, users, slots, fading, fade_corr)
        snr = np.clip(means[:, None] + fade, clip[0], clip[1])
        belief = (belief_util or util).perturbed(pred_noise, rng)
        avg = belief.average() if target is not None else None
        for p in policies:
            hist = run_frame(util, belief, imgs, means, snr, p, alpha, target, margin, offset, avg)
            q = np.array([util.psnr(i, h) for i, h in zip(imgs, hist)])
            r = res[p]
            r["psnr"] += q.tolist()
            r["frame_mean"].append(float(q.mean()))
            r["frame_min"].append(float(q.min()))
            sent = [s for h in hist for s in h]
            r["deep"].append(float(np.mean(np.array(sent) < 0.0)) if sent else 0.0)
            r["served"].append(float(np.mean([len(h) for h in hist])))
            r["lift"] += [s - m for h, m in zip(hist, means) for s in h]
            if target is not None:
                ok = q >= target
                r["sat"] += ok.tolist()
                r["frame_sat"].append(float(ok.mean()))
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
                      f"vs_{base}": float(d.mean()), f"vs_{base}_ci": [lo, hi]}
        if target is not None:
            summary[p]["satisfied"] = float(np.mean(r["sat"]))
            if "padc" in policies:
                e = np.array(r[key]) - np.array(res["padc"][key])
                summary[p]["vs_padc"] = float(e.mean())
                summary[p]["vs_padc_ci"] = list(bootstrap(e, seed=seed))
    return summary, base, dumped


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
    ap.add_argument("--belief", default=None,
                    help="the transmitter's curves (tools/utility_fit.py --calib); default: the truth")
    ap.add_argument("--link", default=None, choices=["cqi", "shannon"],
                    help="digital utility files: link adaptation (default: the file's)")
    ap.add_argument("--gap-db", type=float, default=None, help="digital, shannon link: SNR gap")
    ap.add_argument("--digital-mode", default=None, choices=["step", "interp"],
                    help="digital: best complete file, or an ideal scalable codec")
    ap.add_argument("--digital-max-phases", type=int, default=None,
                    help="digital: slots one user may take (default: the file's, 12)")
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
    if not 0.0 <= a.fade_corr < 1.0:
        ap.error("--fade-corr is in [0, 1)")
    summary, base, dumped = simulate(util, a.users, a.slots, a.frames, a.mean_snr, a.fading,
                                     a.snr_clip, a.policies, a.seed, a.same_image,
                                     a.pred_noise, a.alpha, a.dump, a.target, a.margin,
                                     a.plan_offset, belief, a.fade_corr)
    print(f"{a.users} users, {a.slots} slots ({a.slots / a.users:.2f} phases per user, at most "
          f"{util.P}), {a.frames} frames, mean SNR U{a.mean_snr} dB, fading {a.fading}"
          f"{f' (slot correlation {a.fade_corr:g})' if a.fade_corr else ''}"
          f"{', same image' if a.same_image else ''}"
          f"{f', transmitter curves {a.belief}' if a.belief else ''}"
          f"{f', digital: {util.link} link, {util.mode}, up to {util.P} slots' if getattr(util, 'kind', '') == 'digital' else ''}"
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
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "summary": summary}, f, indent=1)
        print(f"wrote {a.out}")
    if a.dump:
        path = a.dump_out or os.path.splitext(a.out or "sched")[0] + "_histories.json"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"chunking": "phase", "utility": a.utility, "target": a.target,
                       "entries": dumped}, f)
        print(f"wrote {path}: {len(dumped)} delivery histories")


if __name__ == "__main__":
    main()
