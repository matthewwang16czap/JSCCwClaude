"""Opportunistic scheduling of encode-once token prefixes over a block-fading downlink
(docs/PROBLEM.md), simulated on the utility model of alloc/utility.py. CPU, about a minute.

    python tools/schedule_sim.py --utility results/utility_snrc4.json --users 8 --slots 24 \\
        --frames 2000 --mean-snr 0 15 --out results/sched_main.json

A frame: K users, each owed one image drawn from the utility file's pool, and T slots. A
slot carries one PHASE of one user's prefix (CBR 1/48 per phase at 256 px tiles), so a user
holds at most P phases (1/8). User k's SNR in slot t is its mean SNR plus a Rayleigh
block-fading draw (10 log10 |h|^2, |h|^2 ~ Exp(1)), clipped to --snr-clip. The base station
knows every user's SNR in the current slot (CQI), never a future one. Phases go out in
prefix order; at the deadline every user decodes what it holds. Policies:

    rr           round robin over users with phases left (channel- and content-blind)
    maxsnr       the largest current SNR
    pf           proportional fair: log2(1 + snr) over the user's served-rate average
    fixed_eq     PADC-style: before the frame, every user's number of phases from its
                 predicted constant-SNR curve at its MEAN SNR, by PADC's equal-quality rule
                 over the T slots; delivered with PF timing among users below their target
    fixed_slope  the same targets by the equal-slope rule (mean quality)
    adaptive_slope  fixed_slope re-planned every slot: equal-slope targets for the slots left,
                 from what every user already holds and future phases at its mean SNR; PF
                 timing among users below their current target (content-aware lengths on the
                 slow timescale, channel-aware timing on the fast one)
    greedy       the largest predicted PSNR gain of this slot, given what the user already
                 holds and the slot's SNR (myopic marginal utility; --alpha > 0 weighs it by
                 the user's current PSNR^-alpha for alpha-fairness)
    greedy_rel   that gain times its ratio to the gain at the user's mean SNR: tilted towards
                 users at their own peaks, as PF is for rate (a pure ratio starves users that
                 hold nothing yet; --alpha does not apply)

All policies see the same images, mean SNRs and fades (common random numbers). Reported per
policy: the mean PSNR at the deadline, the 5th percentile over users, the worst user of a
frame on average, the share of served slots below 0 dB, and the difference in mean PSNR to
pf with a 95% bootstrap interval over frames. Ablations: --same-image (no content
diversity: only the channel part), --fading none (no fading: only the content and mean-SNR
part), --pred-noise (the transmitter's curves are off: log-normal errors on each image's
noise-free MSE and sensitivity). --dump N writes the delivery histories of the first N
frames for tools/mixed_snr.py --design file, which decodes them with the codec itself.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import Utility  # noqa: E402

POLICIES = ("rr", "maxsnr", "pf", "fixed_eq", "fixed_slope", "adaptive_slope", "greedy",
            "greedy_rel")


def rate(snr_db):
    return np.log2(1.0 + 10.0 ** (np.asarray(snr_db, dtype=np.float64) / 10.0))


def targets(util, imgs, means, T, rule):
    """Phases per user decided before the frame from constant-SNR curves at the mean SNRs."""
    K, P = len(imgs), util.P
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


def run_frame(util, belief, imgs, means, snr, policy, alpha=0.0):
    """One frame under one policy: per-user delivered SNR lists."""
    K, T = snr.shape
    P = util.P
    hist = [[] for _ in range(K)]
    served = np.zeros(K)                                      # PF: served-rate average
    tgt = targets(belief, imgs, means, T, "eq" if policy == "fixed_eq" else "slope") \
        if policy in ("fixed_eq", "fixed_slope") else None
    last = -1
    for t in range(T):
        ok = np.array([len(h) < P for h in hist])
        if tgt is not None:
            ok &= np.array([len(h) < g for h, g in zip(hist, tgt)])
        if not ok.any():
            break
        g = snr[:, t]
        if policy == "adaptive_slope":
            below = ok & (np.array([len(h) for h in hist]) < replan(belief, imgs, means, hist, T - t))
            if below.any():
                ok = below
        if policy == "rr":
            order = [(last + 1 + j) % K for j in range(K)]
            k = next(j for j in order if ok[j])
        elif policy == "maxsnr":
            k = int(np.argmax(np.where(ok, g, -np.inf)))
        elif policy in ("pf", "fixed_eq", "fixed_slope", "adaptive_slope"):
            r = rate(g)
            k = int(np.argmax(np.where(ok, r / (served + 1e-3 * r.mean()), -np.inf)))
        else:
            gain = np.full(K, -np.inf)
            for j in np.flatnonzero(ok):
                now = belief.psnr(imgs[j], hist[j])
                gain[j] = belief.psnr(imgs[j], hist[j] + [float(g[j])]) - now
                if policy == "greedy_rel":          # tilted by the gain at the mean SNR
                    ref = belief.psnr(imgs[j], hist[j] + [float(means[j])]) - now
                    gain[j] *= gain[j] / max(ref, 1e-9)
                elif alpha:
                    gain[j] *= max(now, 1e-3) ** (-alpha)
            k = int(np.argmax(gain))
        hist[k].append(float(g[k]))
        served = (1 - 1.0 / T) * served
        served[k] += rate(g[k]) / T
        last = k
    return hist


def bootstrap(d, n=2000, seed=0):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def simulate(util, users, slots, frames, mean_snr, fading, clip, policies, seed=0,
             same_image=False, pred_noise=0.0, alpha=0.0, dump=0):
    if users > len(util.names) and not same_image:
        raise ValueError(f"{users} users but only {len(util.names)} images in the pool")
    rng = np.random.default_rng(seed)
    res = {p: {"psnr": [], "frame_mean": [], "frame_min": [], "deep": [], "served": []}
           for p in policies}
    dumped = []
    for f in range(frames):
        imgs = (np.full(users, rng.integers(len(util.names))) if same_image
                else rng.choice(len(util.names), users, replace=False))
        means = rng.uniform(mean_snr[0], mean_snr[1], users)
        fade = 10 * np.log10(rng.exponential(1.0, (users, slots))) if fading == "rayleigh" \
            else np.zeros((users, slots))
        snr = np.clip(means[:, None] + fade, clip[0], clip[1])
        belief = util.perturbed(pred_noise, rng)
        for p in policies:
            hist = run_frame(util, belief, imgs, means, snr, p, alpha)
            q = np.array([util.psnr(i, h) for i, h in zip(imgs, hist)])
            r = res[p]
            r["psnr"] += q.tolist()
            r["frame_mean"].append(float(q.mean()))
            r["frame_min"].append(float(q.min()))
            sent = [s for h in hist for s in h]
            r["deep"].append(float(np.mean(np.array(sent) < 0.0)) if sent else 0.0)
            r["served"].append(float(np.mean([len(h) for h in hist])))
            if f < dump:
                for k, (i, h) in enumerate(zip(imgs, hist)):
                    dumped.append({"frame": f, "policy": p, "user": k,
                                   "image": util.names[int(i)], "mean_snr": float(means[k]),
                                   "snrs": h, "predicted_psnr": float(q[k])})
    base = "pf" if "pf" in policies else policies[0]
    summary = {}
    for p in policies:
        r = res[p]
        d = np.array(r["frame_mean"]) - np.array(res[base]["frame_mean"])
        lo, hi = bootstrap(d, seed=seed)
        summary[p] = {"mean_psnr": float(np.mean(r["psnr"])),
                      "p5_psnr": float(np.percentile(r["psnr"], 5)),
                      "worst_user": float(np.mean(r["frame_min"])),
                      "deep_fade_share": float(np.mean(r["deep"])),
                      "phases_per_user": float(np.mean(r["served"])),
                      f"vs_{base}": float(d.mean()), f"vs_{base}_ci": [lo, hi]}
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
    ap.add_argument("--snr-clip", nargs=2, type=float, default=[-2.0, 22.0],
                    help="slot SNRs clipped to the codec's training range")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    ap.add_argument("--same-image", action="store_true", help="every user the same image")
    ap.add_argument("--pred-noise", type=float, default=0.0,
                    help="sd of the log-normal error of the transmitter's curves")
    ap.add_argument("--alpha", type=float, default=0.0, help="greedy: alpha-fair weighting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", type=int, default=0, help="histories of the first N frames")
    ap.add_argument("--dump-out", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    util = Utility.load(a.utility)
    summary, base, dumped = simulate(util, a.users, a.slots, a.frames, a.mean_snr, a.fading,
                                     a.snr_clip, a.policies, a.seed, a.same_image,
                                     a.pred_noise, a.alpha, a.dump)
    print(f"{a.users} users, {a.slots} slots ({a.slots / a.users:.2f} phases per user, at most "
          f"{util.P}), {a.frames} frames, mean SNR U{a.mean_snr} dB, fading {a.fading}"
          f"{', same image' if a.same_image else ''}"
          f"{f', prediction noise {a.pred_noise:g}' if a.pred_noise else ''}"
          f"{f', alpha {a.alpha:g}' if a.alpha else ''}; utility {a.utility}")
    print(f"{'policy':15s} {'mean PSNR':>9s} {'p5 user':>8s} {'worst':>7s} {'<0 dB':>6s} "
          f"{'phases':>6s}   {'vs ' + base + ' [95% CI]':>24s}")
    for p, s in summary.items():
        lo, hi = s[f"vs_{base}_ci"]
        print(f"{p:15s} {s['mean_psnr']:9.3f} {s['p5_psnr']:8.3f} {s['worst_user']:7.3f} "
              f"{100 * s['deep_fade_share']:5.1f}% {s['phases_per_user']:6.2f}   "
              f"{s[f'vs_{base}']:+8.3f} [{lo:+.3f}, {hi:+.3f}]")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "summary": summary}, f, indent=1)
        print(f"wrote {a.out}")
    if a.dump:
        path = a.dump_out or os.path.splitext(a.out or "sched")[0] + "_histories.json"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"chunking": "phase", "utility": a.utility, "entries": dumped}, f)
        print(f"wrote {path}: {len(dumped)} delivery histories")


if __name__ == "__main__":
    main()
