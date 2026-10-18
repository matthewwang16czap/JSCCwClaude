"""The channel a learned code sees under a scheduler, as a bank of SNR histories that
training samples (main.py --snr-profiles; docs/PROBLEM.md, section 7). CPU, about a minute.

    python tools/codesign.py bank --utility results/utility_snrc4.json \\
        --belief results/belief_snrc4.json --policy slope_adv --frames 4000 --out results/bank_sa.json
    python tools/codesign.py bank ... --blind --out results/bank_blind.json
    python tools/codesign.py stats results/bank_sa.json results/bank_blind.json
    python tools/codesign.py stats results/bank_sa_r1.json --against results/bank_sa.json

In a multi-user downlink a code's prefix does not travel over "the channel": every phase
goes out in the slot the scheduler picked for it, so the SNR each phase of the code sees,
and how many phases a user gets, are made by the scheduler (an opportunistic one puts the
first, most sensitive phases into fading peaks). A code trained on i.i.d. chunk SNRs
(--snr-chunks) is trained for a different channel. `bank` runs tools/schedule_sim.py's
frames under one policy and keeps, for every user of every frame, the true SNR of each phase
it received, in prefix order (what its decoder saw), with its mean SNR and image; users
that received nothing are kept for the statistics and skipped by training. Use a --seed
other than the evaluation's (default 1000; schedule_sim.py uses 0), so the frames a code
is trained on are not the frames it is scored on.

--blind keeps every history's length and user and redraws each phase's SNR as a slot taken
without looking at the channel: the user's mean SNR plus a fresh Rayleigh fade. That is the
physical channel at the scheduler's lengths, the control that separates what the
scheduler's slot SELECTION does to the code's channel from what its lengths and the users'
mean SNRs do. --shuffle-phases keeps every history's slots and permutes which phase went out
in which: the same SNR levels per user, but the first phase no longer gets the best slot
(the control for the code using WHICH phase the scheduler favours).

`stats` prints each bank's length distribution, per-phase SNR (mean, sd, lift over the
user's mean) and mean profile per length, next to what --snr-chunks training assumes.
--against B pairs users by frame (banks drawn with the same frames, e.g. the same policy
and seed on the utility of a retrained code): how much the scheduler's channel moved,
the convergence check of code-scheduler iteration.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import load_utility  # noqa: E402
from tools.schedule_sim import NEEDS_TARGET, POLICIES, fades, simulate  # noqa: E402


def build_bank(util, policy, users=8, slots=24, frames=4000, mean_snr=(0.0, 15.0),
               fading="rayleigh", clip=(-2.0, 22.0), seed=1000, belief=None, target=None,
               margin=0.0, offset=0.0, fade_corr=0.0, plan="mean", preempt=0.0, blind=False,
               shuffle=False, cqi_corr=1.0, cqi_aware=False):
    """Every user's delivered history under `policy` on `frames` frames: a bank (dict).
    cqi_corr < 1: the scheduler decides on stale reports (alloc/csi.py; cqi_aware: knowing
    it), the histories hold the slots' true SNRs."""
    summary, _, dumped = simulate(util, users, slots, frames, tuple(mean_snr), fading,
                                  tuple(clip), [policy], seed, dump=frames, target=target,
                                  margin=margin, offset=offset, belief_util=belief,
                                  fade_corr=fade_corr, plan=plan, preempt=preempt,
                                  cqi_corr=cqi_corr, cqi_aware=cqi_aware)
    hist = [[float(s) for s in e["snrs"]] for e in dumped]
    means = [float(e["mean_snr"]) for e in dumped]
    if blind:                                    # the same lengths, slots taken blindly
        rng = np.random.default_rng([seed, 2])
        hist = [np.clip(m + fades(rng, 1, len(h), fading)[0], clip[0], clip[1]).tolist()
                if h else [] for h, m in zip(hist, means)]
    if shuffle:                                  # the same slots, phases in random order
        rng = np.random.default_rng([seed, 3])
        hist = [[h[j] for j in rng.permutation(len(h))] for h in hist]
    return {"format": "snr-bank", "phases": int(util.P), "policy": policy, "target": target,
            "margin": margin, "blind": bool(blind), "shuffled": bool(shuffle), "seed": seed,
            "users": users,
            "slots": slots, "frames": frames, "mean_snr": list(mean_snr), "fading": fading,
            "fade_corr": fade_corr, "cqi_corr": cqi_corr, "cqi_aware": bool(cqi_aware),
            "snr_clip": list(clip), "summary": summary[policy],
            "frame": [int(e["frame"]) for e in dumped], "user": [int(e["user"]) for e in dumped],
            "image": [e["image"] for e in dumped], "user_mean_snr": means, "histories": hist}


def load_bank(path):
    with open(path) as f:
        bank = json.load(f)
    if bank.get("format") != "snr-bank":
        raise SystemExit(f"{path}: not an SNR bank (tools/codesign.py bank)")
    return bank


def stats(bank):
    """Length distribution, per-phase SNR and lift, mean profile per length."""
    P = bank["phases"]
    hist = bank["histories"]
    means = np.asarray(bank["user_mean_snr"], dtype=np.float64)
    L = np.array([len(h) for h in hist])
    served = L > 0
    out = {"users": len(hist), "empty": float(np.mean(~served)),
           "lengths": [float(np.mean(L[served] == n)) if served.any() else 0.0
                       for n in range(1, P + 1)],
           "mean_length": float(L[served].mean()) if served.any() else 0.0, "phase": [],
           "by_length": {}}
    for j in range(P):
        s = np.array([h[j] for h in hist if len(h) > j])
        m = np.array([mu for h, mu in zip(hist, means) if len(h) > j])
        out["phase"].append({"n": int(len(s)), "mean": float(s.mean()) if len(s) else None,
                             "sd": float(s.std()) if len(s) else None,
                             "lift": float((s - m).mean()) if len(s) else None})
    for n in range(1, P + 1):
        rows = np.array([h for h in hist if len(h) == n])
        if len(rows):
            m = means[L == n]
            out["by_length"][n] = {"share": float(len(rows) / max(served.sum(), 1)),
                                   "user_mean": float(m.mean()),
                                   "profile": rows.mean(0).tolist()}
    return out


def print_stats(name, bank, s):
    P = bank["phases"]
    what = (f"{bank['policy']}" + (f" at {bank['target']:g} dB" if bank.get("target") is not None
                                   else "") + (", slots taken blindly" if bank["blind"] else "")
            + (", phase order shuffled" if bank.get("shuffled") else "")
            + (f", stale reports (correlation {bank['cqi_corr']:g}"
               + (", scheduler aware)" if bank.get("cqi_aware") else ")")
               if bank.get("cqi_corr", 1.0) < 1.0 else ""))
    print(f"{name}: {what}; {s['users']} users ({bank['frames']} frames x {bank['users']}, "
          f"{bank['slots']} slots), {100 * s['empty']:.1f}% received nothing")
    print("  phases received (users with any): " + "  ".join(
        f"{n}: {100 * v:4.1f}%" for n, v in enumerate(s["lengths"], 1))
        + f"   mean {s['mean_length']:.2f}")
    print(f"  {'phase':>5s} " + " ".join(f"{j:>6d}" for j in range(1, P + 1)))
    for key, label in (("mean", "SNR"), ("sd", "sd"), ("lift", "lift")):
        print(f"  {label:>5s} " + " ".join(
            f"{p[key]:6.2f}" if p[key] is not None else f"{'-':>6s}" for p in s["phase"]))
    print("  mean SNR of each phase by the number received (the decoder knows that number; "
          "user mean SNR in brackets):")
    for n, r in sorted(s["by_length"].items()):
        print(f"    {n} ({100 * r['share']:4.1f}%, [{r['user_mean']:5.2f}]) "
              + " ".join(f"{v:6.2f}" for v in r["profile"]))


def against(a, b):
    """Users paired by (frame, user) across two banks drawn on the same frames."""
    ka = {(f, u): (h, m) for f, u, h, m in zip(a["frame"], a["user"], a["histories"],
                                                a["user_mean_snr"])}
    kb = {(f, u): (h, m) for f, u, h, m in zip(b["frame"], b["user"], b["histories"],
                                                b["user_mean_snr"])}
    common = sorted(set(ka) & set(kb))
    if not common or any(abs(ka[k][1] - kb[k][1]) > 1e-6 for k in common):
        return None
    La = np.array([len(ka[k][0]) for k in common])
    Lb = np.array([len(kb[k][0]) for k in common])
    same = [k for k in common if len(ka[k][0]) == len(kb[k][0]) > 0]
    d = [np.abs(np.array(ka[k][0]) - np.array(kb[k][0])) for k in same]
    moved = float(np.mean([np.mean(x > 1e-9) for x in d])) if d else None
    P = max(a["phases"], b["phases"])
    ha = np.bincount(La, minlength=P + 1) / len(La)
    hb = np.bincount(Lb, minlength=P + 1) / len(Lb)
    return {"users": len(common), "same_length": float(np.mean(La == Lb)),
            "mean_abs_length_change": float(np.abs(La - Lb).mean()),
            "length_tv": float(0.5 * np.abs(ha - hb).sum()),
            "phases_moved": moved}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank", help="the histories one policy delivers")
    b.add_argument("--utility", required=True, help="the code's utility (tools/utility_fit.py)")
    b.add_argument("--belief", default=None, help="the transmitter's curves (default: the truth)")
    b.add_argument("--policy", default="slope_adv", choices=POLICIES)
    b.add_argument("--target", type=float, default=None)
    b.add_argument("--margin", type=float, default=0.0)
    b.add_argument("--plan-offset", type=float, default=0.0)
    b.add_argument("--users", type=int, default=8)
    b.add_argument("--slots", type=int, default=24)
    b.add_argument("--frames", type=int, default=4000)
    b.add_argument("--mean-snr", nargs=2, type=float, default=[0.0, 15.0])
    b.add_argument("--fading", default="rayleigh", choices=["rayleigh", "none"])
    b.add_argument("--fade-corr", type=float, default=0.0)
    b.add_argument("--plan", default="mean", choices=["mean", "conditional"])
    b.add_argument("--preempt", type=float, default=0.0)
    b.add_argument("--cqi-corr", type=float, default=1.0,
                   help="the scheduler decides on stale reports of this correlation with the "
                        "slots (alloc/csi.py); the bank holds the true SNRs")
    b.add_argument("--cqi-aware", action="store_true",
                   help="--cqi-corr < 1: the scheduler knows its reports are stale")
    b.add_argument("--snr-clip", nargs=2, type=float, default=[-2.0, 22.0])
    b.add_argument("--seed", type=int, default=1000,
                   help="frames to train on: not the evaluation's (schedule_sim.py: 0)")
    b.add_argument("--blind", action="store_true",
                   help="keep the lengths, redraw every phase's SNR as a blindly taken slot")
    b.add_argument("--shuffle-phases", action="store_true",
                   help="keep every history's slots, permute which phase went out in which")
    b.add_argument("--out", required=True)
    s = sub.add_parser("stats", help="what the banks look like")
    s.add_argument("banks", nargs="+")
    s.add_argument("--against", default=None, help="pair every bank's users with this one's")
    s.add_argument("--snr-range", nargs=2, type=float, default=[-2.0, 22.0],
                   help="the --snr-range of --snr-chunks training, for the reference line")
    a = ap.parse_args()
    if a.cmd == "bank":
        if a.policy in NEEDS_TARGET and a.target is None:
            ap.error(f"--policy {a.policy} needs --target")
        if not 0.0 <= a.cqi_corr <= 1.0:
            ap.error("--cqi-corr is a correlation in [0, 1]")
        util = load_utility(a.utility)
        if getattr(util, "kind", "") == "digital":
            ap.error("a bank describes a token prefix code: --utility from tools/utility_fit.py")
        belief = load_utility(a.belief) if a.belief else None
        bank = build_bank(util, a.policy, a.users, a.slots, a.frames, a.mean_snr, a.fading,
                          a.snr_clip, a.seed, belief, a.target, a.margin, a.plan_offset,
                          a.fade_corr, a.plan, a.preempt, a.blind, a.shuffle_phases,
                          a.cqi_corr, a.cqi_aware)
        bank.update(utility=a.utility, belief=a.belief)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(bank, f)
        print_stats(os.path.basename(a.out), bank, stats(bank))
        sm = bank["summary"]
        print(f"  predicted on these frames: mean PSNR {sm['mean_psnr']:.3f} dB"
              + (f", {100 * sm['satisfied']:.1f}% at {a.target:g} dB" if a.target is not None
                 else ""))
        print(f"wrote {a.out}")
        return
    lo, hi = a.snr_range
    print(f"--snr-chunks training (the codes so far): every chunk i.i.d. uniform over [{lo:g}, "
          f"{hi:g}] dB (mean {(lo + hi) / 2:.2f}, sd {(hi - lo) / 12 ** 0.5:.2f}), cut at random "
          f"tokens; lengths uniform over CBR 1/48-1/8")
    ref = load_bank(a.against) if a.against else None
    for path in a.banks:
        bank = load_bank(path)
        print_stats(os.path.basename(path), bank, stats(bank))
        if ref is not None:
            r = against(bank, ref)
            if r is None:
                print(f"  against {os.path.basename(a.against)}: not the same frames (draw both "
                      f"with the same --seed, --frames, --users, --slots, --mean-snr)")
            else:
                moved = ("-" if r["phases_moved"] is None
                         else f"{100 * r['phases_moved']:.1f}%")
                print(f"  against {os.path.basename(a.against)}, the same {r['users']} users: "
                      f"{100 * r['same_length']:.1f}% receive as many phases (mean change "
                      f"{r['mean_abs_length_change']:.3f}, length distributions {r['length_tv']:.3f} "
                      f"apart in total variation); of those, {moved} of phases went out in "
                      f"another slot")


if __name__ == "__main__":
    main()
