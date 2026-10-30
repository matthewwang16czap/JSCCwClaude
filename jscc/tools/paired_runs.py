"""Paired comparison of tools/schedule_sim.py runs on the same frames. CPU, seconds.

Two schedule_sim runs with the same seed, frame count, users, slots and SNR range draw the same
images, mean SNRs and fades, whatever their utility (a learned code or a digital codec), their
policies or their CSI condition (common random numbers: tools/schedule_sim.draw_frames). Run
each with --keep-frames --out FILE: the JSON then holds every policy's per-frame objective
(the frame's mean PSNR, or its share of users at the target). This tool pairs two runs frame
by frame, prints the mean difference with a 95% bootstrap interval over frames, and writes it
as one JSON line with --log.

    python tools/paired_runs.py --label "2 dB CQI error, mean PSNR" \\
        --a results/s30_j_mean_e2.json:lagrange \\
        --b results/s30_d_mean_e2.json:pf,greedy

A side is one or more FILE:POLICY[,POLICY...] entries. A side with several candidates (policies,
or files such as a naive and an aware transmitter) takes the one with the best mean objective,
the best policy and transmitter at each point as the C5 tables do, and says which it took.

It refuses runs that do not share their frames or their objective. "Frames" means the seed,
frame count, users, slots, mean SNR, fading, SNR clip, slot correlation, preemption, same-image
and prediction noise, plus the image names. "Objective" means mean PSNR, or the same target.
The CSI conditions (--cqi-error, --cqi-corr, --cqi-aware) may differ, because the true slots
are the same at any of them: fresh minus stale reports is a paired loss.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from tools.schedule_sim import bootstrap  # noqa: E402

FRAME_KEYS = ("seed", "frames", "users", "slots", "mean_snr", "fading", "snr_clip", "fade_corr",
              "preempt", "same_image", "pred_noise")


def load_run(path):
    with open(path) as f:
        d = json.load(f)
    if "args" not in d or "summary" not in d:
        raise ValueError(f"{path}: not a tools/schedule_sim.py --out file")
    return d


def condition(d):
    """The run's CSI condition and system, for the printout."""
    a = d["args"]
    system = (f"digital {a.get('link') or 'cqi'}" + (f" + HARQ {a['harq']}" if a.get("harq") else "")
              if d.get("kind") == "digital" else "learned code")
    csi = []
    if d.get("cqi_error"):
        csi.append(f"CQI error {d['cqi_error']:g} dB")
    if d.get("cqi_corr", 1.0) < 1.0:
        csi.append(f"r {d['cqi_corr']:g} " + ("aware" if a.get("cqi_aware") else "naive"))
    if a.get("target") is not None and a.get("margin"):
        csi.append(f"margin {a['margin']:g} dB")
    return ", ".join([system] + (csi or ["fresh, exact reports"]))


def check_same_frames(runs):
    """Refuse runs that do not share their frames and objective (ValueError)."""
    (p0, d0), rest = runs[0], runs[1:]
    for p, d in rest:
        for k in FRAME_KEYS:
            if d["args"].get(k) != d0["args"].get(k):
                raise ValueError(f"{p} and {p0} differ in {k} ({d['args'].get(k)} vs "
                                 f"{d0['args'].get(k)}): not the same frames")
        if d["args"].get("target") != d0["args"].get("target"):
            raise ValueError(f"{p} and {p0} score different objectives (target "
                             f"{d['args'].get('target')} vs {d0['args'].get('target')})")
        if "names" in d and "names" in d0 and d["names"] != d0["names"]:
            raise ValueError(f"{p} and {p0} hold different images (or a different order): "
                             f"not the same frames")


def candidates(entries):
    """[(path, policy, per-frame objective)] from FILE:POL[,POL...] entries."""
    out = []
    for e in entries:
        path, sep, pols = e.rpartition(":")
        if not sep or not path:
            raise ValueError(f"{e}: expected FILE:POLICY[,POLICY...]")
        d = load_run(path)
        for p in pols.split(","):
            s = d["summary"].get(p)
            if s is None:
                raise ValueError(f"{path}: no policy {p} (it has {', '.join(d['summary'])})")
            if "frames" not in s:
                raise ValueError(f"{path}: no per-frame values; run tools/schedule_sim.py with "
                                 f"--keep-frames --out")
            out.append((path, p, np.asarray(s["frames"], dtype=np.float64), d))
    return out


def best(cands):
    return max(cands, key=lambda c: float(c[2].mean()))


def compare(a_entries, b_entries, seed=0):
    """The paired difference best(a) - best(b) over frames: a dict."""
    ca, cb = candidates(a_entries), candidates(b_entries)
    runs = [(c[0], c[3]) for c in ca + cb]
    check_same_frames(runs)
    pa, pb = best(ca), best(cb)
    if len(pa[2]) != len(pb[2]):
        raise ValueError(f"{pa[0]} has {len(pa[2])} frames, {pb[0]} {len(pb[2])}")
    d = pa[2] - pb[2]
    lo, hi = bootstrap(d, seed=seed)
    target = pa[3]["args"].get("target")
    return {"objective": "mean PSNR (dB)" if target is None else f"share at {target:g} dB",
            "target": target, "frames": int(len(d)),
            "a": {"file": pa[0], "policy": pa[1], "value": float(pa[2].mean()),
                  "condition": condition(pa[3]), "of": [c[1] for c in ca]},
            "b": {"file": pb[0], "policy": pb[1], "value": float(pb[2].mean()),
                  "condition": condition(pb[3]), "of": [c[1] for c in cb]},
            "diff": float(d.mean()), "ci": [lo, hi]}


def show(label, r):
    pct = r["target"] is not None
    unit, k = ("pts", 100.0) if pct else ("dB", 1.0)
    val = (lambda v: f"{100 * v:6.1f}%") if pct else (lambda v: f"{v:7.3f} dB")
    print(f"{label}  ({r['objective']}, {r['frames']} frames, paired by frame)")
    for side in ("a", "b"):
        s = r[side]
        note = f" (best of {', '.join(s['of'])})" if len(s["of"]) > 1 else ""
        print(f"  {side}: {s['policy']:14s} {val(s['value'])}   {s['condition']}   "
              f"[{os.path.basename(s['file'])}]{note}")
    lo, hi = r["ci"]
    print(f"  a - b: {k * r['diff']:+.3f} {unit} [{k * lo:+.3f}, {k * hi:+.3f}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", nargs="+", required=True, help="FILE:POLICY[,POLICY...] (best taken)")
    ap.add_argument("--b", nargs="+", required=True, help="FILE:POLICY[,POLICY...] (best taken)")
    ap.add_argument("--label", default="")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    ap.add_argument("--log", default=None, help="append the result as one JSON line to this file")
    a = ap.parse_args()
    try:
        r = compare(a.a, a.b, seed=a.seed)
    except (ValueError, OSError) as e:
        ap.error(str(e))
    show(a.label, r)
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps({"label": a.label, **r}) + "\n")


if __name__ == "__main__":
    main()
