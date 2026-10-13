"""Re-reads decoded replays (tools/mixed_snr.py --design file --out) on the CPU and pairs
policies by frame, also across files: two codecs decoding the schedules of the same frames
(tools/schedule_sim.py dumps with the same --seed, --frames, --users and image pool draw
the same images, mean SNRs and fades for every utility file). Seconds, no GPU.

    python tools/replay_report.py results/s12_dec_percbr_tgt.json results/s12_dec_prefix_tgt.json \\
        --tags percbr prefix --base prefix:online

Per policy (tag:policy with --tags): the model's error on the decoded histories, the decoded
mean PSNR, and with a target the share of users at or above it; then every policy minus
--base per frame (mean PSNR, or the share at the target), 95% bootstrap CI over frames. A
user who received nothing scores as the empty prefix the model predicts. Files are paired
only where every frame's users had the same images and mean SNRs in both.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from tools.schedule_sim import bootstrap  # noqa: E402


def score(e):
    a = e.get("actual_psnr")
    return e.get("predicted_psnr", 0.0) if a is None else a


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="decoded replays (mixed_snr.py --design file --out)")
    ap.add_argument("--tags", nargs="+", default=None, help="one name per file (default: none)")
    ap.add_argument("--base", default=None,
                    help="the policy (tag:policy) the others are paired against; default: the "
                         "first of online / padc / pf / fixed_uni present")
    ap.add_argument("--target", type=float, default=None, help="default: the files' target")
    a = ap.parse_args()
    if a.tags and len(a.tags) != len(a.files):
        ap.error("--tags needs one name per file")
    entries, targets, frames_of = [], set(), {}
    for n, path in enumerate(a.files):
        with open(path) as f:
            spec = json.load(f)
        targets.add(spec.get("target"))
        tag = a.tags[n] if a.tags else None
        for e in spec["entries"]:
            if "frame" not in e or "user" not in e:
                raise SystemExit(f"{path}: not a schedule_sim dump (no frame / user)")
            e = dict(e, policy=f"{tag}:{e.get('policy', '?')}" if tag else e.get("policy", "?"))
            entries.append(e)
            key = (e["frame"], e["user"])
            seen = frames_of.setdefault(key, (e["image"], round(float(e["mean_snr"]), 6)))
            if seen != (e["image"], round(float(e["mean_snr"]), 6)):
                raise SystemExit(f"{path}: frame {e['frame']} user {e['user']} differs from the "
                                 f"other files (image / mean SNR): not the same frames")
    tgt = a.target if a.target is not None else (targets.pop() if len(targets) == 1 else None)
    if a.target is None and len(targets) > 1:
        raise SystemExit(f"the files have different targets {sorted(map(str, targets))}: pass --target")
    pols = list(dict.fromkeys(e["policy"] for e in entries))
    rows = [e for e in entries if e.get("actual_psnr") is not None and "predicted_psnr" in e]
    if rows:
        err = np.array([e["predicted_psnr"] - e["actual_psnr"] for e in rows])
        print(f"{len(rows)} decoded histories: predicted - actual PSNR bias {err.mean():+.3f}, "
              f"MAE {np.abs(err).mean():.3f}, p95 {np.percentile(np.abs(err), 95):.3f} dB")
    print(f"{'policy':24s} {'users':>5s} {'decoded mean PSNR':>17s}"
          + (f" {'at ' + format(tgt, 'g') + ' dB':>9s} {'predicted':>9s}" if tgt is not None else ""))
    for p in pols:
        sel = [e for e in entries if e["policy"] == p]
        q = np.array([score(e) for e in sel])
        line = f"{p:24s} {len(sel):5d} {q.mean():17.3f}"
        if tgt is not None:
            pre = np.array([e.get("predicted_psnr", 0.0) >= tgt for e in sel])
            line += f" {100 * (q >= tgt).mean():8.1f}% {100 * pre.mean():8.1f}%"
        print(line)
    base = a.base
    if base is None:
        for b in ("online", "padc", "pf", "fixed_uni"):
            hit = [p for p in pols if p == b or p.endswith(":" + b)]
            if hit:
                base = hit[0]
                break
    if base not in pols:
        raise SystemExit(f"--base {base!r} is not among the policies {pols}")
    per = {}
    for e in entries:
        v = float(score(e) >= tgt) if tgt is not None else float(score(e))
        per.setdefault(e["policy"], {}).setdefault(e["frame"], []).append(v)
    what = "share reaching the target" if tgt is not None else "mean PSNR"
    scale, unit = (100.0, " pts") if tgt is not None else (1.0, " dB")
    print(f"decoded, paired by frame: {what} minus {base}'s, 95% CI over frames")
    for p in pols:
        if p == base:
            continue
        common = sorted(set(per[p]) & set(per[base]))
        if not common:
            continue
        d = np.array([np.mean(per[p][f]) - np.mean(per[base][f]) for f in common])
        lo, hi = bootstrap(d)
        print(f"  {p:24s} {scale * d.mean():+7.3f}{unit} [{scale * lo:+.3f}, {scale * hi:+.3f}]  "
              f"({len(common)} frames)")


if __name__ == "__main__":
    main()
