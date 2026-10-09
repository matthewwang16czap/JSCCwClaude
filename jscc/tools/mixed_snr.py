"""Decoding a token prefix whose chunks arrived at different SNRs.

The scheduling problem of docs/PROBLEM.md delivers each image's token prefix in
chunks, one per slot of a block-fading channel. With receiver CSI and
equalisation every slot is an AWGN channel at its own SNR, so a prefix has a
piecewise SNR along its tokens. This tool measures what a prefix token model
makes of that, on the test set (a few GPU minutes, no training):

    python tools/mixed_snr.py --backbone vit --pretrained <ckpt> --channel-type awgn --amp \\
        --testset Kodak --snrs 1 4 7 10 13 --cbrs 1/24 1/12 1/8 --chunks 2 \\
        --out results/mixed_vit.json

Per CBR the prefix is cut into --chunks equal parts by token index (chunk 1 =
the first tokens) and every combination of chunk SNRs from --snrs is decoded.
The constant profiles give each image's reference curve PSNR(SNR). For every
mixed profile the tables give its PSNR, the constant SNR with the same mean
PSNR (the equivalent SNR) and the error of three rules that predict it from the
chunk SNRs alone, on each image's own curve:

    mean_db    the mean chunk SNR in dB
    noise      the SNR of the mean noise variance
    capacity   the SNR whose capacity is the mean chunk capacity log2(1 + snr)

and the ORDER effect: the same chunk SNRs, the best ones first minus last. A
rule within ~0.1-0.2 dB lets a scheduler work from the constant-SNR curves of
tools/alloc_sweep.py. Every profile of an image and CBR uses the same noise
draw, scaled per chunk (common random numbers).
"""

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from configs.config import Config, parse_cbr  # noqa: E402
from data.datasets import sweep_dataset  # noqa: E402
from engine import autocast, reseed  # noqa: E402
from net.channel import piecewise_snr  # noqa: E402
from net.network import JSCC  # noqa: E402
from utils.common import load_weights  # noqa: E402
from utils.metrics import MetricSuite  # noqa: E402
from utils.parser import create_parser  # noqa: E402

RULES = ("mean_db", "noise", "capacity")


def effective_snr(snrs_db, weights, rule):
    """One SNR (dB) standing for chunks at `snrs_db` with relative sizes `weights`."""
    s = np.asarray(snrs_db, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    if rule == "mean_db":
        return float((w * s).sum())
    lin = 10.0 ** (s / 10.0)
    if rule == "noise":
        return float(-10.0 * np.log10((w / lin).sum()))
    if rule == "capacity":
        return float(10.0 * np.log10(2.0 ** (w * np.log2(1.0 + lin)).sum() - 1.0))
    raise ValueError(f"unknown rule {rule!r}")


def equivalent_snr(p, curve_snr, curve_psnr):
    """The constant SNR at which a constant-SNR curve reaches PSNR p (NaN outside it)."""
    order = np.argsort(curve_psnr)
    x, y = np.asarray(curve_psnr, dtype=np.float64)[order], np.asarray(curve_snr)[order]
    if not x[0] <= p <= x[-1]:
        return float("nan")
    return float(np.interp(p, x, y))


def analyse(scores, profiles, grid, sizes):
    """scores (N images, P profiles) at one CBR -> printable summary + dict."""
    const = {pr[0]: j for j, pr in enumerate(profiles) if len(set(pr)) == 1}
    curve = np.stack([scores[:, const[s]] for s in grid], 1)          # (N, S)
    mean_curve = curve.mean(0)
    mixed = [j for j, pr in enumerate(profiles) if len(set(pr)) > 1]
    rows, err = [], {r: [] for r in RULES}
    for j in mixed:
        pr = profiles[j]
        actual = scores[:, j]
        row = {"snrs": list(pr), "psnr": float(actual.mean()),
               "equivalent_snr": equivalent_snr(actual.mean(), grid, mean_curve)}
        for r in RULES:
            s = effective_snr(pr, sizes, r)
            pred = np.array([np.interp(s, grid, curve[i]) for i in range(len(actual))])
            e = pred - actual
            err[r].append(e)
            row[r] = {"snr": s, "error": float(e.mean())}
        rows.append(row)
    rules = {}
    for r in RULES:
        e = np.concatenate(err[r]) if err[r] else np.zeros(0)
        rules[r] = {"bias": float(e.mean()), "mae": float(np.abs(e).mean()),
                    "p95": float(np.percentile(np.abs(e), 95))} if len(e) else {}
    order = []
    for combo in {tuple(sorted(pr)) for pr in profiles if len(set(pr)) > 1}:
        hi = profiles.index(tuple(sorted(combo, reverse=True)))
        lo = profiles.index(tuple(sorted(combo)))
        order.append(scores[:, hi] - scores[:, lo])
    order = np.stack(order) if order else np.zeros((0, scores.shape[0]))
    return {"constant": {"snrs": list(grid), "psnr": mean_curve.tolist()}, "mixed": rows,
            "rules": rules,
            "order": {"best_first_minus_last": float(order.mean()) if order.size else None,
                      "share_best_first_wins": float((order > 0).mean()) if order.size else None}}


def add_args(parser):
    g = parser.add_argument_group("mixed-SNR prefixes")
    g.add_argument("--cbrs", nargs="+", default=None, help="default: the predefined CBRs")
    g.add_argument("--chunks", type=int, default=2, help="equal chunks per prefix, in token order")
    g.add_argument("--max-images", type=int, default=None)
    g.add_argument("--out", default=None, help="JSON: every profile's per-image PSNR + summary")
    return parser


def main():
    args = add_args(create_parser()).parse_args()
    cfg = Config(args)
    if not cfg.pretrained:
        raise SystemExit("--pretrained is required: this scores a trained codec")
    if not cfg.token or cfg.cascade:
        raise SystemExit("mixed-SNR prefixes need a token model under the prefix scheme")
    if cfg.channel_type != "awgn":
        raise SystemExit("use --channel-type awgn: a chunk is a slot after equalisation, an "
                         "AWGN channel at the slot's SNR")
    if args.chunks < 2:
        raise SystemExit("--chunks must be >= 2")
    torch.manual_seed(cfg.seed)
    model = JSCC(cfg).to(cfg.device).eval()
    rep = load_weights(model, cfg.pretrained)
    print(f"loaded {cfg.pretrained}: {len(rep.missing_keys)} missing, "
          f"{len(rep.unexpected_keys)} unexpected")
    ds = sweep_dataset(cfg, "test", max_images=args.max_images)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    suite = MetricSuite(["psnr"])
    cbrs = [parse_cbr(c) for c in args.cbrs] if args.cbrs else list(cfg.cbrs)
    units = [cfg.units_for_cbr(c, exact=False) for c in cbrs]
    grid = [float(s) for s in cfg.snrs]
    profiles = list(itertools.product(grid, repeat=args.chunks))
    cuts = [[round(u * j / args.chunks) for j in range(1, args.chunks)] for u in units]
    print(f"{cfg.run_name}: {len(ds)} images, CBRs {[str(c) for c in cbrs]}, {args.chunks} "
          f"chunks, {len(profiles)} SNR profiles per CBR from {grid} dB")
    dev = cfg.device
    scores = np.full((len(ds), len(cbrs), len(profiles)), np.nan)
    with torch.no_grad():
        for k, x in enumerate(loader):
            x = x.to(dev, non_blocking=True)
            with autocast(cfg, dev):
                enc = model.encode(x)
                for ci, u in enumerate(units):
                    cut = torch.tensor([cuts[ci]], device=dev, dtype=torch.long)
                    for pi, pr in enumerate(profiles):
                        reseed(cfg, cfg.eval_seed + 100_003 * k + 1000 * ci)
                        snr = piecewise_snr(torch.tensor([pr], device=dev), cut, cfg.n_tokens)
                        out = model.decode(enc, model.send(enc, u, snr), u)
                        scores[k, ci, pi] = float(suite(out, x, keys=[k])["psnr"][0])
            if (k + 1) % 6 == 0 or k + 1 == len(ds):
                print(f"  {k + 1}/{len(ds)} images", flush=True)
    summary = {}
    for ci, (c, u) in enumerate(zip(cbrs, units)):
        sizes = np.diff([0] + cuts[ci] + [u])
        res = analyse(scores[:, ci], profiles, grid, sizes)
        summary[str(c)] = res
        print(f"\nCBR {c}: {u} tokens per tile in chunks of {sizes.tolist()} (token order)")
        print("  constant SNR " + "  ".join(f"{s:g} dB {p:.3f}" for s, p in
                                             zip(res["constant"]["snrs"], res["constant"]["psnr"])))
        print(f"  {'chunk SNRs':>18s} {'PSNR':>8s} {'equiv SNR':>9s}  " +
              "  ".join(f"{r + ' err':>14s}" for r in RULES))
        for row in res["mixed"]:
            print(f"  {str(row['snrs']):>18s} {row['psnr']:8.3f} {row['equivalent_snr']:9.2f}  " +
                  "  ".join(f"{row[r]['error']:+14.3f}" for r in RULES))
        print("  rules (predicted - actual PSNR, per image, over mixed profiles): " +
              "; ".join(f"{r} bias {v['bias']:+.3f} MAE {v['mae']:.3f} p95 {v['p95']:.3f}"
                        for r, v in res["rules"].items()))
        o = res["order"]
        print(f"  order: best chunks first minus last {o['best_first_minus_last']:+.3f} dB, "
              f"best-first better for {100 * o['share_best_first_wins']:.0f}% of (image, set)")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"checkpoint": cfg.pretrained, "run_name": cfg.run_name,
                       "cbrs": [str(c) for c in cbrs], "units": units, "cuts": cuts,
                       "profiles": [list(pr) for pr in profiles], "images": list(ds.names),
                       "psnr": scores.tolist(), "summary": summary}, f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
