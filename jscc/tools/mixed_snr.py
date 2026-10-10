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

Designs (--design):
    grid    every combination of chunk SNRs from --snrs (2-3 chunks): the tables above
    full    the data of the utility model (alloc/utility.py, tools/utility_fit.py): per CBR
            the constant profiles, every chunk alone off --base-snr (the others at it),
            and --random-profiles random ones (chunk SNRs drawn from --snrs) held out to
            validate on. Use with --chunks phase and the phase CBRs (1/48 ... 1/8)
    file    every entry of --profiles-json ({"entries": [{"image", "snrs"}]}, one SNR per
            phase in delivery order; tools/schedule_sim.py --dump writes it): decoded and
            scored, next to the PSNR the utility model predicted for it

--chunks K cuts a prefix into K equal parts; --chunks phase cuts it at phase
boundaries (a phase = one token per patch position: CBR 1/48 at 256 px), so a
prefix of L phases has L chunks.
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
    g.add_argument("--design", default="grid", choices=["grid", "full", "file"])
    g.add_argument("--cbrs", nargs="+", default=None, help="default: the predefined CBRs")
    g.add_argument("--chunks", default="2",
                   help="K (equal chunks, token order) or 'phase' (one chunk per phase)")
    g.add_argument("--base-snr", type=float, default=7.0,
                   help="full: the SNR of the chunks not being varied")
    g.add_argument("--random-profiles", type=int, default=10,
                   help="full: random profiles per CBR (validation)")
    g.add_argument("--profiles-json", default=None, help="file: the profiles to decode")
    g.add_argument("--profile-seed", type=int, default=0)
    g.add_argument("--max-images", type=int, default=None)
    g.add_argument("--out", default=None, help="JSON: every profile's per-image PSNR + summary")
    return parser


def chunk_cuts(units, chunks, per_phase):
    """Chunk boundaries (token positions) of a prefix of `units` tokens."""
    if chunks == "phase":
        return list(range(per_phase, units, per_phase))
    k = int(chunks)
    return [round(units * j / k) for j in range(1, k)]


def make_profiles(design, k, grid, base, n_random, rng):
    """(profiles, kinds) of one CBR with k chunks; constant profiles first."""
    out, kinds = [], []

    def add(pr, kind):
        pr = tuple(float(v) for v in pr)
        if pr not in out:
            out.append(pr)
            kinds.append(kind)

    for s in grid:
        add([s] * k, "constant")
    if design == "grid":
        for pr in itertools.product(grid, repeat=k):
            add(pr, "mixed")
    elif k > 1:
        for j in range(k):
            for s in grid:
                pr = [base] * k
                pr[j] = s
                add(pr, "oat")
        for _ in range(n_random):
            add(rng.choice(grid, size=k), "random")
    return out, kinds


def oat_table(scores, profiles, kinds, grid, base):
    """Per chunk: mean PSNR with that chunk at each grid SNR, the others at `base`."""
    k = len(profiles[0])
    idx = {pr: j for j, pr in enumerate(profiles)}
    rows = []
    for c in range(k):
        vals = []
        for s in grid:
            pr = [base] * k
            pr[c] = s
            j = idx.get(tuple(pr))
            vals.append(float(scores[:, j].mean()) if j is not None else float("nan"))
        rows.append(vals)
    return rows


def score_profiles(model, cfg, enc, x, k, units, cuts, profiles, suite, seed):
    """PSNR of one image under each SNR profile (same noise draw, scaled per chunk)."""
    dev = x.device
    cut = torch.tensor([cuts], device=dev, dtype=torch.long)
    out = []
    for pr in profiles:
        reseed(cfg, seed)
        snr = piecewise_snr(torch.tensor([pr], device=dev), cut, cfg.n_tokens)
        rec = model.decode(enc, model.send(enc, units, snr), units)
        out.append(float(suite(rec, x, keys=[k])["psnr"][0]))
    return out


def run_file(model, cfg, args, ds, loader, suite):
    with open(args.profiles_json) as f:
        spec = json.load(f)
    entries = spec["entries"]
    by_image = {}
    for e_i, e in enumerate(entries):
        by_image.setdefault(e["image"], []).append(e_i)
    missing = sorted(set(by_image) - set(ds.names))
    if missing:
        raise SystemExit(f"{len(missing)} images of {args.profiles_json} are not in the test set, "
                         f"e.g. {missing[:3]}")
    per = cfg.grid_tokens
    with torch.no_grad():
        for k, x in enumerate(loader):
            todo = by_image.get(ds.names[k], [])
            if not todo:
                continue
            x = x.to(cfg.device, non_blocking=True)
            with autocast(cfg, cfg.device):
                enc = model.encode(x)
                for e_i in todo:
                    snrs = [float(v) for v in entries[e_i]["snrs"]]
                    if not snrs:
                        entries[e_i]["actual_psnr"] = None
                        continue
                    if len(snrs) > cfg.phases:
                        raise SystemExit(f"entry {e_i}: {len(snrs)} phases, the model has "
                                         f"{cfg.phases}")
                    units = len(snrs) * per
                    entries[e_i]["actual_psnr"] = score_profiles(
                        model, cfg, enc, x, k, units, chunk_cuts(units, "phase", per), [snrs],
                        suite, cfg.eval_seed + 7919 * e_i)[0]
    rows = [e for e in entries if e.get("actual_psnr") is not None and "predicted_psnr" in e]
    if rows:
        err = np.array([e["predicted_psnr"] - e["actual_psnr"] for e in rows])
        print(f"{len(rows)} histories: predicted - actual PSNR bias {err.mean():+.3f}, MAE "
              f"{np.abs(err).mean():.3f}, p95 {np.percentile(np.abs(err), 95):.3f} dB")
        for pol in sorted({e.get("policy", "?") for e in rows}):
            sel = [e for e in rows if e.get("policy", "?") == pol]
            a = np.mean([e["actual_psnr"] for e in sel])
            pr = np.mean([e["predicted_psnr"] for e in sel])
            print(f"  {pol:12s} users {len(sel):5d}  actual mean PSNR {a:7.3f}  predicted {pr:7.3f}")
    spec["entries"] = entries
    spec["decoded_by"] = {"checkpoint": cfg.pretrained, "run_name": cfg.run_name}
    return spec


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
    if args.chunks != "phase" and (not args.chunks.isdigit() or int(args.chunks) < 2):
        raise SystemExit("--chunks is an integer >= 2 or 'phase'")
    if args.design == "file" and not args.profiles_json:
        raise SystemExit("--design file needs --profiles-json")
    torch.manual_seed(cfg.seed)
    model = JSCC(cfg).to(cfg.device).eval()
    rep = load_weights(model, cfg.pretrained)
    print(f"loaded {cfg.pretrained}: {len(rep.missing_keys)} missing, "
          f"{len(rep.unexpected_keys)} unexpected")
    ds = sweep_dataset(cfg, "test", max_images=args.max_images)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    suite = MetricSuite(["psnr"])
    if args.design == "file":
        out = run_file(model, cfg, args, ds, loader, suite)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
            print(f"wrote {args.out}")
        return
    cbrs = [parse_cbr(c) for c in args.cbrs] if args.cbrs else list(cfg.cbrs)
    units = [cfg.units_for_cbr(c, exact=False) for c in cbrs]
    grid = [float(s) for s in cfg.snrs]
    rng = np.random.default_rng(args.profile_seed)
    levels = []
    for c, u in zip(cbrs, units):
        cuts = chunk_cuts(u, args.chunks, cfg.grid_tokens)
        profiles, kinds = make_profiles(args.design, len(cuts) + 1, grid, args.base_snr,
                                        args.random_profiles, rng)
        levels.append({"cbr": str(c), "units": u, "cuts": cuts, "profiles": profiles,
                       "kinds": kinds})
    print(f"{cfg.run_name}: {len(ds)} images, design {args.design}, chunks {args.chunks}, "
          + ", ".join(f"{lv['cbr']}: {len(lv['cuts']) + 1} chunks x {len(lv['profiles'])} profiles"
                      for lv in levels))
    scores = [np.full((len(ds), len(lv["profiles"])), np.nan) for lv in levels]
    with torch.no_grad():
        for k, x in enumerate(loader):
            x = x.to(cfg.device, non_blocking=True)
            with autocast(cfg, cfg.device):
                enc = model.encode(x)
                for ci, lv in enumerate(levels):
                    scores[ci][k] = score_profiles(model, cfg, enc, x, k, lv["units"], lv["cuts"],
                                                   lv["profiles"], suite,
                                                   cfg.eval_seed + 100_003 * k + 1000 * ci)
            if (k + 1) % 6 == 0 or k + 1 == len(ds):
                print(f"  {k + 1}/{len(ds)} images", flush=True)
    summary = {}
    for ci, lv in enumerate(levels):
        u, sc, profiles = lv["units"], scores[ci], lv["profiles"]
        sizes = np.diff([0] + lv["cuts"] + [u])
        print(f"\nCBR {lv['cbr']}: {u} tokens per tile in chunks of {sizes.tolist()} (token order)")
        const = [sc[:, j].mean() for j, kd in enumerate(lv["kinds"]) if kd == "constant"]
        print("  constant SNR " + "  ".join(f"{s:g} dB {p:.3f}" for s, p in zip(grid, const)))
        if args.design == "grid" and len(sizes) > 1:
            res = analyse(sc, profiles, grid, sizes)
            summary[lv["cbr"]] = res
            print(f"  {'chunk SNRs':>18s} {'PSNR':>8s} {'equiv SNR':>9s}  " +
                  "  ".join(f"{r + ' err':>14s}" for r in RULES))
            for row in res["mixed"]:
                print(f"  {str(row['snrs']):>18s} {row['psnr']:8.3f} {row['equivalent_snr']:9.2f}  "
                      + "  ".join(f"{row[r]['error']:+14.3f}" for r in RULES))
            print("  rules (predicted - actual PSNR, per image, over mixed profiles): " +
                  "; ".join(f"{r} bias {v['bias']:+.3f} MAE {v['mae']:.3f} p95 {v['p95']:.3f}"
                            for r, v in res["rules"].items()))
            o = res["order"]
            print(f"  order: best chunks first minus last {o['best_first_minus_last']:+.3f} dB, "
                  f"best-first better for {100 * o['share_best_first_wins']:.0f}% of (image, set)")
        elif args.design == "full" and len(sizes) > 1:
            rows = oat_table(sc, profiles, lv["kinds"], grid, args.base_snr)
            summary[lv["cbr"]] = {"oat": rows}
            print(f"  one chunk at a time (the others at {args.base_snr:g} dB): mean PSNR with the "
                  f"chunk at " + " / ".join(f"{s:g}" for s in grid) + " dB")
            for c, vals in enumerate(rows):
                print(f"    chunk {c + 1}: " + "  ".join(f"{v:7.3f}" for v in vals) +
                      f"   range {max(vals) - min(vals):.3f}")
        lv["psnr"] = sc.tolist()
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"format": 2, "design": args.design, "chunking": args.chunks,
                       "checkpoint": cfg.pretrained, "run_name": cfg.run_name,
                       "images": list(ds.names), "grid": grid, "base_snr": args.base_snr,
                       "per_phase": cfg.grid_tokens, "levels": levels, "summary": summary},
                      f, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
