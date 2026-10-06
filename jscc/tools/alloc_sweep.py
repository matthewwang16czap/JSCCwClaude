"""Per-image quality-vs-budget curves of a trained codec (input of tools/alloc_policy.py).

Takes main.py's flags (model, --pretrained, channel) plus the ones below.
Test curves -- for the oracle and for scoring a policy -- use a fixed SNR grid
and several noise draws per image. Training curves for the predictor use
deterministic crops and random SNRs per image. Every budget of an image reuses
the same noise draw (common random numbers, see alloc/sweep.py).

    # test curves: Kodak at native resolution, 5 SNRs x 4 noise draws
    python tools/alloc_sweep.py --backbone vit --pretrained <ckpt> --channel-type awgn --amp \\
        --split test --testset Kodak --snrs 1 4 7 10 13 --draws 4 --out curves/vit_kodak.npz

    # training curves: 2 crops of 512 px per DIV2K image, 2 random SNRs in [-2, 22] dB
    python tools/alloc_sweep.py --backbone vit --pretrained <ckpt> --channel-type awgn --amp \\
        --split train --trainset DIV2K --crop 512 --crops-per-image 2 --random-snrs 2 \\
        --draws 2 --out curves/vit_div2k_train.npz
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from alloc.io import save_curves  # noqa: E402
from alloc.sweep import candidate_units, sweep  # noqa: E402
from configs.config import Config  # noqa: E402
from data.datasets import sweep_dataset  # noqa: E402
from net.network import JSCC  # noqa: E402
from utils.common import load_weights  # noqa: E402
from utils.metrics import check_metrics  # noqa: E402
from utils.parser import create_parser  # noqa: E402

DEFAULT_METRICS = ["psnr", "msssim", "lpips", "lpips_vgg", "dists"]


def add_args(parser):
    g = parser.add_argument_group("allocation sweep")
    g.add_argument("--split", default="test", choices=["test", "valid", "train"],
                   help="test: --testset under --test-protocol; valid/train: fixed crops")
    g.add_argument("--crop", type=int, default=512, help="crop size for valid/train")
    g.add_argument("--crops-per-image", type=int, default=1)
    g.add_argument("--max-images", type=int, default=None,
                   help="test: the first n images; valid/train: a seeded sample of n images")
    g.add_argument("--random-snrs", type=int, default=0,
                   help="n SNRs per image drawn from --snr-range (training curves); "
                        "0 = the --snrs grid for every image")
    g.add_argument("--draws", type=int, default=2, help="noise draws per image and SNR")
    g.add_argument("--candidates", type=int, default=11,
                   help="budgets evenly over the trained range (+ the predefined CBRs)")
    g.add_argument("--sweep-metrics", nargs="+", default=DEFAULT_METRICS)
    g.add_argument("--sweep-batch", type=int, default=4, help="images per forward (crops)")
    g.add_argument("--sweep-seed", type=int, default=None, help="default: --seed")
    g.add_argument("--out", required=True, help="curve file (.npz)")
    return parser


def main():
    args = add_args(create_parser()).parse_args()
    cfg = Config(args)
    if not cfg.pretrained:
        raise SystemExit("--pretrained is required: the sweep scores a trained codec")
    if args.random_snrs and args.split == "test":
        print("note: random SNRs on the test split -- the oracle and policy evaluation need a "
              "fixed grid; use this only for training curves")
    metrics = check_metrics(args.sweep_metrics)
    seed = cfg.seed if args.sweep_seed is None else args.sweep_seed
    torch.manual_seed(seed)
    model = JSCC(cfg).to(cfg.device)
    rep = load_weights(model, cfg.pretrained)
    print(f"loaded {cfg.pretrained}: {len(rep.missing_keys)} missing, "
          f"{len(rep.unexpected_keys)} unexpected")
    units = candidate_units(cfg, args.candidates)
    ds = sweep_dataset(cfg, args.split, args.crop, args.crops_per_image, seed, args.max_images)
    full = args.split == "test" and cfg.test_protocol == "full"
    workers = min(4, os.cpu_count() or 1) if cfg.num_workers is None else cfg.num_workers
    loader = DataLoader(ds, batch_size=1 if full else args.sweep_batch, shuffle=False,
                        num_workers=workers, pin_memory=cfg.device.type == "cuda")
    snrs = None if args.random_snrs else list(cfg.snrs)
    print(f"{cfg.run_name}: {len(ds)} images ({args.split}), budgets {units}, "
          f"{'random SNRs x ' + str(args.random_snrs) if args.random_snrs else 'SNRs ' + str(snrs)}"
          f", {args.draws} draw(s), channel {cfg.channel_type}, metrics {metrics}")
    arrays = sweep(model, cfg, loader, units, metrics, args.draws, snrs, args.random_snrs,
                   tuple(cfg.train_snr_range), seed)
    arrays.update(units=np.asarray(units), cbr=np.asarray([cfg.cbr_for_units(u) for u in units]),
                  metrics=metrics, names=list(ds.names))
    meta = {"checkpoint": cfg.pretrained, "backbone": cfg.backbone, "run_name": cfg.run_name,
            "channel": cfg.channel_type, "equalizer": cfg.equalizer, "split": args.split,
            "dataset": (cfg.testset if args.split == "test" else
                        cfg.validset if args.split == "valid" else "+".join(cfg.trainsets)),
            "protocol": cfg.test_protocol if args.split == "test" else f"crop{args.crop}",
            "crops_per_image": args.crops_per_image, "draws": args.draws,
            "snrs": snrs, "random_snrs": args.random_snrs,
            "snr_range": list(cfg.train_snr_range), "seed": seed,
            "predefined_cbrs": [str(c) for c in cfg.cbrs],
            "predefined_units": list(cfg.cbr_units), "command": " ".join(sys.argv)}
    save_curves(args.out, arrays, meta)
    print(f"wrote {args.out}: scores {arrays['scores'].shape} (images, SNRs, draws, budgets, "
          f"metrics)")
    if snrs is not None:
        mean = np.nanmean(arrays["scores"], axis=(0, 2))            # (S, K, M)
        show = [m for m in ("psnr", "lpips", "dists") if m in metrics]
        for s, snr in enumerate(snrs):
            cells = []
            for u in cfg.cbr_units:
                k = units.index(u)
                cells.append(f"{cfg.cbr_for_units(u):.4f}: " +
                             " ".join(f"{m} {mean[s, k, metrics.index(m)]:.4f}" for m in show))
            print(f"  SNR {snr:5.1f} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
