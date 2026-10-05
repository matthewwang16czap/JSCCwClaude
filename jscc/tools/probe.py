"""Peak GPU memory and speed of one configuration, before a long run.

    CUDA_VISIBLE_DEVICES=0 python tools/probe.py --backbone vit --batch-size 16 --amp

Takes the same flags as main.py (plus --steps). Trains a few steps on random
images at the LARGEST budget, where uniform prefix sampling peaks, and prints
the parameter count, the peak memory PyTorch allocated and reserved, and the
time per optimiser step. nvidia-smi shows the reserved figure plus the CUDA
context (0.3-0.5 GB). On a 24 GB card keep reserved at or below ~22 GiB:
validation and the fragmentation of a long run need the rest. 2000 epochs of
DIV2K at batch 16 are 100k steps.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from configs.config import Config  # noqa: E402
from engine import autocast  # noqa: E402
from net.network import JSCC  # noqa: E402
from utils.common import param_groups  # noqa: E402
from utils.parser import create_parser  # noqa: E402


def main():
    parser = create_parser()
    parser.add_argument("--steps", type=int, default=20, help="timed optimiser steps")
    args = parser.parse_args()
    cfg = Config(args)
    dev, cuda = cfg.device, cfg.device.type == "cuda"
    torch.manual_seed(0)
    model = JSCC(cfg).to(dev).train()
    model.objective.schedule = "constant"   # every perceptual term on at the largest budget: worst case
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    units = cfg.cbr_units[-1]
    x = torch.rand(cfg.batch_size, 3, cfg.img_size, cfg.img_size, device=dev)
    snr = torch.full((cfg.batch_size,), 10.0, device=dev)

    def step():
        with autocast(cfg, dev):   # the full training objective, perceptual terms included
            _, loss, _ = model(x, snr, units=[units] * cfg.rates_per_step, want_metrics=False)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.steps
    mem = ""
    if cuda:
        gib = 2 ** 30
        mem = (f" | peak allocated {torch.cuda.max_memory_allocated() / gib:.1f} GiB, "
               f"reserved {torch.cuda.max_memory_reserved() / gib:.1f} GiB")
    print(f"{cfg.run_name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params, "
          f"batch {cfg.batch_size}{mem} | {dt * 1e3:.0f} ms/step, "
          f"{dt * 1e5 / 3600:.1f} h per 100k steps")


if __name__ == "__main__":
    main()
