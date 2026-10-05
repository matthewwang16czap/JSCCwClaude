"""Train and/or test one model (README.md has the commands).

    python main.py --backbone hybrid --training ...          # one GPU
    torchrun --nproc_per_node 2 main.py --backbone ...       # several
"""

import json
import math
import os

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.config import Config
from data.datasets import get_loaders
from engine import evaluate, reseed, train_one_epoch
from net.network import JSCC
from utils.metrics import MetricSuite
from utils.common import (EMA, barrier, build_scheduler, cleanup, get_logger, init_distributed,
                          load_weights, param_groups, save_weights, seed_everything)
from utils.parser import create_parser


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def table(results, keys=None):
    rows = []
    for r in results:
        ks = keys or [k for k in r if k not in ("snr", "cbr", "cbr_nominal", "sec_per_image")]
        vals = "  ".join(f"{k} {r[k]:.4f}" for k in ks if k in r)
        rows.append(f"  snr {r['snr']:5.1f}  cbr {r['cbr']:.5f}  {vals}")
    return "\n".join(rows)


def train(cfg, net, model, loaders, log, rank):
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay, cfg.lr, cfg.mod_lr_mult),
                                  lr=cfg.lr)
    steps_per_epoch = max(1, len(loaders.train) // cfg.accum_steps)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    ema = EMA(model, cfg.ema) if cfg.ema > 0 else None
    suite = MetricSuite(cfg.valid_metrics, cfg.teacher, cfg.teacher_weights) if rank == 0 else None
    state_path = os.path.join(cfg.models_dir, "train_state.pt")
    start, step, best = 0, 0, -math.inf
    if cfg.resume:
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        if ema is not None and st.get("ema") is not None:
            ema.model.load_state_dict(st["ema"])
        start, step, best = st["epoch"], st["step"], st["best"]
        if log:
            log.info(f"resumed after epoch {start} (step {step})")
    for epoch in range(start, cfg.max_epoch):
        if loaders.sampler is not None:
            loaders.sampler.set_epoch(epoch)
        step, skipped = train_one_epoch(epoch, step, net, model, loaders.train, optimizer,
                                        scheduler, cfg, log, ema)
        if skipped and log:
            log.warning(f"epoch {epoch + 1}: {skipped} optimiser step(s) skipped "
                        f"(non-finite gradient norm)")
        if (epoch + 1) % cfg.valid_freq and epoch + 1 != cfg.max_epoch:
            continue
        if rank == 0:
            judged = ema.model if ema is not None else model
            results, _ = evaluate(judged, loaders.valid, cfg, cfg.snrs, cfg.eval_cbrs, suite)
            dump(results, os.path.join(cfg.logs_dir, f"valid_EP{epoch + 1:04d}.json"))
            score = sum(r["psnr"] for r in results) / len(results)
            log.info(f"validation after epoch {epoch + 1}: mean PSNR {score:.4f}\n"
                     f"{table(results, ('psnr',))}")
            save_weights(judged, os.path.join(cfg.models_dir, "last.pt"))
            if score > best:
                best = score
                save_weights(judged, os.path.join(cfg.models_dir, "best.pt"))
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "ema": ema.model.state_dict() if ema is not None else None,
                        "epoch": epoch + 1, "step": step, "best": best}, state_path)
        barrier()


def test(cfg, model, loaders, log):
    if cfg.training:
        load_weights(model, os.path.join(cfg.models_dir, f"{cfg.final_ckpt}.pt"), log)
    suite = MetricSuite(cfg.test_metrics, cfg.teacher, cfg.teacher_weights)
    results, records = evaluate(model, loaders.test, cfg, cfg.snrs, cfg.eval_cbrs, suite,
                                cfg.test_repeats, per_image=True)
    dump(results, os.path.join(cfg.logs_dir, "test.json"))
    dump(records, os.path.join(cfg.logs_dir, "test_per_image.json"))
    log.info(f"test on {cfg.testset} ({cfg.test_protocol}, {cfg.test_repeats} channel "
             f"draw(s) per image):\n{table(results)}")


def main():
    args = create_parser().parse_args()
    ddp = init_distributed()
    cfg = Config(args, world_size=ddp.world_size)
    cfg.device = ddp.device
    if not cfg.training and not cfg.pretrained:
        raise SystemExit("testing needs --pretrained (or pass --training)")
    seed_everything(cfg.seed + ddp.rank)
    reseed(cfg, cfg.seed + ddp.rank)
    log = None
    if ddp.rank == 0:
        os.makedirs(cfg.models_dir, exist_ok=True)
        os.makedirs(cfg.logs_dir, exist_ok=True)
        log = get_logger(os.path.join(cfg.logs_dir, "run.log"))
        log.info("config " + json.dumps(cfg.describe(), default=str))
        for w in cfg.warnings:
            log.warning(w)
    model = JSCC(cfg).to(cfg.device)
    if log:
        log.info(f"{cfg.run_name}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M "
                 f"parameters, effective batch {cfg.batch_size * cfg.world_size * cfg.accum_steps}")
        if cfg.cascade:
            mods = sum(p.numel() for n, p in model.named_parameters()
                       if n.startswith(("enc_cascade.", "dec_cascade.")))
            log.info(f"cascade: {mods / 1e6:.2f} M parameters in the modules "
                     f"(kinds {cfg.mod_kinds}, depths {cfg.mod_depths}, width {cfg.mod_width}, "
                     f"skip {cfg.mod_skip}, grad share {cfg.cascade_grad:g}); levels {cfg.levels}")
    if cfg.pretrained:
        report = load_weights(model, cfg.pretrained, log)
        if cfg.cascade and cfg.token and any(k.startswith("enc_cascade.posts")
                                             for k in report.missing_keys):
            model.seed_levels()   # a prefix checkpoint: levels start from its gain / DC tables
            if log:
                log.info("prefix checkpoint: the cascade levels start from its gain and DC tables")
    if cfg.freeze != "none":
        n = model.freeze(cfg.freeze)
        if log:
            train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            log.info(f"froze the {cfg.freeze} ({n} tensors); {train_n / 1e6:.2f} M "
                     f"parameters still train")
    net = model
    if ddp.world_size > 1:
        net = DDP(model, device_ids=[ddp.local_rank] if cfg.device.type == "cuda" else None,
                  find_unused_parameters=model.token or model.cascade)
    loaders = get_loaders(cfg, ddp.rank, ddp.world_size, train=cfg.training)
    if cfg.training:
        train(cfg, net, model, loaders, log, ddp.rank)
    if ddp.rank == 0:
        test(cfg, model, loaders, log)
    barrier()
    cleanup()


if __name__ == "__main__":
    main()
