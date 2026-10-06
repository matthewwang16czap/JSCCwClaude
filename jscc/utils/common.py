"""Seeding, distributed setup, logging, checkpoints, EMA and the LR schedule."""

import copy
import logging
import math
import os
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_distributed():
    """torchrun sets LOCAL_RANK; a plain `python main.py` is a single process."""
    cuda = torch.cuda.is_available()
    if "LOCAL_RANK" not in os.environ:
        return SimpleNamespace(rank=0, world_size=1, local_rank=0,
                               device=torch.device("cuda:0" if cuda else "cpu"))
    local = int(os.environ["LOCAL_RANK"])
    if cuda:
        torch.cuda.set_device(local)
    dist.init_process_group("nccl" if cuda else "gloo")
    return SimpleNamespace(rank=dist.get_rank(), world_size=dist.get_world_size(),
                           local_rank=local,
                           device=torch.device(f"cuda:{local}" if cuda else "cpu"))


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_logger(path=None):
    log = logging.getLogger("jscc")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%m-%d %H:%M:%S")
    for h in [logging.StreamHandler(sys.stdout)] + ([logging.FileHandler(path)] if path else []):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def amp_dtype(cfg):
    """bf16 or None. fp16 is not offered: the channel-facing code is fp32
    either way, and bf16 needs no loss scaler."""
    if cfg.amp and cfg.device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


TRAIN_ONLY = ("objective.",)   # training-only modules (the alignment head): kept out of weights


def save_weights(model, path):
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith(TRAIN_ONLY)}, path)


def load_weights(model, path, log=None):
    """Load weights by name (strict=False) and report what did not match.

    Accepts plain state dicts, DDP ("module.") prefixes and train_state.pt."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state.get("model"), dict):
        state = state["model"]
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    report = model.load_state_dict(state, strict=False)
    missing = [k for k in report.missing_keys if not k.startswith(TRAIN_ONLY)]
    matched = len(state) - len(report.unexpected_keys)
    if matched == 0:
        raise RuntimeError(f"{path}: no tensor matched this model")
    if log is not None:
        log.info(f"loaded {path}: {matched} tensors, {len(missing)} missing, "
                 f"{len(report.unexpected_keys)} unexpected")
        for name, keys in (("missing", missing),
                           ("unexpected", report.unexpected_keys)):
            if keys:
                log.warning(f"  {name}: {keys[:8]}{' ...' if len(keys) > 8 else ''}")
    return report


class EMA:
    """Exponential moving average of the weights; buffers (the DC estimates)
    are copied from the live model."""

    def __init__(self, model, decay):
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for e, m in zip(self.model.parameters(), model.parameters()):
            e.lerp_(m.detach(), 1.0 - self.decay)
        for e, m in zip(self.model.buffers(), model.buffers()):
            e.copy_(m)


def build_scheduler(optimizer, cfg, steps_per_epoch):
    """Linear warm-up, then cosine to lr_floor * lr at the last step (or constant)."""
    warm = cfg.warmup_steps
    total = max(warm + 1, cfg.max_epoch * steps_per_epoch)

    def factor(step):
        if warm and step < warm:
            return (step + 1) / warm
        if cfg.lr_schedule == "constant":
            return 1.0
        t = min(1.0, (step - warm) / max(1, total - warm))
        return cfg.lr_floor + (1.0 - cfg.lr_floor) * 0.5 * (1.0 + math.cos(math.pi * t))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


NO_DECAY = ("query", "pos", "latent", "phase", "gain", "mask_token", "grid_query",
            "relative_position_bias_table")


CASCADE_MODULES = ("enc_cascade.", "dec_cascade.")


def param_groups(model, weight_decay, lr=None, module_lr_mult=1.0):
    """Decay / no-decay groups; the cascade modules (new parameters) can train at
    `module_lr_mult` x lr, in groups of their own (the scheduler scales each group)."""
    groups = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            plain = p.ndim <= 1 or any(k in n for k in NO_DECAY)
            groups.setdefault((plain, n.startswith(CASCADE_MODULES)), []).append(p)
    out = []
    for (plain, is_module), params in groups.items():
        g = {"params": params, "weight_decay": 0.0 if plain else weight_decay}
        if is_module and module_lr_mult != 1.0:
            g["lr"] = lr * module_lr_mult
        out.append(g)
    return out
