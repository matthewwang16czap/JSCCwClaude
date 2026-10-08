"""Every knob the code reads, resolved and validated once.

Flags left unset on the command line take the backbone's default from
BACKBONE_DEFAULTS; those defaults are the recommended configurations. Problems
that would otherwise surface halfway through a run raise here; soft issues go
to `self.warnings`, which main.py logs once the logger exists.
"""

import importlib.util
import math
import os
from fractions import Fraction

import torch

PREDEFINED_CBRS = ("1/48", "1/24", "1/16", "1/12", "1/8")
SWIN_STRIDE = 16      # SwinJSCC: patch 2, three 2x merges
SWIN_WIDTH = 192      # latent width = real channels per position at CBR 1/8
SWIN_SIZES = {        # stage widths (the last one is SWIN_WIDTH) and depths
    "small": ([48, 64, 96], [2, 2, 4, 2]),
    "base": ([64, 96, 128], [2, 2, 6, 2]),
    "large": ([96, 128, 192], [2, 2, 8, 4]),
}

BACKBONE_DEFAULTS = {
    "swin": dict(warmup_steps=0),
    # the previous tree's best AdaJSCC (wave C: zero-init, latent-space blocks)
    "hybrid": dict(sym_per_token=16, token_dim=256, depth=4, heads=8, rate_mod="film",
                   zero_init=True, pos_scale=0.25, warmup_steps=2000),
    # vit is sized to ~19 GiB of a 24 GB card at batch 16 and 256 px (tools/probe.py
    # measures it). Patch 8 spends the memory on resolution (1024 tokens per tile,
    # 192 pixel values per token); 4-symbol tokens keep hybrid's 6 phases.
    # Previous small size: patch 16, sym 16, 384 wide, 6 blocks, 6 heads.
    "vit": dict(sym_per_token=4, token_dim=768, depth=10, heads=12, patch_size=8,
                rate_mod="film", zero_init=True, warmup_steps=2000),
}

# Depth-cascade modules (--cascade, net/cascade.py), one stage per step down the
# rate ladder (4 stages for the 5 predefined CBRs). `width` is the hidden width of
# a stage body (unused at depth 0). Wave c1 (docs/NOTES.md) chose the defaults: a
# per-rate LINEAR stage (depth 0, the learned truncation-initialised skip) took all
# of the cascade's gain on Kodak (vit +0.048 dB, swin +0.019 dB over the
# grid-trained prefix control); mlp / attention / window bodies, at any depth
# schedule, added <= 0.01 dB. The bodies stay available for ablations:
# --mod-kinds mlp mlp attn attn --mod-depths 1 2 3 4 is wave c1's mixed arm.
MODULE_DEFAULTS = {
    "swin": dict(kinds=["mlp", "mlp", "mlp", "mlp"], depths=[0, 0, 0, 0], width=96),
    "vit": dict(kinds=["mlp", "mlp", "mlp", "mlp"], depths=[0, 0, 0, 0], width=128),
}


def parse_cbr(value):
    return Fraction(str(value)).limit_denominator(1_000_000)


class Config:
    def __init__(self, args, world_size=1):
        self.warnings = []
        self.world_size = max(1, int(world_size))
        self.seed = args.seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.training, self.amp = args.training, args.amp
        self.pretrained, self.resume, self.final_ckpt = args.pretrained, args.resume, args.final_ckpt

        self.backbone = args.backbone
        self.token = self.backbone != "swin"
        defaults = BACKBONE_DEFAULTS[self.backbone]

        def pick(name):
            v = getattr(args, name)
            return defaults.get(name) if v is None else v

        # channel
        self.channel_type, self.channel_backend = args.channel_type, args.channel_backend
        self.equalizer = args.equalizer
        if self.channel_backend == "sionna" and self.channel_type != "none":
            try:
                import sionna
            except ImportError as e:
                raise ValueError("--channel-backend sionna needs Sionna >= 2 (pip install "
                                 "sionna-no-rt); --channel-backend torch is the pure-PyTorch "
                                 "reference") from e
            if int(str(sionna.__version__).split(".")[0]) < 2:
                raise ValueError(f"Sionna {sionna.__version__} is the TensorFlow generation; "
                                 f"this code needs Sionna >= 2 (PyTorch)")
        self.snrs = [float(s) for s in args.snrs]
        self.train_snr_range = (float(args.snr_range[0]), float(args.snr_range[1]))
        if self.train_snr_range[1] < self.train_snr_range[0]:
            raise ValueError(f"--snr-range {args.snr_range}: max < min")
        self.diag_snr = sorted(self.snrs)[len(self.snrs) // 2]

        # Swin geometry (baseline and hybrid)
        self.window = args.window_size
        self.swin_block = SWIN_STRIDE * self.window
        self.model_size = args.model_size
        dims, depths = SWIN_SIZES[self.model_size]
        dims, depths = list(dims) + [SWIN_WIDTH], list(depths)
        heads = [d // 16 for d in dims]
        common = dict(patch_size=2, window_size=self.window, mlp_ratio=4.0)
        self.swin_encoder_kwargs = dict(in_chans=3, embed_dims=dims, depths=depths,
                                        num_heads=heads, **common)
        self.swin_decoder_kwargs = dict(out_chans=3, embed_dims=dims[::-1], depths=depths[::-1],
                                        num_heads=heads[::-1], **common)
        self.width = SWIN_WIDTH

        # token geometry
        self.tile = args.tile
        self.sym = pick("sym_per_token")
        self.token_dim, self.depth, self.heads = pick("token_dim"), pick("depth"), pick("heads")
        self.patch = pick("patch_size")
        self.pos_scale = pick("pos_scale")
        self.zero_init = bool(pick("zero_init")) if self.token else False
        self.rate_mod = pick("rate_mod") if self.token else "none"
        self.rate_anchors = args.rate_anchors
        self.phase_order, self.refine_ch = args.phase_order, args.refine_ch
        self.cascade = bool(args.cascade)

        # rate
        self.cbrs = [parse_cbr(c) for c in PREDEFINED_CBRS]
        self.rate_sampling = args.rate_sampling or ("grid" if self.cascade else "uniform")
        self.rates_per_step = int(args.rates_per_step)
        self.top_prob = float(getattr(args, "top_prob", 0.0))
        self.fixed_cbr = parse_cbr(args.fixed_cbr) if args.fixed_cbr else None
        self.diag_cbr = self.cbrs[len(self.cbrs) // 2]
        if self.token:
            self._setup_tokens()
        self.cbr_units = [self.units_for_cbr(c) for c in self.cbrs]
        self._setup_rates(args)
        self._setup_cascade(args)

        # resolution
        self.img_size = args.img_size
        self._setup_resolution()

        # optimisation
        self.lr, self.lr_schedule, self.lr_floor = args.lr, args.lr_schedule, args.lr_floor
        self.warmup_steps = int(pick("warmup_steps") or 0)
        self.max_epoch, self.valid_freq = args.max_epoch, args.valid_freq
        self.batch_size = args.batch_size or {128: 16, 256: 16, 512: 4}.get(self.img_size, 8)
        self.effective_batch = args.effective_batch
        per_step = self.batch_size * self.world_size
        self.accum_steps = max(1, round(self.effective_batch / per_step))
        if per_step * self.accum_steps != self.effective_batch:
            self.warnings.append(
                f"effective batch is {per_step * self.accum_steps}, not {self.effective_batch} "
                f"({self.batch_size} x {self.world_size} GPU(s) x {self.accum_steps} accumulation)")
        self.grad_clip, self.weight_decay, self.ema = args.grad_clip, args.weight_decay, args.ema
        self.mod_lr_mult = float(args.mod_lr_mult)
        if self.mod_lr_mult <= 0:
            raise ValueError("--mod-lr-mult must be > 0")
        self.loss = args.loss
        self.freeze = getattr(args, "freeze", "none")
        if self.freeze != "none" and not args.pretrained:
            raise ValueError(f"--freeze {self.freeze} needs --pretrained (a frozen random half "
                             f"is not a probe of anything)")
        self._setup_perception(args)
        self.print_step = max(1, args.print_step)

        # data
        self.data_root = args.data_root
        self.trainsets = list(dict.fromkeys(args.trainset))
        self.validset, self.testset = args.validset, args.testset
        self.num_workers = args.num_workers
        self.valid_protocol, self.test_protocol = args.valid_protocol, args.test_protocol
        self.test_repeats = args.test_repeats or (10 if self.channel_type == "rayleigh" else 1)

        self.run_name = self._run_name(args)
        self.workdir = os.path.join(args.out_dir, self.run_name)
        self.models_dir = os.path.join(self.workdir, "models")
        self.logs_dir = os.path.join(self.workdir, "logs")

    def _setup_perception(self, args):
        from utils.metrics import check_metrics
        self.sem_weight, self.align_weight = args.sem_weight, args.align_weight
        self.imp_weight, self.lpips_weight = args.imp_weight, args.lpips_weight
        self.perc_schedule, self.perc_gamma = args.perc_schedule, args.perc_gamma
        self.teacher, self.teacher_weights = args.teacher, args.teacher_weights
        weights = (self.sem_weight, self.align_weight, self.imp_weight, self.lpips_weight)
        if min(weights) < 0 or self.perc_gamma <= 0:
            raise ValueError("perceptual weights must be >= 0 and --perc-gamma > 0")
        if self.imp_weight > 1:
            raise ValueError("--imp-weight is a strength in [0, 1]")
        if self.imp_weight and self.loss != "mse":
            raise ValueError("--imp-weight weights the squared error: it needs --loss mse")
        if (self.sem_weight or self.align_weight or self.imp_weight) \
                and importlib.util.find_spec("timm") is None:
            raise ValueError("the DINOv2 teacher needs timm (pip install timm)")
        if self.lpips_weight and importlib.util.find_spec("lpips") is None:
            raise ValueError("--lpips-weight needs lpips (pip install lpips)")
        self.valid_metrics = check_metrics(args.valid_metrics)
        self.test_metrics = check_metrics(args.test_metrics)

    # -- geometry ---------------------------------------------------------------------
    def _setup_tokens(self):
        if self.token_dim % self.heads or self.token_dim % 4:
            raise ValueError(f"--token-dim {self.token_dim} must be divisible by 4 and by "
                             f"--heads {self.heads}")
        per_tile = 3 * self.tile * self.tile
        exact = [c * per_tile / self.sym for c in self.cbrs]
        if any(v.denominator != 1 for v in exact):
            g = math.gcd(*[int(c * per_tile) for c in self.cbrs])
            raise ValueError(f"--sym-per-token {self.sym} gives fractional token counts at tile "
                             f"{self.tile}; it must divide {g}")
        self.token_list = [int(v) for v in exact]
        self.n_tokens, self.l_min = self.token_list[-1], self.token_list[0]
        if self.backbone in ("hybrid", "vit"):
            unit = SWIN_STRIDE if self.backbone == "hybrid" else self.patch
            if self.tile % unit:
                raise ValueError(f"--tile {self.tile} must be a multiple of {unit}")
            self.grid_side = self.tile // unit
            self.grid_tokens = self.grid_side ** 2
            if self.n_tokens % self.grid_tokens:
                per_pos = int(self.cbrs[-1] * 3 * unit * unit)
                raise ValueError(f"{self.n_tokens} tokens is not a whole number of phases of "
                                 f"{self.grid_tokens}: --sym-per-token must divide {per_pos}")
            self.phases = self.n_tokens // self.grid_tokens
            mid = [str(c) for c, l in zip(self.cbrs, self.token_list) if l % self.grid_tokens]
            if mid:
                self.warnings.append(f"CBR {', '.join(mid)} end mid-phase: the last phase reaches "
                                     f"only part of each block ({self.phase_order} order)")
            if self.phase_order == "raster":
                self.warnings.append("--phase-order raster: a partial phase refines the top rows "
                                     "of each block first (the previous tree's order; use it to "
                                     "evaluate that tree's checkpoints)")
        if self.backbone == "vit" and self.tile % self.patch:
            raise ValueError(f"--tile {self.tile} must be a multiple of --patch-size {self.patch}")

    def _setup_rates(self, args):
        if self.rates_per_step < 1:
            raise ValueError("--rates-per-step must be >= 1")
        if self.rate_sampling == "sandwich" and not 2 <= self.rates_per_step <= len(self.cbrs):
            raise ValueError(f"--rate-sampling sandwich sends the smallest and the largest budget "
                             f"plus random ones: --rates-per-step must be 2..{len(self.cbrs)}")
        if not 0.0 <= self.top_prob <= 1.0:
            raise ValueError("--top-prob is a probability")
        if self.top_prob and (self.rate_sampling != "uniform" or self.fixed_cbr is not None):
            raise ValueError("--top-prob mixes the full budget into UNIFORM sampling; it does "
                             "not combine with --rate-sampling grid or --fixed-cbr")
        if self.rate_sampling == "grid" and self.rates_per_step > len(self.cbrs):
            raise ValueError(f"--rate-sampling grid draws at most {len(self.cbrs)} rates per step")
        if self.fixed_cbr is not None:
            self.units_for_cbr(self.fixed_cbr)   # raises unless exact
            if self.rates_per_step != 1:
                self.warnings.append("--fixed-cbr trains one rate; --rates-per-step ignored")
                self.rates_per_step = 1
        self.eval_cbrs = [parse_cbr(c) for c in args.eval_cbrs] if args.eval_cbrs \
            else list(self.cbrs)
        for c in self.eval_cbrs:
            u = self.units_for_cbr(c, exact=False)
            if abs(self.cbr_for_units(u) - float(c)) > 1e-12:
                self.warnings.append(f"eval CBR {c} rounds to {u} units = CBR "
                                     f"{self.cbr_for_units(u):.6f}; results report the latter")

    def _setup_resolution(self):
        mult = 1
        if self.backbone in ("swin", "hybrid"):
            mult = self.swin_block
            if self.img_size % self.swin_block:
                raise ValueError(f"--img-size {self.img_size} must be a multiple of "
                                 f"{self.swin_block} (stride 16 x window {self.window})")
        if self.token:
            mult = math.lcm(mult, self.tile)
            if self.img_size % self.tile:
                raise ValueError(f"--img-size {self.img_size} must be a multiple of --tile {self.tile}")
        self.size_multiple = mult   # native-resolution tests crop to a multiple of this

    def _setup_cascade(self, args):
        """Levels, code widths and the module spec of the depth cascade."""
        self.levels = list(self.cbr_units[::-1])            # descending rate: level 0 = top
        self.code_dims = list(self.levels)                  # real channels per position
        self.mod_kinds = self.mod_depths = None
        self.mod_width, self.mod_skip = args.mod_width, args.mod_skip
        self.cascade_grad = float(args.cascade_grad)
        if not self.cascade:
            return
        if self.backbone not in MODULE_DEFAULTS:
            raise ValueError(f"--cascade supports --backbone swin (feature transmission) and vit "
                             f"(token transmission), not {self.backbone}")
        if self.fixed_cbr is not None or self.top_prob:
            raise ValueError("--cascade trains the levels themselves: --fixed-cbr and --top-prob "
                             "belong to the prefix scheme")
        if self.rate_sampling == "uniform":
            raise ValueError("--cascade has one code per predefined CBR: use --rate-sampling "
                             "grid (default) or sandwich")
        if not 0.0 <= self.cascade_grad <= 1.0:
            raise ValueError("--cascade-grad is a fraction in [0, 1]")
        if self.token:
            per = self.grid_tokens
            bad = [str(c) for c, u in zip(self.cbrs, self.cbr_units) if u % per]
            if bad:
                raise ValueError(f"--cascade needs every CBR on a phase boundary; {', '.join(bad)} "
                                 f"is not (a level sends whole phases of {per} tokens)")
            self.code_dims = [u // per * 2 * self.sym for u in self.levels]
        n, d = len(self.levels) - 1, MODULE_DEFAULTS[self.backbone]

        def per_stage(name, vals, default, cast):
            vals = default if vals is None else [cast(v) for v in vals]
            if len(vals) == 1:
                vals = vals * n
            if len(vals) != n:
                raise ValueError(f"--{name} takes 1 or {n} values (one per stage), got {len(vals)}")
            return vals

        self.mod_kinds = per_stage("mod-kinds", args.mod_kinds, d["kinds"], str)
        self.mod_depths = per_stage("mod-depths", args.mod_depths, d["depths"], int)
        self.mod_width = int(self.mod_width or d["width"])
        for k in self.mod_kinds:
            if k not in ("mlp", "attn", "swin"):
                raise ValueError(f"--mod-kinds are mlp|attn|swin, got {k!r}")
        if min(self.mod_depths) < 0:
            raise ValueError("--mod-depths must be >= 0")
        if self.mod_skip not in ("proj", "trunc", "none"):
            raise ValueError(f"--mod-skip is proj|trunc|none, got {self.mod_skip!r}")
        for k, dep in zip(self.mod_kinds, self.mod_depths):
            need = {"mlp": 1, "attn": 32, "swin": 16}[k] if dep else 1
            if self.mod_width % need:
                raise ValueError(f"--mod-width {self.mod_width} must be a multiple of {need} "
                                 f"for a {k} stage (heads of {need} channels)")
        if self.mod_skip == "none" and 0 in self.mod_depths:
            raise ValueError("--mod-skip none needs a body in every stage (--mod-depths >= 1)")
        if self.token and "swin" in self.mod_kinds and self.grid_side % self.window:
            raise ValueError(f"swin stages need the tile's {self.grid_side}x{self.grid_side} "
                             f"position grid to be a multiple of --window-size {self.window}")
        for c in self.eval_cbrs:
            if self.units_for_cbr(c, exact=False) not in self.cbr_units:
                raise ValueError(f"--cascade evaluates its levels only: CBR {c} is not one of "
                                 f"{', '.join(PREDEFINED_CBRS)}")

    # -- rate units -------------------------------------------------------------------
    def units_for_cbr(self, cbr, exact=True):
        """Tokens per tile (token models) or real channels per position (baseline)."""
        c = parse_cbr(cbr)
        if self.token:
            v = c * 3 * self.tile * self.tile / self.sym
            u = min(self.n_tokens, max(1, round(v)))
        else:
            v = c * 6 * SWIN_STRIDE ** 2
            u = int(round(v))
            u = min(self.width, max(2, u - u % 2))
        if exact and u != v:
            raise ValueError(f"CBR {c} is {float(v):.3f} "
                             f"{'tokens' if self.token else 'real channels'}, not a whole even count")
        return int(u)

    def cbr_for_units(self, u):
        if self.token:
            return u * self.sym / (3.0 * self.tile * self.tile)
        return u / (6.0 * SWIN_STRIDE ** 2)

    # -- bookkeeping ---------------------------------------------------------------------
    def _run_name(self, args):
        parts = [self.backbone]
        if self.backbone in ("swin", "hybrid"):
            parts.append(self.model_size)
        if self.token:
            patch = f"p{self.patch}" if self.backbone == "vit" else ""
            parts.append(f"{patch}d{self.token_dim}x{self.depth}s{self.sym}")
        parts += [self.channel_type, "+".join(self.trainsets), str(args.img_size)]
        d = BACKBONE_DEFAULTS[self.backbone]
        extra = []
        if self.token and self.rate_mod != d["rate_mod"]:
            extra.append(self.rate_mod)
        if self.token and self.zero_init != d["zero_init"]:
            extra.append("zi" if self.zero_init else "nozi")
        if self.backbone == "hybrid" and self.pos_scale != d["pos_scale"]:
            extra.append(f"ps{self.pos_scale:g}")
        if self.backbone in ("hybrid", "vit") and self.phase_order != "spread":
            extra.append(self.phase_order)
        if self.window != 8:
            extra.append(f"w{self.window}")
        if self.channel_type == "rayleigh" and self.equalizer != "mmse":
            extra.append(self.equalizer)
        if self.rate_sampling != ("grid" if self.cascade else "uniform"):
            extra.append(self.rate_sampling)
        if self.rates_per_step != 1:
            extra.append(f"k{self.rates_per_step}")
        if self.top_prob:
            extra.append(f"top{self.top_prob:g}")
        if self.fixed_cbr is not None:
            extra.append(f"fix{self.fixed_cbr.numerator}-{self.fixed_cbr.denominator}")
        if self.freeze != "none":
            extra.append("frz-" + self.freeze[:3])
        if self.refine_ch:
            extra.append(f"rf{self.refine_ch}")
        if self.cascade:
            letters = "".join(k[0] for k in self.mod_kinds)
            d = MODULE_DEFAULTS[self.backbone]
            extra.append(f"casc-{letters}-d{''.join(str(v) for v in self.mod_depths)}")
            if self.mod_width != d["width"]:
                extra.append(f"w{self.mod_width}")
            if self.mod_skip != "proj":
                extra.append(f"sk{self.mod_skip}")
            if self.cascade_grad != 1.0:
                extra.append(f"cg{self.cascade_grad:g}")
        perc = [f"{k}{v:g}" for k, v in (("sem", self.sem_weight), ("al", self.align_weight),
                                           ("imp", self.imp_weight), ("lp", self.lpips_weight)) if v]
        if perc:
            extra.append("-".join(perc) + ("" if self.perc_schedule == "budget" else "-const")
                         + ("" if self.perc_gamma == 1 else f"-g{self.perc_gamma:g}"))
        if self.lr_schedule != "cosine":
            extra.append(self.lr_schedule)
        if args.ema:
            extra.append(f"ema{args.ema:g}")
        if args.seed != 42:
            extra.append(f"s{args.seed}")
        if args.tag:
            extra.append(args.tag)
        return "_".join(parts + extra)

    def describe(self):
        skip = {"warnings", "swin_encoder_kwargs", "swin_decoder_kwargs"}
        out = {}
        for k, v in vars(self).items():
            if k in skip:
                continue
            if isinstance(v, Fraction):
                v = str(v)
            elif isinstance(v, (list, tuple)):
                v = [str(i) if isinstance(i, Fraction) else i for i in v]
            elif isinstance(v, torch.device):
                v = str(v)
            out[k] = v
        return out
