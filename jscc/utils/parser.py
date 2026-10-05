"""Command line. Model flags left unset take the backbone's defaults
(configs/config.py: BACKBONE_DEFAULTS)."""

import argparse
import os


def create_parser():
    p = argparse.ArgumentParser(
        description="Rate-adaptive deep JSCC: SwinJSCC baseline + three token backbones")

    g = p.add_argument_group("run")
    g.add_argument("--training", action="store_true",
                   help="train, then test; without it only test --pretrained")
    g.add_argument("--amp", action="store_true", help="bf16 autocast (fp32 if unsupported)")
    g.add_argument("--seed", type=int, default=42, help="rank r uses seed + r")
    g.add_argument("--pretrained", default=None, help="weights to start from, or to test")
    g.add_argument("--resume", action="store_true",
                   help="continue from <run dir>/models/train_state.pt")
    g.add_argument("--tag", default=None, help="suffix for the run directory name")
    g.add_argument("--out-dir", default="./history")

    g = p.add_argument_group("model")
    g.add_argument("--backbone", default="swin", choices=["swin", "hybrid", "vit", "adatok"],
                   help="swin = SwinJSCC + linear truncation (baseline); hybrid = Swin + "
                        "attention tokens; vit = plain ViT tokens; adatok = AdaTok/TiTok-S")
    g.add_argument("--model-size", default="base", choices=["small", "base", "large"],
                   help="Swin stage sizes (swin, hybrid)")
    g.add_argument("--window-size", type=int, default=8, help="Swin window (swin, hybrid)")
    g.add_argument("--token-dim", type=int, default=None, help="transformer width")
    g.add_argument("--depth", type=int, default=None,
                   help="blocks per side: token trunk (hybrid) or ViT (vit, adatok)")
    g.add_argument("--heads", type=int, default=None)
    g.add_argument("--patch-size", type=int, default=None, help="vit, adatok")
    g.add_argument("--sym-per-token", type=int, default=None,
                   help="complex symbols per token")
    g.add_argument("--tile", type=int, default=256,
                   help="token models: pixels per tile / latent block side x 16")
    g.add_argument("--pos-scale", type=float, default=None,
                   help="hybrid: scale of the fixed 2D sinusoid added to the grid")
    g.add_argument("--zero-init", action=argparse.BooleanOptionalAction, default=None,
                   help="zero the residual branches of the token transformers")
    g.add_argument("--phase-order", default="spread", choices=["spread", "raster"],
                   help="hybrid, vit: position order within a phase")
    g.add_argument("--rate-mod", default=None, choices=["none", "film", "lora", "both"],
                   help="budget modulation of the token decoder")
    g.add_argument("--rate-anchors", type=int, default=8)
    g.add_argument("--rate-rank", type=int, default=16)
    g.add_argument("--refine-ch", type=int, default=0,
                   help="vit, adatok: width of an optional conv tail (0 = off)")

    g = p.add_argument_group("channel")
    g.add_argument("--channel-type", default="awgn", choices=["awgn", "rayleigh", "none"])
    g.add_argument("--channel-backend", default="sionna", choices=["sionna", "torch"],
                   help="sionna: Sionna 2 blocks (the standard); torch: pure-PyTorch "
                        "reference with the same statistics (CPU tests)")
    g.add_argument("--equalizer", default="mmse", choices=["mmse", "zf"],
                   help="rayleigh receiver with perfect CSI")
    g.add_argument("--snrs", nargs="+", type=float, default=[1.0, 4.0, 7.0, 10.0, 13.0],
                   help="evaluation SNRs (dB)")
    g.add_argument("--snr-range", nargs=2, type=float, default=[-2.0, 22.0],
                   help="training SNR range (dB), uniform per image")

    g = p.add_argument_group("rate")
    g.add_argument("--rate-sampling", default="uniform", choices=["uniform", "grid"],
                   help="uniform: continuous CBR in [1/48, 1/8]; grid: the 5 predefined CBRs")
    g.add_argument("--rates-per-step", type=int, default=1,
                   help="budgets decoded per step from one encoder pass (loss = mean)")
    g.add_argument("--fixed-cbr", default=None, help="train a single-rate specialist")
    g.add_argument("--top-prob", type=float, default=0.0,
                   help="probability that a training budget is the full one (the rest "
                        "uniform): the sampling control for the tail probes")
    g.add_argument("--eval-cbrs", nargs="+", default=None,
                   help="evaluation CBRs (default: 1/48 1/24 1/16 1/12 1/8)")

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"])
    g.add_argument("--lr-floor", type=float, default=0.05,
                   help="cosine: final LR as a fraction of --lr")
    g.add_argument("--warmup-steps", type=int, default=None)
    g.add_argument("--max-epoch", type=int, default=2000)
    g.add_argument("--valid-freq", type=int, default=40)
    g.add_argument("--print-step", type=int, default=100,
                   help="log (with the decoder diagnostic) every N optimiser steps")
    g.add_argument("--batch-size", type=int, default=None, help="per GPU")
    g.add_argument("--effective-batch", type=int, default=16,
                   help="per optimiser step; gradient accumulation fills the gap")
    g.add_argument("--grad-clip", type=float, default=1.0, help="0 = off")
    g.add_argument("--weight-decay", type=float, default=0.0)
    g.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay of the weights used for validation/test (0 = off)")
    g.add_argument("--loss", default="mse", choices=["mse", "msssim"])
    g.add_argument("--freeze", default="none", choices=["none", "encoder", "decoder"],
                   help="train only the other half of a --pretrained model (probes: does "
                        "the frozen encoder's code carry the information / can the frozen "
                        "decoder read it?)")

    g = p.add_argument_group("data")
    g.add_argument("--data-root", default=os.environ.get("JSCC_DATA_ROOT", os.path.expanduser("~")),
                   help="contains datasets/ (default $JSCC_DATA_ROOT or ~)")
    g.add_argument("--trainset", nargs="+", default=["DIV2K"], choices=["DIV2K", "Flickr2K", "CLIC"])
    g.add_argument("--validset", default="DIV2K", choices=["DIV2K", "CLIC"])
    g.add_argument("--testset", default="Kodak", choices=["Kodak", "CLIC"])
    g.add_argument("--img-size", type=int, default=256, help="training crop")
    g.add_argument("--num-workers", type=int, default=None)
    g.add_argument("--valid-protocol", default="native", choices=["native", "crop", "full"])
    g.add_argument("--test-protocol", default="full", choices=["full", "native", "crop"])
    g.add_argument("--test-repeats", type=int, default=None,
                   help="channel draws per test image (default 1 AWGN, 10 Rayleigh)")
    g.add_argument("--final-ckpt", default="last", choices=["last", "best"],
                   help="weights the final test uses after training")
    g.add_argument("--valid-metrics", nargs="+", default=["psnr", "msssim", "lpips"],
                   help="metrics of the periodic validation (utils/metrics.py)")
    g.add_argument("--test-metrics", nargs="+", default=["all"],
                   help="metrics of the final test: 'all' or a list")

    g = p.add_argument_group("perception (net/loss.py); weights are relative to the pixel loss")
    g.add_argument("--sem-weight", type=float, default=0.0,
                   help="DINOv2 patch-feature distance between reconstruction and input")
    g.add_argument("--align-weight", type=float, default=0.0,
                   help="REPA: decoder hidden state predicts DINOv2 features of the input")
    g.add_argument("--imp-weight", type=float, default=0.0,
                   help="strength in [0, 1] of the teacher-attention weighting of the squared error")
    g.add_argument("--lpips-weight", type=float, default=0.0, help="LPIPS (AlexNet) loss")
    g.add_argument("--perc-schedule", default="budget", choices=["budget", "constant"],
                   help="budget: weights fall from 1 at the smallest budget to 0 at the full one")
    g.add_argument("--perc-gamma", type=float, default=1.0, help="exponent of the budget schedule")
    g.add_argument("--teacher", default="vit_small_patch14_reg4_dinov2.lvd142m",
                   help="timm name of the frozen DINOv2 teacher")
    g.add_argument("--teacher-weights", default=None,
                   help="local weights file for the teacher (offline servers)")
    return p
