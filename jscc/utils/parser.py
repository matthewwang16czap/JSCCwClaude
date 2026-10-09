"""Command line. Model flags left unset take the backbone's defaults
(configs/config.py: BACKBONE_DEFAULTS)."""

import argparse
import os


def create_parser():
    p = argparse.ArgumentParser(
        description="Rate-adaptive deep JSCC: feature transmission (Swin) and token "
                    "transmission (ViT), prefix or depth-cascade rate adaptation")

    g = p.add_argument_group("run")
    g.add_argument("--training", action="store_true",
                   help="train, then test; without it only test --pretrained")
    g.add_argument("--amp", action="store_true", help="bf16 autocast (fp32 if unsupported)")
    g.add_argument("--seed", type=int, default=42, help="rank r uses seed + r")
    g.add_argument("--eval-seed", type=int, default=42,
                   help="seeds the channel draws of validation and test, the same for "
                        "every --seed, so runs of different seeds are paired")
    g.add_argument("--pretrained", default=None, help="weights to start from, or to test")
    g.add_argument("--resume", action="store_true",
                   help="continue from <run dir>/models/train_state.pt")
    g.add_argument("--tag", default=None, help="suffix for the run directory name")
    g.add_argument("--out-dir", default="./history")

    g = p.add_argument_group("model")
    g.add_argument("--backbone", default="swin", choices=["swin", "hybrid", "vit"],
                   help="swin = SwinJSCC feature transmission (baseline); vit = plain ViT "
                        "token transmission; hybrid = Swin + attention tokens (prefix only)")
    g.add_argument("--model-size", default="base", choices=["small", "base", "large"],
                   help="Swin stage sizes (swin, hybrid)")
    g.add_argument("--window-size", type=int, default=8, help="Swin window (swin, hybrid)")
    g.add_argument("--token-dim", type=int, default=None, help="transformer width")
    g.add_argument("--depth", type=int, default=None,
                   help="blocks per side: token trunk (hybrid) or ViT (vit)")
    g.add_argument("--heads", type=int, default=None)
    g.add_argument("--patch-size", type=int, default=None, help="vit")
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
                   help="budget conditioning of the token decoder: film (default), lora = "
                        "AdaTok's per-budget LoRA heads on every trunk MLP, both")
    g.add_argument("--rate-anchors", type=int, default=8,
                   help="budget anchors (log-spaced) of the FiLM / LoRA heads")
    g.add_argument("--rate-rank", type=int, default=16, help="rank of each LoRA head")
    g.add_argument("--refine-ch", type=int, default=0,
                   help="vit: width of an optional conv tail (0 = off)")

    g = p.add_argument_group("depth cascade (net/cascade.py): rate k = its own module path")
    g.add_argument("--cascade", action="store_true",
                   help="rate adaptation by a stack of small modules between the backbone "
                        "encoder and decoder, one exit per predefined CBR (swin, vit), "
                        "instead of sending a prefix of one ordered codeword")
    g.add_argument("--mod-kinds", nargs="+", default=None, metavar="KIND",
                   help="mlp | attn | swin, one per stage (4) or one for all; the stages "
                        "step down the rate ladder 1/8 -> 1/12 -> 1/16 -> 1/24 -> 1/48")
    g.add_argument("--mod-depths", nargs="+", type=int, default=None, metavar="N",
                   help="blocks per stage (4 values or one for all); 0 = bare skip")
    g.add_argument("--mod-width", type=int, default=None,
                   help="hidden width of a stage body (default 96 swin, 128 vit)")
    g.add_argument("--mod-skip", default="proj", choices=["proj", "trunc", "none"],
                   help="proj: learned linear skip initialised to channel truncation; trunc: "
                        "fixed truncation (the code stays ordered); none: body only")
    g.add_argument("--cascade-grad", type=float, default=1.0,
                   help="fraction of the gradient that passes each stage boundary on its way "
                        "from a deeper exit into the shared shallower code: 1 = joint training, "
                        "0 = every stage trained on a detached input")
    g.add_argument("--mod-lr-mult", type=float, default=1.0,
                   help="learning-rate multiplier of NEW modules: the cascade stages and the "
                        "LoRA budget heads (randomly initialised; useful when fine-tuning a "
                        "--pretrained model)")

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
    g.add_argument("--snr-chunks", type=int, default=1,
                   help="train on a piecewise SNR along the tokens: K chunks at independent "
                        "SNRs from --snr-range, cut at random token positions (tokens sent "
                        "in slots of a block-fading channel, docs/PROBLEM.md); 1 = one SNR "
                        "per image. Token models, prefix scheme")

    g = p.add_argument_group("rate")
    g.add_argument("--rate-sampling", default=None, choices=["uniform", "grid", "sandwich"],
                   help="uniform: continuous CBR in [1/48, 1/8] (prefix default); grid: the 5 "
                        "predefined CBRs (cascade default); sandwich: grid, always including the "
                        "smallest and the largest budget (--rates-per-step >= 2)")
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
