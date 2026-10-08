# Rate-adaptive deep JSCC: feature and token transmission, prefix and depth-cascade rates

One driver trains and tests the models below. They share the channel, the rate
protocol and the evaluation. Two ways to make the rate adaptive:

* **prefix** (default): one ordered codeword; the receiver gets the first *k*
  channels (feature transmission) or the first *l* tokens (token transmission)
  and knows how many arrived. Nothing else about the channel reaches any network.
* **cascade** (`--cascade`, `net/cascade.py`, `docs/CASCADE.md`): a stack of small
  modules between backbone encoder and decoder, one exit per predefined CBR; each
  rate has its own module path, the backbones are shared. Wave c1 tested it: only
  +0.02 to +0.08 dB over a matched prefix control, all of it from a per-rate linear
  stage (`docs/NOTES.md`; verdict in `docs/ASSESSMENT.md`). `COMMANDS.txt` holds the
  current wave: the gates of `docs/ASSESSMENT.md` (matched compute, per-image
  allocation, budget-specialised decoding).

| `--backbone` | model | rate unit at 256 px | parameters |
|---|---|---|---|
| `swin` (default) | **Feature transmission.** SwinJSCC + linear channel truncation, math unchanged | *k* real channels per position; CBR = *k*/1536 | 5.3 M |
| `vit` | **Token transmission.** Plain ViT, patch 8, width 768, 10 blocks per side: tiles coded and decoded independently | 4-symbol tokens, 6 phases x 1024 positions per tile; CBR = *l*/49152 | 142.4 M |
| `hybrid` | Swin + attention, phase tokens, joint `[grid ; tokens]` trunk: the previous best AdaJSCC. Prefix only | 16-symbol tokens, 6 phases x 256 positions per block; CBR = *l*/12288 | 12.5 M |

The AdaTok backbone (learned 1D latent tokens) was removed: it did not turn tokens
into detail (`docs/NOTES.md`, wave 1). Its per-budget LoRA decoder heads are kept, on
the position-tied token backbones, as `--rate-mod lora|both`. `--cascade` adds
0.09 M parameters to `swin` and 0.02 M to `vit` at the defaults (linear stages); the
module bodies of wave c1 add 1.76 M (+33%) and 2.71 M (+1.9%).

All models fit one 24 GB RTX 3090 at batch 16 and 256 px. Estimated training
peaks: hybrid ~21 GiB, vit ~19, swin ~16; `tools/probe.py` measures the real
figure in a minute. Memory-matched is not compute-matched: per image, vit costs
about 6x the hybrid's FLOPs.

## Quick start

```sh
pip install -r requirements.txt
python tools/fetch_models.py     # once: downloads the teacher and metric networks
python tools/smoke.py            # CPU, ~2 min: all models, invariants, perception code
python tools/check_channel.py    # CPU, seconds: noise and power conventions, measured
python tools/probe.py --backbone vit --amp   # GPU, a minute: peak memory, ms/step

export JSCC_DATA_ROOT=/data      # holds datasets/DIV2K/..., datasets/Kodak, ... (or --data-root)
COMMON="--trainset DIV2K --validset DIV2K --testset Kodak --img-size 256 \
  --channel-type awgn --snr-range -2 22 --valid-freq 40 --amp --training"

python main.py $COMMON                      # baseline
python main.py $COMMON --backbone hybrid    # previous best AdaJSCC (see "Phase order")
python main.py $COMMON --backbone vit
python main.py $COMMON --backbone vit --cascade   # one module path per rate (docs/CASCADE.md)
python main.py $COMMON --backbone swin --cascade
torchrun --nproc_per_node 2 main.py $COMMON --backbone hybrid   # effective batch stays 16

python main.py --backbone hybrid --pretrained history/<run>/models/last.pt     # test only
python tools/compare_runs.py history/<run A> history/<run B> --window 5
python tools/paired.py history/<control> history/<variant> --metric psnr   # paired Kodak, CIs over images
```

A test-only run needs the model flags the checkpoint was trained with.
`COMMANDS.txt` holds the current wave of runs, ready to paste.

## Layout

```
main.py                 train / test driver (one GPU or torchrun)
COMMANDS.txt            the current wave (c2, revised: matched compute, allocation oracle,
                        specialists, LoRA budget heads, seeds), copy-paste ready
                        (the sheets of earlier waves are gone from the tree: they are in git
                        history, commit 6b02b27, and in jscc_clean.zip)
engine.py               one training epoch; the evaluation grid
configs/config.py       every knob, per-backbone defaults, validation at start-up
utils/parser.py         command line
utils/common.py         seeding, DDP, logging, checkpoints, EMA, LR schedule
utils/metrics.py        pixel metrics and the evaluation suite (MetricSuite)
utils/perception.py     frozen DINOv2 teacher, LPIPS, DISTS, classifier, detector
data/datasets.py        DIV2K / Flickr2K / CLIC / Kodak
net/network.py          JSCC: encode -> send (channel) -> decode, and diagnose()
net/channel.py          Sionna 2 AWGN / i.i.d. Rayleigh (+ a torch reference)
net/tokens.py           token interface: order, DC removal, prenorm, folding
net/loss.py             objective and reported metrics
net/cascade.py          depth cascade: stages, module kinds, per-level token post-processing
net/backbones/          swin_linear.py  hybrid.py  vit.py
net/modules/            swin.py  transformer.py  rate_mod.py  common.py
alloc/                  per-image budget policy: core.py (allocation, statistics),
                        sweep.py (curves), predictor.py (the policy), io.py
tools/                  smoke.py  check_channel.py  compare_runs.py  paired.py  probe.py
                        fetch_models.py  toy_nesting.py  alloc_sweep.py  alloc_policy.py  frontier.py
docs/NOTES.md           what the previous tree established; open questions
docs/CASCADE.md         the cascade: design, module choices per rate and interface, training
docs/ASSESSMENT.md      v2: what the paper can claim, the IEEE TWC framing, gates G-A..G-D
docs/SURVEY.md          related work by threat level: rate-adaptive JSCC, token communication,
                        adaptive tokenizers, allocation; the novelty matrix
```

## Defaults and the knobs worth knowing

Flags left unset take the backbone's defaults (`BACKBONE_DEFAULTS` in
`configs/config.py`):

- `hybrid`: `--sym-per-token 16 --token-dim 256 --depth 4 --heads 8 --rate-mod film
  --pos-scale 0.25 --zero-init --warmup-steps 2000`, latent-space blocks, uniform
  prefix sampling. These are the flags of the best previous run
  (`$ADA --tile-space latent --trunk-zero-init`), plus `--phase-order spread`.
- `vit`: `--patch-size 8 --sym-per-token 4 --token-dim 768 --depth 10 --heads 12
  --rate-mod film --zero-init --warmup-steps 2000`. Patch 8 spends the memory on
  resolution (1024 tokens per tile, 192 pixel values per token); 4-symbol tokens
  keep the hybrid's six phases, with all five CBRs on phase boundaries. The
  previous small size is `--patch-size 16 --sym-per-token 16 --token-dim 384
  --depth 6 --heads 6`. `--refine-ch 32` adds a small conv tail against
  patch-edge artefacts; it is off so the arm stays a plain transformer.
- `swin`: `--model-size base --window-size 8`, no warm-up.

- `--cascade` (swin, vit): linear stages (`--mod-depths 0`, the learned skip initialised to
  truncation) `--mod-skip proj --cascade-grad 1`, grid rate sampling; the bodies of wave c1
  are `--mod-kinds mlp mlp attn attn --mod-depths 1 2 3 4` (swin: `swin` for `attn`), see
  `docs/CASCADE.md`. `--mod-lr-mult` trains the new modules (cascade stages and LoRA
  heads) faster when fine-tuning.
- `--rate-mod` (hybrid, vit): how the decoder trunk knows the budget. `film` (default)
  scales and shifts every block, hat-interpolated over `--rate-anchors` log-spaced
  budgets; `lora` gives every block MLP a low-rank head per anchor (rank `--rate-rank`,
  AdaTok's MH-LoRA), interpolated the same way, so it stays continuous in the budget;
  `both` uses the two; `none` removes budget conditioning. The LoRA B factors start at
  zero: a `film` checkpoint loads into `both` and decodes identically at step 0.

All arms: AdamW, lr 1e-4, cosine decay to 5% (`--lr-floor`), batch 16 at 256 px,
effective batch 16 (with a smaller `--batch-size`, gradient accumulation fills
the gap), grad clip 1.0, 2000 epochs, validation every 40.

Rate: `--rate-sampling uniform` (continuous CBR in [1/48, 1/8]; prefix default), `grid`
(the five predefined CBRs; cascade default) or `sandwich` (grid, always the smallest and
largest). `--rates-per-step K` decodes K budgets from one encoder
pass; use the same K in every arm of a comparison to match compute.
`--fixed-cbr 1/16` trains a single-rate specialist. `--eval-cbrs` evaluates
anywhere; results report the CBR actually sent.

**Phase order** (`hybrid`, `vit`): token t = p*N + i carries phase p of grid
position order[i]. `spread` (default) is a bit-reversed Morton order, so a
partially sent phase refines scattered positions across each block; `raster`
refines the top rows first and is the previous tree's order. At the five
predefined CBRs (phase boundaries) the two differ only by a relabelling; they
differ at every budget in between, which is most of what uniform sampling
trains. `spread` is the one deliberate deviation from the validated setup and
is untested at scale; `--phase-order raster` reproduces the previous model
exactly.

Also: `--ema 0.999`, `--lr-schedule constant`, `--final-ckpt best`,
`--channel-type rayleigh|none`, `--equalizer zf`, `--channel-backend torch`,
`--trainset DIV2K Flickr2K CLIC`, `--resume`, `--print-step`, `--valid-metrics`,
`--test-metrics`, and the perception flags below.

## Conventions

- **CBR** = complex symbols sent / (3 H W). Baseline: k/2 complex symbols per
  16x16-px position. Token models: l tokens x sym symbols per 256-px tile.
- **Power**: unit mean power per transmitted complex symbol, per image (baseline)
  or per tile / latent block (token models). Only the prefix enters the channel.
- **Channel**: Sionna 2 blocks (`--channel-backend sionna`, the default):
  `AWGN`, and for `rayleigh` a `FlatFadingChannel` with one antenna at each end
  applied symbol by symbol, i.e. an i.i.d. h ~ CN(0, 1) per complex symbol,
  equalised with perfect CSI at the receiver (MMSE by default; `--equalizer zf`
  divides by h). no = 10^(-SNR/10) per complex symbol. `--channel-backend torch`
  is a pure-PyTorch reference with the same statistics, used by the CPU tests.
- **What a decoder knows**: how many units arrived. Never the SNR, h or the
  equaliser gain.
- **Training**: SNR uniform per image over `--snr-range`; loss = mean MSE over
  the budgets of the step.
- **Evaluation**: 8-bit outputs, per-image metrics averaged: validation PSNR,
  MS-SSIM and LPIPS; the final test the whole suite below. Each (SNR, CBR) cell
  reseeds torch and Sionna (`--eval-seed` + 1000 i + j, default 42, independent of
  `--seed`), so every checkpoint of every configuration and training seed sees the
  same channel draws. Kodak is tested at
  native resolution. `hybrid` runs Swin on the whole image and blocks only the
  bottleneck; `vit` codes and decodes 256 px tiles independently (cascade modules act per
  tile too).

## Output and the training log

`history/<run>/logs/`: `run.log`, `valid_EP####.json`, `test.json`,
`test_per_image.json`. `history/<run>/models/`: `best.pt`, `last.pt`,
`train_state.pt` (for `--resume`). The run name encodes the flags that differ
from the defaults, plus `--tag`.

Every `--print-step` optimiser steps the log also checks the decoder, at CBR
1/16 and the median evaluation SNR:

    diag cbr 0.0625 7 dB: psnr P zeroed Z swapped S common C

`zeroed` decodes with the received symbols set to zero and `swapped` with
another image's symbols. A decoder that uses the channel shows Z and S well
below P; P = Z = S means it has collapsed to a fixed output. C is the share of
the transmitted power that all images of the batch have in common: about
1/batch (0.06 at batch 16) when healthy, higher when power goes into a carrier
the DC estimate misses (it starts near 1 until the estimate has converged).

## Perceptual objective and evaluation suite

The question: can the SHORT prefixes of a variable-length code carry the
important content (objects, semantics) better, while the full budget keeps its
fidelity? `net/loss.py` adds, for every budget u decoded in a step,

- `--sem-weight`: DINOv2 patch-feature distance between reconstruction and input;
- `--align-weight`: REPA-style alignment, a small head maps the decoder's
  hidden state at mid depth to DINOv2 features of the input, which shapes the
  transmitted tokens themselves rather than only the output;
- `--imp-weight`: the squared error weighted by the teacher's CLS attention
  (mean 1, capped at 4), so short prefixes are scored mostly on what the
  teacher looks at;
- `--lpips-weight`: LPIPS-Alex, the plain perceptual-loss baseline.

Each term is scaled by w(u) = ((u_max - u)/(u_max - u_min))^gamma
(`--perc-schedule budget`, `--perc-gamma`): 1 at the smallest budget, 0 at the
full one, so the full budget keeps the plain MSE objective; `constant` applies
the terms at every budget (the ablation). Weights are relative: each term is
rescaled to `weight x` the pixel loss (detached ratio), so 0.5 means half the
pixel loss whatever the backbone or budget. The teacher (default
`vit_small_patch14_reg4_dinov2.lvd142m`, DINOv2 with registers, whose
attention maps lack the original's background artefacts) is frozen and used
only in training; the alignment head is training-only and never saved in
weight files.

The final test scores every image with `utils/metrics.py`: PSNR, SSIM,
MS-SSIM, LPIPS-Alex, LPIPS-VGG, DISTS; `cls_top1` / `cls_prob` (an ImageNet
ConvNeXt-T still sees the original's class); `det_f1` (a COCO Faster R-CNN on
the reconstruction against its detections on the original); `obj_psnr`,
`bg_psnr`, `obj_lpips` (inside / outside those boxes); `dino_sim` (the
teacher: circular for runs trained with it). Undefined values (no object found
in the original) are NaN and left out of the means. Judge a run trained on
LPIPS-Alex by LPIPS-VGG and DISTS.

Networks download on first use (torchvision from download.pytorch.org, timm
from the Hugging Face hub: set `HF_ENDPOINT` for a mirror, or pass a local
file with `--teacher-weights`); `python tools/fetch_models.py` fetches and
checks them all. A metric network that cannot load is skipped with a warning.

## What changed from jscc_rev_20260923b

**Removed.**
- SNR/CSI conditioning anywhere: the DeepJSCC-l++ stem and the SNR inputs of
  adapters and allocator. On Rayleigh the receiver still equalises; no network
  sees h, g or the SNR.
- The `ordered`, `alloc` and `attn` arms, the allocator, the budget policy and
  the stage-2 tools. The policy returns later on the hook below.
- CAB / MDTA / hybrid mixers, image-space tiling (`hybrid` always blocks in
  latent space), the linear / nearest seeds, the mask token, the DC / prenorm /
  mask-attend ablation switches, the ZF equaliser, DWA inverse-PSNR weighting,
  the fp16 scaler path, `--compile`, timm, and the old command and doc files
  (they remain in the previous zip).

**Fixed.**
- Channel: Sionna 2 AWGN and i.i.d. Rayleigh, seeded explicitly (the previous
  tree relied on `torch.manual_seed`, which does not reach Sionna's
  generators), with a pure-PyTorch reference backend; conventions measured by
  `tools/check_channel.py`. Only the prefix enters the channel, so the symbol
  count and power are exact by construction (`tools/smoke.py` checks both).
  The post-MMSE noise bound in the docstring is corrected to 1/4.
- A non-finite gradient discards the whole accumulation window, decided from the
  all-reduced gradient norm, so DDP ranks agree without an extra collective.
  There are no per-step host syncs besides that check.
- Config warnings are logged (the logger did not exist when they were raised);
  impossible geometry raises at start-up.
- FLOW (tautological after DC removal) is replaced by the decoder check above.
- Evaluation runs on rank 0 only, reseeds per cell, reports the transmitted CBR,
  and writes per-image records.
- Protocol: cosine LR with warm-up, final test on the last checkpoint (best is
  still saved), optional EMA, `--resume`.

**Flag renames.** `--backbone SwinJSCC --adapters linear` -> `--backbone swin`;
`--backbone AdaJSCC` -> `hybrid`; `--trunk-depth` -> `--depth`;
`--trunk-zero-init` -> `--zero-init` / `--no-zero-init`; `--ntm-sampling` ->
`--rate-sampling`; `--pass-channel` is gone (the channel is always on; use
`--channel-type none` for a noiseless run); `--tile-space` is gone (latent).

## Revision after wave 1

- `vit` decodes every tile on its own, as in training. The first version
  decoded native-resolution images as one joint grid it had never seen and
  lost ~14 dB on Kodak while leading on the 256 px validation. Outputs on a
  single tile are unchanged, so wave-1 checkpoints stay valid.
- The objective and all metrics run in fp32 even under `--amp`. In bf16, SSIM's
  convolutions lose its variance terms to rounding: wave-1 SSIM, MS-SSIM and
  LPIPS are invalid. PSNR was not affected.
- The log's `between` is replaced by `common`: the pre-DC between-image share
  fell to ~0.03 in the healthy hybrid and ViT runs, so it was no health signal.
- The wave-1 checkpoints were re-scored after these fixes (numbers in `docs/NOTES.md`).
- The channel is Sionna 2 by default (AWGN; i.i.d. Rayleigh through
  `FlatFadingChannel`), seeded explicitly: `torch.manual_seed` does not reach
  Sionna's generators. `--channel-backend torch` keeps the reference.
- After every validation the training random stream restarted from the same
  state (a side effect of reseeding the evaluation cells): image order, SNRs,
  budgets and noise repeated every 40 epochs; crops did not. It now continues
  from a fresh point.

## Old checkpoints

Checked numerically against the previous tree with perturbed weights: the
baseline is bit-identical (all 368 tensors match), and `hybrid --phase-order
raster` reproduces the best AdaJSCC to 6e-7, including a mid-phase budget on a
two-block image. Only `decoder.mask_token`, unused there, is ignored.

```sh
python main.py --backbone swin --pretrained <old linear run>_best.model
python main.py --backbone hybrid --phase-order raster --pretrained <old wc_gpu5_long_zi>_best.model
```

Pass the flags a checkpoint was trained with (the wave-B P6 arms used
`--pos-scale 1.0`). Previous numbers came from a constant LR and best-of-N
selection: retrain the baseline here for comparisons, or reproduce the old
protocol with `--lr-schedule constant --final-ckpt best`.

## Per-image budget policy (`alloc/`, `tools/alloc_sweep.py`, `tools/alloc_policy.py`)

A frozen codec, and a policy that gives each image its own budget so that the
AVERAGE CBR stays at a target. The receiver counts the symbols that arrive, so
a per-image budget costs no side information. The policy is the simple one:
predict every image's quality-vs-budget curve, then allocate on the curves.

1. **Curves** (`tools/alloc_sweep.py`). For each image: encode once, draw one
   noise realisation for the whole codeword, and decode every candidate budget
   (11 by default, spanning the trained range) through the prefix of that same
   noise -- common random numbers, so budgets of one image differ only in how
   much of the code arrives. Metrics: PSNR, MS-SSIM, LPIPS, LPIPS-VGG, DISTS.
   Also stored: the noise-free curve (what the transmitter can simulate
   locally), 14 image statistics and a 24-number codeword energy profile.
   Test curves use a fixed SNR grid and several noise draws per image;
   training curves use deterministic crops and random SNRs per image.
2. **Allocation** (`alloc/core.py`). The equal-slope rule: every image's curve
   is reduced to its upper concave hull and hull segments are bought in order of
   decreasing slope until the symbol budget is spent -- the exact optimum of the
   Lagrangian relaxation, checked against brute force in `tools/smoke.py`. The
   last segment is time-shared, so every method meets the target average CBR
   exactly (in expectation) and is compared with uniform allocation there.
   The objective is any mix of metrics (`--objective lpips:0.7,psnr:0.3`); one
   set of curves serves every objective and every target CBR. The baseline rule
   is PADC's (IEEE TWC 2023), `level_segments`: every image gets the least budget
   that reaches a common quality level, raised until the budget is spent
   (max-min fairness; the same greedy and time-sharing, steps taken in order of
   level instead of slope).
3. **Oracle** (`tools/alloc_policy.py oracle`). Allocation on the measured
   curves themselves: the ceiling of any per-image policy (printed beside
   `equal_q`, PADC's rule on the same curves). It selects on half
   of the noise draws and is scored on the other half (then the halves swap):
   selecting and scoring on the same draws harvests channel noise as gain
   (+0.15 to +0.6 dB on identical synthetic images; cross-fitted: none). With
   few draws the cross-fitted oracle is slightly conservative. The gate: the
   oracle's primary gain must be real (paired 95% CI over images excludes 0)
   AND large, either as quality (0.1 dB PSNR, 0.005 LPIPS/DISTS) or as the
   equal-quality bandwidth saving (5%). A saving alone does not pass: on a
   flat, noisy curve a difference far inside the noise moves the crossing a
   long way. A method that never reaches uniform's quality is charged the
   whole swept range (saving <= 0, marked `*`), never skipped.
4. **Policy** (`tools/alloc_policy.py train / eval`). A small MLP maps what the
   transmitter knows -- image statistics, codeword profile, 3 noise-free
   decodes, and the SNR (fed back) -- to the image's curve for every budget and
   metric at once; the head is monotone by construction. The target CBR and the
   objective are test-time knobs: no retraining per rate. Baselines: `uniform`
   (the fixed-rate system), `sim` (allocate on the transmitter's noise-free
   decodes at every budget: no learning, but K local decodes), `oracle`.

What it can and cannot show. No method gains at the smallest or the largest
CBR (every image already sits at the floor or the ceiling), so targets are the
interior predefined CBRs. Any prefix code can run the same policy, so it is not
an advantage of one backbone: report it on Swin too. The objective metric is
also the one being scored; judge a perceptual objective on the metrics it did
not use (LPIPS-VGG and DISTS for an LPIPS objective). The oracle also reports
whether the perceptual objective allocates differently from PSNR (share of
images whose budget differs, rank correlation): if it does not, a perceptual
reward adds nothing over a fidelity one.

## Budget schedule vs constant weights (`tools/frontier.py`)

The efficiency claim of a budget-scheduled perceptual weight w(u) is that one
model beats every CONSTANT weight somewhere, on metrics it was not trained on.
With a relative loss, a constant weight already sets the same marginal
trade-off (dB of PSNR per percent of LPIPS) at every budget, so a schedule can
only win where the real value of perception differs by budget.

The test: at every budget, the constant-weight models (weight 0 = the MSE
control, plus two or more others) give points (PSNR, perceptual metric); the
scheduled model is placed against the curve through them, on DISTS and
LPIPS-VGG (the training metric LPIPS-Alex is printed for reference only). The
unknown curve between sampled weights is bracketed from below by the chord
(reachable by decoding a share of the images with each of two models) and
from above by the neighbouring chords extended and the segment's own left end
(a concave, non-increasing curve cannot exceed them). Gaps are in
dB-equivalent (the perceptual difference at equal PSNR over the local exchange
rate). ABOVE the bound at two budgets on both metrics and BELOW the chord
nowhere means the schedule beats constant weights (LIVES); never above the
chord means it only chooses operating points constant weights already reach
(DEAD: the preference version remains). Budgets where the schedule's local
weight equals a constant one (1/48, 1/12 and 1/8 for weights 0, 0.2, 0.5 and
gamma 1) also get a direct paired comparison. The inputs are sweep files
(`tools/alloc_sweep.py`) of every model with the same seed, so all models see
the same channel noise; CIs are over images and do not include run-to-run
training noise.

## Tail probes (`--freeze`, `--top-prob`; results in `docs/NOTES.md`)

Noise-free decodes show that a nested code's last third (CBR 1/12 to 1/8)
adds nothing, with or without channel noise, while a fixed-rate 1/8 code
trained from the same checkpoint uses it (+0.6 to +1.0 dB). Two probe flags
locate the loss. `--freeze encoder|decoder` trains only the other half of a
`--pretrained` model (the frozen half stays in eval mode; the run name gets
`frz-enc` / `frz-dec`): a decoder trained for 1/8 on a frozen nested encoder,
minus one trained for 1/12, measures the information the encoder actually
puts in the tail. `--top-prob P` makes a training budget the full one with
probability P and uniform otherwise: the control for "the sampling starves
the tail".

## Hook for the bandwidth-allocation policy

For the token models, `JSCC.send(enc, units, snr)` and
`JSCC.decode(enc, rx, units)` accept an integer tensor of budgets: one per image
`(B,)`, or one per tile `(B*T,)` for `hybrid`. Decoders use key padding (`hybrid`) or
masked folding (`vit`, whose budget modulation is per image). Prefix models only: a
cascade sends one level per batch (its allocation hook would be a per-image choice
among the five levels, not built yet). `tools/smoke.py` checks that mixed budgets decode exactly like
uniform ones. Nothing is trained on non-uniform budgets yet. Read the stage-2
section of `docs/NOTES.md` before comparing a policy with fixed budgets.
