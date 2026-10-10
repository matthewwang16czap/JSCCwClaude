# Notes carried over from the previous tree

What the experiments of `jscc_rev_20260923b` established, what the audit of that
tree found, and what this tree has not tested yet. Numbers are from that tree's
docs: DIV2K training, Kodak test, AWGN.

## Established

1. **Token arm vs linear (wave C).** The best AdaJSCC (now `hybrid --phase-order
   raster`) beat linear truncation at CBR 1/48 (+0.27 dB) and 1/24 (+0.16 dB) and
   lost at 1/8 (-0.58 dB). The crossover sits near 1/16.
2. **Measurement noise.** Identical-seed runs differed by sd 0.67 dB per
   validation epoch. Compare window means (`tools/compare_runs.py`), repeat seeds,
   and prefer the cosine-decayed last checkpoint to best-of-N selection.
3. **SNR conditioning buys nothing**: +0.029 / -0.010 / -0.040 dB. Hence none here.
4. **Collapse and its cure.** The first token build sent a near-constant carrier
   and its decoder learned the dataset mean. What fixed it is kept in every token
   model: per-token DC removal in train and eval, prenorm before the power
   constraint, fp32 symbol heads, a decoder grid stream seeded with received
   content, and tokens tied to positions (`hybrid`, `vit`).
5. **Rank bottleneck.** One head shared by tokens interpolated from the grid
   limited the image-dependent output rank. Per-phase heads (P = 6) removed it.
6. **Zero-init trunk, pos-scale 0.25 and latent-space blocks** were the winning
   settings of wave C and are the `hybrid` defaults.

## Wave 1 (this tree: batch 16, 2000 epochs, AWGN, DIV2K)

PSNR, mean over SNR 1-13 dB, CBR 1/48, 1/24, 1/16, 1/12, 1/8.

| run | validation, 256 px crops | Kodak test |
|---|---|---|
| swin | 25.30 27.53 28.84 29.64 30.28 | 26.01 28.26 29.75 30.59 31.15 |
| hybrid (spread; raster within 0.01) | 25.46 27.68 28.76 29.19 29.71 | 26.17 28.39 29.49 29.84 30.26 |
| vit, patch 8, 142 M | 26.35 28.59 29.87 30.45 31.02 | invalid (joint decoding, fixed) |
| vit, patch 16, 22 M | 25.58 27.44 28.53 29.00 29.49 | invalid (joint decoding, fixed) |
| adatok, TiTok-B (removed from the tree) | 19.34 20.69 20.81 20.82 20.84 | 20.98 22.05 22.13 22.15 22.15 |
| adatok, TiTok-S (removed from the tree) | 18.71 19.56 19.78 19.86 19.90 | 20.20 20.99 21.17 21.23 21.26 |

1. **The crossover reproduces** under the new protocol: the hybrid is ahead of
   channel truncation by ~0.15 dB at 1/48 and 1/24 and behind from 1/16 up
   (-0.9 dB at 1/8 on Kodak).
2. **Spread = raster** at the five CBRs (all phase boundaries). Spread stays.
3. **The big ViT leads on crops at every CBR** (+0.74 to +1.05 dB over swin);
   the small one only at 1/48. Whether patch 8 or size drives it is open. Its
   Kodak numbers came from decoding six tiles as one grid never seen in
   training (-14 dB); now decoded tile by tile.
4. **AdaTok does not turn tokens into detail** (the backbone is gone from the tree;
   the code remains in the previous zip). Its decoder uses the channel
   (late in training: swapped ~10 dB vs psnr ~23 dB), but budgets beyond 1/24
   (128 tokens) add at most 0.15 dB at any point of training (0.34 for
   TiTok-S); at epoch 40 its output did not depend on the budget at all. The
   position-tied phase models gain 2.0-2.4 dB from 1/24 to 1/8 (swin 2.75).
   Untied 1D latents trained from scratch with MSE on DIV2K are the wrong
   inductive bias here. Caveat: this is not AdaTok's own recipe (proxy codes, perceptual and
   adversarial losses, ImageNet-scale data).
5. **The pre-DC between-image share is no health signal**: it fell to ~0.03 in
   the healthy hybrid and ViT (DC removal makes a large constant free).
   Replaced in the log by `common`.
6. All runs trained at ~38 img/s whatever their size (big ViT 30): with seven
   runs on one machine, data loading rather than the GPU probably set the pace.
7. Wave-1 SSIM, MS-SSIM and LPIPS were computed in bf16 and are invalid;
   the retest re-scored them. Kodak retest (fp32 metrics, ViT tile by
   tile): the big ViT leads swin by +0.86 +0.90 +0.78 +0.51 +0.46 dB, with
   better MS-SSIM and LPIPS at every CBR.
8. Wave 1 used the torch channel; after every validation its training random
   stream restarted from the same state (fixed, probably a small effect).

## Perception wave (p1): what would count as evidence

Fine-tunes of the wave-1 ViT and SwinJSCC with the budget-scheduled objective
(README, "Perceptual objective"), against a control fine-tuned identically on
MSE. The claim holds if, against the control,
1. at 1/48-1/16 the object and semantic metrics improve (det_f1, obj_psnr,
   obj_lpips, cls_prob, DISTS, LPIPS-VGG), with PSNR allowed to drop there;
2. at 1/8 PSNR stays within ~0.1 dB (the schedule turns the terms off);
3. the constant-schedule ablation pays PSNR at 1/8 for similar small-budget
   gains (the schedule is the contribution);
4. the LPIPS-only run improves full-image perceptual metrics but not the
   object/semantic ones at small budgets ("smarter than a perceptual loss").
dino_sim is the training teacher: never evidence for the sem/align runs.
Kodak has 24 images and not all contain COCO objects; use the paired
per-image records (test_per_image.json) before trusting small differences.

## Tail probes: where a nested code's tail dies

(Carried over from the retired `COMMANDS_TAIL.txt`; the `--freeze` and `--top-prob`
probe flags stay, see the README.) ViT, Kodak; tail gain = PSNR at 1/8 minus PSNR at 1/12.

| | nested code | fixed-rate 1/8 code |
|---|---|---|
| no channel noise (P3b / P1) | -0.02 dB | +1.00 dB |
| trained 16-22 dB, at 22 dB | +0.04 (P2) | +0.60 (P3a) |
| trained -2..22 dB, at 13 dB | +0.21 (p1) | +1.01 (P3c) |

Nesting, not the channel, empties the tail: even with no noise the nested code's last
third carries nothing. The fixed 1/8 specialist beats the nested model at 1/8 by +0.29 dB
at 1 dB SNR, rising to +0.71 dB at 22 dB (P3c vs p1), while a specialist is useless at
small budgets. Reference, P3b (nested, noise-free): 28.67 31.59 33.71 34.17 34.15 dB at
1/48 1/24 1/16 1/12 1/8.

The question that was open: does the tail's information die (a) in the encoder, which
never puts new information there, (b) in the nested decoder, which does not read it, or
(c) in the sampling, which starves the tail? Probes (all noise-free, 100-200 epoch
fine-tunes): Q2a freeze the nested encoder, train a decoder for 1/8 only; Q2b the same
for 1/12 (tail information sent = Q2a(1/8) - Q2b(1/12); <= ~0.15 dB means (a), >= ~0.5 dB
means (b)); Q1 `--top-prob 0.5` (tail >= +0.5 dB with the small budgets within ~0.1 dB of
P3b means (c), a recipe fix any architecture must beat); Q3 freeze the nested decoder and
train an encoder for 1/8 (>= ~34.6 dB: the nested decoder can read an informative tail).
The results of Q1-Q3 are not recorded in this tree. What each answer calls for: (a) a
rate-causal encoder (tokens attend only to earlier tokens, or a branch fed with the
prefix's reconstruction); (b) budget-gated decoder capacity; (c) a sampling recipe. The
depth cascade (`docs/CASCADE.md`) is a fourth answer: stop making one code serve every
rate.

## Audit findings that shaped this tree

- The CBR accounting was correct: masked symbols were zeroed before and after
  the channel, and power was normalised over transmitted symbols only. This tree
  sends only the prefix, which makes that true by construction.
- `alloc`'s capacity charge used AWGN capacity on Rayleigh, a batch-mean SNR and
  the Shannon limit at short block lengths. `--power-alloc` applied its profile
  after normalisation and broke the power constraint. Both left with the arm.
- The `attn` control was handicapped: bf16 head before DC removal, no zero-init,
  default init, no pos-scale, per-channel DC, 256 vs 1792 tokens. It is dropped
  here. If the cut-axis question returns, rebuild it with parity on those points.
- Grid sampling decoded 5 rates per step against 1 for uniform sampling, so the
  arms were not compute-matched. Use the same `--rates-per-step` in every arm.
- `torch.manual_seed` does not control Sionna's noise (checked with Sionna
  2.0.1: it draws from its own generators), so every evaluation in the previous
  tree drew fresh channel noise. This tree seeds Sionna explicitly.

## Before a policy is compared with fixed budgets (the old stage-2 gate)

- **Time-sharing baseline.** Within a phase, each extra token refines one more
  position, so MSE falls roughly linearly in l and PSNR is convex between phase
  boundaries. A fixed mid-phase budget is then beaten by giving half the images
  the phase budget below and half the one above, at the same average rate
  (Jensen). At 1/32 that gap was about 0.18 dB, roughly the old gate's decision
  threshold, which biased the gate toward the policy. Compare a policy with the
  upper concave hull of the fixed-budget curve, not with the curve itself.
- **Phase-aligned candidates** (1/48, 1/24, 1/16, 1/12, 1/8 at sym 16) avoid the
  issue. If finer candidates are needed, `--phase-order spread` should make the
  within-phase curve more concave (neighbours can borrow refined positions).
  Measure it before relying on it.
- **Common random numbers.** Evaluate every candidate budget of an image on the
  same channel draws. The evaluation here reseeds per cell; a policy evaluator
  should also reuse draws across the budgets of one image.
- **A regressor may suffice.** With per-image PSNR curves over about 9 budgets,
  choosing a budget is a one-step decision. Regressing or ranking the curve may
  be all the policy needs; GRPO is not required.

## Not yet tested in this tree

- `vit` on Kodak with tile-by-tile decoding (done in the retest, above), and whether
  the big ViT's lead comes from patch 8 or from size: a patch-8 ViT at the
  hybrid's compute is `--token-dim 384 --depth 6 --heads 6` (~22 M parameters).
- Seams: `vit` reconstructs 256 px tiles independently.
- EMA, other seeds, budgets between the five CBRs.
- Rayleigh. The decoders are channel-blind by design, and the MMSE equaliser
  assumes unit symbol variance, which the per-token gain violates (up to e^4
  between tokens). Whether that costs anything is open.

## Compute of the backbones (measured 2026-10-08)

`torch.utils.flop_counter` on CPU, one 256x256 image at the top budget, encoder + decoder:

| model | params | encoder GFLOPs | decoder GFLOPs | total |
|---|---|---|---|---|
| swin small | 3.31 M | 4.87 | 4.87 | 9.7 |
| swin base (the baseline) | 5.27 M | 9.26 | 9.26 | 18.5 |
| swin large | 12.67 M | 20.89 | 20.89 | 41.8 |
| hybrid (Swin + token trunk) | 12.54 M | 20.57 | 20.57 | 41.1 |
| vit p16 384x6 s16 (wave-1 small) | 22.11 M | 5.62 | 5.62 | 11.2 |
| vit p8 384x6 s4 | 21.56 M | 21.93 | 21.93 | 43.9 |
| vit p8 512x8 s4 | 50.83 M | 51.79 | 51.79 | 103.6 |
| vit p8 768x10 s4 (wave-1 best) | 142.39 M | 145.33 | 145.33 | 290.7 |

The wave-1 ViT lead over Swin base (+0.46 to +0.90 dB on Kodak) costs 27x the parameters and
16x the FLOPs. The two token models nearest Swin's compute (vit p16 at 11 GFLOPs, hybrid at 41)
lost to it from 1/16 up. ViT p8 384x6 and Swin large are the matched pair (~42-44 GFLOPs);
Swin large is also the published SwinJSCC size class.

## Cascade wave c1: what would have counted as evidence (the plan)

Background: the tail probes (above) showed the nested code's last third
carries almost nothing (-0.02 dB from 1/12 to 1/8 without noise, against +1.00 dB for a
fixed-rate code) and a 1/8 specialist beats the nested ViT by 0.29-0.71 dB.
`--cascade` (docs/CASCADE.md) gives each rate its own module path. `tools/toy_nesting.py`
shows a LINEAR-Gaussian nested code loses <= 0.2 dB, so the penalty is a nonlinear effect.

Fine-tunes of the wave-1 swin and vit checkpoints, 400 epochs, the p1 recipe, against a
control fine-tuned identically on the SAME five budgets (`--rate-sampling grid`, no
modules). Arms per interface: linear (depth 0), all-mlp, mixed (mlp, mlp, spatial,
spatial), all-spatial; then gradient sharing 0.5 and a reversed depth schedule. Reading
(paired Kodak table, `tools/paired.py`, plus 5-validation windows):

1. any arm - control >= +0.3 dB mean over the five CBRs (CI above 0 at >= 3 CBRs) is the
   bar to continue; < +0.1 dB means the nesting penalty is not what a cascade removes;
2. linear - control > 0 at the low CBRs: rate-specific paths help by themselves; if only
   the module arms gain, the effect is capacity: add the parameter-matched adapter control;
3. spatial vs mlp per level: the factorial mmmm / mmss / ssss, read at the levels each
   stage owns (stage 1 -> 1/12, stage 2 -> 1/16, stage 3 -> 1/24, stage 4 -> 1/48);
4. top rate: every arm - control at 1/8 against the 0.3-0.7 dB specialist gap;
5. depth schedule d1234 vs d4321 (same parameters): "deeper for fewer channels";
6. swin pays +33% parameters, vit +1.9%: do not credit Swin's gain to structure before
   the matched control.

Warm starts keep `z0` ordered. The fair test of the top rate is from scratch (`--mod-skip
none`, `--cascade-grad` < 1): wave c2, after these results.

## Cascade wave c1: results (2026-10-08)

Kodak, last checkpoint, PSNR in dB, mean over SNR 1-13, variant minus the grid-trained prefix
control (same w1 start, same 400 epochs, same five budgets), paired over the 24 images
(`tools/paired.py`). Rows marked * are the sum of two paired tables (variant - mixed, mixed -
control), exact for means, no interval printed.

| vit (token) | 1/48 | 1/24 | 1/16 | 1/12 | 1/8 | all |
|---|---|---|---|---|---|---|
| control (absolute) | 27.110 | 29.218 | 30.584 | 31.154 | 31.659 | 29.945 |
| linear (depth 0) | +0.028 | +0.036 | +0.048 | +0.058 | +0.069 | +0.048 |
| all-mlp | +0.011 | +0.033 | +0.049 | +0.054 | +0.069 | +0.043 |
| mixed mlp mlp attn attn | +0.031 | +0.045 | +0.042 | +0.053 | +0.076 | +0.049 |
| all-attention | +0.010 | +0.021 | +0.039 | +0.059 | +0.064 | +0.039 |
| mixed, --cascade-grad 0.5 * | -0.211 | -0.021 | +0.034 | +0.082 | +0.142 | +0.005 |
| mixed, depths 4 3 2 1 * | +0.017 | +0.030 | +0.053 | +0.065 | +0.074 | +0.047 |

| swin (feature) | 1/48 | 1/24 | 1/16 | 1/12 | 1/8 | all |
|---|---|---|---|---|---|---|
| control (absolute) | 26.218 | 28.340 | 29.798 | 30.642 | 31.206 | 29.240 |
| linear (depth 0) | +0.022 | +0.033 | +0.012 | +0.023 | +0.007 | +0.019 |
| all-mlp | +0.043 | +0.033 | +0.013 | +0.032 | +0.018 | +0.028 |
| mixed mlp mlp swin swin | +0.045 | +0.046 | +0.012 | +0.024 | +0.007 | +0.027 |
| all-window | +0.036 | +0.036 | +0.017 | +0.027 | +0.009 | +0.025 |
| mixed, --cascade-grad 0.5 * | -0.013 | +0.022 | +0.022 | +0.070 | +0.074 | +0.035 |
| mixed, depths 4 3 2 1 * | +0.055 | +0.042 | +0.021 | +0.032 | +0.014 | +0.033 |

1. **Real but negligible.** Every arm is above the control with intervals clear of 0 at
   almost every CBR, by +0.02 to +0.08 dB: 6-15x below the +0.3 dB gate (G1 fails), and the
   size of run-to-run training noise (the 5-validation windows have sd 0.05-0.15 dB; no
   second seed was run). The validation windows agree in sign (vit +0.05 to +0.13, swin ~0).
2. **The linear arm takes all of it.** A per-rate linear map with no body (0.02 M parameters
   on vit, 0.09 M on swin) matches every module arm within 0.01 dB. Capacity at the
   bottleneck buys nothing. Answer to "which module at which rate": linear, at every rate,
   on both interfaces (now the default).
3. **No stage wants spatial context.** The factorial reads late stages (mixed - all-mlp at
   1/24, 1/48): vit +0.012 / +0.019, swin +0.013 / +0.001; early stages (all-spatial - mixed
   at 1/16, 1/12): vit -0.002 / +0.005, swin +0.006 / +0.003. Nothing reaches +0.05.
   Early-stage attention costs vit 0.02 dB at 1/48 and 1/24.
4. **The depth schedule does not matter** (1 2 3 4 vs 4 3 2 1 within 0.015 dB): a
   consequence of 2, so "deeper for fewer channels" is not supported.
5. **Gradient sharing moves quality between rates and adds none.** `--cascade-grad 0.5`
   against the default: vit +0.066 at 1/8, -0.242 at 1/48; swin +0.067 / -0.058. This
   knob only scales what the deep exits send into the shared code and ENCODER, so the
   encoder is a contested resource; the code layout is not.
6. **Top rate**: the best 1/8 gain is +0.14 dB (vit, grad share 0.5, paid at 1/48) against
   the 0.29-0.71 dB a 1/8 specialist gained in the tail probes (other protocol). G4 fails.
7. Side metrics: vit LPIPS-VGG better by 0.003-0.005 at every CBR (96-100% of images),
   MS-SSIM +0.001; swin flat (LPIPS-VGG 0.001 worse).
8. Transient: at epoch 40 vit depths 4 3 2 1 was -0.2 to -1.6 dB and swin all-window -0.6 to
   -0.8 dB below the control (4x module learning rate on fresh bodies); both recovered by
   epoch 240. The sheet's "-1 dB at epoch 40 means stop" rule was too strict.

Reading: in these models the nesting penalty does not live in the code layout at the
bottleneck. What is left is the capacity the rates share in the backbones, and point 5
puts at least part of it in the encoder. Wave c2 measures the specialist gap under this
protocol and splits it between encoder and decoder.

## Wave c2, re-planned (2026-10-08)

The wave c2 planned above (the specialist gap, split between encoder and decoder) was
re-planned before it ran, after the reassessment (`docs/ASSESSMENT.md` v2). Kept: the 1/8 and 1/48 specialists (vit and swin) and the vit 1/8 run with the encoder
frozen (the decoder's share of the gap, the ceiling for any decoder-side budget head).
Dropped: the encoder-only runs. Added: the matched-compute pair (vit p8 384x6 against swin
large, from scratch, gate G-A), the allocation oracle on the wave-1 models (G-B), AdaTok's
per-budget LoRA heads in the vit decoder trunk (`--rate-mod both`), alone and with the
linear exits, and a second seed of the c1 control and of the c1 linear exits (G-C).
`--eval-seed` (new, default 42) keeps the test's channel draws fixed across training seeds,
so the seed-43 runs pair with the c1 runs; seed-42 runs are scored exactly as before.

## Problem P1: mixed-SNR prefixes (2026-10-10)

`tools/mixed_snr.py`, Kodak, two equal chunks per prefix (chunk 1 = the first half of the
tokens), every pair of chunk SNRs from 1, 4, 7, 10, 13 dB. Three models: the w1 ViT, its
400-epoch constant-SNR fine-tune (s1) and the same fine-tune on 4-chunk piecewise SNRs
(snrc4, `--snr-chunks 4`).

| | 1/24 | 1/12 | 1/8 |
|---|---|---|---|
| constant 1 / 7 / 13 dB, w1 | 27.27 / 29.40 / 30.56 | 28.94 / 31.35 / 32.76 | 29.73 / 31.86 / 32.98 |
| snrc4 - s1 at constant SNR | -0.06 to -0.07 | -0.06 to -0.10 | -0.06 to -0.10 |
| best symmetric rule, MAE / p95 (s1) | noise 0.23 / 0.63 | noise 0.45 / 0.86 | 0.68 / 1.48 |
| best symmetric rule, MAE / p95 (snrc4) | noise 0.28 / 0.86 | 0.55 / 1.37 | 0.71 / 1.65 |
| order effect, best chunk first - last (s1 / snrc4) | +0.34 / +0.55 | +0.89 / +1.10 | +1.35 / +1.42 |
| [13, 1] dB (s1 / snrc4) | 28.02 / 29.35 | 30.80 / 31.73 | 32.57 / 32.70 |

1. **No symmetric rule works** (mean dB, mean noise, mean capacity: MAE 0.23-0.71 dB, p95 up
   to 2 dB), and none can: the code is ordered, so WHICH tokens a slot carries matters. The
   first half of a prefix is 2-20x more noise-sensitive than the second.
2. **A constant-SNR decoder mis-reads mixed prefixes.** w1 and s1 at 1/24: a clean first chunk
   and a 1 dB second chunk score BELOW a 7 dB first chunk with the same second chunk (27.93 <
   28.16): the SNR-blind decoder infers the noise level from what it sees and over-trusts the
   noisy half. Piecewise training removes it (+1.33 dB on that profile, +0.94 at 1/12) at a
   constant-SNR cost of 0.06-0.10 dB.
3. **After piecewise training the distortion is ADDITIVE over chunks.** A two-way additive fit of
   the 5 x 5 MSE table has max error 0.06 / 0.01 / 0.03 dB (1/24 / 1/12 / 1/8) for snrc4, against
   0.53 / 0.39 / 0.09 for s1. A compact form fits as well (`alloc/utility.py`):
   D = Dsrc(L) + C(L) [rho phi(n1) + (1 - rho) phi(n2)], phi(n) = n / (1 + n / kappa), ONE kappa
   = 1.78 for all CBRs; rho = 0.68 / 0.80 / 0.94; rms error 0.04 / 0.02 / 0.02 dB (max 0.09). This is the
   utility model a scheduler needs: separable over slots, position- and SNR-aware.
4. **The late tokens act like redundancy, not detail.** The fitted noise-free quality is 31.05 /
   33.41 / 33.52 dB at 1/24 / 1/12 / 1/8: from 1/12 to 1/8 almost no new source information,
   while the noise sensitivity C drops from 14.6e-4 to 11.1e-4 (the same "tail carries
   nothing" seen without noise in the tail probes, now with a reason: it buys robustness).
5. Per-SNR, for the literature: w1 at 1/12 is 28.94 / 30.26 / 31.35 dB at 1 / 4 / 7 dB, about
   on par with PADC's DeepJSCC-V (read off its Fig. 11) and 1.8-1.9 dB below JSCCformer-f's
   two-block-feedback numbers (30.84 / 32.05 / 33.16; 1.6-1.8 below its random-SNR "lite"
   version). About 1 dB of that is feedback, judging by its m = 1 vs m = 2 ablation on CIFAR
   (+0.8 to +1.05 dB); the rest is the codec. JSCCformer-f trains on ImageNet crops, this tree
   on DIV2K.

Illustrative scheduling (`tools/schedule_sim.py` on a 6-phase utility EXTRAPOLATED from the
2-chunk means: lengths 1, 3, 5 interpolated, phase shares assumed geometric; 8 users, 24
slots, mean SNRs U[0, 15] dB, Rayleigh block fading; NOT a result, a plausibility check):
PF beats round robin by +1.09 dB (multi-user diversity), and with every user on the same
image the content-aware policies only tie PF (+0.02). With a SYNTHETIC content spread
(half the images saturating after 2 phases), equal-slope lengths fixed before the frame
plus PF timing gain +0.47 dB over PF, re-planning them every slot +0.46, the myopic greedy
+0.38. So the scheduler has two timescales: content-aware lengths, channel-aware timing; the
size of the content part on REAL curves is the open number (G-B's oracle and the per-image
utility of the next run).

## Problem P1, step 2: the utility per phase, first scheduling numbers (2026-10-10)

COMMANDS.txt section 6 on the server: `tools/mixed_snr.py --design full --chunks phase` on
Kodak (24 images; per CBR 1/48 ... 1/8: the 5 constant profiles, each phase alone off 7 dB,
10 held-out random profiles), `tools/utility_fit.py` per image, `tools/schedule_sim.py` on
the fitted file (2000 frames), and 30 simulated frames of every policy decoded by the codec.

| | snrc4 (piecewise-trained) | s1 (constant-SNR control) |
|---|---|---|
| kappa; fit rms | 1.769; 0.037 dB | 2.441; 0.092 dB |
| rho at 1/8 (phase 1 ... 6) | 0.51 0.25 0.17 0.05 0.010 0.006 | 0.51 0.27 0.18 0.04 0.000 0.000 |
| held-out MAE / p95 per image, 1/24 ... 1/8 | 0.03-0.07 / 0.10-0.23 dB | 0.12-0.29 / 0.39-0.79 dB |
| best symmetric (noise) rule, MAE | 0.15-0.51 dB | 0.21-0.54 dB |
| replay of 1862 scheduled histories: bias, MAE, p95 | +0.009, 0.083, 0.29 dB | +0.102, 0.195, 0.79 dB |
| decoded mean PSNR on those histories, rr / pf / fixed_slope | 29.84 / 31.03 / 31.07 | 29.47 / 30.98 / 30.98 |

1. **P-B passes for the piecewise-trained decoder.** Per image and per phase the additive
   model predicts held-out profiles within 0.03-0.07 dB (p95 <= 0.23) and real schedules
   within 0.08 dB (p95 0.29), with no bias; per policy the predicted means are within 0.04 dB
   of the decoded ones. The constant-SNR decoder is predicted 2-4x worse and decodes the same
   schedules 0.04-0.36 dB worse under every policy but max-SNR (most under round robin, which
   sends 22% of its phases below 0 dB; max-SNR's slots are all good, where snrc4's constant-SNR
   cost of 0.06 shows): piecewise training is worth having whatever the scheduler.
2. **rho depends on the position, not on the prefix length** (for L >= 3: 0.51 / 0.25 / 0.17
   / 0.05 / 0.01 / 0.01), and phases 5-6 carry under 2% of the noise sensitivity while still
   adding 0.35-0.44 dB each at 1 dB SNR (0.08-0.13 at 13 dB): the tail is redundancy for the
   head, as note 4 of P1 said.
3. **Mean-PSNR scheduling on the real curves (gate P-C as stated): fails.**

   | vs pf, dB [95% CI] | main (K 8, T 24) | same image | no fading | T 16 | T 36 | K 16, T 48 | pred noise 0.1 |
   |---|---|---|---|---|---|---|---|
   | rr | -1.12 | -1.09 | 0 | -0.90 | -1.17 | -1.42 | -1.12 |
   | fixed_eq (PADC's rule) | -1.06 | -0.46 | -1.03 | -0.86 | -0.37 | -1.17 | -1.07 |
   | fixed_slope | +0.050 [0.047, 0.053] | +0.05 | 0 | -0.02 | +0.05 | +0.05 | +0.05 |
   | adaptive_slope | +0.050 | +0.05 | 0 | -0.02 | +0.06 | +0.05 | +0.05 |
   | greedy | -0.23 | +0.03 | 0 | -0.20 | -0.26 | -0.24 | -0.23 |
   | greedy_rel | -0.01 | +0.24 | 0 | -0.04 | -0.05 | -0.02 | -0.02 |

   PF takes the multi-user diversity gain (+1.1 dB over round robin) and content-aware LENGTHS
   add +0.05 dB on Kodak: per-image curves differ too little in slope for the equal-slope rule
   to matter at 2-5 phases per user (G-B's oracle will say the same for one link). PADC's
   equal-quality rule trades 1.06 dB of mean for +1.0 dB on the 5th-percentile user. One hint:
   greedy_rel gains +0.24 dB when content is removed (same image), i.e. a channel-timing gain
   PF does not see, which content diversity then drowns in its myopia.

Designing the next policies on a stand-in (a utility with the measured MEAN curves of the
snrc4 table, the fitted kappa and rho, and a Kodak-like content spread of 1.8 dB sd; script
not in the tree): it reproduces the real ordering (fixed_slope +0.09 against the real +0.05,
greedy_rel +0.04 against -0.01), so the numbers below are plausible, NOT results. Section 7
runs them on the real utility and decodes them.

4. **Equal shares with PF timing (`fixed_uni`) tie PF (+0.05)**: what PF needs is the
   timing, not online lengths. For the mean objective one model per CBR with PF timing would
   do as well as the prefix code (and skip the nesting penalty).
5. **The advantage index (`slope_adv`) gains +0.52 dB over PF** (+0.39 to +0.74 over T 16-36,
   K 16, equal mean SNRs, prediction noise; +0.47 with the same image; p5 user +0.6-0.9 dB):
   serve the user whose predicted PSNR gain from THIS slot exceeds its gain from a slot at its
   mean SNR by the most (lengths: fixed_slope's). Decomposition on the stand-in: with rho made
   uniform +0.31, with phi made linear (kappa 100) +0.61. So about 0.3 dB is valuing a slot
   in PSNR rather than in rate (a user near its noise-free quality gains little from a good
   slot and takes the bad ones; PF, on log2(1 + snr), cannot see that) and about 0.2 dB is the
   position (a noise-sensitive head phase waits for a peak, an insensitive tail phase takes a
   fade). Variants: the gain ratio instead of the difference +0.39, the reference at mean + 2
   dB +0.56, equal shares instead of equal-slope lengths +0.47.
6. **A quality target changes the picture** (`--target`: the share of users whose image
   reaches Q dB by the deadline, the QoS form of PADC's own objective). On the stand-in, Q =
   30 dB, K = 8:

   | satisfied | T 16 | T 24 | T 36 |
   |---|---|---|---|
   | pf (blind to the target) | 34.8% | 60.8% | 76.7% |
   | pf_len (content-blind lengths at the mean SNR) | 34.2% | 57.0% | 77.9% |
   | padc (per-image length at the mean SNR, shortest first, PF timing) | 57.0% | 70.0% | 75.7% |
   | padc_adv (the same, advantage timing) | 64.0% | 77.8% | 82.8% |
   | stop (online: stop when the realised SNRs got the image there) | 38.1% | 71.5% | 81.5% |
   | stop_adv | 44.7% | 78.2% | 87.1% |
   | online (stop_adv + shortest remaining need first, spare slots open to all) | 66.5% | 84.5% | 87.1% |
   | online - padc, points [95% CI] | +9.5 [8.2, 10.9] | +14.5 [13.1, 16.0] | +11.5 [10.2, 12.8] |

   Three parts, each needing a property of this codec: per-image sizing (the curves; padc vs
   pf_len: +13 / +23 points at T 24 / 16, none at T 36), online stopping from the realised
   slot SNRs (encode-once prefix + the mixed-SNR utility; stop_adv vs padc_adv: +0.4 / +4.3
   points at T 24 / 36, what makes loose frames work), and advantage timing (+6-8 points on
   padc, +6-7 on stop). Admission matters when the frame is tight (online vs stop_adv: +22
   points at T 16, where stopping alone loses to padc's shortest-first). Tuning
   on the stand-in: the planning SNR for future phases at mean + 0-3 dB and an overcommitted
   admission change little once spare slots open to all; mean SNR, no overcommit is kept.

So the scheduler's value is not where docs/PROBLEM.md first put it (content-aware lengths for
mean quality): it is quality- and position-aware TIMING (any objective) and, for a quality
target, per-image sizing with online stopping. Section 7 decides both on the real utility.
