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

## Problem P1, step 3: the advantage index and the quality target, real utility (2026-10-10)

COMMANDS.txt section 7 on the server: `tools/schedule_sim.py` on `results/utility_snrc4.json`
(2000 frames per case), then 60 frames of schedules per objective decoded by the snrc4 codec
(and the mean ones by s1), with decoded differences paired by frame.

**Mean PSNR, vs pf (dB, simulator; 95% CIs within +-0.012):**

| | main (K 8, T 24) | same image | T 16 | T 36 | K 16, T 48 | equal means 5 dB | pred noise 0.1 |
|---|---|---|---|---|---|---|---|
| fixed_uni | +0.050 | +0.051 | -0.021 | -0.002 | +0.046 | +0.040 | +0.053 |
| fixed_slope | +0.050 | +0.051 | -0.017 | +0.049 | +0.045 | +0.040 | +0.053 |
| slope_adv | **+0.476** | +0.489 | +0.343 | +0.464 | +0.384 | +0.452 | +0.477 |

slope_adv also lifts the 5th-percentile user by +0.46 dB and the worst user of a frame by
+0.69 dB, and sends 1.1% of its phases below 0 dB (pf 3.3%). **Decoded** (snrc4, 60 frames x 8
users): slope_adv - pf **+0.465 dB [0.393, 0.533]**, fixed_uni +0.057, fixed_slope +0.061; the
model's error on these schedules: bias -0.002, MAE 0.087, p95 0.27 dB. With the constant-SNR
decoder s1 on the same schedules slope_adv still gains +0.461 [0.371, 0.544] (s1 is predicted
worse: bias +0.12, MAE 0.22), and piecewise training is worth +0.04 dB under slope_adv, +0.03
under pf, +0.33 under round robin: the advantage index avoids the fades a mis-trained decoder
handles worst. **Gate P-C1 passes** (>= 0.3 dB decoded).

**Quality target, share of users at the target (simulator; online - padc with 95% CI):**

| | pf | pf_len | padc | padc_adv | stop | stop_adv | online | online - padc |
|---|---|---|---|---|---|---|---|---|
| 30 dB, K 8, T 24 | 63.5% | 60.7% | 65.7% | 68.9% | 69.6% | 73.0% | **75.0%** | **+9.4 [8.9, 9.8]** |
| 28 dB | 81.3 | 80.3 | 83.6 | 86.5 | 87.9 | 90.8 | 90.9 | +7.3 |
| 32 dB | 41.9 | 42.1 | 44.6 | 50.1 | 45.2 | 49.5 | 55.0 | +10.4 |
| T 16 | 48.5 | 48.8 | 58.8 | 65.0 | 54.1 | 56.3 | 67.4 | +8.6 |
| T 36 | 71.0 | 69.7 | 69.0 | 73.1 | 72.7 | 76.1 | 76.2 | +7.2 |
| K 16, T 48 | 66.1 | 62.7 | 69.2 | 71.4 | 73.9 | 77.2 | 78.1 | +8.9 |
| same image | 64.1 | 61.0 | 66.2 | 69.5 | 66.3 | 67.9 | 73.2 | +6.9 |
| pred noise 0.1 | 63.5 | 60.5 | 65.4 | 68.3 | 65.5 | 68.3 | 69.5 | +4.2 |
| plan at mean + 2 dB | 63.5 | 61.4 | 64.5 | 70.3 | 69.6 | 72.8 | 74.7 | +10.2 |

**Decoded** (30 dB, 60 frames): online 76.0% (predicted 76.9%), padc 66.0% (66.2%);
online - padc **+10.0 points [7.3, 12.9]**, online - pf +13.5; aiming 0.2 dB above the target:
+9.6 [7.1, 12.1] with 0.2% of users predicted to make it but decoding below (0.8% without
the margin). The model's error on these schedules: bias -0.011, MAE 0.064, p95 0.18 dB. **Gate
P-C2 passes** (>= 5 points decoded, missed <= 3%).

Reading the target table (main case): per-image sizing is worth +5.0 points (padc - pf_len),
advantage timing +3.2 on padc's lengths and +3.4 on online stopping, stopping on the realised
SNRs +3.9 (stop - padc; +4.1 with advantage timing on both), admission +2.0 (online -
stop_adv; +11.1 at T 16, where stopping alone loses to padc's shortest-first). With every user
on the same image online still leads by +6.9: most of the gain is the channel side (stopping,
timing, admission), which only the encode-once prefix with a mixed-SNR utility allows; the
content side adds ~2.5 points on Kodak. The one weak case: curves known only to ~10% in log
MSE (pred noise 0.1) halve the lead to +4.2 -- the transmitter's knowledge is the next
question (section 8: calibrating each image from 2P of its own constant-SNR decodes).

## Problem P1, step 4: a realistic transmitter, correlated fading (2026-10-11)

COMMANDS.txt section 8 on the server. The digital baseline (8d) did not run: `bpgenc` was not
on PATH and the sheet's `| tee` hid the error (fixed in section 9: HEIC through pip, errors
kept in the files).

**What a transmitter can know** (`utility_fit.py --calib`): kappa and rho cross-fitted on the
other half of the images, each image's Dsrc and C from its own constant-SNR decodes only.

| errors on the profiles it did not use, MAE / p95 (dB) | 1/24 | 1/16 | 1/12 | 5/48 | 1/8 |
|---|---|---|---|---|---|
| calibrated at 1 + 13 dB (12 decodes per image) | 0.051 / 0.15 | 0.064 / 0.17 | 0.047 / 0.13 | 0.042 / 0.13 | 0.030 / 0.10 |
| calibrated at 4 + 10 dB (12 decodes) | 0.044 / 0.14 | 0.054 / 0.17 | 0.037 / 0.11 | 0.037 / 0.12 | 0.029 / 0.10 |
| calibrated at all 5 SNRs (30 decodes) | 0.048 / 0.17 | 0.058 / 0.19 | 0.038 / 0.11 | 0.039 / 0.14 | 0.027 / 0.09 |
| full fit (every profile of the image), held-out random | 0.050 / 0.19 | 0.069 / 0.19 | 0.034 / 0.10 | 0.056 / 0.23 | 0.044 / 0.12 |

Twelve decodes per image are as good as the full mixed-SNR measurement of that image: the
codec-level constants (kappa, rho) transfer across images, and the per-image part is two
numbers per prefix length. **Schedulers deciding on that belief** (users scored on the fit):
slope_adv - pf +0.477 dB (with exact curves +0.476); online - padc at 30 dB +8.4 points [7.9,
8.9] (exact +9.4), +8.9 aiming 0.3 dB above, +8.1 at T 16, +9.5 at 32 dB, +7.0 at 28 dB.
**Decoded** (60 frames): slope_adv - pf **+0.464 dB [0.392, 0.529]**; online - padc **+11.0
points [8.3, 13.8]** (online - pf +14.0), no user predicted to make it decoded below (margin
0.3); the model's error on these schedules MAE 0.065-0.089 dB. **Gate P-C3 passes.**

**Correlated fading** (Gauss-Markov slot gains, belief, plan at the mean SNR):

| slot correlation | 0 | 0.5 | 0.9 | 0.98 |
|---|---|---|---|---|
| pf - rr, dB | +1.12 | +1.02 | +0.53 | +0.13 |
| slope_adv - pf, dB | +0.48 | +0.52 | +0.58 | +0.34 |
| online - padc at 30 dB, points | +8.4 | +8.5 | +8.9 | +5.6 (online_pf +7.8) |

The multi-user diversity gain fades with the correlation (24 slots at 0.98 is nearly one
channel state per frame), the advantage index keeps most of its lead, and the QoS lead stays
>= 5.6 points. At 0.98 advantage timing starts to hurt (padc_adv < padc, online < online_pf):
the index compares the slot with the MEAN SNR, but a user in a long fade will not see its
mean again within the frame. `--plan conditional` (section 9) compares with the SNR expected
for the rest of the frame given the current slot.

## Problem P1, step 5: one model per CBR, slow fading (2026-10-11)

COMMANDS.txt section 10 on the server. The digital baseline did not run again: the sheet
pointed `digital_rd.py` at `$JSCC_DATA_ROOT/Kodak`, the data is under `datasets/Kodak`
(fixed: the tool now resolves test sets like the trainer). No Flickr2K, so no data run.

**The nesting penalty, measured right** (six specialists, each trained like snrc4_s1 plus
`--fixed-cbr`, from w1, 400 epochs, piecewise SNR; Kodak test, paired over the 24 images,
mean over SNRs 1-13; 5/48 from the matched-noise constant profiles of `mixed_snr.py`):

| CBR (phases) | 1/48 (1) | 1/24 (2) | 1/16 (3) | 1/12 (4) | 5/48 (5) | 1/8 (6) |
|---|---|---|---|---|---|---|
| specialist - prefix, dB | +0.446 [0.41, 0.49] | +0.179 | +0.166 | +0.253 | ~+0.37 | +0.470 [0.44, 0.51] |
| at 1 dB / 13 dB SNR | +0.32 / +0.59 | +0.15 / +0.22 | +0.10 / +0.19 | +0.16 / +0.30 | +0.22 / +0.47 | +0.27 / +0.62 |

Every specialist wins on all 24 images. The penalty is U-shaped in the length (largest at
the two ends, 0.17-0.25 dB at 2-4 phases, where schedules mostly are) and grows with SNR.

**The specialists obey the additive model too** (one per-CBR utility from the six runs):
held-out MAE 0.077-0.089 dB (prefix 0.034-0.069), calibration from 12 decodes 0.061-0.085.
Their noise sensitivity is much flatter along the code (1/8: rho 0.34 0.26 0.20 0.13 0.05
0.03, prefix 0.51 0.25 0.17 0.05 0.01 0.01): the prefix model's ordering comes from nesting.

**Per-CBR codes through the same schedulers** (`--fixed-codes`, belief, same frames):

| | prefix code (s8, 10.4) | one model per CBR |
|---|---|---|
| mean PSNR, fixed_uni / slope_adv | 30.999 / **31.426** | 31.203 / **31.576** (+0.15) |
| at 30 dB: padc / padc_adv / online | 65.6 / 68.9 / **74.0** | 62.6 / **69.8** / - |
| 28 / 32 dB, T 16, K 16 (best each) | 90.5 / 54.0 / 66.8 / 76.9 | 85.1 / 51.2 / 66.1 / 73.8 |

1. **Mean PSNR: one model per CBR wins, +0.15 dB**, with the advantage index (+0.37 dB over
   its own fixed_uni; +0.43 for the prefix code, whose ordered sensitivities give the index
   more to work with). The mean claim is about the scheduler and the abstraction, not the
   prefix code: it is stated as codec-agnostic.
2. **Quality target: the prefix code's online leads by +0.7 to +5.4 points -- against a
   baseline that was too weak.** The per-CBR padc planned only the shortest lengths and
   dropped the rest: 1.74 phases per user, ~10 of 24 slots never used. A real per-CBR
   system plans those slots before the frame. Fixed (`fill`, default "admit": left-out
   users first, each at the longest code that fits; on the stand-in it beats lengthening
   the admitted users and even the prefix code's padc_adv with leftovers). To rerun.
3. Where a prefix code is REQUIRED: when the slots a user will get are not known when the
   image is encoded (`--preempt p`: other traffic takes each slot with probability p). A
   fixed code cut short decodes to nothing; a prefix decodes what arrived. Stand-in, 20%
   preempted, mean PSNR: prefix slope_adv 30.11 dB against per-CBR slope_adv 29.39 at its
   best hedge (lengths planned for the 10% quantile of the slots; 28.76 planned at the
   expectation, with users decoding nothing); for the target the two stay within ~1 point.

**Slow fading with `--plan conditional`** (belief): the mean leads are unchanged (+0.584 at
0.9, +0.349 at 0.98); at 30 dB the conditional plan helps padc (+2.0 points at 0.98) and
online (+1.5) alike, online - padc +8.9 / +5.2. It does not bring the advantage index back
at 0.98: online_pf (PF timing) stays best there (+6.4 over padc). When a frame is shorter
than the coherence time there are no peaks to wait for; PF timing is the right choice then.

## Problem P1, step 6: digital, fair per-CBR, lost slots, decoder heads (2026-10-11)

COMMANDS.txt section 11 on the server.

**Separation baseline** (`tools/digital_rd.py`: HEVC intra through libheif / x265, 26 quality
points per Kodak image, 0.1 / 0.25 / 0.5 / 1.0 bpp = 27.1 / 30.1 / 32.9 / 36.1 dB; JPEG 2000
24.7 / 26.9 / 29.0 / 31.7). Error-free link adaptation, free switching between files, and up
to 12 slots per user where JSCC is capped at 6 (CBR 1/8): optimistic for digital.

| mean PSNR, K 8, T 24 (dB) | PF | best policy |
|---|---|---|
| JSCC prefix code (snrc4, calibrated transmitter) | 30.95 | 31.43 (slope_adv) |
| JSCC, one model per CBR | 31.20 (fixed_uni) | 31.58 (slope_adv) |
| HEIC + LTE CQI table, best complete file | 31.08 | 31.29 (greedy) |
| HEIC as an ideal scalable code + CQI | 31.61 | 31.76 |
| HEIC + Shannon with a 3 dB gap / capacity | 31.91 / 33.17 | 32.13 / 33.28 |
| JPEG 2000 (scalable) + CQI | 28.03 | 28.13 |

| share at the target, best policy each | 28 dB | 30 dB | 32 dB | 30 dB, T 16 |
|---|---|---|---|---|
| JSCC prefix code, online | 90.5% | 74.0% | 54.0% | 66.8% |
| JSCC per CBR, padc_adv with planned slots | 92.5% | 78.1% | 60.0% | 69.8% |
| HEIC + CQI (online_pf; padc at 32 dB) | 90.2% | 76.1% | 59.8% | 60.6% |
| HEIC + 3 dB gap / capacity | - | 82.7% / 93.0% | - | - |

Against HEVC intra with ideal practical link adaptation the JSCC codec is ON PAR, not ahead:
+0.13 dB mean PSNR (prefix code, both at their best policy; per-CBR +0.28), ahead at the
target in tight frames (+6 to +9 points) and behind at 32 dB (-6 prefix, per-CBR +0.2). A
3 dB-gap code (closer to practice than ideal CQI with no errors) is ahead of JSCC by 0.5-0.7
dB; capacity by 1.7-1.9 dB. The codec gap to the state of the art (~0.8-0.9 dB, see P1) is about
what separates on-par from clearly ahead. The schedulers transfer to digital too (online_pf
+3.7 points over padc with HEIC).

**One model per CBR, fair slot planning** (`fill`): padc_adv 78.1% at 30 dB (padc 71.4%), 92.5%
at 28, 60.0% at 32, 69.8% at T 16, 82.5% at K 16: ahead of the prefix code's online by 2.0-6.0
points. Most of that is the planner, not the code: on the stand-in, the same planner on the
prefix code (`plan_adv`) equals per-CBR's padc_adv on the same curves by construction, and
re-planned every slot (`replan_adv`) it ties per-CBR with a 0.2 dB quality bonus and beats
`online` (+3.2 at T 16, +4.0 at 32 dB). To run on the real curves (section 12).

**Lost slots** (`--preempt`, per-CBR planned for the expected share or hedged):

| | prefix code | per CBR, expected / best hedge |
|---|---|---|
| mean PSNR, 10% lost | 30.94 (slope_adv) | 29.52 / 30.59 |
| mean PSNR, 25% lost | 30.16 (30.29 with the decoder fine-tune) | 28.76 / 29.62 |
| at 30 dB, 10% lost | 71.4% (online) | 72.7% / 73.8% |
| at 30 dB, 25% lost | 66.4% (online) | 68.4% / 68.9% |

Mean PSNR: the prefix code wins by +0.36 dB (10%) and +0.53 dB (25%) against per-CBR's best
hedge (+1.4 dB against planning for the expected slots, where users whose code is cut short
decode nothing). The target: per-CBR still +2.4 / +2.5 points against `online`; `replan_adv`
(stand-in: +3.9 / +3.1 over hedged per-CBR) is the open test.

**Where the nesting penalty lives** (frozen snrc4 encoder, decoder fine-tuned 400 epochs;
paired over Kodak, mean over SNRs):

| | at 1/48 | at 1/8 | all CBRs |
|---|---|---|---|
| HC: decoder fine-tuned at all rates (more training alone) | +0.08 | +0.15 | +0.115 |
| H48 / H8: decoder specialised to that length | +0.22 | +0.28 | - |
| HL: LoRA budget heads (rank 16) on the decoder | +0.07 | +0.16 | +0.116 |
| full specialist (encoder + decoder), section 10 | +0.45 | +0.47 | - |

Beyond more training, a decoder of its own recovers ~40% of the penalty ((0.22-0.08) /
(0.45-0.08) at 1/48, (0.28-0.15) / (0.47-0.15) at 1/8); ~60% lives in the shared ENCODER --
the price of encoding before the length is known. The LoRA heads recover none of it (HL = HC
within 0.01 dB everywhere): gate P-F fails for heads; framing A is off. HL / HC as utilities
(model errors as snrc4's): slope_adv 31.58 / 31.57 dB (= per-CBR's 31.58, with twice the
decoder training: not a fair codec comparison), online 75.5% at 30 dB, 68.0% at T 16.

## Problem P1, step 7: the same planner on both codes, a decoder per length, digital at an equal cap (2026-10-12)

COMMANDS.txt section 12 on the server.

**The per-CBR planner on the prefix code** (snrc4, calibrated transmitter, 2000 frames; share
of users at the target):

| | 28 dB | 30 dB | 32 dB | 30 dB, T 16 | 30 dB, K 16 |
|---|---|---|---|---|---|
| prefix: padc_adv (spare slots by the index, unplanned) | 86.4 | 68.9 | 50.1 | 64.9 | 71.3 |
| prefix: online | 90.5 | 74.0 | 54.0 | 66.8 | 76.9 |
| prefix: plan_adv (padc_adv + fill) | **91.5** | **74.9** | **56.2** | 67.9 | **78.1** |
| prefix: replan_adv | 90.5 | 74.2 | 55.5 | **68.9** | 76.9 |
| prefix + a decoder per length (below), best | - | 77.2 | 58.6 | 71.0 | - |
| one model per CBR: padc_adv + fill (s11) | 92.5 | 78.1 | 60.0 | 69.8 | 82.5 |

With the same planner the prefix code still trails per-CBR codes by 1.0-4.4 points with known
slots (0.9 at T 16): what is left is the nesting penalty itself. The planner is worth +3.0 to
+6.8 points over padc_adv on the prefix code too. Among the prefix policies, plan_adv is best
on the count with known slots (+0.3 to +1.2 over replan_adv) but pays in mean PSNR (29.94 vs
online's 30.54 at 30 dB: users outside the plan get little); replan_adv is best when slots are
tight (T 16) or lost:

| at 30 dB, slots lost | online | plan_adv | replan_adv | per CBR, best hedge (s11) |
|---|---|---|---|---|
| 10% | 71.4 | 71.8 | **72.4** | 73.8 |
| 25% | 66.4 | 67.2 | **69.4** | 68.9 |

replan_adv takes the target at 25% from per-CBR (+0.5) and cuts its lead at 10% to 1.4 points
(from 2.4); the prefix code wins mean PSNR at both (+0.35 / +0.54 dB, s11).

**Decoded** (60 frames each, the codecs decoding the schedules; target runs aim 0.3 dB high):

| | model bias / MAE / p95 (dB) | decoded |
|---|---|---|
| per CBR, mean: fixed_uni / slope_adv | -0.046 / 0.078 / 0.214 | 31.423 / 31.762: +0.339 dB (predicted +0.353) |
| per CBR, 30 dB: padc / padc_adv | -0.000 / 0.059 / 0.180 | 75.4 / 81.0%: +5.6 points [3.3, 8.1] (= predicted) |
| prefix, 30 dB: padc_adv / online / plan_adv / replan_adv | -0.018 / 0.053 / 0.143 | 67.7 / 76.5 / 76.7 / 77.7% (predicted within 0.2) |

The abstraction holds, decoded, for six separately trained codes too (the model is codec-
agnostic), and so do the advantage index (+0.34 dB) and the planner (+5.6 points). On the same
60 frames per-CBR padc_adv decodes 81.0% against the prefix code's best 77.7% (replan_adv):
+3.3 points (paired CI: section 13, `tools/replay_report.py`).

**A decoder of its own per length, behind the shared (frozen) encoder** (decoder fine-tuned 400
epochs at one CBR; HC: the same at all CBRs, i.e. more training alone; paired over Kodak, mean
over SNRs, dB over snrc4; 5/48 from the constant profiles of the per-phase measurement):

| length (phases) | 1/48 (1) | 1/24 (2) | 1/16 (3) | 1/12 (4) | 5/48 (5) | 1/8 (6) |
|---|---|---|---|---|---|---|
| HC: one decoder, all lengths | +0.08 | +0.095 | +0.118 | +0.131 | - | +0.15 |
| a decoder of its own | +0.22 | +0.147 | +0.168 | +0.200 | +0.240 | +0.28 |
| full specialist (section 10) | +0.446 | +0.179 | +0.166 | +0.253 | +0.371 | +0.470 |
| share of the penalty the decoder recovers | 49% | 82% | 101% | 79% | 65% | 60% |
| beyond more training: (own - HC) / (spec - HC) | 38% | 62% | ~100% | 57% | - | 41% |

Section 11's "~60% of the penalty lives in the encoder" holds only at the two ends. At 2-4
phases, where the schedulers put most users, a decoder of its own recovers 79-101% of it
(57-100% beyond more training): there the penalty is mostly a DECODER matter, and the receiver,
which knows how many phases arrived, can pay it. The ends stay encoder-bound (1 phase: the
base layer must serve every length; 6 phases).

**The system: encode once, decode per length** (utility_hd: the six per-length decoders behind
the one snrc4 encoder; the transmitter encodes once and needs no length, the receiver picks the
decoder for the length that arrived). Its model is as accurate as snrc4's (fit 0.018-0.042 dB,
held-out 0.048-0.093, calibration from 12 decodes 0.039-0.072), and its noise sensitivity
keeps the prefix code's steep order (1/8: rho 0.49 0.25 0.17 0.07 0.02 0.01; the specialists'
0.34 0.26 0.20 0.13 0.05 0.03): the order comes from the nested encoder, not the decoder.

| | prefix (snrc4) | + one more-trained decoder (HC) | + a decoder per length | per CBR |
|---|---|---|---|---|
| mean PSNR, slope_adv (dB) | 31.43 | 31.57 | **31.63** | 31.58 |
| 30 dB, best policy | 74.9 | 76.0 | 77.2 | **78.1** |
| 32 dB / 30 dB at T 16 | 56.2 / 68.9 | - | 58.6 / **71.0** | **60.0** / 69.8 |
| 25% lost: mean PSNR / 30 dB | 30.16 / 69.4 | - | **30.35 / 71.7** | 29.62 / 68.9 |

Encode-once with per-length decoders is the best JSCC system on mean PSNR and whenever slots
can be lost (+0.73 dB and +2.8 points over per-CBR's best hedge at 25%), and within 0.9-1.4
points of per-CBR codes at the target with known slots (ahead at T 16). Caveat: its decoders
trained 400 epochs more than the specialists; HC, with the same extra training, shows that
+0.10-0.15 dB of it is training alone. The equal-budget comparison is section 13.

**Digital at the same per-user cap as JSCC** (6 slots = CBR 1/8; s11 allowed 12): the mean is
unchanged (HEIC + CQI 31.29 dB best; it rarely needs more than 6 slots), the target loses 0.1-3.7
points (online_pf 86.5 / 74.4 / 57.8 / 60.5% at 28 / 30 / 32 dB / T 16). Best policy each:

| vs HEIC + CQI, 6 slots | mean PSNR | 28 dB | 30 dB | 32 dB | 30 dB, T 16 |
|---|---|---|---|---|---|
| prefix code | +0.14 dB | +5.0 | +0.5 | -1.6 | +8.4 |
| prefix + a decoder per length | +0.34 dB | - | +2.8 | +0.8 | +10.5 |
| one model per CBR | +0.29 dB | +6.0 | +3.7 | +2.2 | +9.3 |
| each against a 3 dB-gap code (6 slots) | -0.70 / -0.50 / -0.55 dB | - | -5.5 / -3.2 / -2.3 | - | - |

At equal caps JSCC with a decoder per length is ahead of HEVC intra with ideal, error-free link
adaptation everywhere, by small margins except in tight frames; a code 3 dB from capacity is
still ahead. The assumption left that favours digital most is error-free CQI: JSCC degrades
smoothly when the reported SNR is off, a digital link backs off or loses blocks. Section 13
measures it (`tools/schedule_sim.py --cqi-error`: decisions on the reports, codes decoded at the
true SNR; a digital slot sent at its report minus a backoff tuned per policy on other frames,
lost when its rate exceeds what the slot supported, HARQ ACK to the transmitter).

## Problem P2, round 1 (2026-10-14): the code trained for the channel its scheduler makes

COMMANDS.txt section 14 (section 13 deferred: the story comes first). The question
(docs/PROBLEM.md, section 7; docs/PAPER.md C8-C10): under a scheduler, the SNR each phase of a
code sees and the number of phases each user gets are the scheduler's decisions, so the code is
trained for the wrong channel; does training it on the channel the scheduler makes pay, on the
same slots and with the same training budget?

Built: `tools/codesign.py bank` (every user's delivered per-phase SNR history under one policy,
from tools/schedule_sim.py's frames on a code's utility; `--blind` keeps the lengths and draws
every phase in a blindly taken slot), `codesign.py stats` (length distribution, per-phase SNR and
lift, mean profile per length; `--against` pairs the users of two banks drawn on the same
frames: the loop's convergence check), and `main.py --snr-profiles BANK` (each step one prefix
length from the bank's distribution and per image a history of that length, one SNR per
phase; replaces --snr-chunks and the rate sampling; run-name suffix prof-<bank>). On the
stand-in utility (K 8, T 24) the scheduler's first phases sit +3.3 dB above the user's mean SNR,
blind slots -1.9 dB, and the per-phase SNR falls along the code (10.7 dB at phase 1, 1.1 at 6
under slope_adv). The CPU smoke test checks the bank, the sampler and the refusals; a tiny CPU
run checked the whole section-14 chain (bank, training, dumps, decodes, fit, re-planned
schedules, paired reports).

Arms (snrc4 + 400 epochs each, the FTF recipe): hf (--snr-chunks 4, the control), blind, sa
(slope_adv's channel), sat (replan_adv's at 30 dB). Each decodes the original system's
schedules on 150 frames (same slots for every code) and, after its own measurement and fit, its
own re-planned schedules.

**The channel the scheduler makes** (real snrc4 utility, calibrated transmitter, 4000 frames):

| bank | phases received | SNR of phase 1 / 2 / 3 ... (dB) | lift over the user's mean |
|---|---|---|---|
| sa (slope_adv) | 3 for 99.9% of users | 10.68 / 10.13 / 9.31 (sd 4.6) | +3.21 / +2.66 / +1.83 |
| blind (same lengths, slots taken blindly) | 3 for 99.9% | 5.67 / 5.67 / 5.74 (sd 5.7) | -1.80 each |
| sat (replan_adv, 30 dB) | 1-6: 13 / 36 / 20 / 7 / 4 / 19%, 3.3% nothing | 10.61 / 9.88 / 9.11 / 7.31 / 5.87 / 5.43 | +3.06 ... -1.25 |
| snrc4's training | uniform CBR | i.i.d. 10.0 (sd 6.9), random cuts | - |

With equal-slope quotas on the real curves every user gets exactly 3 phases, so sa is also a
3-phase training set: sa (and blind) are 1/16 specialists of sorts.

**Same slots** (the original system's schedules, 150 frames, every code decoding the same
users, images and slot SNRs; mean PSNR in dB, share at 30 dB in %):

| code | pf | slope_adv | online | plan_adv | replan_adv |
|---|---|---|---|---|---|
| snrc4 (original) | 31.036 | 31.512 | 74.7 | 74.8 | 76.1 |
| hf (control, +400 epochs) | 31.139 | 31.607 | 75.4 | 76.2 | 76.9 |
| blind | 30.779 | 31.563 | 29.3 | 35.0 | 29.5 |
| sa | 31.228 | **31.885** | 62.7 | 66.1 | 62.9 |
| sat | 31.161 | 31.688 | 75.7 | 76.9 | **77.1** |

Paired by frame: sa - hf under slope_adv +0.278 dB [0.273, 0.284]; sa - blind +0.32; blind -
hf -0.044 (training on the physical channel LOSES to generic training); under PF slots sa - hf
+0.089 [0.072, 0.106] (the lead is specific to the scheduler it trained for). At 30 dB sat - hf
+0.17 points [0.00, 0.42] (replan_adv), +0.67 (plan_adv), +0.25 (online): nothing. sa and blind,
trained on 3 phases only, collapse under the QoS schedulers (1-6 phases).

**The loop closed once** (each code measured, fitted, and scheduled by its own model; the
model's error on these schedules 0.07 / 0.055 dB MAE): slope_adv sa 31.859 vs snrc4 31.512
(+0.347 [0.331, 0.363]) and hf 31.610 (+0.249 [0.232, 0.265]); 30 dB replan_adv sat 77.2% vs
snrc4 76.1 (+1.1 [0.4, 1.8]) and hf 77.1 (+0.1 [-0.5, 0.7]): at the target all of the gain
over the original system is the extra training. Re-planning for sa added nothing over the
original schedules (31.859 vs 31.885 decoded), and the channel barely moves: re-banked on
sa's own utility, 99.7% of users keep their length, the lifts change by <= 0.14 dB (sat: 79.6%
keep their length, lifts within 0.3 dB). One round is the fixed point; the "game" is shallow.

**Mechanism** (fits): kappa hf 1.73, blind 1.60, sa 2.78, sat 2.43; rho at 1/16 hf 0.565 /
0.266 / 0.169, sa the same (0.566 / 0.267 / 0.167), blind flatter (0.416 / 0.316 / 0.269), sat
steeper (0.619 / 0.237 / 0.145). sa keeps the phase weights and gives up robustness to deep
noise (a larger kappa: its distortion keeps growing where hf's saturates), which its scheduler
never asks of it. Constant SNR at 7 dB (Kodak, 1/48 ... 1/8): hf 27.23 / 29.51 / 30.91 / 31.52
/ 32.03; sa 25.75 / 29.06 / **31.16** / 31.61 / 32.01 (-1.5 dB at 1 phase, +0.25 at 3); sat
27.39 / 29.65 / 30.97 / 31.55 / 32.07 (no cost); blind 20.22 / 26.86 / 30.90 / 31.28 / 31.56.

Verdict against the pre-registered rules (COMMANDS 14): CD1 mean +0.28 dB, just under the
+0.3 bar ("supporting" band), QoS +0.2 points ("dead" for the target); CD2 passes (sa - blind
+0.32 at the same lengths); CD3 passes (+0.09 under PF, the cross table); CD4 +0.35 dB over the
original system on mean PSNR, nothing beyond extra training at the target; CD5 passes. Open:
whether sa's +0.28 over hf is the scheduler's channel or the one length it trains (a 3-phase
specialist on the generic channel, FS16, is the control), whether the code uses which phase
gets the good slot (a shuffled bank), and whether the gain grows with the number of users
(more selection): COMMANDS section 15.

## Problem P2, round 2 (2026-10-14): the scheduler's channel, or one prefix length?

COMMANDS.txt section 15. Four more fine-tunes at the same recipe and budget: fs16 (snrc4 + 400
epochs at 1/16 on the usual chunk SNRs: a 3-phase specialist on the generic channel), shuf
(sa's histories with each user's phases permuted over its own slots), k16 (slope_adv's channel
with 16 users, 48 slots), pf (PF's channel).

**Section 14's slots** (K 8, 150 frames, the same users, images and slot SNRs for every code;
decoded mean PSNR, dB):

| code | slope_adv | minus fs16 | pf slots | minus hf (pf slots) |
|---|---|---|---|---|
| snrc4 | 31.512 | -0.255 | 31.036 | -0.103 |
| hf | 31.607 | -0.160 | 31.139 | 0 |
| blind | 31.563 | -0.204 | 30.779 | -0.360 |
| fs16 (3-phase specialist, generic channel) | 31.767 | 0 | 30.252 | -0.888 |
| sat | 31.688 | -0.079 | 31.161 | +0.021 |
| pf (PF's channel) | 31.722 | -0.045 | **31.319** | **+0.180** |
| shuf | 31.843 | +0.076 | 31.194 | +0.055 |
| sa | 31.885 | **+0.118** [0.110, 0.124] | 31.228 | +0.089 |
| k16 | **31.899** | +0.132 | 31.282 | +0.143 |

**16 users, 48 slots** (150 frames): fs16 31.970, sa 32.099 (+0.129 [0.119, 0.142] over fs16),
k16 32.122 (+0.151 [0.141, 0.164]); over hf: fs16 +0.163, sa +0.292, k16 +0.314.

**The selection the code trains for** (slope_adv's lift of phases 1 / 2 / 3 over the user's
mean SNR, dB): K 2 +0.60 / -0.46 / -0.96; K 4 +2.16 / +1.28 / +0.40; K 8 +3.21 / +2.66 / +1.83;
K 16 +3.92 / +3.63 / +2.99; K 24 +4.29 / +4.10 / +3.57 (it grows like an order statistic of the
users' fades). PF's channel is the other way round: 2-4 phases (70% get 3), lifts +1.26 /
+2.07 / +2.70 / +3.65 rising along the code (a user is served when its rate ratio is high,
which comes after its average has fallen).

**What each code learned** (fit; kappa, then rho at 1/16):

| code | its channel | kappa | rho (phase 1 / 2 / 3) |
|---|---|---|---|
| blind | blind slots, -1.8 dB lift, many deep fades | 1.60 | 0.416 / 0.316 / 0.269 |
| hf | i.i.d. chunks over [-2, 22] dB | 1.73 | 0.565 / 0.266 / 0.169 |
| fs16 | the same, 3 phases | 1.88 | 0.440 / 0.282 / 0.278 |
| pf | PF: later phases in better slots | 1.96 | 0.366 / **0.418** / 0.216 |
| sat | replan_adv | 2.43 | 0.619 / 0.237 / 0.145 |
| shuf | sa's slots, order shuffled (+2.6 dB each) | 2.55 | 0.504 / 0.291 / 0.205 |
| sa | slope_adv: +3.2 / +2.7 / +1.8 dB | 2.78 | 0.566 / 0.267 / 0.167 |
| k16 | slope_adv, 16 users: +3.9 / +3.6 / +3.0 dB | 3.06 | 0.543 / 0.285 / 0.172 |

Two regularities. The noise saturation kappa rises as the channel's deep fades get rarer
(blind < generic < PF < the selected channels, highest at K 16): the code gives up robustness
to noise its scheduler never delivers. The phase weights follow which phase the scheduler
favours: the PF code makes phase 2 its most sensitive phase, slope_adv's codes phase 1; with the
order shuffled the profile flattens. A learned code does unequal error protection by itself,
matched to its scheduler.

**Constant SNR at 1/16** (Kodak, mean over 1-13 dB, against fs16): hf -0.164, sa +0.043, shuf
+0.030, k16 +0.048, blind -0.143. Every scheduler-trained code collapses at lengths its
scheduler never gives (fs16 -7.4 dB at 1-2 phases, pf -5.6 at 1, shuf -3.9 at 1, k16 -1.9 at 1).
sat, the only one trained on all lengths, costs nothing anywhere (+0.045 over hf on average).

**Verdict** (COMMANDS 15): D1 sa - fs16 +0.118 dB, between the bars: of sa's +0.28 over hf,
+0.16 is training for the one length slope_adv's quotas give, +0.12 the scheduler's channel --
real (tight CI, +0.13 / +0.15 again with 16 users) but small. D2 the order is a third of it
(sa - shuf +0.042), the SNR levels the rest. D3 it grows a little with the users' number
(+0.118 at K 8, +0.151 at K 16), as the lift does. D4 holds both ways (the PF code +0.09 over sa
under PF slots, sa +0.16 over it under slope_adv). The mechanism is clean and the training
distribution clearly matters (training on the physical channel loses 0.2-0.3 dB), but beyond
the length the decoded value is about a tenth of a dB: a contribution, not a headline. Open: a
seed replicate of sa and fs16 (COMMANDS 16), since the frame CIs do not cover training noise.
