# Results notes (condensed 2026-10-15)

What the experiments established, by topic; the last section indexes every experiment round.
The full, chronological notes (970 lines, every table) are in git: `git show ec9cc68:jscc/docs/NOTES.md`.

Unless stated: Kodak (24 images); the snrc4 ViT (below); K = 8 users, T = 24 slots (3 phases
per user on average), mean SNRs U[0, 15] dB, i.i.d. Rayleigh block fading, a calibrated
transmitter, 2000 simulated frames. "Decoded" = the schedules were decoded by the codec itself
(60-150 frames), differences paired by frame with 95% bootstrap CIs.

## 1. The codec

* **snrc4**: ViT, patch 8, 768 wide x 10 blocks, 4 complex symbols per token, a prefix (nested)
  code of 6 phases, one phase = CBR 1/48 at 256 px tiles (1/8 in all). DIV2K, AWGN; w1 (2000
  epochs) fine-tuned 400 epochs on piecewise SNR (`--snr-chunks 4`: chunks of the prefix at
  independent SNRs over [-2, 22] dB). Encoder and decoder never see the SNR.
* **Piecewise training is required.** A decoder trained at one SNR per image mis-reads a prefix
  whose slots had different SNRs (it over-trusts the noisy part: -1.3 dB on [13 dB, 1 dB] at
  1/24); piecewise training fixes it at a constant-SNR cost of 0.06-0.10 dB.
* **Against other codecs.** The ViT beats SwinJSCC (base) by +0.5 to +0.9 dB on Kodak but with
  27x the parameters and 16x the FLOPs. About 0.8-0.9 dB below JSCCformer-f's codec after
  discounting its feedback (w1 at 1/12: 28.9 / 30.3 / 31.4 dB at 1 / 4 / 7 dB), about on par with
  PADC's. All claims below are relative (the same codec under every policy).
* **Dead ends (kept out of the paper):** SNR-conditioned decoders (+0.03 dB at best), per-rate
  cascade exits (+0.02-0.08 dB, a linear map takes all of it), LoRA budget heads (= more
  training), AdaTok-style 1D tokens (do not turn budget into detail).

## 2. The link abstraction (the paper's anchor)

**The model** (`alloc/utility.py`): an image decoded from L phases, phase j received at noise
variance n_j, has MSE

    D = Dsrc(L) + C(L) * sum_j rho_L[j] * phi(n_j),    phi(n) = n / (1 + n / kappa)

phi is the MMSE of a Gaussian of variance kappa. Per image: Dsrc, C (noise-free distortion and
noise sensitivity per length); per codec: kappa, rho (each phase's share of the sensitivity).
snrc4: kappa 1.77; rho at 1/8 = 0.51 0.25 0.17 0.05 0.01 0.01 (the head is 2-20x more sensitive).

| accuracy (dB) | MAE | p95 |
|---|---|---|
| held-out mixed profiles, per image and length | 0.03-0.07 | 0.10-0.23 |
| scheduled histories, decoded (every replay since step 2) | 0.05-0.09, no bias | 0.14-0.29 |
| transmitter calibrated from 12 constant-SNR decodes of its own image | 0.03-0.06 | 0.10-0.17 |

**Learned codes need their own abstraction.** Held-out profiles mapped to one effective SNR on
each image's own curve (MAE, dB; sensitivity weights unless stated):

| code | MIESM (capacity, equal weights) | capacity | EESM | noise | **MMSE** |
|---|---|---|---|---|---|
| snrc4 (prefix) | 0.487 | 0.337 | 0.194 | 0.133 | **0.055** |
| a decoder per length (hd) | 0.499 | 0.270 | 0.182 | 0.173 | **0.089** |
| one model per CBR | 0.410 | 0.293 | 0.182 | 0.204 | **0.110** |
| s1 (constant-SNR-trained) | 0.721 | 0.614 | 0.214 | **0.150** | 0.222 |

For every piecewise-trained code, slots add as sensitivity-weighted MMSEs: 2.5-6x more accurate
than any other rule and ~9x more accurate than MIESM, digital system simulation's standard. No
symmetric rule works (the code is ordered: which phase a slot carries matters). The constant-
SNR-trained code obeys no rule well and fits the additive model 2-4x worse: the training
decides the abstraction. The nested form (one master rho for all lengths) costs <= 0.007 dB.

## 3. Scheduling on the abstraction

* **Mean PSNR: the advantage index** (`slope_adv`: serve the user whose predicted PSNR gain from
  this slot most exceeds its gain from a slot at its mean SNR): +0.476 dB over PF simulated,
  **+0.465 [0.39, 0.53] decoded**, +0.46 dB for the 5th-percentile user; +0.34 to +0.58 over K,
  T, curve errors and slot correlation up to 0.9 (+0.34 at 0.98). Content-aware LENGTHS add only
  +0.05 dB: the gain is in valuing slots in quality (users near saturation take bad slots) and
  by position (sensitive head phases wait for peaks).
* **Quality target** (share of users reaching 30 dB): `online` (admission by remaining need,
  stopping on the realised slot SNRs, index timing) **+11.0 points [8.3, 13.8] decoded** over
  PADC's per-image rule (+14 over PF). Parts: per-image sizing +5, stopping +4, timing +3,
  admission +2 (+11 in tight frames). Slot planners: `plan_adv` (plan the spare slots before
  the frame) best with known slots and under CQI errors; `replan_adv` best with lost slots.
* **How close to optimal** (400 frames, `tools/headroom.py`, `alloc/whittle.py`):

  | | mean PSNR | share at 30 dB |
  |---|---|---|
  | one rollout step on our rule (achievable, causal) | +0.04 dB | +0.7 to +2.0 points |
  | Lagrangian upper bound on ANY causal scheduler | +0.26 dB | +5.5 points |
  | clairvoyant optimum (all SNRs known; certified, bound - optimum <= 0.001 dB) | +0.23 dB | +4.7 to +5.9 points |
  | the same slots with phases mapped by sensitivity (ex post) | +0.13 dB | +3.0 to +4.3 points |

  With the nested model a user's state is two numbers and the frame is a finite-horizon
  restless bandit; the Whittle index policy equals our rule (+0.005 dB; +0.15 dB with 25% lost
  slots). Slot counts: the greedy quotas equal the exact DP in 100% of frames. So: for the mean,
  our rule is certified within 0.26 dB of the best causal scheduler; for the target the gap is
  2-5.5 points, and the largest piece a causal rule could still take is the phase-to-slot
  mapping (a sequential-assignment rule, owed).

## 4. Prefix code, a decoder per length, one model per CBR

* **Nesting penalty** (one model per CBR over the prefix code, Kodak, mean over 1-13 dB): +0.45 /
  +0.18 / +0.17 / +0.25 / +0.37 / +0.47 dB at 1-6 phases. At EQUAL training budget (all snrc4 +
  400 epochs) it is 0.15-0.20 dB at 2-4 phases, and a decoder of its own per received length,
  behind the one shared encoder, recovers about half of it (44-52%); the encoder holds 0.08-0.11.
* **The systems through the same schedulers:**

  | | prefix | encode once + a decoder per length (hd) | one model per CBR |
  |---|---|---|---|
  | mean PSNR, decoded | 31.59 | **31.82** | 31.76 |
  | 30 dB, decoded (best policy) | 77.7% | 80.2-80.8% | **81.0%** (tie with hd) |
  | 25% of slots lost: mean / 30 dB | 30.16 / 69.4% | **30.35 / 71.7%** | 29.62 / 68.9% |
  | 25% lost, decoded, hd minus per-CBR | | **+0.78 dB, +1.9 points** | |

  Encode once with a decoder per length is the best JSCC system on mean PSNR and when slots can
  be lost (a fixed-length code cut short decodes nothing); per-CBR codes only tie it at the
  target with known slots. The abstraction and the schedulers hold on all three (codec-agnostic).

## 5. Against digital (HEVC intra / HEIC + link adaptation, same schedulers, 6-slot cap)

| CQI error sd (dB) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| mean PSNR, hd / HEIC + CQI table / 3 dB-gap code | 31.63 / 31.29 / 32.13 | 31.59 / 30.59 / 31.12 | 31.48 / 29.84 / 30.26 | 31.33 / 29.02 / 29.44 |
| hd minus HEIC (dB) | +0.34 | +1.00 | **+1.64** | +2.31 |
| at 30 dB, hd / HEIC / gap-3 | 77.2 / 74.4 / 80.4 | 76.9 / 67.7 / 71.7 | 75.6 / 61.1 / 63.9 | 73.2 / 54.3 / 56.7 |
| hd minus HEIC (points) | +2.8 | +9.2 | **+14.5** | +18.9 |

With error-free CQI JSCC is on par (a code 3 dB from capacity is 0.5 dB ahead). With errors
(decisions on the reports; JSCC decodes at the true SNR, digital backs off 1-6 dB at its best,
tuned per policy, and still loses 2-17% of its blocks) JSCC loses 0.3 dB from 0 to 3 dB, digital
2.3 dB: from 1 dB of CQI error on, JSCC is ahead even of the 3 dB-gap code.

## 6. The scheduler-made channel (scheduler-aware training)

* **The channel a code sees is made by the scheduler** (`tools/codesign.py`): slope_adv gives every
  user 3 phases and puts them +3.2 / +2.7 / +1.8 dB above the user's mean SNR; slots taken
  blindly sit -1.8 dB. The lift grows with the number of users like an order statistic (phase
  1: +0.6 / +2.2 / +3.2 / +3.9 / +4.3 dB at K 2 / 4 / 8 / 16 / 24). PF is the other way round
  (later phases in better slots).
* **Training for it** (`main.py --snr-profiles`, snrc4 + 400 epochs, decoded on the SAME slots):

  | code (trained on) | mean PSNR under slope_adv | minus hf |
  |---|---|---|
  | hf (usual chunks, the same-budget control) | 31.607 | 0 |
  | blind (the physical channel, same lengths) | 31.563 | -0.04 |
  | fs16 (usual chunks, 3 phases only) | 31.767 | +0.16 |
  | sa (slope_adv's channel), seeds 1 / 2 | 31.885 / 31.846 | +0.28 / +0.24 |

  Of the +0.28, +0.16 is training for the one length the quotas give; the scheduler's channel
  adds +0.08 to +0.12 dB beyond it (two seeds each; fs16's seeds agree within 0.001), and none at
  constant SNR. The slots' order is a third of it (a bank with each user's phases shuffled over
  its own slots keeps +0.08). With 16 users +0.15.
  A PF-trained code wins under PF slots (+0.18 over hf; sa +0.09): a code per scheduler. At the
  30 dB target nothing (+0.2 points): that scheduler's channel is close to the usual one.
  Re-planning for the retrained code barely moves the channel (99.7% of users keep their
  length): one round is the fixed point.
* **The codes learn their scheduler** (fits of eight codes): kappa rises as deep fades get rarer
  (blind 1.60, usual 1.73, PF 1.96, slope_adv 2.78, 16 users 3.06), and the phase weights move to
  the phase the scheduler favours (PF code: phase 2 most sensitive; slope_adv code: phase 1;
  shuffled: flatter). An unequal error protection learned from the scheduler.

## 7. Open

1. An online phase-to-slot mapping rule for fixed-quota frames (sequential stochastic assignment,
   Derman-Lieberman-Ross; exact per user because distortion is linear in the per-phase noise
   terms): up to +0.13 dB / +3-4 points ex post. CPU.
2. Review robustness: a second test set (CLIC), a perceptual metric (MS-SSIM, LPIPS) at the
   target. GPU.
3. Optional: a Jacobian probe (the decoder's per-chunk Jacobian energy against the fitted C rho:
   the first-order mechanism behind the abstraction); the codec gap (more data).

## 8. Experiment index

| round | COMMANDS section | date | settled |
|---|---|---|---|
| wave 1, tail probes, cascade c1 | 1-4 (retired) | to 10-08 | the codec (section 1): ViT vs Swin, nesting empties the tail, cascade exits negligible |
| P1 (step 1) | 5 | 10-10 | piecewise training makes distortion additive; symmetric rules fail |
| P1, step 2 | 6 | 10-10 | per-phase model 0.03-0.07 dB; content-aware lengths +0.05 only |
| P1, step 3 | 7 | 10-10 | advantage index +0.465 dB, online +10 points, decoded |
| P1, step 4 | 8 | 10-11 | calibration from 12 decodes; correlated fading |
| P1, step 5 | 10 | 10-11 | nesting penalty per length; per-CBR codes through the schedulers |
| P1, step 6 | 11 | 10-11 | digital on par at ideal CQI; lost slots favour the prefix code; LoRA heads nothing |
| P1, step 7 | 12 | 10-12 | the same planner on both codes; a decoder per length; digital at an equal cap |
| P2, round 1 | 14 | 10-14 | scheduler-trained code +0.28 dB on the same slots; nothing at the target; one round suffices |
| P2, round 2 | 15, 16 | 10-14 | +0.16 is the length, +0.08-0.12 the channel (two seeds); codes learn their scheduler |
| P1, step 8 | 17 (= 13) | 10-14 | MMSE abstraction ~9x MIESM; CQI errors: JSCC clearly ahead; scheduler certified; equal budget |
