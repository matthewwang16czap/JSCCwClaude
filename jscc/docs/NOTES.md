# Results notes (condensed 2026-10-15; sections 18-19 added 2026-10-19, section 20 2026-10-20)

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

The JSCC numbers in this section are the per-length decoders' (hd). With the single decoder, the
paper's main system, see section 6e: every lead is 0.17-0.21 dB and 1.4-2.5 points smaller, and
the verdict is the same.

| CQI error sd (dB) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| mean PSNR, hd / HEIC + CQI table / 3 dB-gap code | 31.63 / 31.29 / 32.13 | 31.59 / 30.59 / 31.12 | 31.48 / 29.84 / 30.26 | 31.33 / 29.02 / 29.44 |
| hd minus HEIC (dB) | +0.34 | +1.00 | **+1.64** | +2.31 |
| at 30 dB, hd / HEIC / gap-3 | 77.2 / 74.4 / 80.4 | 76.9 / 67.7 / 71.7 | 75.6 / 61.1 / 63.9 | 73.2 / 54.3 / 56.7 |
| hd minus HEIC (points) | +2.8 | +9.2 | **+14.5** | +18.9 |

With error-free CQI JSCC is on par (a code 3 dB from capacity is 0.5 dB ahead). With errors
(decisions on the reports; JSCC decodes at the true SNR, digital backs off 1-6 dB at its best,
tuned per policy, and still loses 2-17% of its blocks) JSCC loses 0.3 dB from 0 to 3 dB, digital
2.3 dB: from 1 dB of CQI error on, JSCC is ahead even of the 3 dB-gap code. These digital
numbers have no HARQ (a lost block's slot is wasted); with it see below (the stale-report
paragraph): the lead at 2 dB is +1.33 dB / +12.9 points.

**Stale reports: the physical axis (COMMANDS 18.1, 2026-10-19).** A report taken tau before its
slot has correlation r = |J0(2 pi f_D tau)| with it (3.5 GHz, 5 ms-old reports: r 0.98 at 3 km/h,
0.76 at 10, 0.50 at 15; 1 ms-old: 0.91 at 30 km/h, 0.66 at 60). Best policy of each system at
each r, and the better of a naive transmitter (takes the reports as exact; digital with its
backoff tuned per policy) and an aware one (decides on each slot's posterior: JSCC its expected
distortion, digital the rate of best expected goodput); 2000 frames, the same frames and true
slots at every r; simulated on the fitted curves (JSCC decodes at the true SNR; a digital block
the slot cannot carry is lost):

| report correlation r | 1 | 0.95 | 0.9 | 0.8 | 0.6 | 0.4 | 0 |
|---|---|---|---|---|---|---|---|
| mean PSNR, hd / HEIC + CQI table / 3 dB-gap code | 31.63 / 31.29 / 32.13 | 31.51 / 30.07 / 30.51 | 31.38 / 29.48 / 29.93 | 31.09 / 28.62 / 29.03 | 30.60 / 27.32 / 27.72 | 30.27 / 26.31 / 26.70 | 30.00 / 26.53 / 25.93 |
| hd minus HEIC (dB) | +0.34 | +1.44 | +1.89 | **+2.48** | +3.27 | +3.96 | +3.48 |
| hd minus the 3 dB-gap code (dB) | -0.50 | +0.99 | +1.44 | +2.06 | +2.88 | +3.57 | +4.07 |
| at 30 dB, hd / HEIC (%) | 77.2 / 74.4 | 75.4 / 63.0 | 73.9 / 58.2 | 70.3 / 51.2 | 64.0 / 43.7 | 59.4 / 41.0 | 56.7 / 40.9 |
| hd minus HEIC (points) | +2.8 | +12.4 | +15.7 | **+19.1** | +20.3 | +18.4 | +15.8 |

Without HARQ (the table above) JSCC leads by ~2.6 dB and ~19 points at 10 km/h with 5 ms-old
reports (r ~0.76). From r = 1 to 0 JSCC loses 1.6 dB, HEIC 4.8 (its backoff grows to 2-7 dB and it
still loses 7-50% of its blocks).

**With HARQ (COMMANDS 20.2, 2026-10-20): the numbers to quote.** A block the slot could not
carry is retransmitted in the user's next slot and combined (`schedule_sim.py --harq ir`:
incremental redundancy, the transmissions' efficiency adds; up to 4 transmissions), the backoff
(naive) or an offset on the aware rate tuned with HARQ on per policy and r. Best digital policy
and transmitter at each point; the JSCC numbers are those above (a prefix has no block to lose):

| report correlation r | 1 | 0.95 | 0.9 | 0.8 | 0.6 | 0.4 | 0 |
|---|---|---|---|---|---|---|---|
| mean PSNR, HEIC + HARQ / 3 dB-gap code + HARQ | 31.29 / 32.13 | 30.19 / 30.65 | 29.82 / 30.19 | 29.36 / 29.71 | 28.92 / 29.18 | 28.78 / 28.96 | 28.75 / 28.87 |
| what HARQ gives HEIC (dB) | 0 | +0.12 | +0.34 | +0.74 | +1.60 | +2.47 | +2.22 |
| hd minus HEIC + HARQ (dB) | +0.34 | +1.32 | +1.56 | **+1.73** | +1.69 | +1.49 | +1.25 |
| hd minus the gap code + HARQ (dB) | -0.50 | +0.86 | +1.19 | +1.38 | +1.42 | +1.31 | +1.13 |
| at 30 dB, HEIC + HARQ (%) | 74.4 | 63.6 | 59.5 | 54.8 | 49.8 | 46.8 | 47.0 |
| hd minus HEIC + HARQ (points) | +2.8 | +11.8 | +14.4 | **+15.5** | +14.2 | +12.6 | +9.7 |

| CQI error sd (dB) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| mean PSNR, HEIC + HARQ / gap code + HARQ | 31.29 / 32.13 | 30.68 / 31.15 | 30.15 / 30.53 | 29.73 / 30.04 |
| hd minus HEIC + HARQ (dB) | +0.34 | +0.91 | **+1.33** | +1.60 |
| hd minus the gap code + HARQ (dB) | -0.50 | +0.44 | +0.96 | +1.29 |
| at 30 dB, HEIC + HARQ (%) / hd minus it (points) | 74.4 / +2.8 | 68.4 / +8.5 | 62.7 / **+12.9** | 58.1 / +15.1 |

HARQ takes 0.3 dB off the CQI-error lead at 2 dB and 0.75 dB off the stale-report lead at
r = 0.8, far less than a toy codec suggested (it recovered half of digital's loss there): even
with IR, HEIC loses 1.9 dB from fresh reports to r = 0.8, JSCC 0.5. So at 10 km/h with 5 ms-old
reports JSCC leads HEIC with HARQ by ~1.7 dB and ~15 points, at 3 km/h by ~0.7 dB, and a code
3 dB from capacity with HARQ from r = 0.95 (+0.9 dB) and from 2 dB of CQI error (+1.0). The
tuned digital transmitter turns bold with HARQ (greedy's aware rate up to 7 dB above its
single-transmission goodput optimum at r = 0.4; 30-60% of its slots then decode no block).
Chase combining is worse than IR (r = 0.8 / 0.6: 29.03 / 28.19 dB vs 29.36 / 28.92). Digital
without HARQ stays in the tables above as the reference; the paper quotes these.

**The JSCC side decoded (COMMANDS 20.1).** The stale-report schedules of the C5 table (aware
transmitter, 60 frames, the per-length decoders) decoded by the codec: the model's error on
1408-1440 histories per point is unbiased to the conservative side (predicted minus decoded
-0.02 to -0.045 dB) with MAE 0.08-0.12 dB (p95 0.28-0.39), slightly above the fresh reports'.
Decoded slope_adv - pf: +0.500 dB [0.40, 0.61] at r = 0.8, +0.361 [0.25, 0.48] at r = 0.6
(simulated +0.475 / +0.342). Paired with section 17's fresh decodes on the same frames,
slope_adv loses 0.49 dB at r = 0.8 and 0.98 at r = 0.6 (simulated 0.54 / 1.03). So the JSCC
side of the stale-report axis holds as decoded.

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

## 6b. The scheduler as a restless bandit, re-examined (COMMANDS 18, run 2026-10-19)

Our frame is a PLANNING restless bandit (every user's state reported each slot, the quality
model calibrated), not a learning one (docs/SURVEY.md F5; docs/PROBLEM.md section 3, item 6).
Two assumptions tested: fresh reports, and one user per slot. Methods: stale reports h_rep =
r h + sqrt(1 - r^2) e (`alloc/csi.py`; the users' programs decide on the report through a
report-to-channel kernel and the bound holds for every scheduler that sees only such reports;
program and simulation agreed within 0.02 dB on the stand-in); the Whittle index as the top of
its indifference range; B = K/8 units per slot (`tools/rmab_scale.py`) with PF, our rule, the
Whittle and the Lagrangian (LP-priority) index, the CMU blog's ID-guided policy and its
random-order baseline.

* **The opportunistic part fades with the reports, and at K 8 nothing else is there** (hd, 2000
  frames, mean PSNR; naive / aware transmitter):

  | r | 1 | 0.95 | 0.9 | 0.8 | 0.6 | 0.4 | 0 |
  |---|---|---|---|---|---|---|---|
  | slope_adv - PF (dB) | +0.475 | +0.471 / +0.493 | +0.452 / +0.498 | +0.385 / +0.475 | +0.256 / +0.342 | +0.148 / +0.190 | +0.078 / -0.011 |
  | PF - RR (dB) | +1.16 | +1.02 | +0.89 | +0.65 | +0.29 | +0.08 | -0.08 / 0.00 |
  | advantage timing on the same quotas (dB) | +0.437 | +0.434 / +0.450 | +0.410 / +0.449 | +0.336 / +0.426 | +0.187 / +0.318 | +0.073 / +0.190 | -0.01 |

  The content part (fixed_slope - fixed_uni: lengths) is 0.000 at every r: at K 8, T 24 the
  equal-slope quotas give 3 phases to 99.9% of users, so all of our rule's gain is how it
  values slots. An aware transmitter keeps the whole gain over PF down to r = 0.8 and 72% of it
  at r = 0.6 (naive: 81% and 54%). At r = 0 every policy is round robin (30.00 dB).
* **The scheduler-made channel is a fresh-report effect** (banks of 4000 frames, slope_adv): the
  SNR lift of phases 1 / 2 / 3 over random selection, +5.0 / +4.5 / +3.6 dB with fresh reports,
  keeps 91% of itself at r = 0.95, 65% at 0.8, 36% at 0.6 and 8% at 0.3. Even at r = 0.95, 58% of
  the phases go out in another slot than with fresh reports.
* **Quality target: stopping on reported SNRs breaks** (hd, 30 dB):

  | r | 1 | 0.95 | 0.9 | 0.8 | 0.6 | 0.4 | 0 |
  |---|---|---|---|---|---|---|---|
  | online, naive / aware (%) | 76.4 | 69.8 / 72.2 | 65.8 / 70.0 | 59.0 / 65.3 | 49.1 / 58.7 | 43.3 / 54.9 | 39.2 / 53.0 |
  | plan_adv, naive / aware | 77.2 | 75.1 / 75.4 | 72.1 / 73.9 | 66.4 / 70.3 | 56.8 / 64.0 | 50.3 / 58.9 | 45.7 / 54.4 |
  | PADC, naive / aware | 67.8 | 66.5 / 66.1 | 65.0 / 64.8 | 62.9 / 62.1 | 59.3 / 58.8 | 57.3 / 57.5 | 56.1 / 56.7 |

  A user served at a reported peak looks done although its true slot was worse: naive online
  loses 7 points already at r = 0.95 and falls below PADC by r = 0.8 (replan_adv, which also
  stops on the reports, the same). Deciding on the posterior and planning the spare slots
  before the frame (plan_adv, aware) keeps +8 points over PADC to r = 0.8 and +5 at r = 0.6; at
  r = 0 PADC is best. At r = 0 with an aware transmitter every advantage index tied and the
  first user took every spare slot (padc_adv mean PSNR 26.4 dB); ties now go to PF.
  Rerun after the fix (COMMANDS 20.4b): padc_adv = padc (56.7%), online 51.9% (53.0 before),
  plan_adv 54.4% unchanged, every mean-PSNR policy 30.00 dB; PADC stays best at r = 0.
* **Decoded at r = 0.8 / 0.6** (COMMANDS 20.1, aware, 60 frames): PADC 62.1 / 60.2%, online 68.5
  / 60.0%, plan_adv 72.7 / 65.4% at 30 dB decoded (predicted 63.3 / 67.9 / 72.7 and 61.9 / 59.8 /
  66.0); plan_adv - PADC +10.6 points [7.5, 14.0] and +5.2 [1.7, 8.8], online - PADC +6.5 [3.1,
  10.2] and -0.2 [-3.8, +3.5]. As simulated: plan the spare slots, decide on the posterior.
* **Margins** (COMMANDS 20.4a, 2000 frames; every policy aims at 30 + m dB):

  | r | 0.95 | 0.8 | 0.6 |
  |---|---|---|---|
  | naive online, m 0 / 0.5 / 1 (%) | 69.8 / 73.7 / 72.7 | 59.0 / 64.1 / 66.1 | 49.1 / 54.1 / 57.1 |
  | naive plan_adv, m 0 / 0.5 / 1 | 75.1 / 75.3 / 74.9 | 66.4 / 69.0 / 70.3 | 56.8 / 61.4 / 64.3 |
  | aware plan_adv, m 0 / 0.5 / 1 | 75.4 / 75.3 / 74.5 | 70.3 / 71.1 / 70.5 | 64.0 / 66.1 / 66.6 |
  | aware replan_adv, m 0.5 / 1 | 75.9 / 75.8 | 69.7 / 71.1 | 64.0 / 66.1 |
  | PADC, best over m and transmitter | 66.5 | 63.0 | 60.2 |

  A margin is what stale reports call for: the reported SNRs of the slots a user was served in
  are biased up (it was served at its reported peaks), so the model's stopping point is early.
  1 dB brings naive online back above PADC at r = 0.8 (66.1 vs 63.0; not at 0.6), and the best
  rule gains +0.5 / +0.8 / +2.6 points over margin 0 (best JSCC 75.9 / 71.1 / 66.6%, against
  PADC's best +9.4 / +8.1 / +6.4 points; the margin is chosen on the evaluation frames, a mild
  optimism). PADC gains at most 0.9 points from a margin. The rule: decide on the posterior, plan the
  spare slots, aim 0.5-1 dB above the target.
* **The bound and the indices on stale reports** (K 8, nested utility, 300 frames, mean): the
  bound falls with r (31.68 / 31.43 / 30.85 / 30.18 dB at r = 1 / 0.9 / 0.7 / 0.4) and our rule's
  gap to it shrinks (0.25 / 0.25 / 0.21 / 0.12 dB): less information, less to gain. The
  Lagrangian index stays the best policy (+0.065 dB [0.052, 0.079] over our rule with fresh
  reports, +0.005 / +0.041 / +0.033 stale); the Whittle index falls below our rule (-0.06 /
  -0.12 / -0.15) and below PF at r = 0.4: its price-range convention is fragile when the
  reports say little. Aware vs naive at r = 0.7: our rule +0.04 dB. At 30 dB and r = 0.7 the
  bound is loose (71.4% against 59.9% for the best policy: each program knows its own true
  state) and the guided policies collapse (ID 35.6%, random 46.3%): with flat ideal actions the
  order alone decides.
* **Many users per slot: the theory's regime holds** (B = K/8, 100 frames per K):

  | K (B units) | 8 (1) | 16 (2) | 32 (4) | 64 (8) | 128 (16) |
  |---|---|---|---|---|---|
  | gap to the bound, mean PSNR (dB): Lagrangian / Whittle / ours / PF | 0.207 / 0.265 / 0.267 / 0.710 | 0.090 / 0.130 / 0.211 / 0.610 | 0.039 / 0.066 / 0.178 / 0.571 | 0.009 / 0.029 / 0.152 / 0.529 | 0.005 / 0.016 / 0.155 / 0.515 |
  | the same, ID / random guided (dB) | 0.580 / 0.564 | 0.389 / 0.358 | 0.245 / 0.243 | 0.160 / 0.137 | 0.110 / 0.090 |
  | gap at 30 dB (points): Lagrangian / Whittle / ID / random / PF | 1.6 / 5.7 / 4.3 / 5.1 / 14.7 | 1.9 / 4.1 / 3.5 / 4.2 / 14.1 | 1.4 / 3.0 / 3.0 / 3.6 / 14.4 | 1.7 / 2.9 / 3.0 / 3.1 / 13.8 | 1.5 / 2.4 / 2.6 / 3.1 / 13.2 |

  The index policies converge to the bound faster than 1/sqrt(K) (Lagrangian gap x sqrt(K):
  0.59 -> 0.06): with 16 units per slot the Lagrangian index is within 0.005 dB of the best
  possible scheduler. Our closed form plateaus 0.15 dB below the bound, so at scale the
  programs' indices are worth +0.14 dB over it (+0.19 dB already at K 8 with 25% of units lost;
  there the Lagrangian index is 0.000 from the bound at K 128). At 30 dB the Lagrangian index
  stays ~1.5 points from the bound at every K; ID beats random by +0.5 to +0.7 points
  (persistency, the blog's lesson, modest here), and for the mean value ranking beats both
  guided policies. Grid states where the decision flips back as the price rises: 1-1.7% (mean),
  6-13% (target).

## 6c. Delay (COMMANDS 19, run 2026-10-19)

A frame has one deadline, so delay is read three ways (docs/PROBLEM.md section 3, item 7): the
latency budget at equal quality (`tools/latency.py`), time to the target within a frame
(`schedule_sim.py --delay`), and continuous traffic with a deadline per request
(`tools/stream_sim.py`, which serves a frame exactly as schedule_sim does when all deadlines
coincide). snrc4 with the calibrated transmitter, K 8, 2000 frames; the Whittle index on the
nested utility (500 frames; it ties our rule on the mean there, 31.33 dB both).

* **The latency budget (the headline).** For the mean PSNR our rule reaches at T = 16 / 24, PF
  needs 17.6 / 29.5 slots: 9.3% [8.7, 9.9] / 18.6% [16.3, 20.7] fewer slots for ours (31 dB takes
  PF 24.5 slots, ours 21.5). For online's share at 30 dB at T = 16 (66.8%) PADC needs 26.0 slots
  (38.6% [34.9, 41.9] fewer for ours), PF 27.4; online's share at T = 24 (74.0%) no other policy
  reaches within 40 slots (PADC saturates near 70%). 60% at 30 dB takes online 12.4 slots, PADC
  17.0, PF 21.5.
* **Time to 30 dB inside the frame** (T 24, % of all users by slots 6 / 12 / 18 / 24): online 18.3
  / 42.1 / 61.9 / 74.0, PADC 19.5 / 48.9 / 61.6 / 65.6, PF 16.2 / 33.3 / 53.6 / 63.5. Online trails
  PADC in the first half and overtakes in the last quarter (T 16 and 32 alike): its mean slot
  with misses at T + 1 is 0.36 [0.27, 0.45] later than PADC's, and 1.8 slots later on the users
  both bring to 30 dB. The **Whittle index** is ahead at every slot: 38.4 / 60.0 / 69.3 / 72.8
  (PADC 20.1 / 48.4 / 60.8 / 64.6, online 17.6 / 41.2 / 60.7 / 72.6 on the same 500 frames), 2.6
  slots earlier than PADC with misses counted, 2.9 [2.7, 3.1] earlier on the same users, at
  online's final share. With the target reward a user's index is about its success gain per
  remaining phase, so it serves the shortest remaining need first, weighted by the channel.
  For the mean, slope_adv's first phase comes at slot 6.7 (PF 4.5) and its PSNR over the frame
  is 1.3 dB lower (25.0 vs 26.2): the deadline objective does not reward early coarse images.
* **Continuous traffic** (30 dB, D 16 / 24 / 32, 0.1-0.45 requests per slot, 20000 slots): at the
  frame's load (~0.33 per slot, D 24) online brings 70-71% of requests to 30 dB, PADC 46%,
  stop_adv 66-68%, PF timing with stopping 58-62%, content-blind lengths 51-53%. Online's share
  RISES with load, from 66% at 0.1 requests per slot (a request mostly alone, served in every
  slot: no choice of slot) to 71% at 0.3 (multiuser diversity); PADC stays near 45%. Capacity
  at 60% of requests at 30 dB: online >= 0.45 per slot at every D, stop_adv 0.38 / 0.43 / 0.45,
  stop 0.28 / 0.32 / 0.35, PADC and content-blind lengths never (about a third of requests cannot
  reach 30 dB at any load, so 70% and more is out of reach for every policy). Online's delay is
  long: its p90 is the deadline at high load and it is 2.9 to 7 slots slower than PADC on the
  same requests (its deadline guard waits for the slack to run out). The **Whittle index**
  (nested, 10000 slots, D 24, 0.3 / 0.4 requests per slot): 68.3 / 68.8% at 30 dB (online 70.5 /
  69.3, PADC 45.0 / 45.2), 57.3 / 54.1% within D/4 (online 39.2 / 27.3, PADC 38.4 / 34.9), mean
  delay 4.2 / 4.8 slots (online 8.0 / 11.1), and 0.6 / 1.0 slots FASTER than PADC on the same
  requests; D 16 alike. With 4 units per slot the ordering is the same (online +21 to +30 points
  over PADC).
* **The Lagrangian index's time to 30 dB** (COMMANDS 20.3, `rmab_scale.py --delay`, nested
  utility; % of users at 30 dB by slots 6 / 12 / 18 / 24, mean slot with misses at 25):

  | | K 8, one unit (300 frames) | K 32, 4 units (100 frames) | K 8, r = 0.8 aware |
  |---|---|---|---|
  | Lagrangian | 32.3 / 55.5 / 70.3 / **76.6**, 12.67 | 33.5 / 59.2 / 74.8 / **77.1**, 12.06 | 24.2 / 45.1 / 57.5 / 63.6, 15.15 |
  | Whittle | **38.2 / 61.4** / 70.9 / 73.8, **11.95** | **41.0 / 63.8** / 72.8 / 75.4, **11.47** | 28.2 / 48.6 / 58.0 / 63.4, 14.61 |
  | ID-guided | 25.4 / 50.7 / 68.0 / 74.5, 13.66 | 26.1 / 52.5 / 69.8 / 75.5, 13.37 | 9.8 / 19.5 / 28.7 / 36.4, 19.96 |
  | PF | 16.5 / 34.5 / 55.2 / 63.9, 16.37 | 16.8 / 34.3 / 54.3 / 64.1, 16.34 | 13.8 / 30.0 / 48.6 / 59.0, 17.36 |
  | gap to the bound at the deadline (points): Lagrangian / Whittle | 2.2 / 5.1 | 1.4 / 3.0 | 10.0 / 10.3 |

  A trade-off between the two indices: the Lagrangian index brings the most users to 30 dB by
  the deadline (+2.8 points [1.9, 3.8] over the Whittle index at K 8, +1.7 at K 32) but later:
  0.7 slots later on average (1.25 [1.05, 1.47] on the users both bring there), behind until
  about slot 18. Its prices hold a slot back for the user it is worth most to at the deadline;
  the Whittle index's private prices serve the shortest remaining need first. Both are ahead of
  PF by 3.7-4.9 slots and, against 19.1's nested frame (same utility, other frames), the
  Lagrangian index is also ahead of PADC (20 / 48 / 61 / 65) and of online (18 / 41 / 61 / 73) at
  every checkpoint. On stale reports the two tie at the deadline and the bound is loose (each
  program knows its own true state); the guided policies collapse (ID 36%).

## 6d. The runs before writing (COMMANDS 21-22, run 2026-10-23 to -25)

Kodak, the snrc4 prefix code (one encoder, one decoder), nested curves, calibrated transmitter;
K 8, T 24 unless stated. "Decoded": 60 frames decoded by the codec, paired by frame.

**The Lagrangian index on the main frames (21.2, 500 frames, simulated).** Gain over PF (mean)
or over PADC (share at 30 dB):

| setting | lagrange | slope_adv / plan_adv | whittle | best other |
|---|---|---|---|---|
| mean, fresh | **+0.518** | +0.464 | +0.458 | |
| mean, 2 dB CQI error | **+0.485** | +0.420 | +0.429 | |
| mean, 25% of slots lost | **+0.483** | +0.269 | +0.440 | |
| mean, r = 0.8 aware | **+0.486** | +0.473 | +0.374 | |
| mean, T 16 | **+0.430** | +0.329 | +0.391 | |
| 30 dB, fresh | **+11.1** | +8.8 (plan_adv) | +8.2 | online +8.0 |
| 30 dB, 2 dB CQI error | +1.6 | **+8.3** (plan_adv) | +1.2 | online +1.7 |
| 30 dB, 25% lost | +9.1 | | **+10.7** | replan_adv +9.5 |
| 30 dB, r = 0.8 aware, 0.5 dB margin | +6.1 | **+9.7** (plan_adv) | +5.4 | |
| 30 dB, T 16 | +9.2 | +9.6 (plan_adv) | **+10.8** | online +8.6 |

For the mean the index is best everywhere. At 30 dB it is best with fresh, exact reports and
fragile otherwise: its programs model stale reports (the report kernel) but take a Gaussian CQI
error as exact, and they stop users at the predicted threshold; plan_adv (planned spare slots)
is the robust target policy under CQI errors and stale reports, the Whittle index with lost slots
and tight frames. Time to 30 dB: the Whittle index earliest (-2.63 slots against PADC over all
users, -2.89 on the same users), the Lagrangian index -1.92 / -1.21. First decodable phase
later than PF (7.1 vs 4.5).

**Decoded (21.1, 22.1).** The model's error on every file: bias -0.022 ... +0.013 dB, MAE
0.062-0.095: C2 holds on the index's schedules as on ours.

| decoded, Kodak | mean, fresh | mean, r 0.8 | 30 dB, fresh | 30 dB, r 0.8 |
|---|---|---|---|---|
| lagrange minus slope_adv / padc | **+0.060** [0.021, 0.100] dB | +0.004 [-0.07, 0.08] | **+12.3** [9.6, 15.4] pts (22.1: +11.5) | +3.1-4.0 |
| online / plan_adv minus padc | | | online +8.3-9.6 | **plan_adv +9.4-10.4** |
| whittle minus the same base | -0.002 | -0.10 | +8.3-9.0 | +6.3 |
| pf minus slope_adv | -0.468 | -0.512 | | |

**Latency at equal quality (21.3).** The Lagrangian index needs 12.7% [10.8, 15.3] fewer slots
than PF for its mean PSNR at T 16 and 19.9% [13.3, 24.1] at T 24 (2.4-3.1% fewer than our closed
form). At 30 dB, online needs 32 slots for the index's 76% at T 24 (25% fewer for the index) and
PADC more than 32; the Whittle index needs 4.4% fewer than the index at T 16; 70% at 30 dB takes
17.5 slots (Whittle), 17.8 (Lagrangian), 20.4 (online), more than 32 (PADC).

**The PI-type ratio index (22.1, 22.2e-f).** Worse than PF: -0.48 dB fresh, -0.42 at 2 dB CQI
error, -0.25 at r 0.8 (main frames); decoded -0.96 [-1.20, -0.75] dB against our closed form
(-0.82 at r 0.8); at scale 1.06-1.34 dB from the bound (our closed form 0.15-0.27, PF 0.52-0.71).
It starves users: worst user 23.0 dB against PF's 25.9, and 8 of 480 decoded users got nothing.
The ratio rewards a slot that is near the user's own best, whatever its absolute gain, and with
a saturating quality the high-SNR users' gains are nearly flat, so their ratios are large. The
difference form (C3) is the right certainty-equivalent rule.

**Perceptual metrics (22.1, 22.4).** Mean PSNR gains carry over: our closed form vs PF +0.0057
MS-SSIM, -0.0115 LPIPS (Kodak); the Lagrangian index vs our closed form +0.0009 MS-SSIM, +0.0025
LPIPS (slightly worse; CLIC -0.0010, not significant). At the target the share costs the users
who miss: the Lagrangian index's +11.5 points come with -0.0013 MS-SSIM (n.s.) and +0.011 LPIPS
against PADC (its mean PSNR 30.53 vs 30.75), and on stale reports it gives up 44 of 480 users
entirely (+0.085 LPIPS; plan_adv 32 users, +0.053). Online is the balanced target policy: +8.3
points with +0.0073 MS-SSIM and unchanged LPIPS.

**CLIC (22.1, 22.3; the first 60 test images).**
* The abstraction holds: nested fit held-out MAE 0.05-0.11 dB per CBR (free fit the same), the
  transmitter's belief from 12 decodes 0.05-0.09; master profile 0.574 0.227 0.131 0.046 0.013
  0.009, kappa 1.94. Link abstractions on the held-out profiles (MAE, dB, sensitivity weights):
  MMSE **0.083**, noise 0.135, EESM 0.159, mean dB 0.230, capacity 0.290, MIESM (equal weights)
  0.439. The constant-SNR control s1 again obeys no rule (noise best at 0.147, MMSE 0.203, its
  additive fit 2x worse: 0.13-0.24). Piecewise training costs 0.05-0.09 dB at constant SNR.
* Decoded: model error bias -0.029 / -0.000, MAE 0.076 / 0.087 dB. Mean: our closed form +0.51
  over PF, the Lagrangian index +0.12 [0.08, 0.17] over ours. 30 dB: the Lagrangian index +7.7
  [5.4, 10.0] points over PADC, online +5.2 (PADC already at 83.5%: CLIC images reach 30 dB more
  easily). Simulated (500 frames): mean +0.60 / +0.55 / +0.50 dB over PF (Lagrangian / Whittle /
  ours); 30 dB +8.8 / +8.0 / +6.8 points over PADC (Lagrangian / plan_adv / online).

**The 30 dB gap at scale (22.2, B = K/8, 100 frames).** Per-user gap to the bound (points):

| K | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| lagrange (as before) | 1.59 | 1.90 | 1.36 | 1.67 | 1.46 |
| lagrange_relax (the relaxation's tie rule) | 3.34 | 2.52 | 1.30 | 1.48 | 1.19 |
| whittle | 5.72 | 4.09 | 3.05 | 2.92 | 2.39 |
| lagrange, Y grid 256 / 32 bins | 2.99 | | 2.27 | | 2.25 |
| lagrange_relax, Y grid 256 / 32 bins | 3.61 | | 1.68 | | 1.71 |

* Indifferent users at lambda*: 1.9% of the supply at K 8, 0.0% from K 16 on (sure demand 76-79%).
  The tie rule has almost no one to act on; it costs 1.75 points at one unit per slot (the
  Whittle-index order among equal gains is the better one there) and gains 0.27 [0.07, 0.50] at
  K 128. Caveat: the band (half the smallest grid price) is narrower than the golden-section
  precision of lambda*, so users exactly at the kink may show gains just outside it.
* Prices: the supply-scaled subgradient at the kink (30 steps) never improved on the constant
  price at the target (bound and schedules identical to the old solver); for the mean it lowers
  the bound by 0.007-0.008 dB per user, policies unchanged.
* LP-update (re-solving every 6 slots) schedules exactly as lagrange_relax at K 32 and 128 (the
  re-solved prices move, 0.33 -> 0.36 on the stand-in, but a price common to all users in a slot
  does not change their ranking) and -0.6 points against lagrange at K 8.
* Finer grids raise the bound by 0.15-0.17 points and make both policies worse by 0.3-1.4.
* So none of the four closes it. The suspect left (COMMANDS 23.1): every one of these runs, and
  18.3, scores and bounds on the fitted curves while the policies decide on the calibrated
  belief. For the mean that costs nothing measurable (gap 0.005 dB at K 128, -0.004 with the new
  prices); at a 0/1 target a user stopped at its predicted 30 dB is a miss whenever the belief
  is a little optimistic, at every K. Finer grids (a sharper threshold, more users stopped right
  at it) making the policies worse fits. A stand-in check with a 0.025 dB belief error showed no
  flat component (7.3 vs 6.9 points at K 8, 3.1 vs 3.2 at K 32), which neither confirms nor rules
  it out (the stand-in's gaps are 2-4x the real ones).

**Mean at scale with the new prices (22.2e).** Unchanged: the Lagrangian index 0.207 -> 0.005 dB
per user from K 8 to 128 (0.206 -> -0.004 with supply-scaled prices); the relaxation's tie rule
identical for the mean.

## 6e. The 30 dB gap explained; digital against the single decoder (COMMANDS 23, run 2026-10-25)

**The 30 dB gap was the transmitter's curve errors (23.1).** Every earlier scaling run bounded
and scored on the fitted curves while the policies decided on the calibrated belief. Given the
fitted curves, the Lagrangian index meets the bound (B = K/8, 100 frames, the frames of 18.3 and
22.2; per-user gap in points, 95% CI):

| K | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| bound (share at 30 dB, %) | 78.09 | 78.34 | 78.45 | 79.01 | 78.88 |
| lagrange, true curves | +0.84 [0.05, 1.70] | -0.23 [-0.80, 0.36] | -0.27 [-0.61, 0.08] | +0.06 [-0.20, 0.32] | -0.19 [-0.38, -0.01] |
| lagrange, calibrated curves (22.2a) | +1.59 | +1.90 | +1.36 | +1.67 | +1.46 |
| cost of the curve errors (difference) | 0.75 | 2.12 | 1.62 | 1.61 | 1.66 |
| whittle, true / calibrated | 4.84 / 5.72 | 3.09 / 4.09 | 1.80 / 3.05 | 1.73 / 2.92 | 1.34 / 2.39 |
| lagrange, calibrated, policies aim at 30.25 dB | +2.97 | | +1.61 | | +1.86 |
| lagrange, calibrated, policies aim at 30.5 dB | +6.47 | | +6.11 | | +5.24 |
| lagrange, true curves, Y grid 256 / 32 bins (its own bound +0.15) | +1.24 | | -0.07 [-0.42, 0.27] | | |

* On its own model the index is near-optimal at the target too: from K 16 on its per-user gap is
  within +-0.3 points of 0, i.e. a bounded total gap (the non-degenerate O(1) regime of Zhang
  and Frazier), as for the mean. The small negative gaps are the coarse bound's own
  discretisation error: finer grids raise the bound by 0.15-0.17 points, and with them the gap at
  K 32 is -0.07.
* The curve errors cost ~1.6 points per user from K 16 on (0.75 at one unit per slot), at every
  K: a hard target makes the abstraction's calibration error first-order, while for the mean the
  same errors cost nothing measurable (gap 0.005 dB per user at K 128 with them). The decodes show
  the same effect against the codec: at 30 dB the policies that stop users closest to the
  threshold decode furthest below their predicted share (Lagrangian index 77.7% vs 79.0%, Whittle
  74.6 vs 75.4, online 74.6 vs 75.2, PADC 66.2 vs 66.5), while every mean decodes within 0.02 dB.
* A uniform margin makes it worse (0.25 dB: +0.25 to +1.4 points more gap; 0.5 dB: +3.8 to
  +4.9): the errors are per image and go both ways, so a margin spends slots on every user to
  protect the few whose curves are optimistic, and admits fewer users. On stale reports, where
  the error is several dB, a margin helped (section 6b); here it is a few hundredths.
* The finer grid hurt in 22.2d only with calibrated curves; with the true curves it changes
  nothing (77.0 vs 77.25 at K 8, 78.69 vs 78.72 at K 32): the coarse grid's smoothing of the 0/1
  reward was a hedge against the curve errors.
* The Whittle index's gap falls with K given the true curves too (4.8 -> 1.3) but stays above
  the Lagrangian index's; the curve errors cost it ~1 point.
* Built 2026-10-25, run in COMMANDS 24: programs whose terminal reward is the calibrated
  success probability Phi((q + bias - Q) / sigma), sigma and bias per prefix length pooled over
  the images, in place of 1{q >= Q} (a soft target; the hedge a margin cannot be):
  `whittle_soft` / `lagrange_soft` (docs/PAPER.md, owed item 13). Smoke stand-in (T 10, 29 dB,
  60 frames; curves perturbed to a measured error of sigma ~0.10 dB, bias +0.03, close to the
  real calibration's): the curve errors cost the hard Lagrangian index 0.21 / 1.41 points per
  user at K 8 / 32 (gap 7.07 / 4.59 against 6.86 / 3.18 with the true curves); the soft index
  recovers +0.63 [-0.21, +1.41] at K 32 (about 45%) and -0.42 [-1.88, +1.04] at K 8 (nothing to
  recover); the soft Whittle index gains nothing (0.00 / -0.42). Suggestive, not significant:
  the real curves' runs (100 frames, K to 128) decide. They did (section 6f): with the real
  curves the soft index recovers all of it.

**C5 with the single decoder (23.2).** The prefix code snrc4 (one encoder, one decoder), its
nested curves and calibrated transmitter, 2000 frames, the frames of section 20.2; best policy
and transmitter (naive or aware) at each point, closed-form policies (the Lagrangian index adds
~0.06 dB for the mean); digital with IR-HARQ as in section 5. In brackets: the per-length
decoders' lead (section 5).

| report correlation r | 1 | 0.95 | 0.9 | 0.8 | 0.6 | 0.4 | 0 |
|---|---|---|---|---|---|---|---|
| mean PSNR, single decoder | 31.42 | 31.30 | 31.17 | 30.90 | 30.42 | 30.09 | 29.84 |
| minus HEIC + HARQ (dB) | +0.13 (+0.34) | +1.11 (+1.32) | +1.35 (+1.56) | **+1.54** (+1.73) | +1.49 (+1.69) | +1.31 (+1.49) | +1.09 (+1.25) |
| minus the 3 dB-gap code + HARQ (dB) | -0.71 | +0.65 | +0.98 | +1.19 | +1.23 | +1.13 | +0.96 |
| at 30 dB, single decoder (%) | 74.7 | 73.3 | 71.7 | 68.3 | 61.9 | 56.9 | 55.3 |
| minus HEIC + HARQ (points) | +0.3 (+2.8) | +9.7 (+11.8) | +12.2 (+14.4) | **+13.5** (+15.5) | +12.1 (+14.2) | +10.1 (+12.6) | +8.3 (+9.7) |

| CQI error sd (dB) | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| mean PSNR, single decoder | 31.42 | 31.39 | 31.27 | 31.13 |
| minus HEIC + HARQ (dB) | +0.13 (+0.34) | +0.71 (+0.91) | **+1.12** (+1.33) | +1.40 (+1.60) |
| minus the 3 dB-gap code + HARQ (dB) | -0.71 | +0.24 | +0.74 | +1.09 |
| at 30 dB (%) / minus HEIC + HARQ (points) | 74.7 / +0.3 (+2.8) | 74.6 / +6.2 (+8.5) | 73.4 / **+10.7** (+12.9) | 70.9 / +12.8 (+15.1) |

The verdict stands with the single decoder: on par with exact CSI (+0.13 dB, +0.3 points; a
3 dB-gap code is 0.7 dB ahead), ahead of HEIC with HARQ from 1 dB of CQI error or r = 0.95, and of
the 3 dB-gap code from 1 dB / r = 0.95. Per-length decoders add a uniform 0.17-0.21 dB and 1.4-2.5
points to every lead. On stale reports the naive target policies break as in section 6b (online
-4.0 points under PADC at r = 0.8 naive, +3.3 aware); plan_adv aware is the best target policy
from r = 0.95 to 0.4, PADC at r = 0. The JSCC side holds decoded at r = 0.8 with this decoder
(section 6d: model error MAE 0.076-0.095 dB, bias -0.012 / +0.013).

## 6f. Calibration-aware programs recover the curve errors' cost (COMMANDS 24, run 2026-10-26)

The soft target (alloc/whittle.Arm(soft=...), soft_spread): at the target a user's terminal
reward is the probability that its true PSNR reaches 30 dB when the transmitter's curves are off
by N(bias, sigma^2), Phi((q + bias - Q) / sigma), not 1{q >= Q}. Measured on the Kodak curves
(fitted against calibrated, pooled over images): sigma 0.031 0.044 0.047 0.040 0.033 0.028 dB and
bias +0.021 +0.012 -0.009 -0.008 +0.000 +0.010 dB at 1-6 phases.

**At scale (24.1, calibrated curves, B = K/8, the frames of 18.3-23).** Per-user gap to the bound
(points):

| K | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| lagrange, hard (as 22.2a) | 1.59 | 1.90 | 1.36 | 1.67 | 1.46 |
| **lagrange_soft** (measured sigma) | **0.84** | **-0.10** | **-0.30** | **+0.01** | **-0.19** |
| lagrange, hard, TRUE curves (23.1a) | 0.84 | -0.23 | -0.27 | +0.06 | -0.19 |
| lagrange_soft - lagrange (paired) | +0.75 [0.00, 1.63] | +2.00 [1.38, 2.69] | +1.66 [1.25, 2.16] | +1.66 [1.36, 1.97] | +1.66 [1.45, 1.89] |
| whittle hard / soft / true curves | 5.72 / 4.34 / 4.84 | 4.09 / 2.71 / 3.09 | 3.05 / 1.49 / 1.80 | 2.92 / 1.51 / 1.73 | 2.39 / 1.12 / 1.34 |
| lagrange_soft, sigma 0.05 / 0.1 / 0.2 dB (no bias) | 1.09 / 0.72 / 0.97 | | -0.23 / -0.11 / +0.14 | | -0.23 / -0.09 / +0.05 |

* The calibration-aware Lagrangian index, deciding on the calibrated curves, is exactly as close
  to the bound as the hard index given the true curves: it recovers 100% of the curve errors'
  cost at K 8 and 99% pooled over K 16-128 (identical values at K 8 and 128). The flat ~1.6
  points were a decision artefact of trusting the curves at a hard threshold, not an information
  limit: the curves' error is 0.03-0.05 dB, and a program that knows it hedges by a fraction of
  a phase where that is cheap.
* Insensitive to sigma: anything from the measured 0.03-0.05 dB up to 0.2 dB recovers it (at
  K >= 32 every sigma lands within +-0.3 points of the bound; 0.2 dB slightly over-hedges). No
  tuning is needed beyond the offline calibration statistics.
* The soft Whittle index gains 1.3-1.6 points too and ends above even the hard Whittle index
  with the true curves (the smooth reward also helps the index itself, whose 0/1 programs are
  6-13% non-indexable), but stays 1.3-3.5 points behind the soft Lagrangian index.

**Main frames (24.2, K 8, T 24, 500 frames; share at 30 dB, gain over PADC):**

| policy | fresh | 2 dB CQI error | r = 0.8 aware, 0.5 dB margin |
|---|---|---|---|
| lagrange | +11.1 | +1.6 | +6.1 |
| **lagrange_soft** | **+12.7** (77.2%) | +6.1 | +7.1 |
| whittle / whittle_soft | +8.2 / +9.8 | +1.2 / +5.1 | +5.4 / +6.3 |
| plan_adv | +8.8 | **+8.3** | **+9.7** |
| online | +8.0 | +1.7 | |

With fresh reports the soft index is the best target policy (+1.5 points over the hard one). It
also hedges part of the CQI errors' damage (+4.5 points over the hard index), though the planner
plan_adv stays best under CQI errors and stale reports, whose errors are dB, not hundredths.
Time to 30 dB: like the hard index (-1.85 slots vs PADC over all users, -1.04 on the same users;
the Whittle indices remain earliest). Its mean PSNR is 0.22 dB below the hard index's: the share
objective gives up a little more on the users who miss.

**Decoded by the codec (24.3, 60 frames; share at 30 dB):**

| file | lagrange | lagrange_soft | soft - hard (paired) | soft over PADC | missed (soft / hard) |
|---|---|---|---|---|---|
| Kodak, measured sigma | 78.8% | **80.4%** | **+1.67 [0.42, 2.92]** | +14.8 [11.7, 17.9] | 0.0% / 1.2% |
| Kodak, sigma 0.15 dB | 78.1% | 80.2% | +2.08 [0.00, 4.38] | +14.4 | 0.2% / 1.7% |
| CLIC, measured sigma | 91.2% | 91.7% | +0.42 [-0.63, 1.67] | +8.1 | 0.0% / 0.8% |
| Kodak, r = 0.8 aware | 63.3% | 64.0% | +0.63 [-1.67, 2.92] | +4.4 (plan_adv +10.4) | 2.3% / 1.7% |

* The soft schedules decode exactly as the curves predict (Kodak: 80.4% decoded, 80.4% predicted,
  0.0% missed; the hard index loses 1.2% of its users to the codec's error): no user is left
  within the curves' error of the threshold, so the hedge also covers the codec's own error,
  which the measured sigma does not include. The soft Whittle index does the same (77.3 = 77.3).
* CLIC is near its ceiling (PADC 83.5%, the index 91%); the gain there is within noise.
* On stale reports the misses come from the reports, not the curves, and the planner stays best.
* The price, as in the simulation: lower mean PSNR (30.27 vs 30.52 dB decoded) and worse LPIPS
  (+0.013 against the hard index, +0.025 against PADC) for the users who miss.

## 6g. The calibration-aware index in every condition (COMMANDS 26, run 2026-10-27)

After the first figure review the calibration-aware Lagrangian index is the proposed target
scheduler (docs/PAPER.md). 26.2 ran it where only the hard index had run: the main frames
with 25% of slots lost, at T 16 and on CLIC (500 frames, the frames of 21.2 and 22.3: PADC,
online and the planner reproduce exactly), and the latency budget (300 frames per length, as
21.3). Share at 30 dB, gain over PADC (points; 95% CIs within about +-1):

| setting | lagrange, hard (6d) | **lagrange_soft** | whittle, hard | whittle_soft | planner | online |
|---|---|---|---|---|---|---|
| fresh (24.2) | +11.1 | **+12.7** | +8.2 | +9.8 | +8.8 | +8.0 |
| 25% of slots lost | +9.1 | **+11.2** [9.8, 12.5] | +10.7 | **+11.6** [10.7, 12.5] | +9.5 (replan_adv) | +6.3 |
| T 16 | +9.2 | **+11.5** [10.3, 12.6] | +10.8 | **+11.6** [10.7, 12.5] | +9.6 | +8.6 |
| CLIC | +8.8 | **+9.5** [8.7, 10.5] | | +7.4 | +8.0 | +6.8 |
| 2 dB CQI error (24.2) | +1.6 | +6.1 | +1.2 | +5.1 | **+8.3** | +1.7 |
| r = 0.8 aware, 0.5 dB margin (24.2) | +6.1 | +7.1 | +5.4 | +6.3 | **+9.7** | |

* The calibration-aware reward raises the Lagrangian index by +2.1, +2.3 and +0.7 points with
  lost slots, at T 16 and on CLIC (the Whittle index by +0.9 and +0.8): never lower, in every
  condition run (6f, 6g).
* With exact reports the calibration-aware Lagrangian index is now best or tied-best
  everywhere: alone with fresh reports and on CLIC, within 0.4 points of the calibration-aware
  Whittle index with lost slots and at T 16, both ~2 points above the planner. The hard index's
  shortfall there (6d: the Whittle index led) was the curve errors' cost too. With CSI errors
  (CQI errors, stale reports) the planner stays best: C7 has two cases.
* The price for the users who miss stays: with lost slots mean PSNR 28.25 dB (the index)
  against PADC 29.05 and online 29.42.
* Time to 30 dB at T 16: the Whittle index earliest (-1.91 slots against PADC over all users,
  -1.89 on the same users); the Lagrangian index -0.54 over all users but +0.38 [0.17, 0.57] on
  the same users: in a short frame it waits for good slots and finishes near the deadline.

**Latency at 30 dB (26.2; % of users at 30 dB by frame length):**

| T | 12 | 16 | 20 | 24 | 28 | 32 |
|---|---|---|---|---|---|---|
| lagrange_soft | 62.0 | 70.5 | 74.1 | **77.7** | 78.5 | 79.4 |
| whittle_soft | 62.9 | 70.4 | 72.3 | 74.8 | 76.2 | 76.6 |
| online | 59.8 | 67.2 | 69.7 | 72.8 | 74.7 | 76.0 |
| padc | 51.9 | 58.2 | 61.8 | 64.8 | 67.9 | 68.0 |

* For the index's 70.5% at T 16 online needs 21.1 slots [18.7, 23.9] (the index 24.2% fewer
  [14.6, 32.9]), the Whittle index 16.3 (a tie), PADC more than 32.
* For its 77.7% at T 24 no other policy gets there within 32 slots (the index needs at least
  25% fewer; the hard index's 76% took online 32).
* 70% of users at 30 dB: 15.8 slots (both calibration-aware indices) against 17.8 for the hard
  Lagrangian index (21.3), 20.4 for online and more than 32 for PADC: the reward alone saves
  2 slots (11%).

## 7. Open

1. An online phase-to-slot mapping rule for fixed-quota frames (sequential stochastic assignment,
   Derman-Lieberman-Ross; exact per user because distortion is linear in the per-phase noise
   terms): up to +0.13 dB / +3-4 points ex post. CPU.
2. Review robustness: a second test set (CLIC), a perceptual metric (MS-SSIM, LPIPS) at the
   target. GPU.
3. Optional: a Jacobian probe (the decoder's per-chunk Jacobian energy against the fitted C rho:
   the first-order mechanism behind the abstraction); the codec gap (more data).
4. The scheduler to recommend is the programs' Lagrangian index (0.005 dB from the bound at
   K 128, robust to stale reports, the most users at 30 dB by the deadline; the Whittle index is
   earlier). Built into `schedule_sim.py` as `lagrange` (2026-10-20, identical to rmab_scale's
   with one unit per slot): COMMANDS 21 runs it on the main frames (CQI errors, stale reports,
   lost slots, the latency budget) and decodes its schedules.
5. Done 2026-10-20: margins on stale reports (aware + plan_adv + 0.5-1 dB), the decoded
   stale-report schedules, digital with HARQ (C5 restated: +1.3 dB at 2 dB CQI error, +1.7 dB /
   +15.5 points at r = 0.8).
6. The 30 dB gap of the Lagrangian index (~1.5 points below the bound at every K, B = K/8;
   section 6b). Two leads from 2026-10-20: Hao, Xu and Tong (IEEE TAC, read in full) get the
   same flat gap when ties are not broken by the relaxation's randomized policy (ours go to the
   Whittle index); and `alloc/whittle.lagrangian_bound`'s subgradient step is not divided by B.
   Smoke stand-in check (8 images, P 6, T 24, one draw per K, mean PSNR per user): constant
   price 31.9393 / 31.3366 / 31.6951 dB at K 8 / 32 / 64; 60 steps as coded 31.9319 / 31.3366 /
   31.6951 (no gain at B 4 and 8); 60 steps divided by the supply 31.9319 / 31.3302 / 31.6882.
   The stand-in's 30 dB target does not bind (prices 0). One-unit runs are unaffected. Plan:
   docs/PAPER.md, owed item 9. CPU.
   Batch 2 of the full texts (2026-10-20) adds the multi-action counterpart: McDowell et al.,
   citing Chakraborty, Sun, Hou and Eryilmaz (2026, in preparation), report that the per-slot
   knapsack with arbitrary ties is not asymptotically optimal and an LP-priority tie rule fixes
   it. The tie rule is the first suspect to test.
   Batch 5 (2026-10-21; Gast, Gaujal and Yan; Zhang and Frazier) sharpens it: a flat per-user
   gap is a total gap linear in K, which an LP-compatible policy never has (a degenerate problem
   still gives 1/sqrt(K) per user), and no priority rule (ours: the Whittle index among
   indifferent users) is LP-compatible when several users are mixed in the relaxation. Price
   check on the stand-in with a binding target (T 10, 29 dB, 60 supply-scaled steps): bound per
   user 0.7826 -> 0.7819 at K 8 (B 1) and 0.7043 -> 0.7028 at K 32 (B 4), prices within +-4% of
   the constant price except the last slot (0.84 of it): prices explain at most ~0.15 points.
   Order of the diagnosis: tie rule, then the 0/1 reward's discretisation, then prices, then
   LP-update (docs/PAPER.md, owed item 9).
   Built 2026-10-22, run in COMMANDS 22.2a-d: `lagrange_relax` (tools/rmab_scale.py and
   tools/schedule_sim.py: sure gains first, then the users indifferent at lambda*_t in a random
   order, then the rest), `lagrange_lpu` (re-solves lambda* from the users' states every
   --resolve slots), `--price-scale supply`, `--kink-grad` (alloc/whittle.kink_demand: the
   relaxation's common mix theta_t of the indifferent states, which meets the supply wherever it
   is interior), `--ny` / `--ng`; every run prints the sure demand and the indifferent mass at
   lambda* as shares of the supply. The defaults reproduce every earlier bound and schedule bit
   for bit. Stand-in, T 10, 29 dB, 30 frames: the indifferent users hold 38% / 44% of the supply
   at K 8 / 32 and the sure demand 92% / 95%; `lagrange_relax` - `lagrange` +0.00 / -0.10
   points; the gap 6.7 -> 3.9 points per user already falls like 1/sqrt(K); a 4x finer grid
   moves the bound by 0.004 per user. The stand-in does not show the flat gap, so only the real
   utility's runs can decide.
   Run 2026-10-25 (COMMANDS 22.2, section 6d): none of the four closes it (the indifferent mass
   at lambda* is ~0; the tie rule costs 1.75 points at one unit and gains 0.27 at K 128; prices
   and re-solving change nothing; finer grids make the policies worse). Last suspect: the
   transmitter's curve errors, which the bound does not have (COMMANDS 23.1).
   Resolved 2026-10-25 (COMMANDS 23.1, section 6e): with the true curves the index meets the
   bound from K 16 on (within +-0.3 points); the curve errors cost ~1.6 points per user at the
   target and nothing measurable for the mean; uniform margins make it worse. And recovered
   2026-10-26 (COMMANDS 24, section 6f): calibration-aware programs (the success probability
   under the curves' measured error) bring the index on the calibrated curves to the bound
   (99-100% of the cost), and their schedules decode exactly as predicted.
7. A PI-type ratio baseline (Ayesta, Erausquin and Jacko, Performance Evaluation 2010, read in
   full 2026-10-20: nu^PI = c mu_n / sum_{m>n} q_m (mu_m - mu_n), infinite in the best channel
   state). For our arm: serve the largest g_k(s) / E[(g_k(S') - g_k(s))^+], g_k(s) the predicted
   PSNR gain of user k's next phase at slot SNR s. Built 2026-10-22 as `--policies pi` in both
   tools (S' on 32 equiprobable fades at the user's mean SNR; aware SNRs on stale reports; ties,
   including infinite indices, by the largest gain): COMMANDS 22.1 (decoded), 22.2e-f
   (docs/PAPER.md, owed item 10). CPU. Run 2026-10-25 (section 6d): it loses to PF (-0.48 dB) and
   to our closed form by 0.96 dB decoded, and starves users; C3 stands.
8. Optional, only if a referee asks: a two-layer superposition digital baseline with IR per layer
   (Steiner and Shamai, IEEE TWC 2008, read in full 2026-10-21) at r <= 0.6, where our 0-15 dB
   users sit in its best range (docs/PAPER.md, owed item 11). Its full text supports omitting
   it otherwise: it is designed for no transmit CSI, and its gain needs a channel fixed between
   rounds and few retransmissions, while our chain has CQI, newly faded retransmission slots and
   4 IR rounds.
9. The paper's figures (2026-10-26): `tools/paper_figs.py` draws the result figures and two
   LaTeX tables from results/ (COMMANDS 25-26, CPU, ~1 min; PDF + PNG at IEEE sizes);
   docs/FIGURES.md lists each one's claim, inputs and suggested use. Checked on synthetic inputs
   in the tools' own formats (`--self-test`; smoke: the parsers read schedule_sim.py's and
   utility_fit.py's tables back and pair decodes exactly as replay_report.py does). First review
   of the real figures 1-5 (2026-10-26): the abstraction figure became Table I; the phase-weight
   figure compares the two trainings directly; the predicted-vs-decoded scatter became the
   error along the PSNR; the mean and 30 dB scaling figures merged (gap per user only); and the
   calibration-aware index became the proposed target scheduler, the hard-target index an
   ablation (fig05, Table II). COMMANDS 26 ran it with lost slots, at T 16, on CLIC and in the
   latency budget (section 6g); the figures draw from those runs.

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
| RMAB re-examined | 18 | 10-19 | stale reports: JSCC over digital +0.3 -> +2.5 dB / +19 points at r 0.8; the opportunistic part and the lift fade; aware + plan_adv for the target; with many units the indices reach the bound (0.005 dB at K 128), our closed form plateaus at 0.15 |
| Delay | 19 | 10-19 | latency budget 9-19% (mean, vs PF) and 39% (30 dB, vs PADC) shorter; the Whittle index reaches 30 dB first (2.9 slots before PADC, same users) at online's share; traffic: online +25 points over PADC, the Whittle index with half its delay |
| HARQ, decoded stale, Lagrangian delay, margins | 20 | 10-20 | C5 with HARQ: +1.3 dB at 2 dB CQI error, +1.7 dB / +15.5 points at r 0.8 (HARQ took 0.3 / 0.75 dB off); stale-report schedules decoded as simulated; the Lagrangian index has the best share at 30 dB, the Whittle index the earliest; margins of 0.5-1 dB on stale reports |
| Before writing | 21, 22 | 10-23 to 10-25 | the Lagrangian index best for the mean everywhere and at 30 dB with fresh reports; the PI index loses to PF; CLIC, MS-SSIM and LPIPS hold; the 30 dB gap survives the tie rule, prices, LP update, grids |
| The 30 dB gap; C5 with one decoder | 23 | 10-25 | the gap is the curve errors' cost (true curves meet the bound), margins worse; the single decoder leads digital by +1.1 / +1.5 dB at 2 dB CQI error / r 0.8 |
| Calibration-aware programs | 24 | 10-26 | 99-100% of the curve errors' cost recovered at every K, any assumed sigma 0.03-0.2 dB; decoded +1.7 points, 0.0% missed |
| Figures | 25 | 10-26 | tools/paper_figs.py; first review: the calibration-aware index proposed, the hard target an ablation |
| Calibration-aware everywhere | 26 | 10-27 | +2.1 / +2.3 / +0.7 points with lost slots / T 16 / CLIC; best or tied-best with exact reports; at 30 dB 24% fewer slots than online at T 16 |
