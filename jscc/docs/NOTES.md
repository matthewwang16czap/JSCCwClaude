# Results notes (condensed 2026-10-15; sections 18-19 added 2026-10-19)

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
2.3 dB: from 1 dB of CQI error on, JSCC is ahead even of the 3 dB-gap code. Caveat found
2026-10-19: digital here has no HARQ (a lost block's slot is wasted). With HARQ-IR and its rates
tuned for it, on a toy codec digital's best policy gains +0.1 / +0.9 / +1.9 dB at 1 / 2 / 3 dB of
error, so this lead may shrink to well under 1 dB; COMMANDS 20.2 measures it.

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

So at 10 km/h with 5 ms-old reports (r ~0.76) JSCC leads by ~2.6 dB and ~19 points, at 3 km/h
(r ~0.98) by ~0.8 dB. From r = 1 to 0 JSCC loses 1.6 dB, HEIC 4.8 (its backoff grows to 2-7 dB
and it still loses 7-50% of its blocks); a code 3 dB from capacity falls behind from r = 0.95.
The JSCC side is decoded at r = 0.8 / 0.6 in COMMANDS 20.1 (pending). The digital side
drops a block the slot could not carry; NR would retransmit it and combine (HARQ), which is
what stale reports call for: `schedule_sim.py --harq ir|chase` (built 2026-10-19) does it,
and on a toy codec (stand-in R-D curves; rates tuned with HARQ on, which makes them bolder than
the report) digital's best policy gains +2.2 dB at r = 0.8 and +3.1 dB at r = 0.6, about half of
what stale reports cost it. The lead above is an upper estimate, possibly by 1 dB or more,
until COMMANDS 20.2 measures it with HARQ.

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

## 7. Open

1. An online phase-to-slot mapping rule for fixed-quota frames (sequential stochastic assignment,
   Derman-Lieberman-Ross; exact per user because distortion is linear in the per-phase noise
   terms): up to +0.13 dB / +3-4 points ex post. CPU.
2. Review robustness: a second test set (CLIC), a perceptual metric (MS-SSIM, LPIPS) at the
   target. GPU.
3. Optional: a Jacobian probe (the decoder's per-chunk Jacobian energy against the fitted C rho:
   the first-order mechanism behind the abstraction); the codec gap (more data).
4. The scheduler to recommend at scale is the programs' Lagrangian index (0.005 dB from the bound
   at K 128, robust to stale reports); its time to the target is measured in COMMANDS 20.3.
5. Quality targets on stale reports: stopping needs the posterior; does a margin restore online?
   (COMMANDS 20.4). The decoded check of the stale-report schedules: COMMANDS 20.1 (GPU).
6. Digital with HARQ (incremental redundancy, Chase combining; built 2026-10-19): C5 on stale
   reports and CQI errors again, COMMANDS 20.2. The stale-report lead is the one at stake.

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
