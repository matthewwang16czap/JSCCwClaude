# Where every number in the Introduction outline comes from

This record covers each number in `docs/INTRO_OUTLINE.md`. For each one it gives:
- the result file on your server (`~/projects/JSCCwClaude-main/jscc/results/`);
- the round that wrote it (a COMMANDS.txt section and subsection) and the date you sent the
  output back;
- the line to look for in the file;
- whether the number is simulated or decoded.

Every line quoted below was matched against the outputs you pasted into the session.

- **Simulated**: schedules scored on the fitted quality curves (`tools/schedule_sim.py`,
  `tools/rmab_scale.py`).
- **Decoded**: the codec itself decoded the schedules (`tools/replay_report.py` reads the
  decodes).
- **Setup**: unless a file says otherwise, Kodak (24 images), the snrc4 code, K = 8 users,
  T = 24 slots, mean SNR U[0, 15] dB, i.i.d. Rayleigh block fading. Every output file prints
  this in its header line.

## What the trace changed in the outline

Five numbers were stated more strongly than their files support. `INTRO_OUTLINE.md` is
corrected to match:

1. **MIESM and EESM were not compared like for like.** The old "0.49 dB" for MIESM was its
   equal-weight form. The old "0.19 dB" for EESM, like MMSE's 0.055 dB, used the per-phase
   sensitivity weights. With the same weights for all three rules: MMSE 0.055, EESM 0.194,
   MIESM 0.337 dB (Kodak). In the standard equal-weight form: MMSE 0.367, EESM 0.373, MIESM
   0.487 dB.
2. **Not everything was decoded.** The gap to the bound (0.207 → 0.005 dB; within ±0.3 points)
   is simulated, over 100 frames per K. So is the comparison with digital, over 2000 frames.
   The old outline called all results decoded.
3. **The digital comparison is not CLASP's.** Its JSCC side used the best policy at each point:
   - for mean PSNR, the closed form `slope_adv` (the Lagrangian index adds about 0.06 dB,
     NOTES 6e);
   - for the 30 dB share, the planner `plan_adv`.

   Under CSI errors the planner beats CLASP's calibration-aware index: +8.3 vs +6.1 points over
   PADC at 2 dB CQI error, and +9.7 vs +7.1 at r = 0.8 (simulated, round 24.2). A direct number
   for CLASP against digital needs one more CPU run.
4. **Scheduler-aware training (+0.28 dB) was measured under the closed-form scheduler**, not
   CLASP's index. Of the +0.28, +0.16 is training for the single prefix length the scheduler
   gives. The scheduler's channel itself adds +0.08-0.12 dB.
5. **The model error on scheduled deliveries (0.05-0.09 dB) holds with fresh reports only.**
   With aged reports it is 0.076-0.095 dB for the single decoder and 0.082-0.121 dB for
   per-length decoders.

## Round 30: the numbers it measures for CLASP

COMMANDS section 30 measures directly the numbers that rested on another scheduler's runs or
on arithmetic. When it is back, these replace the entries below:

| number in the outline | now (trace above) | round 30 file | what changes |
|---|---|---|---|
| +0.5-0.6 dB over PF, decoded | 31.651 - 31.127 and 34.192 - 33.561 (two means) | `s30_direct.txt` (30.1) | the same gains, with their own paired intervals |
| against digital: +1.1-1.5 dB, +10.7-13.5 points | best JSCC policy at each point (closed form; planner) | `s30_vs_digital.txt`, rows "CLASP - digital" (30.2) | CLASP's own lead, paired over 2000 frames, with intervals; the rows "best JSCC - digital" reproduce C5 |
| fresh → r = 0.8: digital -1.9 dB, JSCC -0.5 dB | the closed form's means | `s30_loss.txt` (30.2) | CLASP's loss and digital's, with intervals |
| the JSCC side holds decoded | checked at r = 0.8 only | `s30_dec_e2.txt` (30.3) | also at 2 dB CQI error, with CLASP's schedules |
| scheduler-aware training +0.28 dB (+0.08-0.12 beyond the length) | the closed form's channel and schedules | `s30_codesign.txt` (30.4) | training on CLASP's channel, scored under CLASP's schedules; lag - lagblind is the part due to the slot selection |

Check before using them: 30.5 (2) must print the old numbers, which means the frames are the
same ones (slope_adv 31.424 / 31.274 / 30.902; greedy 31.291 / 30.146 / 29.359).

## A. The codec (paragraph 4)

| fact | source |
|---|---|
| ViT, patch 8, width 768, 10 blocks, 4 complex symbols per token; six phases of CBR 1/48 at 256 px (1/8 in all) | checkpoint `history/vit_p8d768x10s4_awgn_DIV2K_256_snrc4_s1`, NOTES 1 |
| trained with the chunks of each codeword at independent SNRs | COMMANDS 5c (E2b, log `s1_gpu5_E2b_snrc4.out`): from w1, 400 epochs, DIV2K, `--snr-chunks 4`, `--snr-range -2 22` |
| the constant-SNR control (s1) | COMMANDS 5c (E2a, log `s1_gpu5_E2a_const.out`): the same recipe without `--snr-chunks` |
| the encoder and decoder never see the SNR | `net/network.py`: `encode(self, x)` and `decode(self, enc, rx, units)` take no SNR; only the channel model does |
| held-out profiles | COMMANDS 6b, `results/p2_full_snrc4.json` / `p2_full_s1.json`: 10 random mixed-SNR profiles per CBR per image |

## B. The link abstraction (paragraphs 4 and 6)

All values are mean absolute error / p95 in dB, from decoded held-out profiles. The two columns
are equal weights / sensitivity weights.

| number | file, round (date) | line to look for |
|---|---|---|
| Kodak, snrc4: MMSE **0.055** | `s13_abstractions.txt`, block 1 of 4 (snrc4); written by COMMANDS 17.4a (2026-10-14) | `MMSE, kappa 1.77  0.367 / 0.951  0.055 / 0.187` |
| Kodak, snrc4: EESM 0.373 / **0.194** | same block | `exponential (EESM)  0.373 / 1.144  0.194 / 0.608` |
| Kodak, snrc4: MIESM **0.487** / 0.337 | same block | `capacity (MIESM-like)  0.487 / 1.480  0.337 / 0.896` |
| Kodak, constant-SNR code: MMSE **0.222**; best rule (noise) **0.150** | block 2 (s1) | `MMSE, kappa 2.44 ... 0.222 / 0.674`; `noise variance ... 0.150 / 0.487` |
| CLIC (60 images), snrc4: MMSE **0.083**; EESM 0.374 / **0.159**; MIESM **0.439** / 0.290 | `s22_clic.txt`, first abstraction block; COMMANDS 22.3 (2026-10-25) | `MMSE, kappa 1.93 ... 0.083 / 0.275`; `exponential (EESM)  0.374 / 1.152  0.159 / 0.533`; `capacity (MIESM-like)  0.439 / 1.546  0.290 / 0.842` |
| CLIC, constant-SNR code: MMSE 0.203; best rule (noise) 0.147 | `s22_clic.txt`, second block | `MMSE, kappa 2.65 ... 0.203 / 0.609` |
| **12 decodes** per image; calibration error 0.030-0.064 dB | `s8_calib.txt`, COMMANDS 8a (2026-10-11) | `... constant decodes at [1.0, 13.0] dB (12 decodes)`, then the "all unused" column: 0.037 0.051 0.064 0.047 0.042 0.030 |
| calibration spread **0.03-0.05** dB | `s24_soft_auto.txt` header, COMMANDS 24.1 (2026-10-26) | `sigma 0.031 0.044 0.047 0.040 0.033 0.028 dB` |
| trusting the curves at a hard target costs **1.7-2.0** points (K 16-128); simulated, 100 frames per K | `s24_soft_auto.txt`, per-K blocks | `lagrange_soft - lagrange: +2.000 pts [+1.375, +2.688]` (K 16); `+1.656` (K 32, 64, 128); `+0.750` (K 8) |

**The model's error on scheduled deliveries, decoded.** Each `replay_report` block starts with a
line of the form `N decoded histories: predicted - actual PSNR bias ..., MAE ...`.

| condition | files (round) | MAE (dB) | bias (dB) |
|---|---|---|---|
| fresh reports, single decoder, Kodak | `s8_dec.txt` (8), `s21_replays.txt` (21.1), `s22_replays.txt` (22.4), `s24_replays.txt` (24.4) | 0.062-0.089 | -0.022 to +0.001 |
| fresh reports, CLIC | `s22_replays.txt`, `s24_replays.txt` | 0.076-0.087 | -0.029 to -0.000 |
| fresh reports, per-length decoders / per-CBR codes | `s13_replays.txt`, `s13_hd_more.txt`, `s13_hf.txt` (17.4-17.5a) | 0.051-0.078 | -0.046 to -0.012 |
| aged reports (r = 0.8), single decoder | `s21`, `s22` and `s24` replays | 0.076-0.095 | -0.012 to +0.013 |
| aged reports (r = 0.8 / 0.6), per-length decoders | `s20_replays.txt` (20.1) | 0.082-0.121 | -0.045 to -0.015 |

The constant-SNR control's replays (MAE 0.195-0.221, bias +0.10 to +0.12) are the code that
obeys no rule; they are not part of these ranges.

## C. CLASP's results (paragraphs 5 and 6)

| number | file, round (date) | line to look for | kind |
|---|---|---|---|
| mean PSNR over PF, Kodak: **+0.524 dB** | `s22_replays.txt`, first report (`s22_dec_mean.json`); 22.1 decoded, 22.4 reported (2026-10-25) | `pf  480  31.127`, `lagrange  480  31.651`; 31.651 - 31.127 = +0.524 | decoded, 60 frames |
| mean PSNR over PF, CLIC: **+0.631 dB** | `s22_replays.txt`, report of `s22_clic_dec_mean.json` | `pf  480  33.561`, `lagrange  480  34.192` | decoded, 60 frames |
| 30 dB share over PADC, Kodak: **+14.8** points | `s24_replays.txt`, report of `s24_dec_auto.json`; 24.3 decoded, 24.4 reported (2026-10-26) | `lagrange_soft  +14.792 pts [+11.667, +17.917]  (60 frames)` | decoded |
| 30 dB share over PADC, CLIC: **+8.1** points | `s24_replays.txt`, report of `s24_dec_clic.json` | `lagrange_soft  +8.125 pts [+5.833, +10.625]` | decoded |
| calibration-aware over hard: **+1.7** points; users missed **0.0% vs 1.2%** | `s24_replays.txt`; `s24_dec_auto.txt` | `lagrange_soft  +1.667 pts [+0.417, +2.917]`; `lagrange ... missed  1.2%`; `lagrange_soft ... missed  0.0%` | decoded |
| gap to the bound, mean PSNR: **0.207 → 0.005** dB per user, K 8 → 128 (B = K/8) | `s22_scale_mean.txt`, summary table; COMMANDS 22.2e (2026-10-25) | the `lagrange` column: 0.207 0.090 0.039 0.009 0.005 | simulated, 100 frames per K |
| gap to the bound, 30 dB: within **±0.3** points from K 16 | `s24_soft_auto.txt`, summary table; 24.1 | the `lagrange_soft` column: 0.841 -0.103 -0.296 0.011 -0.194 | simulated, 100 frames per K |

The decoded "over PF" numbers are differences of the two policies' decoded means on the same
480 users. `replay_report` printed their intervals against `slope_adv` only. `table03_main`
(COMMANDS 28) prints the gain over PF directly, with its interval.

## D. Against digital (paragraphs 2, 5 and 6)

All of these are simulated, over 2000 frames. JSCC is the single decoder: `s23_c5_single.txt`,
COMMANDS 23.2 (2026-10-25). Digital is HEIC with CQI-table link adaptation and IR-HARQ:
`s20_harq_mean.txt`, `s20_harq_tgt.txt`, `s20_harq_tgt_aware.txt` and `s20_harq_tgt_cqi.txt`,
COMMANDS 20.2 (2026-10-20). Each side's best policy and transmitter at each point:
- JSCC: `slope_adv` for the mean, `plan_adv` for the target;
- digital: `greedy` for the mean, `online_pf` for the target.

| number | JSCC line | digital line | arithmetic |
|---|---|---|---|
| fresh reports: **+0.13 dB, +0.3** points | `slope_adv  31.424`; `plan_adv  74.7%` | `greedy  31.291`; `online_pf  74.4%` (`s20_harq_tgt.txt`, first block) | +0.133; +0.3 |
| 2 dB CQI error: **+1.13 dB, +10.7** points | `CQI error 2 dB` blocks: `slope_adv  31.274`; `plan_adv  73.4%` | `CQI error 2 dB`: `greedy  30.146`; `online_pf  62.7%` (`s20_harq_tgt_cqi.txt`) | +1.128; +10.7 |
| r = 0.8: **+1.54 dB, +13.5** points | `correlation 0.8; transmitter aware`: `slope_adv  30.902`; `plan_adv  68.3%` | `correlation 0.8; transmitter naive`: `greedy  29.359` (aware 29.268); `online_pf  54.8%` (aware 54.3%) | +1.543; +13.5 |
| fresh → r = 0.8: digital **-1.9 dB**, the learned code **-0.5 dB** | 31.424 → 30.902 | 31.291 → 29.359 | -0.522; -1.932 |

## E. Scheduler-aware training (paragraphs 5 and 6)

All decoded on the same slots, under `slope_adv`'s schedules, over 150 frames.

| number | file, round (date) | line to look for |
|---|---|---|
| **+0.28 dB** (the code trained on the scheduler's channel, against the same-budget control) | `s14_same_slots.txt`, COMMANDS 14.3a (2026-10-14) | `sa:slope_adv  +0.278 dB [+0.273, +0.284]  (150 frames)` (31.885 against `hf:slope_adv` 31.607) |
| second seed: +0.24 | `s16_seeds.txt`, COMMANDS 16.4 (2026-10-14) | `sab:slope_adv  1200  31.846` |
| beyond training for one length: **+0.08-0.12** | `s16_seeds.txt` (against `fs16`) | `sa:slope_adv  +0.118 dB [+0.110, +0.124]`; `sab:slope_adv  +0.079 dB [+0.071, +0.086]` |

## To check on the server

```
cd ~/projects/JSCCwClaude-main/jscc
grep -n "MMSE, kappa\|exponential (EESM)\|capacity (MIESM-like)\|noise variance" results/s13_abstractions.txt results/s22_clic.txt
grep -n "12 decodes" -A 8 results/s8_calib.txt
grep -n "decoded histories" results/s13_replays.txt results/s20_replays.txt results/s21_replays.txt results/s22_replays.txt results/s24_replays.txt
head -n 8 results/s22_replays.txt
grep -n "lagrange_soft" results/s24_replays.txt; grep -n "missed" results/s24_dec_auto.txt
grep -n "sigma\|lagrange_soft - lagrange" results/s24_soft_auto.txt
grep -A 7 "^summary" results/s22_scale_mean.txt results/s24_soft_auto.txt
grep -n "slope_adv \|plan_adv \|correlation 0.8\|CQI error 2 dB" results/s23_c5_single.txt
grep -n "greedy \|correlation 0.8\|CQI error 2 dB" results/s20_harq_mean.txt
grep -n "online_pf \|correlation 0.8\|CQI error 2 dB" results/s20_harq_tgt.txt results/s20_harq_tgt_aware.txt results/s20_harq_tgt_cqi.txt
grep -n "slope_adv" results/s14_same_slots.txt results/s16_seeds.txt
```
