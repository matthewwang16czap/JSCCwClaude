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
3. **The digital comparison was not CLASP's.** Its JSCC side used the best policy at each
   point: the closed form `slope_adv` for the mean, the planner `plan_adv` for the 30 dB share.
   Round 30 ran CLASP itself on the same frames, and the outline now quotes CLASP (section D).
4. **Scheduler-aware training (+0.28 dB) was measured under the closed-form scheduler**, not
   CLASP's index. Round 30 measured it under CLASP: +0.21 dB (section E).
5. **The model error on scheduled deliveries (0.05-0.09 dB) holds with fresh reports**, and
   (round 30) with 2 dB CQI errors. With aged reports it is 0.076-0.095 dB for the single
   decoder and 0.082-0.121 dB for per-length decoders.

## Round 30: what it replaced (COMMANDS 30, returned 2026-10-31)

| number in the outline | before round 30 | now | file |
|---|---|---|---|
| decoded gain over PF | +0.524 (Kodak) / +0.631 dB (CLIC), differences of two decoded means | the same, with their own intervals; also +0.514 at r = 0.8 and +0.512 at 2 dB CQI error | `s30_direct.txt`, `s30_dec_e2.txt` |
| decoded gain over PADC at 30 dB | +14.8 (Kodak) / +8.1 points (CLIC) | the same; also +8.1 points at 2 dB CQI error | `s30_direct.txt`, `s30_dec_e2.txt` |
| against digital | +1.1-1.5 dB, +10.7-13.5 points (the best JSCC policy) | CLASP: +1.19 / +1.56 dB and +8.1 / +12.0 points at 2 dB CQI error / r = 0.8; +0.20 dB and +3.6 points with exact reports | `s30_vs_digital.txt` |
| fresh → r = 0.8 | digital -1.9 dB, the closed form -0.5 dB (two means each) | digital -1.93 dB, CLASP -0.57 dB, paired | `s30_loss.txt` |
| scheduler-aware training | +0.28 dB under the closed form | +0.21 dB under CLASP, two seeds; CLASP's lengths alone -0.09 dB | `s30_codesign.txt` |
| model error on deliveries | fresh reports: 0.05-0.09 dB | also at 2 dB CQI error: 0.068-0.070 dB | `s30_dec_e2.txt` |

The check passed: 30.5 (2) printed `slope_adv` 31.424 / 31.274 / 30.902 and `greedy` 31.291 /
30.146 / 29.359, so round 30's frames are those of 23.2 and 20.2.

## A. The codec (paragraph 4)

| fact | source |
|---|---|
| ViT, patch 8, width 768, 10 blocks, 4 complex symbols per token; six phases of CBR 1/48 at 256 px (1/8 in all) | checkpoint `history/vit_p8d768x10s4_awgn_DIV2K_256_snrc4_s1`, NOTES 1 |
| trained with the chunks of each codeword at independent SNRs | COMMANDS 5c (E2b, log `s1_gpu5_E2b_snrc4.out`): from w1, 400 epochs, DIV2K, `--snr-chunks 4`, `--snr-range -2 22` |
| the constant-SNR control (s1) | COMMANDS 5c (E2a, log `s1_gpu5_E2a_const.out`): the same recipe without `--snr-chunks` |
| the encoder and decoder never see the SNR | `net/network.py`: `encode(self, x)` and `decode(self, enc, rx, units)` take no SNR; only the channel model does |
| held-out profiles | COMMANDS 6b, `results/p2_full_snrc4.json` / `p2_full_s1.json`: 10 random mixed-SNR profiles per CBR per image |

## B. The link abstraction (paragraphs 4, 5 and 6)

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
| **12 decodes** per image; the calibrated curves predict held-out mixed-SNR decodes within **0.03-0.06** dB (MAE per prefix length; these files print MAE / p95 only, no bias) | `s13_nested.txt`, COMMANDS 17.3c (2026-10-14): the nested curves every run since round 17 schedules on (`belief_snrc4_nested.json`) | the block after `transmitter belief: ... (12 decodes)` for the 24-image snrc4 fit, column "all unused": 0.037 0.050 0.063 0.047 0.042 0.033 |
| the same on CLIC: **0.05-0.09** dB | `s22_clic.txt`, COMMANDS 22.3 (2026-10-25) | "all unused": 0.062 0.075 0.091 0.064 0.060 0.049 |
| (superseded) the free-model curves of rounds 8-16: 0.030-0.064 dB | `s8_calib.txt`, COMMANDS 8a (2026-10-11), wrote `belief_snrc4.json` | "all unused": 0.037 0.051 0.064 0.047 0.042 0.030 |
| paragraph 5: the spread of the calibrated curves around the fitted ones, **0.03-0.05** dB per prefix length (the error the calibration-aware reward assumes; a different comparison from the two above) | `s24_soft_auto.txt` header, COMMANDS 24.1 (2026-10-26) | `sigma 0.031 0.044 0.047 0.040 0.033 0.028 dB, bias (truth - belief) +0.021 +0.012 -0.009 -0.008 +0.000 +0.010 dB` |
| trusting the curves at a hard target costs **1.7-2.0** points (K 16-128); simulated, 100 frames per K | `s24_soft_auto.txt`, per-K blocks | `lagrange_soft - lagrange: +2.000 pts [+1.375, +2.688]` (K 16); `+1.656` (K 32, 64, 128); `+0.750` (K 8) |

**The model's error on scheduled deliveries, decoded.** The outline's "0.05-0.09 dB, bias within
±0.05 dB" comes from these replay files, not from the calibration files above. Each
`replay_report` block starts with a line of the form `N decoded histories: predicted -
actual PSNR bias ..., MAE ...`.

| condition | files (round) | MAE (dB) | bias (dB) |
|---|---|---|---|
| fresh reports, single decoder, Kodak | `s8_dec.txt` (8), `s21_replays.txt` (21.1), `s22_replays.txt` (22.4), `s24_replays.txt` (24.4) | 0.062-0.089 | -0.022 to +0.001 |
| fresh reports, CLIC | `s22_replays.txt`, `s24_replays.txt` | 0.076-0.087 | -0.029 to -0.000 |
| fresh reports, per-length decoders / per-CBR codes | `s13_replays.txt`, `s13_hd_more.txt`, `s13_hf.txt` (17.4-17.5a) | 0.051-0.078 | -0.046 to -0.012 |
| 2 dB CQI error, single decoder, Kodak (CLASP's schedules) | `s30_dec_e2.txt` (30.3, 2026-10-31) | 0.068-0.070 | -0.009 to +0.003 |
| aged reports (r = 0.8), single decoder | `s21`, `s22` and `s24` replays | 0.076-0.095 | -0.012 to +0.013 |
| aged reports (r = 0.8 / 0.6), per-length decoders | `s20_replays.txt` (20.1) | 0.082-0.121 | -0.045 to -0.015 |

The constant-SNR control's replays (MAE 0.195-0.221, bias +0.10 to +0.12) are the code that
obeys no rule; they are not part of these ranges.

## C. CLASP's results (paragraphs 5 and 6)

Decoded rows: `replay_report` against the baseline CLASP replaces, paired by frame, 60 frames.
`s30_direct.txt` (COMMANDS 30.1, 2026-10-31) re-reports the decodes of 22.1 and 24.3; the 2 dB
CQI-error rows are new decodes (30.3).

| number | file, round (date) | line to look for | kind |
|---|---|---|---|
| mean PSNR over PF, Kodak: **+0.524 dB** [0.464, 0.585] | `s30_direct.txt`, report 1 of 5 (`s22_dec_mean.json`) | `lagrange  +0.524 dB [+0.464, +0.585]  (60 frames)` (pf 31.127, lagrange 31.651) | decoded |
| mean PSNR over PF, CLIC: **+0.631 dB** [0.558, 0.703] | `s30_direct.txt`, report 2 (`s22_clic_dec_mean.json`) | `lagrange  +0.631 dB [+0.558, +0.703]  (60 frames)` | decoded |
| mean PSNR over PF, r = 0.8: +0.514 dB [0.406, 0.625] | `s30_direct.txt`, report 3 (`s22_dec_r80_mean.json`) | `lagrange  +0.514 dB [+0.406, +0.625]  (60 frames)` | decoded |
| mean PSNR over PF, 2 dB CQI error: +0.512 dB [0.445, 0.583] | `s30_dec_e2.txt`, report 1 (`s30_dec_e2_mean.json`), COMMANDS 30.3 | `lagrange  +0.512 dB [+0.445, +0.583]  (60 frames)` | decoded |
| 30 dB over PADC, Kodak: **+14.8** points [11.7, 17.9] | `s30_direct.txt`, report 4 (`s24_dec_auto.json`) | `lagrange_soft  +14.792 pts [+11.667, +17.917]  (60 frames)` | decoded |
| 30 dB over PADC, CLIC: **+8.1** points [5.8, 10.6] | `s30_direct.txt`, report 5 (`s24_dec_clic.json`) | `lagrange_soft  +8.125 pts [+5.833, +10.625]  (60 frames)` | decoded |
| 30 dB over PADC, 2 dB CQI error: **+8.1** points [5.2, 10.8] | `s30_dec_e2.txt`, report 2 (`s30_dec_e2_tgt.json`) | `lagrange_soft  +8.125 pts [+5.208, +10.833]  (60 frames)` | decoded |
| calibration-aware over hard: **+1.7** points; users missed **0.0% vs 1.2%** | `s24_replays.txt`; `s24_dec_auto.txt`; COMMANDS 24.3-24.4 (2026-10-26) | `lagrange_soft  +1.667 pts [+0.417, +2.917]`; `lagrange ... missed  1.2%`; `lagrange_soft ... missed  0.0%` | decoded |
| gap to the bound, mean PSNR: **0.207 → 0.005** dB per user, K 8 → 128 (B = K/8) | `s22_scale_mean.txt`, summary table; COMMANDS 22.2e (2026-10-25) | the `lagrange` column: 0.207 0.090 0.039 0.009 0.005 | simulated, 100 frames per K |
| gap to the bound, 30 dB: within **±0.3** points from K 16 | `s24_soft_auto.txt`, summary table; 24.1 | the `lagrange_soft` column: 0.841 -0.103 -0.296 0.011 -0.194 | simulated, 100 frames per K |

The outline states the 8-15 points for fresh reports and 2 dB CQI errors. With aged reports
(r = 0.8) the decoded gain over PADC is +4.4 points [0.8, 8.1] (`s24_dec_r80.json`, COMMANDS
24.3; printed by `table03_main` in `s30_figs.txt`, row "reports r = 0.8").

## D. Against digital (paragraphs 2, 5 and 6)

All simulated, over the same 2000 frames, paired by frame (`tools/paired_runs.py`). Files:
`s30_vs_digital.txt` (one JSON line per comparison in `s30_vs_digital.jsonl`) and
`s30_loss.txt`, COMMANDS 30.2 and 30.5 (2026-10-31).
- CLASP: `lagrange` for the mean, `lagrange_soft` for the 30 dB share; aware on aged reports,
  where the target run takes the better of no margin and a 0.5 dB margin (`s30_j_tgtm_*`). The
  margin is better at r <= 0.8. At r = 0.8 the quoted 66.8% is the margin run's
  `lagrange_soft  66.8%` in `s30_j_tgtm_r80.txt`, against `lagrange_soft  66.2%` in
  `s30_j_tgt_r80.txt` without it (every r: the grep in `s31_checks.txt`, COMMANDS 31.1).
- Digital: HEIC, the LTE CQI table, IR-HARQ, 6 slots; its best policy (`pf`, `greedy`; `padc`,
  `stop`, `online_pf`) and transmitter (naive or aware) at each point.

Each comparison prints a label line, the two sides (`a:` CLASP, `b:` digital), then `a - b:`.

| number | label line | line to look for |
|---|---|---|
| exact reports: **+0.20 dB** | `[f] mean PSNR: CLASP - digital` | `a - b: +0.196 dB [+0.174, +0.217]` (lagrange 31.487, greedy 31.291) |
| exact reports: **+3.6** points | `[f] 30 dB: CLASP - digital` | `a - b: +3.612 pts [+3.137, +4.081]` (78.0% against online_pf 74.4%) |
| 2 dB CQI error: **+1.19 dB** | `[e2] mean PSNR: CLASP - digital` | `a - b: +1.192 dB [+1.164, +1.221]` (31.339 against 30.146) |
| 2 dB CQI error: **+8.1** points | `[e2] 30 dB: CLASP - digital` | `a - b: +8.106 pts [+7.500, +8.694]` (70.8% against 62.7%) |
| r = 0.8: **+1.56 dB** | `[r80] mean PSNR: CLASP - digital` | `a - b: +1.561 dB [+1.530, +1.591]` (30.921 against the naive digital 29.359) |
| r = 0.8: **+12.0** points | `[r80] 30 dB: CLASP - digital` | `a - b: +11.969 pts [+11.350, +12.594]` (66.8%, 0.5 dB margin, against 54.8%) |
| ahead at every CSI condition | the ten `CLASP - digital` blocks of each objective | the smallest leads: +0.196 dB and +3.612 pts, both `[f]` |
| fresh → r = 0.8: CLASP **-0.6 dB** | `s30_loss.txt`: `CLASP, mean PSNR: fresh - r 0.8` | `a - b: +0.567 dB [+0.555, +0.579]` |
| fresh → r = 0.8: digital **-1.9 dB** | `s30_loss.txt`: `digital, mean PSNR: fresh - r 0.8 (its best of naive / aware)` | `a - b: +1.932 dB [+1.907, +1.957]` |

The outline does not use the `best JSCC - digital (as C5)` rows. They take the best JSCC policy
at each point, as C5 did. For the mean that is CLASP everywhere but r = 0.4 (the closed form, by
0.004 dB). For the 30 dB share it is the planner `plan_adv` from 2 dB of CQI error and from
r = 0.9, 1.1-4.6 points above CLASP (for example `[e2] 30 dB: best JSCC - digital (as C5)`:
`a - b: +10.650 pts [+10.056, +11.225]`). Section VII reports them (C7). The planner minus
CLASP, paired, is in `s31_checks.txt` (COMMANDS 31.1), under labels such as
`[e2] 30 dB: planner - CLASP`: `a - b: +2.544 pts [+2.069, +2.987]`.

Superseded: the old rows took the best JSCC policy from `s23_c5_single.txt` (COMMANDS 23.2):
+0.13 dB / +0.3 points with fresh reports, +1.13 dB / +10.7 points at 2 dB CQI error, +1.54 dB
/ +13.5 points at r = 0.8, and fresh → r = 0.8 -0.52 dB for the closed form.

## E. Scheduler-aware training (paragraphs 5 and 6)

All decoded on the same slots: the 150 frames of section 14 (`s30_s0_mean`, 30.3), scheduled
once on the snrc4 belief and decoded by every code (snrc4 + 400 epochs each). File
`s30_codesign.txt`, COMMANDS 30.4-30.5 (2026-10-31): report 1 is against `hf:lagrange`,
report 2 against `lagblind:lagrange`. Tags: `hf`, the same-budget control (the usual chunks);
`lagblind`, CLASP's lengths with every slot taken blindly; `lag`, CLASP's channel; `lagb`, the
same with seed 2; `sa`, the closed form's channel (section 14).

| number | line to look for |
|---|---|
| **+0.21 dB** (CLASP's channel against the same-budget control, under CLASP's schedules) | report 1: `lag:lagrange  +0.211 dB [+0.204, +0.219]  (150 frames)` (31.875 against `hf:lagrange` 31.663) |
| second seed: +0.21 | report 1: `lagb:lagrange  +0.206 dB [+0.199, +0.214]  (150 frames)` |
| CLASP's lengths with blind slots gain nothing: **-0.09** | report 1: `lagblind:lagrange  -0.089 dB [-0.095, -0.082]  (150 frames)` |
| the slot selection is worth +0.30 | report 2: `lag:lagrange  +0.300 dB [+0.292, +0.308]  (150 frames)` |

Superseded in the outline: +0.28 dB under the closed form (`s14_same_slots.txt`, COMMANDS 14.3a:
`sa:slope_adv  +0.278 dB [+0.273, +0.284]`). It reproduces on round 30's frames: `sa:slope_adv`
31.887 against `hf:slope_adv` 31.609 in either report's table.

## To check on the server

```
cd ~/projects/JSCCwClaude-main/jscc
grep -n "MMSE, kappa\|exponential (EESM)\|capacity (MIESM-like)\|noise variance" results/s13_abstractions.txt results/s22_clic.txt
grep -n "12 decodes" -A 8 results/s13_nested.txt results/s22_clic.txt
grep -n "decoded histories" results/s13_replays.txt results/s20_replays.txt results/s21_replays.txt results/s22_replays.txt results/s24_replays.txt results/s30_dec_e2.txt
grep -n "decoded histories\|lagrange \|lagrange_soft " results/s30_direct.txt results/s30_dec_e2.txt
grep -n "lagrange_soft" results/s24_replays.txt; grep -n "missed" results/s24_dec_auto.txt
grep -n "sigma\|lagrange_soft - lagrange" results/s24_soft_auto.txt
grep -A 7 "^summary" results/s22_scale_mean.txt results/s24_soft_auto.txt
grep -A 3 "CLASP - digital" results/s30_vs_digital.txt
cat results/s30_loss.txt
grep -n "lag:lagrange \|lagb:lagrange \|lagblind:lagrange \|sa:slope_adv \|hf:slope_adv " results/s30_codesign.txt
```
