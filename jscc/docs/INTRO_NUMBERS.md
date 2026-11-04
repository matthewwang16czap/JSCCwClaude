# Where every number in the Introduction outline comes from

One entry per number, in the order the numbers appear in `docs/INTRO_OUTLINE.md`. Each entry
gives the number as the outline writes it and where; the lines the run printed, copied from the
outputs you pasted back into the session (with the date you sent them); the file on the server
(`~/projects/JSCCwClaude-main/jscc/results/`) and the COMMANDS.txt step that wrote it; and how
the outline's number follows from the printed one. A number that appears again points to its
first entry. Entries 1-3, the aged-report setting and what aged reports cost, come first because
entries 19 and 53 in paragraphs 4 and 6 point to them. Settings (a report correlation, a CQI error, the 30 dB target, user counts) are
listed where they first appear.

- **Simulated**: schedules scored on the fitted quality curves (`tools/schedule_sim.py`,
  `tools/rmab_scale.py`), paired by frame where two runs are compared.
- **Decoded**: the codec itself decoded the schedules (`tools/replay_report.py`).
- **Shares of users**: a share is the percentage of users whose decoded PSNR is at or above the
  target at the deadline. The outline gives a difference between two shares in % of all users,
  the difference of the two percentages. The tools print that difference with the unit `pts`
  (percentage points). Example, entry 38: CLASP 78.0% of the users at 30 dB, the 3 dB-gap
  code 80.4%; printed difference -2.394 pts, i.e. the gap code brings an extra 2.4% of the users
  to 30 dB.
- **Setup**, unless an entry says otherwise: Kodak (24 images), the snrc4 code, K = 8 users,
  T = 24 slots, mean SNRs U[0, 15] dB, i.i.d. Rayleigh block fading.

The commands that print every quoted line on the server are at the end, in the same order.

## Aged reports: the setting and its cost (quoted in paragraphs 4 and 6)

**1. "correlation 0.8"** (aged reports): a setting. The aged-report condition is report
correlation r = 0.8 (`--cqi-corr 0.8`): CLASP with the aware transmitter (`--cqi-aware`), the
digital chain with its better transmitter (here the naive one). The labels of entries 2 and 3
print it (`results/s30_loss.txt`, COMMANDS 30.5 (4), sent back 2026-10-31):
```
CLASP, mean PSNR: fresh - r 0.8  (mean PSNR (dB), 2000 frames, paired by frame)
  b: lagrange        30.921 dB   learned code, r 0.8 aware   [s30_j_mean_r80.json]
digital, mean PSNR: fresh - r 0.8 (its best of naive / aware)  (mean PSNR (dB), 2000 frames, paired by frame)
  b: greedy          29.359 dB   digital cqi + HARQ ir, r 0.8 naive   [s30_d_mean_n80.json] (best of pf, greedy, pf, greedy)
```

**2. "1.9 dB"** (what aged reports cost the digital chain). Simulated, 2000 frames, paired
by frame. `results/s30_loss.txt`, COMMANDS 30.5 (4), sent back 2026-10-31:
```
digital, mean PSNR: fresh - r 0.8 (its best of naive / aware)  (mean PSNR (dB), 2000 frames, paired by frame)
  a: greedy          31.291 dB   digital cqi + HARQ ir, fresh, exact reports   [s30_d_mean_f.json] (best of pf, greedy)
  b: greedy          29.359 dB   digital cqi + HARQ ir, r 0.8 naive   [s30_d_mean_n80.json] (best of pf, greedy, pf, greedy)
  a - b: +1.932 dB [+1.907, +1.957]
```
1.932 dB, rounded to 1.9.

**3. "0.6 dB"** (what they cost a learned code under CLASP). Same file and frames:
```
CLASP, mean PSNR: fresh - r 0.8  (mean PSNR (dB), 2000 frames, paired by frame)
  a: lagrange        31.487 dB   learned code, fresh, exact reports   [s30_j_mean_f.json]
  b: lagrange        30.921 dB   learned code, r 0.8 aware   [s30_j_mean_r80.json]
  a - b: +0.567 dB [+0.555, +0.579]
```
0.567 dB, rounded to 0.6.

## Paragraph 4

**4. "six phases"** (item 1): the code's six prefix lengths, CBR 1/48 to 1/8, one phase of
CBR 1/48 each. `results/s13_abstractions.txt`, block 1 (the snrc4 code), COMMANDS 17.4a, sent
back 2026-10-14; the table's rows run from 1/48 (1 chunk) to 1/8 (6 chunks):
```
24 images, 6 levels; noise response phi(n) = n / (1 + n / 1.769); fit rms 0.037 dB
  1/48      1  1.000                                              0.022/0.069                 -                   -
   1/8      6  0.509 0.254 0.169 0.052 0.010 0.006                0.015/0.051       0.044/0.122         0.470/1.136
```

**5. "0.055 dB (0.37)"** (item 2, table, Kodak, MMSE rule). Decoded held-out profiles,
each cell mean absolute error / p95 (dB), equal weights first, sensitivity weights second.
`results/s13_abstractions.txt`, block 1, COMMANDS 17.4a, sent back 2026-10-14:
```
24 images, 6 levels; noise response phi(n) = n / (1 + n / 1.769); fit rms 0.037 dB
  averaging rule                 equal weights   sensitivity weights
  MMSE, kappa 1.77               0.367 / 0.951         0.055 / 0.187
```
0.055 (sensitivity weights) as printed; 0.367 (equal weights), rounded to 0.37.

**6. "0.19 dB (0.37)"** (Kodak, EESM). Same block:
```
  exponential (EESM)             0.373 / 1.144         0.194 / 0.608
```
0.194, rounded to 0.19; 0.373, rounded to 0.37.

**7. "0.34 dB (0.49)"** (Kodak, MIESM). Same block:
```
  capacity (MIESM-like)          0.487 / 1.480         0.337 / 0.896
```
0.337, rounded to 0.34; 0.487, rounded to 0.49.

**8. "0.083 dB (0.41)"** (CLIC, MMSE). CLIC (60 images). `results/s22_clic.txt`, its
abstraction block, COMMANDS 22.3 (printed by 22.4, sent back 2026-10-25):
```
60 images, 6 levels; noise response phi(n) = n / (1 + n / 1.934); fit rms 0.058 dB
  averaging rule                 equal weights   sensitivity weights
  MMSE, kappa 1.93               0.407 / 1.067         0.083 / 0.275
```
0.083 as printed; 0.407, rounded to 0.41.

**9. "0.16 dB (0.37)"** (CLIC, EESM). Same block:
```
  exponential (EESM)             0.374 / 1.152         0.159 / 0.533
```
0.159, rounded to 0.16; 0.374, rounded to 0.37.

**10. "0.29 dB (0.44)"** (CLIC, MIESM). Same block:
```
  capacity (MIESM-like)          0.439 / 1.546         0.290 / 0.842
```
0.290, rounded to 0.29; 0.439, rounded to 0.44.

**11. "0.22 dB"** (item 3: the MMSE rule on the code trained at one constant SNR per
image). `results/s13_abstractions.txt`, block 2 (the s1 code), COMMANDS 17.4a, sent back 2026-10-14:
```
24 images, 6 levels; noise response phi(n) = n / (1 + n / 2.441); fit rms 0.092 dB
  MMSE, kappa 2.44               0.408 / 1.101         0.222 / 0.674
```
0.222 (sensitivity weights), rounded to 0.22.

**12. "0.15 dB"** (item 3: no rule does better). The five rules of the same block 2:
```
  mean dB                        0.652 / 1.740         0.541 / 1.438
  noise variance                 0.382 / 1.017         0.150 / 0.487
  capacity (MIESM-like)          0.721 / 1.905         0.614 / 1.600
  exponential (EESM)             0.426 / 1.228         0.214 / 0.507
  MMSE, kappa 2.44               0.408 / 1.101         0.222 / 0.674
```
The smallest mean absolute error in the block, under either weighting, is the noise-variance
rule's 0.150.

**13. "Twelve decodes per image"** (item 4): each image's own curves are calibrated from its
constant decodes at 2 SNRs (1 and 13 dB) x 6 prefix lengths. `results/s13_nested.txt`, the
snrc4 block (the one followed by `wrote results/belief_snrc4_nested.json`), COMMANDS 17.3c, sent
back 2026-10-14:
```
transmitter belief: kappa and rho cross-fitted over 2 folds of images, each image's Dsrc and C from its constant decodes at [1.0, 13.0] dB (12 decodes); errors (belief - actual PSNR, MAE/p95) on the profiles it did not use
wrote results/belief_snrc4_nested.json
```

**14. "0.03-0.06 dB per prefix length"** (item 4: the calibrated curves on held-out
mixed-SNR decodes). The table under the line of entry 13, column "all unused" (mean absolute
error / p95, dB), one row per prefix length:
```
   CBR  other constants  one at a time       random   all unused  full fit, held-out
  1/48      0.037/0.131              -            -  0.037/0.131                   -
  1/24      0.039/0.087    0.050/0.122  0.059/0.207  0.050/0.143         0.048/0.187
  1/16      0.047/0.096    0.058/0.154  0.078/0.204  0.063/0.175         0.069/0.193
  1/12      0.035/0.070    0.048/0.124  0.050/0.142  0.047/0.127         0.034/0.095
  5/48      0.024/0.052    0.032/0.094  0.066/0.251  0.042/0.133         0.054/0.213
   1/8      0.021/0.054    0.025/0.077  0.055/0.154  0.033/0.102         0.051/0.138
```
0.037, 0.050, 0.063, 0.047, 0.042, 0.033: from 0.033 to 0.063, i.e. 0.03-0.06.

**15. "0.05-0.09 dB on CLIC"** (item 4). `results/s22_clic.txt`, the CLIC block (followed by
`wrote results/belief_clic_nested.json`), COMMANDS 22.3, sent back 2026-10-25:
```
transmitter belief: kappa and rho cross-fitted over 2 folds of images, each image's Dsrc and C from its constant decodes at [1.0, 13.0] dB (12 decodes); errors (belief - actual PSNR, MAE/p95) on the profiles it did not use
   CBR  other constants  one at a time       random   all unused  full fit, held-out
  1/48      0.062/0.200              -            -  0.062/0.200                   -
  1/24      0.048/0.110    0.078/0.197  0.087/0.303  0.075/0.223         0.073/0.233
  1/16      0.055/0.113    0.081/0.211  0.119/0.315  0.091/0.253         0.109/0.303
  1/12      0.043/0.092    0.065/0.166  0.069/0.189  0.064/0.169         0.052/0.145
  5/48      0.031/0.067    0.048/0.137  0.092/0.318  0.060/0.189         0.081/0.304
   1/8      0.025/0.059    0.038/0.109  0.083/0.197  0.049/0.149         0.073/0.177
wrote results/belief_clic_nested.json
```
"all unused": 0.062, 0.075, 0.091, 0.064, 0.060, 0.049: from 0.049 to 0.091, i.e. 0.05-0.09.

**16. "0.06-0.09 dB"** and **17. "±0.03 dB"** (item 4: the model's mean absolute error
and bias on complete scheduled deliveries, decoded). Every schedule decoded for the
Introduction's results, all policies in each file. Each `replay_report.py` report begins with
one line `N decoded histories: predicted - actual PSNR bias ..., MAE ..., p95 ... dB`.

`results/s30_direct.txt` (COMMANDS 30.1, printed by 30.5 (1), sent back 2026-10-31), its five
reports in order: Kodak fresh, mean PSNR (`s22_dec_mean.json`); CLIC fresh, mean
(`s22_clic_dec_mean.json`); Kodak r = 0.8, mean (`s22_dec_r80_mean.json`); Kodak fresh, 30 dB
(`s24_dec_auto.json`); CLIC fresh, 30 dB (`s24_dec_clic.json`):
```
2872 decoded histories: predicted - actual PSNR bias -0.022, MAE 0.062, p95 0.171 dB
2388 decoded histories: predicted - actual PSNR bias -0.029, MAE 0.076, p95 0.207 dB
2875 decoded histories: predicted - actual PSNR bias -0.012, MAE 0.076, p95 0.214 dB
2870 decoded histories: predicted - actual PSNR bias -0.006, MAE 0.070, p95 0.213 dB
1920 decoded histories: predicted - actual PSNR bias -0.005, MAE 0.083, p95 0.256 dB
```

`results/s30_dec_e2.txt` (COMMANDS 30.5 (5), sent back 2026-10-31): Kodak, 2 dB CQI error, mean
(`s30_dec_e2_mean.json`), then 30 dB (`s30_dec_e2_tgt.json`):
```
1440 decoded histories: predicted - actual PSNR bias -0.009, MAE 0.068, p95 0.183 dB
1415 decoded histories: predicted - actual PSNR bias +0.003, MAE 0.070, p95 0.212 dB
```

`results/s33_dec.txt` (COMMANDS 33.3 (4), sent back 2026-11-01): r = 0.8, 30 dB (`s24_dec_r80.json`
and `s33_dec_r80.json`, pooled), then 2 dB CQI error, 30 dB (`s30_dec_e2_tgt.json` and
`s33_dec_e2.json`, pooled); each line is printed twice, once per base:
```
2701 decoded histories: predicted - actual PSNR bias +0.009, MAE 0.092, p95 0.312 dB
2337 decoded histories: predicted - actual PSNR bias +0.002, MAE 0.072, p95 0.226 dB
```
MAE from 0.062 to 0.092, i.e. 0.06-0.09; bias from -0.029 to +0.009, i.e. within ±0.03.

**18. "2 dB CQI errors"** (item 4): a setting, a Gaussian error of standard deviation 2 dB on
each reported slot SNR (`--cqi-error 2`; the decoded 2 dB schedules of entry 16 were drawn
with it by COMMANDS 30.3 and 33.1).

**19. "correlation 0.8"** (item 4): as entry 1.

**20. "an extra 1.7-2.0% of the users"** (item 4: the users a scheduler that takes the
calibrated curves as exact leaves below 30 dB). Simulated, 100 frames per K, B = K/8 units per slot. The calibration-aware
index minus the hard-target one, paired, at each cell size. `results/s24_soft_auto.txt`,
COMMANDS 24.1 (printed by 24.4, sent back 2026-10-26):
```
K 16 users, B 2 units per slot, 100 frames (636 s; 49.2 s per frame and worker)
  lagrange_soft - lagrange: +2.000 pts [+1.375, +2.688]
K 32 users, B 4 units per slot, 100 frames (1141 s; 87.1 s per frame and worker)
  lagrange_soft - lagrange: +1.656 pts [+1.250, +2.156]
K 64 users, B 8 units per slot, 100 frames (2246 s; 174.1 s per frame and worker)
  lagrange_soft - lagrange: +1.656 pts [+1.359, +1.969]
K 128 users, B 16 units per slot, 100 frames (3452 s; 267.8 s per frame and worker)
  lagrange_soft - lagrange: +1.656 pts [+1.445, +1.891]
```
+2.000 at K 16 and +1.656 at K 32, 64 and 128: 1.7-2.0. At K 8, which the claim leaves out:
```
K 8 users, B 1 units per slot, 100 frames (278 s; 21.3 s per frame and worker)
  lagrange_soft - lagrange: +0.750 pts [+0.000, +1.625]
```

**21. "30 dB"** (item 4): the quality target, a setting (`--target 30`). The header of the
same file:
```
24 slots, 8 users per unit, mean SNR U[0.0, 15.0] dB, Rayleigh; share at 30 dB; transmitter curves results/belief_snrc4_nested.json; utility results/utility_snrc4_nested.json
```

**22. "from 16 to 128 users"** (item 4): the cell sizes K = 16, 32, 64, 128 of entry 20.

**23. "two numbers"** (item 5): not a measurement. Under the link abstraction a user's
predicted distortion depends on its past slots only through the phases received and their
accumulated weighted noise Y = sum_j rho_j phi(n_j): the state of each user's program
(`alloc/whittle.py`).

## Paragraph 5

**24. "12 decodes"** (item 3, challenge (1)): as entry 13.

**25. "a two-number state"** (item 3, challenge (1)): as entry 23.

**26. "0.03-0.05 dB per prefix length"** (item 3, challenge (3): the spread with exact
reports, the calibrated curves' error alone). `results/s24_soft_auto.txt`, header, COMMANDS
24.1, sent back 2026-10-26:
```
calibration-aware programs (*_soft): the curves' error per prefix length, sigma 0.031 0.044 0.047 0.040 0.033 0.028 dB, bias (truth - belief) +0.021 +0.012 -0.009 -0.008 +0.000 +0.010 dB (measured)
```
The same line with exact reports in `results/s33_csi_f.txt` (COMMANDS 33.1, printed by
33.3 (1), sent back 2026-11-01):
```
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves' only: exact reports), sigma 0.031 0.044 0.047 0.040 0.033 0.028 dB, bias (truth - belief) +0.021 +0.012 -0.009 -0.008 +0.000 +0.010 dB (measured)
```
From 0.028 to 0.047, i.e. 0.03-0.05.

**27. "0.18-0.97 dB"** (item 3: the spread with noisy or aged reports). The first line of
each `results/s33_csi_*.txt` (COMMANDS 33.1), printed by `grep -h "calibration-aware"
results/s33_csi_*.txt` (33.3 (1), sent back 2026-11-01); in file order e1, e2, e3, f (exact reports,
entry 26), r0, r40, r60, r80, r90, r95:
```
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves', and the reports' at CQI error 1 dB, correlation 1, naive), sigma 0.242 0.224 0.239 0.205 0.188 0.175 dB, bias (truth - belief) +0.018 -0.003 -0.032 -0.038 -0.014 -0.028 dB (measured)
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves', and the reports' at CQI error 2 dB, correlation 1, naive), sigma 0.472 0.429 0.448 0.386 0.362 0.343 dB, bias (truth - belief) +0.022 -0.010 -0.048 -0.058 -0.014 -0.050 dB (measured)
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves', and the reports' at CQI error 3 dB, correlation 1, naive), sigma 0.679 0.625 0.634 0.561 0.526 0.501 dB, bias (truth - belief) +0.030 -0.007 -0.053 -0.066 -0.000 -0.055 dB (measured)
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves' only: exact reports), sigma 0.031 0.044 0.047 0.040 0.033 0.028 dB, bias (truth - belief) +0.021 +0.012 -0.009 -0.008 +0.000 +0.010 dB (measured)
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves', and the reports' at CQI error 0 dB, correlation 0, aware), sigma 0.964 0.952 0.955 0.955 0.918 0.755 dB, bias (truth - belief) +0.135 +0.108 +0.074 +0.069 +0.091 +0.089 dB (measured)
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves', and the reports' at CQI error 0 dB, correlation 0.4, aware), sigma 0.968 0.940 0.954 0.943 0.919 0.753 dB, bias (truth - belief) +0.125 +0.100 +0.074 +0.062 +0.097 +0.088 dB (measured)
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves', and the reports' at CQI error 0 dB, correlation 0.6, aware), sigma 0.963 0.902 0.930 0.905 0.899 0.739 dB, bias (truth - belief) +0.120 +0.085 +0.070 +0.053 +0.094 +0.089 dB (measured)
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves', and the reports' at CQI error 0 dB, correlation 0.8, aware), sigma 0.877 0.800 0.830 0.789 0.810 0.673 dB, bias (truth - belief) +0.107 +0.054 +0.053 +0.033 +0.062 +0.082 dB (measured)
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves', and the reports' at CQI error 0 dB, correlation 0.9, aware), sigma 0.730 0.680 0.711 0.651 0.677 0.572 dB, bias (truth - belief) +0.081 +0.026 +0.044 +0.013 +0.025 +0.064 dB (measured)
calibration-aware programs (*_soft): the prediction's error per prefix length (the curves', and the reports' at CQI error 0 dB, correlation 0.95, aware), sigma 0.586 0.551 0.590 0.523 0.534 0.460 dB, bias (truth - belief) +0.062 +0.003 +0.040 +0.002 +0.005 +0.045 dB (measured)
```
Leaving out the exact-report line: smallest 0.175 (CQI error 1 dB, prefix length 6), largest
0.968 (r = 0.4, prefix length 1): 0.18-0.97.

**28. "all four decoded cases"** (item 4): Kodak with fresh reports, 2 dB CQI errors or aged
reports (r = 0.8), and CLIC with fresh reports: the four conditions with decoded schedules in
entries 29 and 30.

**29. "0.5-0.6 dB of mean PSNR over proportional fair"** (item 4, decoded, 60 frames, paired
by frame). The `lagrange` line of each report against `pf`: `results/s30_direct.txt` reports 1-3
(Kodak fresh, CLIC fresh, Kodak r = 0.8; COMMANDS 30.1, sent back 2026-10-31):
```
  lagrange                  +0.524 dB [+0.464, +0.585]  (60 frames)
  lagrange                  +0.631 dB [+0.558, +0.703]  (60 frames)
  lagrange                  +0.514 dB [+0.406, +0.625]  (60 frames)
```
and `results/s30_dec_e2.txt` report 1 (Kodak, 2 dB CQI error; COMMANDS 30.5 (5), sent back
2026-10-31):
```
  lagrange                  +0.512 dB [+0.445, +0.583]  (60 frames)
```
From 0.512 to 0.631, i.e. 0.5-0.6.

**30. "an extra 8-15% of the users to 30 dB over per-image rate selection"** (item 4, decoded, 60
frames, paired by frame; PADC is the per-image rate selection [Zhang23]). `results/s30_direct.txt`
reports 4 (Kodak fresh) and 5 (CLIC fresh), the `lagrange_soft` line against `padc`
(sent back 2026-10-31):
```
decoded, paired by frame: share reaching the target minus padc's, 95% CI over frames
  lagrange_soft            +14.792 pts [+11.667, +17.917]  (60 frames)
decoded, paired by frame: share reaching the target minus padc's, 95% CI over frames
  lagrange_soft             +8.125 pts [+5.833, +10.625]  (60 frames)
```
and `results/s33_dec.txt` (COMMANDS 33.3 (4), sent back 2026-11-01): 2 dB CQI error, then r = 0.8,
the measured-spread CLASP (`csi:lagrange_soft`) against the PADC schedules of the same frames
(`old:padc`):
```
decoded, paired by frame: share reaching the target minus old:padc's, 95% CI over frames
  csi:lagrange_soft        +13.333 pts [+10.620, +16.042]  (60 frames)
decoded, paired by frame: share reaching the target minus old:padc's, 95% CI over frames
  csi:lagrange_soft        +12.500 pts [+9.792, +15.214]  (60 frames)
```
+14.792, +8.125, +13.333, +12.500: from 8.1 to 14.8, i.e. 8-15.

**31. "0.005 dB per user at 128 users"** (item 4: the gap to the bound, mean PSNR).
Simulated, 100 frames per K, B = K/8. `results/s22_scale_mean.txt`, the summary table,
COMMANDS 22.2e (printed by 22.4, sent back 2026-10-25); the `lagrange` column:
```
summary: gap to the Lagrangian bound per user (dB)
      K   B         pf        adv         pi    whittle   lagrange lagrange_relax
    128  16      0.515      0.155      1.063      0.016      0.005      0.005
```

**32. "within ±0.3% of the bound from 16 users on"** (item 4: the gap to the bound at 30 dB).
Simulated, 100 frames per K. `results/s24_soft_auto.txt`, the summary table (sent back 2026-10-26); the
`lagrange_soft` column, K = 16 to 128:
```
summary: gap to the Lagrangian bound per user (pts)
      K   B         pf    whittle   lagrange whittle_soft lagrange_soft
     16   2     14.085      4.085      1.897      2.710     -0.103
     32   4     14.360      3.047      1.360      1.485     -0.296
     64   8     13.808      2.917      1.667      1.511      0.011
    128  16     13.197      2.392      1.462      1.118     -0.194
```
-0.103, -0.296, 0.011, -0.194: all within ±0.3.

**33. "at every CSI condition tested"** (item 4): the ten conditions (exact; CQI error 1, 2,
3 dB; r = 0.95, 0.9, 0.8, 0.6, 0.4, 0). Mean PSNR, `results/s30_vs_digital.txt` (sent back
2026-10-31):
```
[f] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +0.196 dB [+0.174, +0.217]
[e1] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +0.770 dB [+0.743, +0.794]
[e2] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +1.192 dB [+1.164, +1.221]
[e3] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +1.465 dB [+1.436, +1.494]
[r95] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +1.162 dB [+1.132, +1.191]
[r90] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +1.381 dB [+1.351, +1.410]
[r80] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +1.561 dB [+1.530, +1.591]
[r60] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +1.535 dB [+1.505, +1.565]
[r40] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +1.306 dB [+1.270, +1.346]
[r0] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +1.099 dB [+1.064, +1.135]
```
30 dB, `results/s33_csi.txt` (sent back 2026-11-01):
```
[f] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +3.612 pts [+3.137, +4.081]
[e1] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +8.831 pts [+8.281, +9.400]
[e2] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +12.744 pts [+12.156, +13.300]
[e3] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +14.663 pts [+14.037, +15.256]
[r95] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +12.194 pts [+11.637, +12.806]
[r90] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +14.537 pts [+13.944, +15.144]
[r80] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +15.825 pts [+15.200, +16.425]
[r60] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +15.837 pts [+15.212, +16.463]
[r40] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +15.775 pts [+15.100, +16.406]
[r0] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +14.162 pts [+13.594, +14.788]
```
Every difference and every interval is above zero.

**34. "1.2-1.6 dB of mean PSNR"** (item 4, against the digital chain: HEIC, the LTE CQI table,
IR-HARQ, its best policy and transmitter at each point). Simulated, the same 2000 frames,
paired. `results/s30_vs_digital.txt`, COMMANDS 30.5 (3), sent back 2026-10-31:
```
[e2] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a: lagrange        31.339 dB   learned code, CQI error 2 dB   [s30_j_mean_e2.json]
  b: greedy          30.146 dB   digital cqi + HARQ ir, CQI error 2 dB   [s30_d_mean_e2.json] (best of pf, greedy)
  a - b: +1.192 dB [+1.164, +1.221]
[r80] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a: lagrange        30.921 dB   learned code, r 0.8 aware   [s30_j_mean_r80.json]
  b: greedy          29.359 dB   digital cqi + HARQ ir, r 0.8 naive   [s30_d_mean_n80.json] (best of pf, greedy, pf, greedy)
  a - b: +1.561 dB [+1.530, +1.591]
```
+1.192 and +1.561: 1.2-1.6.

**35. "an extra 13-16% of the users at 30 dB"** (item 4). `results/s33_csi.txt`, COMMANDS 33.3 (2)-(3), sent
back 2026-11-01:
```
[e2] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a: lagrange_soft    75.5%   learned code, CQI error 2 dB   [s33_csi_e2.json]
  b: online_pf        62.7%   digital cqi + HARQ ir, CQI error 2 dB   [s30_d_tgt_e2.json] (best of padc, stop, online_pf)
  a - b: +12.744 pts [+12.156, +13.300]
[r80] 30 dB: CLASP (measured spread) - digital  (share at 30 dB, 2000 frames, paired by frame)
  a: lagrange_soft    70.6%   learned code, r 0.8 aware   [s33_csi_r80.json]
  b: online_pf        54.8%   digital cqi + HARQ ir, r 0.8 naive   [s30_d_tgt_n80.json] (best of padc, stop, online_pf, padc, stop, online_pf)
  a - b: +15.825 pts [+15.200, +16.425]
```
+12.744 and +15.825: 13-16.

**36. "0.2 dB and 3.6% with exact reports"** (item 4). `results/s30_vs_digital.txt`
(sent back 2026-10-31):
```
[f] mean PSNR: CLASP - digital  (mean PSNR (dB), 2000 frames, paired by frame)
  a - b: +0.196 dB [+0.174, +0.217]
[f] 30 dB: CLASP - digital  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +3.612 pts [+3.137, +4.081]
```
0.196, rounded to 0.2; 3.612, rounded to 3.6 (the `[f]` line of `results/s33_csi.txt` is the
same, entry 33).

**37. "3 dB from capacity, with the same HARQ"** (item 4): a setting, the digital link at
Shannon capacity minus 3 dB (`--link shannon --gap-db 3 --harq ir`; COMMANDS 35.1). Its line
prints `digital shannon + HARQ ir` (entry 38).

**38. "an extra 2.4% of the users to 30 dB over CLASP with exact reports"** (item 4). Simulated, the
same 2000 frames as CLASP, paired. `results/s35_gap3.txt`, COMMANDS 35.2 (2), sent back 2026-11-01:
```
[f] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a: lagrange_soft    78.0%   learned code, fresh, exact reports   [s33_csi_f.json]
  b: online_pf        80.4%   digital shannon + HARQ ir, fresh, exact reports   [s35_g_tgt_f.json] (best of padc, stop, online_pf)
  a - b: -2.394 pts [-2.856, -1.919]
```
CLASP 78.0%, the gap code 80.4%; the paired difference -2.394: the gap code brings an extra 2.4%
of the users to 30 dB.

**39. "from 1 dB CQI errors or correlation 0.95 on"** (item 4): the conditions `[e1]` and
`[r95]` of entry 40, the mildest of each kind.

**40. "an extra 5-14% of the users"** (item 4). Same file (sent back 2026-11-01):
```
[e1] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +5.319 pts [+4.800, +5.844]
[e2] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +9.775 pts [+9.212, +10.319]
[e3] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +11.931 pts [+11.325, +12.519]
[r95] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +8.694 pts [+8.156, +9.275]
[r90] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +11.494 pts [+10.919, +12.081]
[r80] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +13.556 pts [+12.944, +14.169]
[r60] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +13.719 pts [+13.106, +14.307]
[r40] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +12.775 pts [+12.169, +13.425]
[r0] 30 dB: CLASP - 3 dB-gap code + IR-HARQ  (share at 30 dB, 2000 frames, paired by frame)
  a - b: +12.244 pts [+11.656, +12.837]
```
From +5.319 to +13.719: 5-14.

**41. "0.21 dB (two seeds)"** (item 4: training the code on the channel CLASP creates, against
the same-budget code trained on the usual channel). Decoded on the same 150 frames, paired.
`results/s30_codesign.txt`, report 1 (base `hf:lagrange`), COMMANDS 30.5 (6), sent back 2026-10-31;
`lag` and `lagb` are the two seeds:
```
decoded, paired by frame: mean PSNR minus hf:lagrange's, 95% CI over frames
  lag:lagrange              +0.211 dB [+0.204, +0.219]  (150 frames)
  lagb:lagrange             +0.206 dB [+0.199, +0.214]  (150 frames)
```
0.211 and 0.206: 0.21.

**42. "-0.09 dB"** (item 4: CLASP's prefix lengths with blindly taken slots). Same report:
```
  lagblind:lagrange         -0.089 dB [-0.095, -0.082]  (150 frames)
```
-0.089, rounded to -0.09.

## Paragraph 6

**43. "0.055-0.083 dB on two test sets"** (item 1): entries 5 (Kodak, 0.055) and
8 (CLIC, 0.083); the two test sets are Kodak (24 images) and CLIC (60 images).

**44. "EESM errs by 0.16-0.19 dB and MIESM by 0.29-0.34 dB"** (the same weights): entries
6 and 9 (EESM 0.194, 0.159), 7 and 10 (MIESM 0.337, 0.290).

**45. "in their standard form, by 0.37 dB and 0.44-0.49 dB"** (equal weights): entries
6 and 9 (EESM 0.373, 0.374), 7 and 10 (MIESM 0.487, 0.439).

**46. "12 decodes per image"** (item 1): entry 13.

**47. "a two-number user state"** (item 2): as entry 23.

**48. "0.5-0.6 dB over proportional fair"** (item 3): entry 29.

**49. "an extra 8-15% of the users at 30 dB"** (item 3): entry 30.

**50. "from 0.21 to 0.005 dB per user between 8 and 128 users"** (item 3).
`results/s22_scale_mean.txt`, the summary table (sent back 2026-10-25), the `lagrange` column at K 8 and
K 128:
```
summary: gap to the Lagrangian bound per user (dB)
      K   B         pf        adv         pi    whittle   lagrange lagrange_relax
      8   1      0.710      0.267      1.338      0.265      0.207      0.207
    128  16      0.515      0.155      1.063      0.016      0.005      0.005
```
0.207, rounded to 0.21; 0.005.

**51. "within ±0.3% of the bound from 16 users on"** (item 3): entry 32.

**52. "1.2-1.6 dB and an extra 13-16% of the users at 30 dB with 2 dB CQI errors or reports aged
to correlation 0.8"**
(item 3): entries 34, 35; the settings, entries 18 and 1.

**53. "Aged reports cost the digital chain 1.9 dB and CLASP 0.6 dB"** (item 3): entries 2 and 3.

**54. "from 1 dB CQI errors or correlation 0.95 on ... with an extra 5-14% of the users at 30 dB"**
(item 3): entries 39
and 40.

**55. "0.21 dB"** (item 3): entry 41.

## The table "How paragraphs 3, 5 and 6 line up"

**56. Row (1):** "12 decodes": entry 13; "0.055-0.083 dB": entries 5 and 8; "EESM 0.16-0.19 and
MIESM 0.29-0.34": entries 6, 9, 7, 10; "0.06-0.09 dB, bias within ±0.03 dB": entries 16 and 17.

**57. Row (2):** "0.21 → 0.005 dB, 8 → 128 users": entry 50; "+0.5-0.6 dB over PF": entry 29.

**58. Row (3), "an extra 1.7% of the users at 30 dB over the hard target"** (decoded, 60 frames, paired by frame,
Kodak with fresh reports). `results/s24_replays.txt`, the report of `s24_dec_auto.json` against
the hard-target index `lagrange`, COMMANDS 24.4, sent back 2026-10-26:
```
2870 decoded histories: predicted - actual PSNR bias -0.006, MAE 0.070, p95 0.213 dB
decoded, paired by frame: share reaching the target minus lagrange's, 95% CI over frames
  lagrange_soft             +1.667 pts [+0.417, +2.917]  (60 frames)
```
+1.667, rounded to +1.7.

**59. Row (3), "users missed 0.0% against 1.2%"** (the users predicted to reach 30 dB who
decoded below it). `results/s24_dec_auto.txt`, printed by `grep -A 8 "target 30 dB: share"
results/s24_dec_*.txt` (COMMANDS 24.4, sent back 2026-10-26; grep prefixes each line with the file
name):
```
results/s24_dec_auto.txt-  lagrange     users   480  actual  78.8%  predicted  79.0%  missed  1.2%  extra  1.0%
results/s24_dec_auto.txt-  lagrange_soft users   480  actual  80.4%  predicted  80.4%  missed  0.0%  extra  0.0%
```
`lagrange` (the hard target) 1.2%, `lagrange_soft` (calibration-aware) 0.0%.

**60. Row (3), "an extra 13.3% / 12.5% over PADC at 2 dB CQI errors / r = 0.8 (decoded)"**:
entry 30.

**61. Row (3), "+1.2-1.6 dB and an extra 13-16% over digital (simulated)"**: entries 34 and 35.

**62. Row (3), "aged reports cost CLASP 0.6 dB, digital 1.9 dB"**: entries 2 and 3.

## To check on the server, in the order above

```
cd ~/projects/JSCCwClaude-main/jscc
cat results/s30_loss.txt                                                    # 1-3
grep -n "levels\|averaging rule\|MMSE, kappa\|exponential (EESM)\|capacity (MIESM-like)\|noise variance\|mean dB " results/s13_abstractions.txt   # 4-7, 11-12
grep -n "levels\|averaging rule\|MMSE, kappa\|exponential (EESM)\|capacity (MIESM-like)" results/s22_clic.txt          # 8-10
grep -n -A 8 "12 decodes" results/s13_nested.txt results/s22_clic.txt     # 13-15
grep -n "decoded histories" results/s30_direct.txt results/s30_dec_e2.txt results/s33_dec.txt   # 16-17
grep -n "^K \|lagrange_soft - lagrange\|share at 30 dB\|sigma" results/s24_soft_auto.txt    # 20-21, 26
grep -h "calibration-aware" results/s33_csi_*.txt                          # 26-27
grep -n "minus pf's\|minus padc's\|^  lagrange \|^  lagrange_soft " results/s30_direct.txt results/s30_dec_e2.txt   # 29-30
grep -n "minus old:padc's\|csi:lagrange_soft " results/s33_dec.txt          # 30
grep -A 7 "^summary" results/s22_scale_mean.txt results/s24_soft_auto.txt  # 31-32, 50
grep -A 3 "CLASP - digital" results/s30_vs_digital.txt                     # 33, 34, 36
grep -A 3 "CLASP (measured spread) - digital" results/s33_csi.txt          # 33, 35
grep -A 3 "3 dB-gap code" results/s35_gap3.txt                             # 38, 40
grep -n "minus hf:lagrange's\|lag:lagrange \|lagb:lagrange \|lagblind:lagrange " results/s30_codesign.txt   # 41-42
grep -n "decoded histories\|minus lagrange's\|^  lagrange_soft " results/s24_replays.txt      # 58
grep -A 8 "target 30 dB: share" results/s24_dec_auto.txt                   # 59
```
