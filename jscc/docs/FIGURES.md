# The paper's figures

`tools/paper_figs.py` draws every result graph of the paper outline (docs/PAPER_OUTLINE.md,
section V) with matplotlib, from the files COMMANDS sections 13-24 wrote to `results/`. It
re-reads those files only: no simulation, no decoding, CPU, under a minute.

    python tools/paper_figs.py --results results --out figures     # COMMANDS 25.1
    python tools/paper_figs.py --only fig05 fig09                  # some of them
    python tools/paper_figs.py --list                              # each figure's inputs
    python tools/paper_figs.py --self-test --out figures_selftest  # synthetic inputs, layout only

Each figure is written twice: `figures/<name>.pdf` (vector, fonts embedded, for the paper) and
`figures/<name>.png` (300 dpi, to look at). A figure whose required input is missing is skipped
with a note naming the file; optional inputs (marked [ ] below) only add series or groups.
`--self-test` builds every input from synthetic numbers in the tools' own formats
(`tools/paper_figs_selftest.py`) and stamps each figure "synthetic self-test data": it checks
the drawing and the layout, never the results.

## Format

* IEEE sizes: one column 3.5 in or two columns 7.16 in wide; all text 8 pt at that size
  (IEEE asks for 8-10 pt). The script warns if a figure comes out wider than its column.
* One legend per figure, above it. Panels are lettered (a), (b), ... at their top left.
* Gains and gaps carry the tools' 95% bootstrap intervals over frames. Decoded gains are paired
  by frame exactly as `tools/replay_report.py` pairs them (smoke-checked).
* The same encoding in every figure, checked for colour-vision deficiency on white; every series
  also has its own marker and line style, so the figures survive grayscale printing:

| entity | colour | marker / style |
|---|---|---|
| Lagrangian index | blue | circle; calibration-aware solid, filled; the hard-target index beside it dashed, hollow (hatched bars) |
| closed form (advantage index for the mean, online rule for the target) | orange | square |
| Whittle index | aqua | triangle; calibration-aware / hard as above |
| planner (plan_adv; replan_adv where slots are lost) | yellow | diamond |
| PI ratio index | magenta | plus; cross-hatched bars |
| PF, PADC | grey, dark grey | down triangle, cross |
| JSCC, one decoder / a decoder per length | green solid / green dashed, hollow | circle |
| HEIC + IR-HARQ / 3 dB-gap code + IR-HARQ | dark grey solid / grey dashed | square / triangle |
| true curves (oracle) | ink, dotted | x |
| phases, user counts | one blue ramp, dark = phase 1 or the most users | |

## The figures

"Use" is a suggestion for a TWC paper of ~12 figures: main text, or optional (appendix, or one
sentence in the text).

| figure | width | shows | claim, section | reads | use |
|---|---|---|---|---|---|
| fig01_abstraction | 2 col | held-out MAE of each rule that maps a history to one effective SNR (MIESM, capacity, mean dB, EESM, noise, MMSE): four codes on Kodak, two on CLIC | C12, V-A | s13_abstractions.txt [s22_clic.txt] | main, or as Table I |
| fig02_phase_weights | 2 col | mean PSNR with one phase's SNR varied, the others at 7 dB: piecewise-trained vs constant-SNR-trained code | C1, III/IV | p2_full_snrc4.json [p2_full_s1.json] | optional (motivates the weights) |
| fig03_predicted_vs_decoded | 2 col | the PSNR the model predicts for every decoded schedule against the decoded PSNR; the error's histogram with bias and MAE | C2, V-A | s22_dec_*.json, s24_dec_*.json, [s22_clic_dec_*.json, s24_dec_clic.json] | main |
| fig04_mean_scaling | 2 col | mean PSNR: gap to the Lagrangian bound vs K (B = K/8), per user and per frame, log scale | C13, C3, V-B | s22_scale_mean.json | main |
| fig05_target_scaling | 1 col | 30 dB: gap to the bound vs K; hard indices on the calibrated curves, calibration-aware on the same curves, hard on the true curves | C13, V-C | s24_soft_auto.json (else s22_gap_ties.json) [s23_gap_truth.json] | main |
| fig06_soft_sigma | 1 col | 30 dB: the calibration-aware index's gap against the curve error it assumes (0 = hard, measured, 0.05 / 0.1 / 0.2 dB), K 8 / 32 / 128; dotted: the true curves | C13, V-C | s24_soft_auto.json [s24_soft_s05/s10/s20.json, s24_soft_auto.txt, s23_gap_truth.json] | optional (one sentence) |
| fig07_target_variants | 1 col | 30 dB: every variant tried on the gap (tie rule, prices, LP update, grids, margins, calibration-aware, true curves) at K 8 / 32 / 128 | C13, V-C | s22_gap_ties/prices/lpu/grid.json, s23_gap_m25/m50/truth.json, s24_soft_auto.json | optional (appendix) |
| fig08_main_mean | 2 col | main frames (K 8, T 24): mean PSNR gain over PF, fresh / CQI error 2 dB / r = 0.8 / 25% of slots lost / T 16 / CLIC | C3, V-B | s22_main_mean.txt [s21_lag_mean_more.txt, s21_lag_t16.txt, s22_clic.txt] | main |
| fig09_main_target | 2 col | main frames: gain in users at 30 dB over PADC under the same conditions | C4, C7, V-C | s24_main_soft.txt [s24_main_soft_r80.txt, s21_lag_tgt_more.txt, s21_lag_t16.txt, s22_clic.txt] | main |
| fig10_decoded_target | 2 col | decoded, 30 dB: (a) gain in users over PADC; (b) users predicted at 30 dB that decoded below it. Kodak, Kodak r = 0.8, CLIC | C4, C13, V-C | s24_dec_auto.json [s24_dec_r80.json, s24_dec_clic.json] | main |
| fig11_decoded_mean | 2 col | decoded, mean: gains over PF in PSNR, MS-SSIM and LPIPS. Kodak, Kodak r = 0.8, CLIC | C3, V-B | s22_dec_mean.json [s22_dec_r80_mean.json, s22_clic_dec_mean.json] | main |
| fig12_latency | 2 col | quality against the frame length T; the arrow: the slots PF (PADC) needs for the Lagrangian index's quality at T 24 | C14, V-C | s21_latency_mean.json and/or s21_latency_tgt.json | main |
| fig13_time_to_target | 1 col | share of all users at 30 dB by slot 6 / 12 / 18 / 24 | C14, V-C | s24_main_soft.txt (its --delay table) | optional |
| fig14_digital | 2 col | JSCC vs digital with IR-HARQ, best policy and transmitter per point: mean PSNR and users at 30 dB against the CQI error and the report correlation | C5, V-D | s23_c5_single.txt, s20_harq_mean.txt [s20_harq_tgt*.txt, s18_stale_jscc.txt, s13_cqi_jscc.txt] | main |
| fig15_scheduler_channel | 2 col | served SNR minus the user's mean for phases 1-3 under the closed form, against report age and K | C8, C11, V-E | bank_sa.json [bank_sa_cqi*.json, bank_sa_k16.json, bank_lift_k*.json] | main |
| fig16_decoders | 1 col | what a decoder per length and one model per CBR add over the single decoder at each prefix length (constant SNRs) | C6, V-F | p2_full_snrc4.json, p4_full_h*.json and/or p3_full_fix*.json | main or optional |

## Notes on reading them

* fig04-fig07 are per user. A target run's gaps are shares; the figures show points (x 100).
* fig08 and fig09 mix sections: the 25%-lost, T 16 and CLIC groups come from runs made before
  the calibration-aware policies existed (sections 21-22), so those groups have no
  calibration-aware bars. The r = 0.8 target group uses the 0.5 dB margin those runs used.
* fig14 takes, at every point, the best policy and transmitter (naive or aware) of each system,
  as docs/NOTES.md 5 and 6e do; JSCC with a decoder per length comes from the utility_hd runs.
* fig15 reads the banks with `tools/codesign.py`'s own `stats`.
* To change a figure (a series, an axis, a size, a label), edit its function in
  `tools/paper_figs.py`; the encoding sits at the top (`ENT`), the IEEE sizes in `COL1`, `COL2`
  and `FONT`.
