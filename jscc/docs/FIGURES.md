# The paper's figures and tables

`tools/paper_figs.py` draws every result graph of the paper outline (docs/PAPER_OUTLINE.md,
section V) with matplotlib, and writes two LaTeX tables, from the files COMMANDS sections
13-26 wrote to `results/`. It re-reads those files only: no simulation, no decoding, CPU,
under a minute.

    python tools/paper_figs.py --results results --out figures     # COMMANDS 26.1
    python tools/paper_figs.py --only fig05 table02                # some of them
    python tools/paper_figs.py --list                              # each one's inputs
    python tools/paper_figs.py --self-test --out figures_selftest  # synthetic inputs, layout only

Each figure is written twice: `figures/<name>.pdf` (vector, fonts embedded, for the paper) and
`figures/<name>.png` (300 dpi, to look at). Each table is written as `figures/<name>.tex`
(LaTeX with `\hline` rules, caption and label included) and printed as text. Anything whose
required input is missing is skipped with a note naming the file; optional inputs (marked [ ]
below) only add series or groups. `--self-test` builds every input from synthetic numbers in
the tools' own formats (`tools/paper_figs_selftest.py`) and stamps each figure "synthetic
self-test data": it checks the drawing and the layout, never the results.

## The proposed scheduler, in the figures

The paper proposes the Lagrangian index; for a quality target its programs use the
calibration-aware reward (the probability that a user's true PSNR reaches 30 dB given the
transmitter's curve error; docs/PAPER_OUTLINE.md IV-C). So in every target figure "Lagrangian
index" and "Whittle index" mean the calibration-aware versions (`lagrange_soft`,
`whittle_soft` in the tools). The hard-target versions appear in the ablation (fig05,
table02), and elsewhere only where a run had no calibration-aware version, labelled "hard
target" (none since COMMANDS 26: every target figure has the calibration-aware runs).

Terms used in the figures:

* **gap to the bound per user**: the Lagrangian bound caps what any causal scheduler can
  achieve; the gap is the bound minus the policy's quality, averaged over users (dB for the
  mean, points of the share at 30 dB for the target). Near zero = near-optimal. It is per user
  because the bound and the policies scale with K; the gap summed over the frame's users is
  K times it.
* **true curves (oracle)**: the scheduler decides on each image's quality curves as the
  transmitter estimates them (calibrated from 12 decodes); the simulator scores users on the
  fitted curves. "True curves" gives the scheduler those exact curves instead: the best a
  curve-based scheduler can do, used only to show what the curve errors cost.

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
| Lagrangian index | blue | circle, solid; with a hard target (ablation) dashed, hollow, hatched bars |
| closed form (advantage index for the mean, online rule for the target) | orange | square |
| Whittle index | aqua | triangle; hard target as above |
| planner (plan_adv; replan_adv where slots are lost) | yellow | diamond |
| PI ratio index | magenta | plus; cross-hatched bars |
| PF, PADC | grey, dark grey | down triangle, cross |
| piecewise-trained code / JSCC, one decoder | green, solid | circle |
| a decoder per length | green, dashed, hollow | circle |
| constant-SNR-trained code, digital (HEIC + IR-HARQ / 3 dB-gap code) | greys | |
| true curves (oracle) | ink, dotted | x |
| users K, prefix lengths | one blue ramp, dark = the most users | |

## The list

"Use" is a suggestion for a TWC paper of ~12 figures: main text, or optional (appendix, or one
sentence in the text). fig01 and fig06 no longer exist (fig01 is table01; fig06 is fig05b);
the other numbers are kept so the review can go on by number.

| name | width | shows | claim, section | reads | use |
|---|---|---|---|---|---|
| table01_abstraction | 1 col | held-out MAE of each rule that maps a history to one effective SNR (MIESM, capacity, mean dB, EESM, noise, MMSE): four codes on Kodak, two on CLIC; best per code in bold | C12, V-A | s13_abstractions.txt [s22_clic.txt] | Table I |
| fig02_codes | 2 col | the two trainings compared at CBR 1/8: (a) the PSNR gain when one phase's slot goes from 1 to 13 dB, per phase, both codes; (b) piecewise- minus constant-SNR-trained PSNR at each constant SNR | C1, C12, V-A | p2_full_snrc4.json [p2_full_s1.json] | main or optional |
| fig03_model_error | 2 col | the model's error (predicted minus decoded PSNR) on every decoded history: (a) median and 10-90% range per 1 dB of predicted PSNR; (b) its distribution, with bias, MAE and the share beyond 0.5 dB | C2, V-A | s22_dec_*.json, s24_dec_*.json, [s22_clic_dec_*.json, s24_dec_clic.json] | main |
| fig04_scaling | 2 col | gap to the bound per user vs K (B = K/8): (a) mean PSNR, log scale; (b) users at 30 dB (PF's range as a note: off the axis) | C13, C3, C4, V-B, V-C | s22_scale_mean.json [s24_soft_auto.json, else s22_gap_ties.json] | main |
| fig05_ablation_calibration | 2 col | ablation: (a) the Lagrangian index's gap vs K with a hard target, with the calibration-aware reward (proposed), and with a hard target on the true curves (oracle); (b) the calibration-aware index against the curve error its reward assumes (0 = hard), K 8 / 32 / 128, the oracle dotted | C13, V-F | s24_soft_auto.json [s23_gap_truth.json, s24_soft_s05/s10/s20.json, s24_soft_auto.txt] | main (ablation) |
| table02_calibration | 2 col | ablation, decoded: users at 30 dB with a hard target and calibration-aware, the paired gain, users missed, mean PSNR, LPIPS; Kodak, Kodak σ 0.15 dB, CLIC, Kodak r = 0.8 | C13, V-F | s24_dec_auto.json [s24_dec_s15.json, s24_dec_clic.json, s24_dec_r80.json] | Table II |
| fig07_target_variants | 1 col | 30 dB: every variant tried on the gap (tie rule, prices, LP update, grids, margins, calibration-aware, true curves) at K 8 / 32 / 128 | C13, V-F | s22_gap_ties/prices/lpu/grid.json, s23_gap_m25/m50/truth.json, s24_soft_auto.json | optional (appendix) |
| fig08_main_mean | 2 col | main frames (K 8, T 24): mean PSNR gain over PF, fresh / CQI error 2 dB / r = 0.8 / 25% of slots lost / T 16 / CLIC | C3, V-B | s22_main_mean.txt [s21_lag_mean_more.txt, s21_lag_t16.txt, s22_clic.txt] | main |
| fig09_main_target | 2 col | main frames: gain in users at 30 dB over PADC under the same conditions | C4, C7, V-C | s24_main_soft.txt [s24_main_soft_r80.txt, s26_tgt_lost/t16/clic.txt (older runs if missing: s21_lag_tgt_more.txt, s21_lag_t16.txt, s22_clic.txt)] | main |
| fig10_decoded_target | 2 col | decoded, 30 dB: (a) gain in users over PADC; (b) users predicted at 30 dB that decoded below it. Kodak, Kodak r = 0.8, CLIC | C4, C13, V-C | s24_dec_auto.json [s24_dec_r80.json, s24_dec_clic.json] | main |
| fig11_decoded_mean | 2 col | decoded, mean: gains over PF in PSNR, MS-SSIM and LPIPS. Kodak, Kodak r = 0.8, CLIC | C3, V-B | s22_dec_mean.json [s22_dec_r80_mean.json, s22_clic_dec_mean.json] | main |
| fig12_latency | 2 col | quality against the frame length T; the arrow: the slots PF (PADC) needs for the Lagrangian index's quality at T 24 (dashed: more than the longest frame run) | C14, V-C | s21_latency_mean.json and/or s26_latency_tgt.json (older run if missing: s21_latency_tgt.json) | main |
| fig13_time_to_target | 1 col | share of all users at 30 dB by slot 6 / 12 / 18 / 24 | C14, V-C | s24_main_soft.txt (its --delay table) | optional |
| fig14_digital | 2 col | JSCC vs digital with IR-HARQ, best policy and transmitter per point: mean PSNR and users at 30 dB against the CQI error and the report correlation | C5, V-D | s23_c5_single.txt, s20_harq_mean.txt [s20_harq_tgt*.txt, s18_stale_jscc.txt, s13_cqi_jscc.txt] | main |
| fig15_scheduler_channel | 2 col | served SNR minus the user's mean for phases 1-3 under the closed form, against report age and K | C8, C11, V-E | bank_sa.json [bank_sa_cqi*.json, bank_sa_k16.json, bank_lift_k*.json] | main |
| fig16_decoders | 1 col | what a decoder per length and one model per CBR add over the single decoder at each prefix length (constant SNRs) | C6, V-F | p2_full_snrc4.json, p4_full_h*.json and/or p3_full_fix*.json | main or optional |

## Notes on reading them

* fig02a explains why the abstraction weights slots: each phase is the next block of tokens,
  sent in its own slot, and the decoder is far more sensitive to the first (the coarse image)
  than to the last. Both codes are ordered that way; what training changes is whether the slots
  combine by a rule (table01), at a small cost at constant SNR (fig02b).
* fig08 and fig09 mix sections: the r = 0.8 target group uses the 0.5 dB margin its runs used.
* fig14 takes, at every point, the best policy and transmitter (naive or aware) of each system,
  as docs/NOTES.md 5 and 6e do; JSCC with a decoder per length comes from the utility_hd runs.
* fig15 reads the banks with `tools/codesign.py`'s own `stats`.
* To change a figure (a series, an axis, a size, a label), edit its function in
  `tools/paper_figs.py`; the encoding sits at the top (`ENT`), the IEEE sizes in `COL1`, `COL2`
  and `FONT`.
