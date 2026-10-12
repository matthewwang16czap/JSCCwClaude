# Paper plan (IEEE TWC)

2026-10-12. What the measurements so far support, how the paper is laid out, and what is
still owed. Numbers are Kodak (24 images), the snrc4 ViT (p8, 768 x 10, 4 symbols per
token, 6 phases of CBR 1/48 at 256 px), 8 users, 24 slots, mean SNRs U[0, 15] dB, Rayleigh
block fading, unless stated; "decoded" means the schedules were decoded by the codec
itself (`docs/NOTES.md`, P1 steps 1-7).

## Working title

Quality-Aware Opportunistic Scheduling of Deep JSCC Images over Fading Multi-User Downlinks
(framing B since P1 step 6: the abstraction and the schedulers are the contribution, shown on a
prefix code, on one model per CBR, and on the system P1 step 7 recommends: one encode-once
prefix encoder with a decoder per received length. The prefix-centred title "Scheduling
Encode-Once Deep JSCC Prefixes ..." stays off: per-length decoders close most of the nesting
penalty at 2-4 phases, not at the ends, and per-CBR codes still win the target with known slots)

## Claims and their evidence

| # | claim | evidence | status |
|---|---|---|---|
| C1 | A prefix (nested) ViT JSCC trained on piecewise SNR profiles decodes a prefix whose slots had different SNRs, and its distortion is ADDITIVE over slots: D = Dsrc(L) + C(L) sum_j rho_j phi(n_j). Symmetric effective-SNR rules (EESM-like) fail. | P1: additive within 0.06 dB (2-chunk means); symmetric rules 0.23-0.71 dB MAE; constant-SNR decoders do not obey it (s1: 2-4x worse) | done |
| C2 | The model is a link-to-system abstraction: per image and phase it predicts held-out profiles within 0.03-0.07 dB and real schedules within 0.05-0.09 dB, unbiased; the transmitter can calibrate it from 2P constant-SNR decodes of its own image (kappa, rho transfer across images). Codec-agnostic: the same holds, decoded, for six per-CBR codes and for per-length decoders | P1 step 2 (fit), step 4 (calibration 0.03-0.06 dB), step 7 (per-CBR decoded MAE 0.06-0.08; per-length decoders 0.04-0.07) | done |
| C3 | Mean quality: PF on rate leaves ~0.5 dB on the table; the ADVANTAGE index (this slot's predicted PSNR gain minus a typical slot's) gains +0.46 dB decoded, +0.46 dB at the 5th-percentile user, robust over K, T, curve errors and slot correlation; content-aware lengths add only +0.05 | P1 steps 2-4 | done (theory owed) |
| C4 | Quality target (QoS): online admission by remaining need + stopping on the realised slot SNRs + advantage timing raises the share of users reaching 30 dB by +11 points decoded over PADC's per-image rule (given shortest-first admission and PF timing), +14 over PF; decomposition: sizing +5, stopping +4, timing +3, admission +2 (+11 in tight frames) | P1 steps 3-4 | done |
| C5 | Against separation (HEIC intra + error-free CQI link adaptation, the same schedulers, the same 6-slot cap per user): ahead by small margins -- mean PSNR +0.14 dB (prefix) / +0.34 (a decoder per length) / +0.29 (per CBR); target +0.5 to +10.5 points with per-length decoders (largest in tight frames), the prefix code alone -1.6 at 32 dB; a 3 dB-gap code is 0.5-0.7 dB ahead. Under CQI errors (sd 1-3 dB; digital with its best backoff per policy) the gap is expected to open: to measure | P1 steps 6-7; section 13 (`--cqi-error`) | done at ideal CQI; CQI errors owed |
| C6 | Against one model per CBR (six piecewise-trained specialists): nesting penalty 0.17-0.47 dB; a decoder per received length behind the one encoder recovers 79-101% of it at 2-4 phases (49-65% at 1, 5, 6). Encode once + per-length decoders is the best JSCC system on mean PSNR (31.63 vs 31.58 dB) and when slots can be lost (+0.73 dB, +2.8 points at 25%); per-CBR codes keep +0.9 to +1.4 points at the target with known slots. The model and the schedulers work on all three (codec-agnostic) | P1 steps 5-7; section 13 (equal training budget) | done, equal-budget check owed |
| C7 | Which QoS policy when: plan_adv (shortest-first admission + slots planned before the frame + advantage timing) with known slots; replan_adv (the same re-planned every slot from what users hold, with stopping) when slots are tight, lost or (to measure) the CQI is noisy; online when mean PSNR matters too | P1 step 7 (decoded: replan_adv 77.7, plan_adv 76.7, online 76.5, padc_adv 67.7%) | done; CQI errors owed |

## Outline

1. Introduction: semantic / JSCC image delivery to many users under deadlines; why
   per-rate codes and feedback schemes (JSCCformer-f) do not fit opportunistic scheduling;
   contributions C1-C7.
2. Related work (`docs/SURVEY.md`): DeepJSCC and variable rate (DeepJSCC-V/PADC, SwinJSCC,
   DeepJSCC-MIMO, JSCCformer-f, DRJSCC), scalable multi-user DeepJSCC, semantic resource
   allocation (static, mean-SNR), opportunistic and gradient scheduling (PF, Stolyar,
   Agrawal-Subramanian), link abstraction (EESM / MIESM).
3. System model: frame of T slots, K users, block fading with (possibly erroneous) CQI,
   slots that other traffic may take, encode-once prefix, one phase per slot, objectives (mean
   PSNR; share at a target).
4. The codec and its abstraction: architecture, piecewise-SNR training, encode once with a
   decoder per received length (where the nesting penalty lives), the additive model, fitting,
   transmitter-side calibration; accuracy on all three codes (C1, C2, C6).
5. Schedulers: the advantage index (derivation as the finite-deadline, fixed-quota form of
   the gradient scheduler, with the opportunity cost of a typical slot as reference); the
   QoS policy (shortest-remaining-need admission, i.e. Moore-Hodgson with stochastic sizes,
   spare slots, stopping); conditional planning under correlated fading.
6. Evaluation: decoded results; scenarios (K, T, targets, same image, curve errors, slot
   correlation); baselines (RR, max-SNR, PF, PADC-style, per-CBR codes, digital); ablations
   (the decomposition tables); the cost (calibration decodes, scheduler complexity).
7. Conclusion and limits (single test set, AWGN-equalised slots, PSNR).

## Figures and tables (source)

* Fig. "mixed-SNR decoding": PSNR of [n1, n2] profiles, w1 / s1 / snrc4 (results/p1_mixed_*.json)
* Fig. "rho per phase" + table of model errors per CBR (p2_fit_snrc4.txt, s8_calib.txt)
* Fig. "predicted vs decoded" scatter of scheduled histories (replay_*.json, s7_dec_*.json, s8_dec_*.json)
* Table "mean PSNR by policy and scenario" (s7_mean.txt, s8_belief.txt, s8_corr.txt)
* Fig. "share at target vs target" for 28 / 30 / 32 dB, and vs T (s7_tgt.txt, s8_belief.txt)
* Table "decomposition of the QoS gain" (s7_tgt.txt)
* Fig. "vs slot correlation" (s8_corr.txt + section 9's conditional plan)
* Fig. "vs digital" and "vs per-CBR codes" (s11_*, s12_*), "vs CQI error" (s13_cqi)
* Table "where the nesting penalty lives" (s11_heads, s12_heads, s13_budget)

## Still owed, in order

1. CQI errors (section 13, CPU): JSCC (prefix, per-length decoders, per-CBR) against HEIC with
   its best backoff, sd 1-3 dB. Decides whether C5 is "on par" or "ahead under realistic CSI".
2. Equal training budget (section 13, GPUs): the prefix code and per-CBR codes with the same
   400 extra epochs as the per-length decoders (HF; FS at 1/24, 1/16, 1/12), paired; HF also as
   a utility (the fair prefix baseline for C6).
3. Decoded per-length-decoder schedules (section 13, GPU 5 first) and the paired cross-codec
   replay (per-CBR vs prefix on the same frames).
4. The codec gap: about 0.8-0.9 dB below JSCCformer-f's codec after discounting feedback; more
   training (HF tells how much 400 epochs give) or data. Relative claims do not depend on it.
5. Theory for section 5, now layered (docs/PROBLEM.md, section 3, "Optimality"): exact slot
   counts (dynamic program; greedy exact for concave curves), exact count admission
   (shortest first), the clairvoyant frame problem solved with a certificate (column
   generation + integer program, `tools/headroom.py`), the online problem as a stochastic DP
   with the index as a one-step approximation and rollout as one improvement step, and the
   phase-order result (rearrangement; Derman-Lieberman-Ross thresholds online). Section
   13.3d measures every gap on the real curves (stand-in: rollout +0.06 dB, clairvoyant
   optimum +0.19, any phase order +0.27; at 30 dB +1.6 / +4.7 / +4.7 points). Small rollout
   gains -> "the rules are near the best causal policy for the model"; large ones -> an RL
   scheduler trained on the model, with our rules as baselines. A large phase-order gain ->
   a DLR phase-mapping rule for fixed-quota frames (a new, provable component).
6. Robustness a reviewer may ask for: a second test set (CLIC), MS-SSIM targets, OFDMA
   (several users per slot).
