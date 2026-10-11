# Paper plan (IEEE TWC)

2026-10-11. What the measurements so far support, how the paper is laid out, and what is
still owed. Numbers are Kodak (24 images), the snrc4 ViT (p8, 768 x 10, 4 symbols per
token, 6 phases of CBR 1/48 at 256 px), 8 users, 24 slots, mean SNRs U[0, 15] dB, Rayleigh
block fading, unless stated; "decoded" means the schedules were decoded by the codec
itself (`docs/NOTES.md`, P1 steps 1-6).

## Working title

Quality-Aware Opportunistic Scheduling of Deep JSCC Images over Fading Multi-User Downlinks
(the framing after P1 step 6: the abstraction and the schedulers are the contribution, shown
on a prefix code and on one model per CBR; the prefix code is the choice when the slots a
user will get are not known at encoding. The prefix-centred title "Scheduling Encode-Once
Deep JSCC Prefixes ..." needed per-length decoder heads to close the nesting penalty, and the
LoRA heads did not)

## Claims and their evidence

| # | claim | evidence | status |
|---|---|---|---|
| C1 | A prefix (nested) ViT JSCC trained on piecewise SNR profiles decodes a prefix whose slots had different SNRs, and its distortion is ADDITIVE over slots: D = Dsrc(L) + C(L) sum_j rho_j phi(n_j). Symmetric effective-SNR rules (EESM-like) fail. | P1: additive within 0.06 dB (2-chunk means); symmetric rules 0.23-0.71 dB MAE; constant-SNR decoders do not obey it (s1: 2-4x worse) | done |
| C2 | The model is a link-to-system abstraction: per image and phase it predicts held-out profiles within 0.03-0.07 dB and real schedules within 0.065-0.09 dB, unbiased; the transmitter can calibrate it from 2P constant-SNR decodes of its own image (kappa, rho transfer across images) | P1 step 2 (fit), step 4 (calibration 0.03-0.06 dB) | done |
| C3 | Mean quality: PF on rate leaves ~0.5 dB on the table; the ADVANTAGE index (this slot's predicted PSNR gain minus a typical slot's) gains +0.46 dB decoded, +0.46 dB at the 5th-percentile user, robust over K, T, curve errors and slot correlation; content-aware lengths add only +0.05 | P1 steps 2-4 | done (theory owed) |
| C4 | Quality target (QoS): online admission by remaining need + stopping on the realised slot SNRs + advantage timing raises the share of users reaching 30 dB by +11 points decoded over PADC's per-image rule (given shortest-first admission and PF timing), +14 over PF; decomposition: sizing +5, stopping +4, timing +3, admission +2 (+11 in tight frames) | P1 steps 3-4 | done |
| C5 | Against separation (HEVC intra + error-free CQI link adaptation, the same schedulers): on par -- +0.13 dB mean PSNR (prefix) / +0.28 (per CBR), ahead in tight frames, behind at high targets; below a 3 dB-gap code (0.5-0.7 dB) and capacity. Stated as parity at the current codec, with the codec gap to the state of the art (~0.8-0.9 dB) as the lever | P1 step 6 | done (equal per-user cap owed) |
| C6 | Against one model per CBR (six piecewise-trained specialists): nesting penalty 0.17-0.47 dB, ~60% of it in the shared encoder; with known slots per-CBR codes + our model and schedulers are the best system; with slots lost to other traffic the prefix code wins mean PSNR (+0.36 / +0.53 dB at 10% / 25%). The model and both schedulers work on both codes (codec-agnostic) | P1 steps 5-6; section 12 (the same planner on both codes, per-length decoders) | done, last checks owed |

## Outline

1. Introduction: semantic / JSCC image delivery to many users under deadlines; why
   per-rate codes and feedback schemes (JSCCformer-f) do not fit opportunistic scheduling;
   contributions C1-C6.
2. Related work (`docs/SURVEY.md`): DeepJSCC and variable rate (DeepJSCC-V/PADC, SwinJSCC,
   DeepJSCC-MIMO, JSCCformer-f, DRJSCC), scalable multi-user DeepJSCC, semantic resource
   allocation (static, mean-SNR), opportunistic and gradient scheduling (PF, Stolyar,
   Agrawal-Subramanian), link abstraction (EESM / MIESM).
3. System model: frame of T slots, K users, block fading with CQI, encode-once prefix,
   one phase per slot, objectives (mean PSNR; share at a target).
4. The codec and its abstraction: architecture, piecewise-SNR training, the additive
   model, fitting, transmitter-side calibration; accuracy (C1, C2).
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
* Fig. "vs digital" and "vs per-CBR codes" (section 9)

## Still owed, in order

1. P-D and P-E (COMMANDS.txt section 10: HEIC / JPEG 2000 baselines; six piecewise-trained
   specialists replace the retired wave c2).
2. The codec gap: about 0.8-0.9 dB below JSCCformer-f's codec after discounting feedback;
   more training data (7e) is the cheap first step. Relative claims do not depend on it, but a
   reviewer will ask.
3. Theory for section 5 (the advantage index as a gradient scheduler with a deadline;
   conditions under which shortest-remaining-need admission is optimal for the count).
4. Robustness a reviewer may ask for: a second test set (CLIC), MS-SSIM targets, OFDMA
   (several users per slot), imperfect CQI.
