# Paper plan (IEEE TWC), condensed 2026-10-15

Every number is in `docs/NOTES.md` (section numbers below). Earlier versions of this plan, with
the framings that were tried and dropped, are in git: `git show ec9cc68:jscc/docs/PAPER.md`.

## Working title

**Learned JSCC in Scheduled Multi-User Downlinks: An MMSE Link Abstraction, Near-Optimal Index
Scheduling, and Scheduler-Aware Training** (short form: "Scheduling Learned Image Codes over
Fading Multi-User Downlinks")

## The story

Deep JSCC codes are designed and evaluated on one link. In a cellular downlink they would be
delivered slot by slot, by a scheduler, to many users, before a deadline. Two things must cross
the code-scheduler interface: the scheduler must know what a slot is worth to the code, and the
code only ever sees the slots the scheduler gives it. We show that

1. a learned code trained for slot-varying SNR admits a link abstraction of its own: its slots
   add as sensitivity-weighted MMSEs, not as mutual information (MIESM, digital's standard, is
   ~9x less accurate), and the transmitter calibrates it from 12 decodes per image;
2. with it, deadline scheduling of many users is a restless bandit whose index policy gains
   +0.46 dB over PF and +11 points over PADC's rule (decoded) and is certified within 0.26 dB of
   the best possible causal scheduler;
3. the scheduler in turn makes the code's channel (selection lifts the first phases ~3 dB), the
   right channel to train for: +0.1 dB beyond rate specialisation, and the codes learn their
   scheduler's fade statistics;
4. under realistic CSI errors this JSCC system beats HEVC intra with tuned link adaptation by
   +1.6 dB and +15 points at 2 dB CQI error.

## Contributions and evidence

Claim IDs are stable (COMMANDS.txt and the other docs cite them). All "decoded" results are the
codec itself decoding the schedules.

| contribution | ID | claim | key evidence | NOTES | status |
|---|---|---|---|---|---|
| 1. Link abstraction | C1 | A piecewise-trained prefix code decodes slot-split prefixes, and its distortion is additive over slots: D = Dsrc + C sum_j rho_j phi(n_j) | held-out 0.03-0.07 dB per image and phase; constant-SNR decoders do not obey it | 1, 2 | done |
| | C2 | It is a usable link-to-system model | scheduled histories decoded within 0.05-0.09 dB, no bias; calibrated from 12 decodes per image (0.03-0.06 dB); holds for prefix, per-length decoders and per-CBR codes | 2 | done |
| | C12 | Learned codes need their own abstraction, and training decides it | sensitivity-weighted MMSE 0.055 dB vs MIESM 0.49, EESM 0.19; a constant-SNR-trained code obeys no rule | 2 | done: the anchor |
| 2. Scheduling on it | C3 | Mean PSNR: the advantage index | +0.465 dB [0.39, 0.53] decoded over PF, +0.46 for the 5th-percentile user, robust over K, T, curve errors, correlation; content-aware lengths only +0.05 | 3 | done |
| | C4 | Quality target: admission + stopping on realised SNRs + index timing | +11.0 points [8.3, 13.8] decoded over PADC's rule, +14 over PF; sizing +5, stopping +4, timing +3, admission +2 | 3 | done |
| | C13 | Near-optimal, with a theory | two-number user state, restless bandit; Whittle index = our rule (+0.15 dB with lost slots); Lagrangian bound: within 0.26 dB of any causal scheduler (mean), 2-5.5 points at the target; clairvoyant optimum certified | 3 | done |
| | C7 | Which QoS policy when | plan_adv with known slots and under CQI errors (online loses 13 points at 3 dB error), replan_adv with lost slots | 3, 5 | done |
| 3. The scheduler-made channel | C8 | Training for it pays | +0.28 dB on the same slots over the same-budget control; +0.08-0.12 beyond length specialisation (two seeds); nothing at the target | 6 | holds, small |
| | C9 | The physical channel is the wrong training channel | blind slots -0.04 dB vs generic training, -0.32 vs the scheduler's channel | 6 | done |
| | C10 | A code per scheduler; codes learn their scheduler; one round is the fixed point | PF-trained code wins under PF slots; kappa tracks the deep-fade rate, rho the favoured phase; channel unchanged after retraining | 6 | done |
| | C11 | It grows with the number of users | the lift grows like an order statistic; the decoded gain only +0.12 to +0.15 dB (K 8 to 16) | 6 | weak: a remark |
| 4. System verdicts | C6 | Encode once + a decoder per length is the JSCC system to use | best on mean PSNR and with lost slots (+0.78 dB, +1.9 points decoded vs per-CBR), ties per-CBR at the target; at equal budget a decoder per length recovers about half of the nesting penalty | 4 | done |
| | C5 | Against digital under realistic CSI | on par with error-free CQI; at 1 / 2 / 3 dB CQI error +1.0 / +1.6 / +2.3 dB and +9 / +15 / +19 points over HEIC; ahead of a 3 dB-gap code from 1 dB on | 5 | done |

## Why it is a TWC paper now (and was not on 10-13)

TWC wants a wireless problem that matters, a non-obvious contribution backed by analysis, and a
fair, thorough evaluation. Before section 17 we had the problem and the evaluation, but the
contributions were heuristic schedulers with moderate gains, a tie with digital, and a
co-design idea worth a tenth of a dB: engineering. Three results changed that:

* **A general, falsifiable finding** (C12): learned codes obey an MMSE link abstraction, not
  MIESM, and only when trained for it. It is reusable by anyone doing system-level design with
  learned codes, the role EESM/MIESM played for OFDM.
* **Theory that follows from it** (C13): with the abstraction a user's state is two numbers, the
  frame is a restless bandit, and our simple index is certified near-optimal by an upper bound
  on every causal scheduler. Heuristics became analysed policies.
* **A practical verdict** (C5): under the CSI errors real systems have, JSCC with these
  schedulers beats digital with tuned link adaptation by 1.6 dB and 15 points at 2 dB error,
  a concrete answer to "why JSCC in a cellular downlink?".

The reverse interface (C8-C10) completes the picture: modest in dB, clean in mechanism. The
evidence standard is high throughout: decoded by the real codec, paired CIs, three code
families, two seeds where it matters, digital with tuned backoff and lost blocks counted.

## Outline

1. Introduction: learned JSCC meets the MAC; the two interfaces; contributions 1-4.
2. Related work: DeepJSCC and variable rate (PADC, SwinJSCC, DeepJSCC-MIMO, JSCCformer-f);
   SNR-varying training (CFA-JSCC, DRJSCC, euJSCC); semantic resource allocation; opportunistic
   scheduling and restless bandits (PF, Hosseini-Panwar, Yu-Xu-Tong, Borkar et al.); link
   abstraction (EESM, MIESM). `docs/SURVEY.md`.
3. System model: K users, T slots, deadline, block fading, CQI with errors, lost slots,
   encode-once prefix codes with per-length decoders; mean PSNR and share at a target.
4. The link abstraction: the model and why MMSE, the training condition, transmitter
   calibration, accuracy against MIESM / EESM / noise rules (C1, C2, C12).
5. Scheduling: the restless bandit, Whittle index and the advantage index as its
   certainty-equivalent form, the Lagrangian bound and the clairvoyant optimum; QoS admission,
   stopping and planning (C3, C4, C7, C13).
6. The scheduler-made channel: the induced distribution and its order statistics,
   scheduler-aware training, what the codes learn (C8-C11).
7. Evaluation: policies against PF / PADC, codes against per-CBR codes (C6), JSCC against
   digital under CQI errors (C5); lost slots, correlated fading, K, T; complexity.
8. Conclusion and limits.

## Key figures (sources in `results/` on the server)

1. System diagram: encode once, slots, scheduler, the two interfaces.
2. Mixed-SNR decoding, piecewise vs constant-SNR training (`p1_mixed_*`).
3. Abstraction accuracy, four rules x four codes (`s13_abstractions.txt`).
4. Predicted vs decoded PSNR of scheduled histories (replays).
5. Policies against the bounds: rollout, causal bound, clairvoyant optimum (`s13_headroom.txt`, `s13_rmab.txt`).
6. JSCC vs digital against CQI error (`s13_cqi_*.txt`).
7. Prefix / per-length decoders / per-CBR, known and lost slots (`s12_*`, `s13_replays.txt`).
8. The scheduler-made channel: lift per phase vs K, kappa and rho of eight codes (`s15_*`).

## Owed before submission

1. Online phase-to-slot mapping rule (sequential assignment): the largest piece our scheduler
   leaves (+0.13 dB / +3-4 points ex post). CPU; I build it.
2. Second test set (CLIC) and a perceptual metric (MS-SSIM / LPIPS) for the main tables. GPU,
   about a day, if CLIC is on the server.
3. Optional: a Jacobian probe (the first-order mechanism behind section 4); a better codec
   (more data) for the absolute numbers.
4. Writing and figures.

## Expected reviewer objections

| objection | answer |
|---|---|
| Kodak only, PSNR only | CLIC and MS-SSIM / LPIPS (owed, item 2) |
| The codec is ~0.8 dB behind the best published one | every claim is relative; the abstraction and the schedulers hold on three code families |
| Gains are moderate | the claims are the abstraction, certified optimality and the CSI verdict, not a single gain |
| The digital baseline is weak | HEIC with tuned backoff, lost blocks counted, plus a 3 dB-gap code and capacity; ideal-CQI results reported too |
| TDMA, one user per slot | OFDMA adds a dimension to the same index; discussed as an extension |
