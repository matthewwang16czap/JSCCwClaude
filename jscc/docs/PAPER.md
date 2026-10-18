# Paper plan (IEEE TWC), condensed 2026-10-15, story revised 2026-10-18

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
2. with it, each user is a small MDP with a two-number state and deadline scheduling of many
   users is a finite-horizon restless bandit. Relaxing the slot budget gives one program per
   user, hence an upper bound on every scheduler and each user's ideal action; any scheduler is
   those ideal actions plus a conflict rule (the restless-bandit literature, the CMU blog among
   it, shows the rule decides near-optimality). Our closed-form rule gains +0.46 dB over PF and
   +11 points over PADC's rule (decoded), ties the Whittle index, and is certified within
   0.23 dB of the best causal scheduler at the regime's hardest point (8 users, one per slot);
   the gap should shrink as users per slot grow (COMMANDS 18.3, pending). Read as latency, the
   same gains should be a shorter deadline for the same quality and, with requests arriving
   over time, more images at the target within any delay (C14; COMMANDS 19, pending);
3. with fresh reports the scheduler makes the code's channel (selection lifts the first phases
   ~3 dB), the right channel to train for: +0.1 dB beyond rate specialisation, and the codes
   learn their scheduler's fade statistics; with stale reports the lift must vanish
   (COMMANDS 18.2, pending), so this is a low-mobility effect;
4. under realistic CSI this JSCC system beats HEVC intra with tuned link adaptation: +1.6 dB and
   +15 points at 2 dB CQI error, and (COMMANDS 18.1, pending) along a physical axis, speed x
   feedback delay, on which the opportunistic part of every scheduler fades and the content part
   (how many phases each user gets) remains.

What changed on 2026-10-18 (the doc "Scheduling Learned Image Codes as a Restless Bandit",
https://claude.ai/code/artifact/0801c93d-9404-4fb3-acd8-e88c03641f8f): item 2 is now stated as
relaxation + conflict rule with its regime; items 3-4 gain the report-staleness axis; the index
takes the top of its indifference range; the positioning adds planning vs learning restless
bandits. If 18.3 shows no shrinking gap, item 2 keeps "certified within 0.23 dB" without the
asymptotic claim; if JSCC's lead does not grow with staleness, item 4 stays on the dB-error axis;
if the lift survives stale reports, item 3 needs no scope.

Delay (added the same day, docs/PROBLEM.md section 3 item 7): the frame's deadline is the
latency budget, so the scheduling gains are restated as the frame length needed for the same
quality, and under continuous traffic (requests arriving over time, a deadline each) as the
share of images at the target within every delay d and the load the cell carries at a given
share (C14). Reported with them, because a reviewer will ask: the first coarse image comes later
and early quality is lower (the deadline objective does not reward them), and on the requests
two rules both serve the closed-form rule can be slower (its deadline guard waits for the slack
to run out). There the theory earns its keep: with deadlines that differ, the Whittle index
(each request's own program) weighs urgency against the channel and, on the stand-in, keeps the
closed form's success with ~40% less delay. If 19 shows the reliability-latency curves crossing,
C14 is the latency budget and the reliability at the deadline, not dominance at every d.

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
| | C13 | Near-optimal, with a theory | two-number user state, restless bandit; one program per user (relaxation) + a conflict rule; Whittle index = our rule (+0.15 dB with lost slots); Lagrangian bound: within 0.26 dB of any causal scheduler (mean; the clairvoyant optimum, +0.23, is the tighter cap), 2-5.5 points at the target; clairvoyant optimum certified. Re-examined (SURVEY F5): the bound and the index carry over to stale reports; the theory's regime is many users per slot, where the gap should fall like 1/sqrt(K); conflict rules: value (indices) for the mean, persistency (ID, admission) for the target | 3 | done; COMMANDS 18.2-18.3 pending (stale-report bound, gap vs K, guided policies) |
| | C14 | Delay: the same gains as latency | the frame length each policy needs for a quality (`tools/latency.py`); time to the target within a frame, a CDF by slot and on the same users (`schedule_sim.py --delay`); continuous traffic, a deadline per request (`tools/stream_sim.py`, the frame exactly when all deadlines coincide): share at the target within d slots for every d, capacity at a share, delay on the same requests. Stand-in only: ours needs 11% fewer slots than PF for the mean, 24% fewer than PADC at 30 dB; at 0.3 requests per slot 85% vs 52% at 30 dB, ahead at every d, but 1.9 slots slower on the requests both serve; the Whittle index (each request's program, its own deadline) keeps online's success with ~40% less delay and is faster than PADC on the same requests; in the frame too it brings users to 30 dB ~3.5 slots earlier at the same final share (its index ~ success gain per remaining phase: shortest remaining need first). The theory's index, not the closed form, carries the time-to-target claim | 3 | pending (COMMANDS 19) |
| | C7 | Which QoS policy when | plan_adv with known slots and under CQI errors (online loses 13 points at 3 dB error), replan_adv with lost slots | 3, 5 | done |
| 3. The scheduler-made channel | C8 | Training for it pays | +0.28 dB on the same slots over the same-budget control; +0.08-0.12 beyond length specialisation (two seeds); nothing at the target. Only as far as the reports are fresh (the lift is the scheduler's information about the slot) | 6 | holds, small; lift vs report freshness pending (18.2) |
| | C9 | The physical channel is the wrong training channel | blind slots -0.04 dB vs generic training, -0.32 vs the scheduler's channel | 6 | done |
| | C10 | A code per scheduler; codes learn their scheduler; one round is the fixed point | PF-trained code wins under PF slots; kappa tracks the deep-fade rate, rho the favoured phase; channel unchanged after retraining | 6 | done |
| | C11 | It grows with the number of users | the lift grows like an order statistic; the decoded gain only +0.12 to +0.15 dB (K 8 to 16) | 6 | weak: a remark |
| 4. System verdicts | C6 | Encode once + a decoder per length is the JSCC system to use | best on mean PSNR and with lost slots (+0.78 dB, +1.9 points decoded vs per-CBR), ties per-CBR at the target; at equal budget a decoder per length recovers about half of the nesting penalty | 4 | done |
| | C5 | Against digital under realistic CSI | on par with error-free CQI; at 1 / 2 / 3 dB CQI error +1.0 / +1.6 / +2.3 dB and +9 / +15 / +19 points over HEIC; ahead of a 3 dB-gap code from 1 dB on. Next: stale reports, r = J0(2 pi f_D tau), a physical axis (speed x CSI delay), digital with tuned backoff or the goodput-optimal rate | 5 | done; stale-report axis pending (18.1) |

## Prior work: what is known and what is new (docs/SURVEY.md, F-F4)

* Known: the Gaussian MMSE; for LINEAR analog codes (SoftCast) the distortion is a sum of
  per-component MMSE errors by construction, and ParCast puts important components on good
  subchannels; digital link abstraction by EESM / MIESM; single-link adaptive DeepJSCC (PADC,
  JSCCformer-f, DRJSCC); static multi-user semantic allocation from a constant-SNR quality curve;
  PF, gradient and restless-bandit scheduling of digital (layered) video.
* New: a learned NONLINEAR code behaves at the system level like a linear analog code with
  learned per-phase weights, and only when trained for slot-varying SNR (C12) -- which explains
  why MIESM fails for it; that abstraction makes deadline scheduling of learned codes a tractable
  restless bandit with a certified index policy (C13); the scheduler-made channel (C8-C10); and a
  comparison with digital under a CQI-driven link-adaptation loop with report errors (C5). No
  paper found schedules deep JSCC codes slot by slot over a fading multi-user downlink.
* Restless bandits (SURVEY F5): ours is a PLANNING restless bandit (known model, every user's
  state reported each slot), not a learning one (unknown pay-offs, bandit feedback: Grunewalder
  and Khaleghi, JMLR 2019; Gornet and Sinopoli 2024); the index-policy theory for finite horizons
  (Hu and Frazier; Brown and Smith; Zhang and Frazier; Gast, Gaujal and Yan) supplies the bound
  we use and says where it is tight (many users per slot); the learning papers supply the
  principle that switching gains only what the scheduler knows about the current slot, which
  is what stale reports test.
* Baselines in the paper: RR, max-SNR, PF (the LTE/NR default), PADC's per-image rule (TWC 2023),
  the exact optimum of the static allocation problem (what static semantic-allocation papers
  solve), fixed-rate codes per CBR, encode once with per-length decoders, HEIC with LTE CQI link
  adaptation and tuned backoff, a 3 dB-gap code and capacity; for the abstraction mean dB,
  noise, MIESM, EESM; above all of them the causal upper bound and the clairvoyant optimum.

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
   stopping and planning; the unifying view (one program per user + a conflict rule: index,
   advantage, ID-guided); stale reports (the programs decide on the report, the bound holds)
   and many users per slot (the Lagrangian index, the gap against K) (C3, C4, C7, C13).
6. The scheduler-made channel: the induced distribution and its order statistics,
   scheduler-aware training, what the codes learn (C8-C11).
7. Evaluation: policies against PF / PADC, codes against per-CBR codes (C6), JSCC against
   digital under CQI errors (C5); lost slots, correlated fading, K, T; delay (C14: the latency
   budget at equal quality, time to the target, continuous traffic); complexity.
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
9. Reliability against latency: the share of users / requests at 30 dB within d slots (frame and
   continuous traffic), and the frame length each policy needs for a quality (`s19_*`).

## Owed before submission

1. Online phase-to-slot mapping rule (sequential assignment): the largest piece our scheduler
   leaves (+0.13 dB / +3-4 points ex post). CPU; I build it.
2. Second test set (CLIC) and a perceptual metric (MS-SSIM / LPIPS) for the main tables. GPU,
   about a day, if CLIC is on the server.
3. HARQ with soft combining in the digital baseline: today a block the slot could not carry is
   lost (the transmitter learns it); real NR retransmits and combines. It narrows C5 somewhat
   (likely tenths of a dB at 2 dB CQI error); CPU, I add it.
4. Optional: a Jacobian probe (the first-order mechanism behind section 4); a better codec
   (more data) for the absolute numbers.
5. Writing and figures.
6. COMMANDS section 18 (CPU, ~3 h, built 2026-10-18): stale reports (18.1: JSCC vs digital,
   static vs opportunistic part; 18.2: the lift and the bound on stale reports) and many users
   per slot (18.3: gap to the bound vs K, the blog's guided policies, indexability).
7. COMMANDS section 19 (CPU, ~1.5 h, built 2026-10-18): delay (C14): 19.1 time to the target in
   the frame, 19.2 the latency budget at equal quality, 19.3 continuous traffic.

## Expected reviewer objections

| objection | answer |
|---|---|
| Kodak only, PSNR only | CLIC and MS-SSIM / LPIPS (owed, item 2) |
| The codec is ~0.8 dB behind the best published one | every claim is relative; the abstraction and the schedulers hold on three code families |
| Gains are moderate | the claims are the abstraction, certified optimality and the CSI verdict, not a single gain |
| The digital baseline is weak | HEIC with tuned backoff, lost blocks counted, plus a 3 dB-gap code and capacity; ideal-CQI results reported too |
| TDMA, one user per slot | a slot is any resource unit (a group of RBs over a TTI); B units per slot, K users, is the regime of the index-policy theory (gap to the bound O(1/sqrt(K)) per user); measured for K = 8 ... 128 with B = K/8 (COMMANDS 18.3) |
| Why an index and not the ID policy of the recent theory (Hong et al.)? | both are the same relaxation with a different conflict rule; their theorems are for average reward, ours is a finite horizon where index policies need no attractor condition; we run the ID-guided policy and its random-order baseline beside the indices (COMMANDS 18.3): value ranking for smooth rewards, persistency (our admission rules) for all-or-nothing ones |
| Is the Whittle index well defined (indexability)? | the Lagrangian index (gain at the bound's prices) needs no indexability (Hu and Frazier; Brown and Smith) and is run beside it; the programs' grid check reports the share of non-monotone states (18.3); the index takes the top of the indifference range where now and later are alike |
| Real CSI is stale, not just noisy | the outdated-CSI model h_rep = r h + sqrt(1 - r^2) e, r = J0(2 pi f_D tau): the programs decide on the report and the bound still holds; JSCC vs digital, the scheduling gain and the lift against r (COMMANDS 18.1-18.2) |
| A scheduler that waits for peaks must add delay | the deadline is the latency budget: the same quality with a shorter frame (C14, 19.2); within the frame and with requests arriving over time, more images at the target within every delay d (19.1, 19.3). With deadlines that differ, the Whittle index (each request's program weighs urgency against the channel) keeps the success with much less delay (stand-in: ~40%), faster than PADC even on the same requests. Slower only for the first coarse image, which the deadline objective does not reward; a per-slot reward in the same programs covers that if an application needs it |
| Why not learn the scheduler (bandits, RL)? | the model is known: every user reports its channel each slot (full, not bandit, feedback) and the quality model is calibrated from 12 decodes per image; learning restless bandits address a different information structure; and any learned causal policy is capped by our bound (+0.26 dB mean) |
| "The MMSE form is SoftCast's" | yes for linear analog codes, by construction; the finding is that a learned nonlinear code obeys it only when trained for it, and that this, not MIESM, is the abstraction for scheduling learned codes |
| Analog symbols are not NR-compatible | a 6G mode (as all DeepJSCC work assumes); the scheduler and abstraction need only per-slot noise, so they carry over to quantised (DeepJSCC-Q-style) codes, to measure |
