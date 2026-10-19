# Paper plan (IEEE TWC), condensed 2026-10-15, story revised 2026-10-18 and 2026-10-19

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
   those ideal actions plus a conflict rule. The programs' Lagrangian (LP-priority) index is
   near-optimal where the theory says it should be: with B = K/8 units per slot its gap to the
   bound falls from 0.21 dB at 8 users to 0.005 dB at 128, and it stays the best policy on stale
   reports. Our closed-form rule, the index's certainty-equivalent form, captures most of it
   (+0.46 dB over PF and +11 points over PADC's rule, decoded; ties the Whittle index at 8 users,
   within 0.23 dB of the best causal scheduler there) but plateaus 0.15 dB below the bound at
   scale. Read as latency, the gains are a 9-19% shorter frame for the same mean PSNR and a 39%
   shorter one for the same share at 30 dB; and the Whittle index brings users to 30 dB 2.9 slots
   before PADC at the same final share (with requests arriving over time: our closed form's
   success with half its delay);
3. with fresh reports the scheduler makes the code's channel (selection lifts the first phases
   ~5 dB above random), the right channel to train for: +0.1 dB beyond rate specialisation, and
   the codes learn their scheduler's fade statistics; the lift keeps 65% of itself at report
   correlation 0.8 and 8% at 0.3, so this is a low-mobility effect;
4. under realistic CSI this JSCC system beats HEVC intra with tuned link adaptation, the more so
   the older the reports: +0.3 dB / +3 points with fresh reports, +1.6 dB / +15 points at 2 dB
   CQI error, and +2.5 dB / +19 points at report correlation 0.8 (about 10 km/h with 5 ms-old
   reports at 3.5 GHz), ahead even of a code 3 dB from capacity from r = 0.95 (digital without
   HARQ so far; with it, COMMANDS 20.2, the lead will be smaller). There the opportunistic part
   of every scheduler fades, and stopping rules must decide on the posterior of each slot, not
   on its report.

What sections 18-19 changed (2026-10-19; docs/NOTES.md 5, 6b, 6c): item 2's scheduler is now the
programs' index, with the closed form as its cheap approximation (they tie at 8 users with one
unit per slot; with many units the index reaches the bound and the closed form does not); the
theory's regime is confirmed, not argued. The delay claims rest on the latency budget (robust)
and on the Whittle index, whose time to the target dominates PADC's at every slot; the closed
form's online rule wins only at the deadline (it trails PADC early in the frame). Item 3 is
scoped to fresh reports, measured. Item 4 gains its physical axis, the paper's strongest system
result. New caveats, stated: stopping on reported SNRs breaks with stale reports (an aware
transmitter and plan_adv fix most of it), the Whittle index degrades on stale reports (the
Lagrangian index does not), and the first coarse image comes later than under PF. Open: the
digital baseline drops a block the slot could not carry; with HARQ (built the same day,
COMMANDS 20.2) it recovers part of that, and item 4's stale-report lead is restated with it.

Earlier (2026-10-18, the doc "Scheduling Learned Image Codes as a Restless Bandit",
https://claude.ai/code/artifact/0801c93d-9404-4fb3-acd8-e88c03641f8f): item 2 restated as
relaxation + conflict rule with its regime; the index takes the top of its indifference range;
the positioning adds planning vs learning restless bandits; delay read as the latency budget,
time to the target, and continuous traffic (docs/PROBLEM.md section 3, item 7).

## Contributions and evidence

Claim IDs are stable (COMMANDS.txt and the other docs cite them). All "decoded" results are the
codec itself decoding the schedules.

| contribution | ID | claim | key evidence | NOTES | status |
|---|---|---|---|---|---|
| 1. Link abstraction | C1 | A piecewise-trained prefix code decodes slot-split prefixes, and its distortion is additive over slots: D = Dsrc + C sum_j rho_j phi(n_j) | held-out 0.03-0.07 dB per image and phase; constant-SNR decoders do not obey it | 1, 2 | done |
| | C2 | It is a usable link-to-system model | scheduled histories decoded within 0.05-0.09 dB, no bias; calibrated from 12 decodes per image (0.03-0.06 dB); holds for prefix, per-length decoders and per-CBR codes | 2 | done |
| | C12 | Learned codes need their own abstraction, and training decides it | sensitivity-weighted MMSE 0.055 dB vs MIESM 0.49, EESM 0.19; a constant-SNR-trained code obeys no rule | 2 | done: the anchor |
| 2. Scheduling on it | C3 | Mean PSNR: the advantage index | +0.465 dB [0.39, 0.53] decoded over PF, +0.46 for the 5th-percentile user, robust over K, T, curve errors, correlation; content-aware lengths only +0.05. Stale reports: +0.385 / +0.256 dB over PF at r = 0.8 / 0.6 (an aware transmitter: +0.475 / +0.342); all of it is how slots are valued (the content part is 0.000 at K 8, T 24) | 3, 6b | done; decoded at r = 0.8 / 0.6 pending (COMMANDS 20.1) |
| | C4 | Quality target: admission + stopping on realised SNRs + index timing | +11.0 points [8.3, 13.8] decoded over PADC's rule, +14 over PF; sizing +5, stopping +4, timing +3, admission +2. Stale reports: stopping on the reports breaks (naive online -7 points at r = 0.95, below PADC by 0.8); deciding on the posterior and planning the spare slots (plan_adv, aware) keeps +8 points over PADC at r = 0.8, +5 at 0.6; PADC is best at r = 0 | 3, 6b | done; margins on stale reports pending (20.3) |
| | C13 | Near-optimal, with a theory | two-number user state, finite-horizon restless bandit; one program per user (relaxation) + a conflict rule; Lagrangian bound: our rule within 0.26 dB of any causal scheduler at K 8 (mean; the clairvoyant optimum, +0.23, is the tighter cap), 2-5.5 points at the target. In the theory's regime (B = K/8 units per slot) the Lagrangian index's gap falls 0.21 -> 0.005 dB from K 8 to 128 (faster than 1/sqrt(K)), Whittle 0.27 -> 0.016, our closed form plateaus at 0.15, PF 0.52; at 30 dB the Lagrangian index stays ~1.5 points from the bound. On stale reports the bound holds and the Lagrangian index stays best (the Whittle index degrades: 0.15 dB below our rule at r = 0.4). Conflict rules: value ranking (indices) for the mean; for the target ID beats random by +0.5-0.7 points but trails the Lagrangian index | 3, 6b | done |
| | C14 | Delay: the same gains as latency | latency budget (`tools/latency.py`): our rule needs 9.3% [8.7, 9.9] / 18.6% [16.3, 20.7] fewer slots than PF for its mean PSNR at T 16 / 24, online 38.6% [34.9, 41.9] fewer than PADC for its share at 30 dB at T 16 (PADC never reaches its T 24 share). Time to 30 dB in the frame (`--delay`): the Whittle index is ahead at every slot (38 / 60 / 69 / 73% of users by T/4 ... T; PADC 20 / 48 / 61 / 65), 2.9 [2.7, 3.1] slots before PADC on the same users, at online's final share (its index ~ success gain per remaining phase: shortest remaining need first); online trails PADC in the first half and wins at the deadline. Continuous traffic (`tools/stream_sim.py`, 30 dB, D 24, the frame's load): online 70-71% vs PADC 46%, capacity at 60% >= 0.45 requests per slot vs never; the Whittle index 68% with about half online's delay (4.2 vs 8.0 slots), faster than PADC on the same requests. The first coarse image comes later (slot 6.7 vs PF 4.5) | 3, 6c | done; the Lagrangian index's time to the target pending (20.2) |
| | C7 | Which QoS policy when | plan_adv with known slots, under CQI errors (online loses 13 points at 3 dB error) and on stale reports (aware: 70.3% vs online 65.3% at r = 0.8), replan_adv with lost slots; the Whittle index when delay within the deadline matters | 3, 5, 6b, 6c | done |
| 3. The scheduler-made channel | C8 | Training for it pays | +0.28 dB on the same slots over the same-budget control; +0.08-0.12 beyond length specialisation (two seeds); nothing at the target. A fresh-report effect: the lift (+5.0 / +4.5 / +3.6 dB, phases 1-3) keeps 65% of itself at r = 0.8 and 8% at 0.3, so the claim is for low mobility | 6, 6b | holds, small; scoped to fresh reports |
| | C9 | The physical channel is the wrong training channel | blind slots -0.04 dB vs generic training, -0.32 vs the scheduler's channel | 6 | done |
| | C10 | A code per scheduler; codes learn their scheduler; one round is the fixed point | PF-trained code wins under PF slots; kappa tracks the deep-fade rate, rho the favoured phase; channel unchanged after retraining | 6 | done |
| | C11 | It grows with the number of users | the lift grows like an order statistic; the decoded gain only +0.12 to +0.15 dB (K 8 to 16) | 6 | weak: a remark |
| 4. System verdicts | C6 | Encode once + a decoder per length is the JSCC system to use | best on mean PSNR and with lost slots (+0.78 dB, +1.9 points decoded vs per-CBR), ties per-CBR at the target; at equal budget a decoder per length recovers about half of the nesting penalty | 4 | done |
| | C5 | Against digital under realistic CSI | on par with error-free CQI; at 1 / 2 / 3 dB CQI error +1.0 / +1.6 / +2.3 dB and +9 / +15 / +19 points over HEIC; on stale reports (r = J0(2 pi f_D tau); digital with tuned backoff or the goodput-optimal rate) +1.4 / +2.5 / +3.3 dB and +12 / +19 / +20 points at r = 0.95 / 0.8 / 0.6, about +2.6 dB at 10 km/h with 5 ms-old reports; ahead of a 3 dB-gap code from 1 dB of CQI error and from r = 0.95. All against digital WITHOUT HARQ: with HARQ-IR (rates tuned for it) a toy codec's digital gains +0.9 dB at 2 dB CQI error and +2.2 at r = 0.8, so both axes may shrink to about +1 dB or less | 5 | at stake: with HARQ (20.2) and decoded (20.1) pending |

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
  on every causal scheduler; with many units per slot the programs' index reaches that bound
  (0.005 dB at 128 users). Heuristics became analysed policies.
* **A practical verdict** (C5; at stake until COMMANDS 20.2 adds HARQ to digital, see
  ASSESSMENT 19): under the CSI errors real systems have, JSCC with these
  schedulers beats digital with tuned link adaptation by 1.6 dB and 15 points at 2 dB CQI
  error, and by 2.5 dB and 19 points with reports as old as 10 km/h and 5 ms make them (digital
  without HARQ; with it pending): a concrete answer to "why JSCC in a cellular downlink?".

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
   and many users per slot (the Lagrangian index reaches the bound; the closed form is its
   cheap approximation) (C3, C4, C7, C13).
6. The scheduler-made channel: the induced distribution and its order statistics,
   scheduler-aware training, what the codes learn (C8-C11).
7. Evaluation: policies against PF / PADC, codes against per-CBR codes (C6), JSCC against
   digital under CQI errors and stale reports (C5); lost slots, correlated fading, K, T; delay
   (C14: the latency budget at equal quality, time to the target, continuous traffic);
   complexity.
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
10. JSCC's lead over digital against report age, with a speed x report-delay axis
   (`s18_stale_jscc.txt`, `s18_stale_dig.txt`); the scheduling gain and the lift against r
   (`s18_bank_stale.txt`).
11. Gap to the bound against K with B = K/8 units: Lagrangian, Whittle, ours, PF, guided
   policies (`s18_scale.txt`).

## Owed before submission

1. Online phase-to-slot mapping rule (sequential assignment): the largest piece our scheduler
   leaves (+0.13 dB / +3-4 points ex post). CPU; I build it.
2. Second test set (CLIC) and a perceptual metric (MS-SSIM / LPIPS) for the main tables. GPU,
   about a day, if CLIC is on the server.
3. HARQ with combining in the digital baseline: built 2026-10-19 (`schedule_sim.py --harq ir|
   chase`: the user's next slot retransmits a lost block and the receiver combines, up to 4
   transmissions; the backoff, or an offset on the aware rate, tuned with it). On a toy codec,
   with its rates tuned (bolder than the report), digital's best policy gains +2.2 dB at r = 0.8
   and +3.1 at 0.6, half of its stale-report loss: C5's stale-report lead may shrink to about
   +1 dB. Measured in COMMANDS 20.2.
4. Optional: a Jacobian probe (the first-order mechanism behind section 4); a better codec
   (more data) for the absolute numbers.
5. Writing and figures.
6. Done 2026-10-19: COMMANDS 18 (stale reports, many users per slot; NOTES 5, 6b) and 19
   (delay; NOTES 6c).
7. COMMANDS section 20 (built 2026-10-19): 20.1 the stale-report schedules decoded by the codec
   (GPUs 1-4, ~30 min); 20.2 C5 with HARQ, stale reports and CQI errors (CPU, ~1.5 h); 20.3 the
   time to the target of the Lagrangian index and the guided policies; 20.4 margins for
   quality targets on stale reports, and the r = 0 aware rerun after the tie fix.

## Expected reviewer objections

| objection | answer |
|---|---|
| Kodak only, PSNR only | CLIC and MS-SSIM / LPIPS (owed, item 2) |
| The codec is ~0.8 dB behind the best published one | every claim is relative; the abstraction and the schedulers hold on three code families |
| Gains are moderate | the claims are the abstraction, certified optimality and the CSI verdict, not a single gain |
| The digital baseline is weak | HEIC with tuned backoff (or, on stale reports, the rate of best expected goodput), lost blocks counted, plus a 3 dB-gap code and capacity; ideal-CQI results reported too; with HARQ (incremental redundancy or Chase combining, COMMANDS 20.2) |
| TDMA, one user per slot | a slot is any resource unit (a group of RBs over a TTI); with B = K/8 units per slot, the regime of the index-policy theory, the Lagrangian index's gap to the bound falls from 0.21 dB at K 8 to 0.005 at K 128, and the ordering of every policy is the same as with one unit (NOTES 6b) |
| Why an index and not the ID policy of the recent theory (Hong et al.)? | both are the same relaxation with a different conflict rule; their theorems are for average reward, ours is a finite horizon where index policies need no attractor condition; measured: for the mean the indices beat both guided policies at every K (0.005 vs 0.09-0.11 dB from the bound at K 128); at 30 dB ID beats its random-order baseline by +0.5-0.7 points (persistency) but stays behind the Lagrangian index (2.6 vs 1.5 points from the bound at K 128) |
| Is the Whittle index well defined (indexability)? | the Lagrangian index (gain at the bound's prices) needs no indexability and is the best policy we run, with fresh and stale reports and at every K; on the programs' grid 1-1.7% of states flip back as the price rises for the mean, 6-13% for the target; the Whittle index takes the top of the indifference range where now and later are alike |
| Real CSI is stale, not just noisy | the outdated-CSI model h_rep = r h + sqrt(1 - r^2) e, r = J0(2 pi f_D tau): the programs decide on the report and the bound still holds. Measured: JSCC's lead over digital grows with report age (+2.5 dB / +19 points at r = 0.8, about 10 km/h with 5 ms-old reports at 3.5 GHz); our gain over PF keeps 81% of itself at r = 0.8 (all of it with an aware transmitter); quality-target stopping must use the posterior; the scheduler-made channel is a low-mobility effect (NOTES 5, 6b) |
| A scheduler that waits for peaks must add delay | the deadline is the latency budget: the same mean PSNR with 9-19% fewer slots than PF, the same share at 30 dB with 39% fewer than PADC (C14). Within the frame the Whittle index brings users to 30 dB earlier than PADC at every slot (2.9 slots on the same users) at our rule's final share, and with requests arriving over time keeps online's success with about half its delay. Our closed-form online rule does wait: it trails PADC early in the frame and wins at the deadline. Slower only for the first coarse image (slot 6.7 vs PF's 4.5), which the deadline objective does not reward; a per-slot reward in the same programs covers that if an application needs it |
| Why not learn the scheduler (bandits, RL)? | the model is known: every user reports its channel each slot (full, not bandit, feedback) and the quality model is calibrated from 12 decodes per image; learning restless bandits address a different information structure; and any learned causal policy is capped by our bound (+0.26 dB mean) |
| "The MMSE form is SoftCast's" | yes for linear analog codes, by construction; the finding is that a learned nonlinear code obeys it only when trained for it, and that this, not MIESM, is the abstraction for scheduling learned codes |
| Analog symbols are not NR-compatible | a 6G mode (as all DeepJSCC work assumes); the scheduler and abstraction need only per-slot noise, so they carry over to quantised (DeepJSCC-Q-style) codes, to measure |
