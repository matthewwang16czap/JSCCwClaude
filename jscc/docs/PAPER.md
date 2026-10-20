# Paper plan (IEEE TWC), condensed 2026-10-15, story revised 2026-10-18, -19 and -20

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
   those ideal actions plus a conflict rule. The programs' Lagrangian (LP-priority) index, Hao,
   Xu and Tong's Lagrangian priority policy applied to this arm (imported, not ours), is
   near-optimal where the theory says it should be: with B = K/8 units per slot its gap to the
   bound falls from 0.21 dB at 8 users to 0.005 dB at 128, and it stays the best policy on stale
   reports. Our closed-form rule, the index's certainty-equivalent form, captures most of it
   (+0.46 dB over PF and +11 points over PADC's rule, decoded; ties the Whittle index at 8 users,
   within 0.23 dB of the best causal scheduler there) but plateaus 0.15 dB below the bound at
   scale. Read as latency, the gains are a 9-19% shorter frame for the same mean PSNR and a 39%
   shorter one for the same share at 30 dB. At a quality target the Lagrangian index brings
   the most users there by the deadline (+2.8 points over the Whittle index) and is ahead of
   PADC at every slot (so far on other frames; COMMANDS 21 pairs them); the Whittle index is
   the earliest (2.9 slots before PADC on the same
   users; with requests arriving over time, our closed form's success with half its delay);
3. with fresh reports the scheduler makes the code's channel (selection lifts the first phases
   ~5 dB above random), the right channel to train for: +0.1 dB beyond rate specialisation, and
   the codes learn their scheduler's fade statistics; the lift keeps 65% of itself at report
   correlation 0.8 and 8% at 0.3, so this is a low-mobility effect;
4. under realistic CSI this JSCC system beats HEVC intra with tuned link adaptation and HARQ
   (incremental redundancy), the more so the older the reports: +0.3 dB / +3 points with fresh
   reports, +1.3 dB / +13 points at 2 dB CQI error, and +1.7 dB / +15.5 points at report
   correlation 0.8 (about 10 km/h with 5 ms-old reports at 3.5 GHz), ahead even of a code 3 dB
   from capacity with HARQ from r = 0.95; decoded by the codec, the JSCC side holds. JSCC
   degrades gracefully (0.5 dB from fresh reports to r = 0.8) where digital, even with HARQ,
   loses 1.9 dB. There the opportunistic part of every scheduler fades, and stopping rules must
   decide on the posterior of each slot, not on its report, and aim 0.5-1 dB above the target.

What sections 18-19 changed (2026-10-19; docs/NOTES.md 5, 6b, 6c): item 2's scheduler is now the
programs' index, with the closed form as its cheap approximation (they tie at 8 users with one
unit per slot; with many units the index reaches the bound and the closed form does not); the
theory's regime is confirmed, not argued. The delay claims rest on the latency budget (robust)
and on the Whittle index, whose time to the target dominates PADC's at every slot; the closed
form's online rule wins only at the deadline (it trails PADC early in the frame). Item 3 is
scoped to fresh reports, measured. Item 4 gains its physical axis, the paper's strongest system
result. New caveats, stated: stopping on reported SNRs breaks with stale reports (an aware
transmitter and plan_adv fix most of it), the Whittle index degrades on stale reports (the
Lagrangian index does not), and the first coarse image comes later than under PF.

What section 20 changed (2026-10-20; NOTES 5, 6b, 6c): digital now has HARQ, and item 4 is quoted
with it; HARQ took 0.3 dB off the lead at 2 dB CQI error and 0.75 dB at r = 0.8, so C5 holds as
the system result (+1.7 dB / +15.5 points at r = 0.8, not +2.5 / +19). The stale-report
schedules decode as simulated. The Lagrangian index is the scheduler to present: best share at
the target and ahead of PADC at every slot; the Whittle index is earlier, so the delay claim
names both (a trade-off between final share and earliness). On stale reports the target rule
gains a margin.

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
| 2. Scheduling on it | C3 | Mean PSNR: the advantage index | +0.465 dB [0.39, 0.53] decoded over PF, +0.46 for the 5th-percentile user, robust over K, T, curve errors, correlation; content-aware lengths only +0.05. Stale reports: +0.385 / +0.256 dB over PF at r = 0.8 / 0.6 (an aware transmitter: +0.475 / +0.342; decoded +0.500 [0.40, 0.61] / +0.361 [0.25, 0.48]); all of it is how slots are valued (the content part is 0.000 at K 8, T 24) | 3, 5, 6b | done |
| | C4 | Quality target: admission + stopping on realised SNRs + index timing | +11.0 points [8.3, 13.8] decoded over PADC's rule, +14 over PF; sizing +5, stopping +4, timing +3, admission +2. Stale reports: stopping on the reports breaks (naive online -7 points at r = 0.95, below PADC by 0.8); deciding on the posterior, planning the spare slots (plan_adv, aware) and a 0.5-1 dB margin keeps +8 points over PADC at r = 0.8 and +6 at 0.6 (decoded without margin: +10.6 [7.5, 14.0] / +5.2 [1.7, 8.8]); PADC is best at r = 0 | 3, 6b | done |
| | C13 | Near-optimal, with a theory | two-number user state, finite-horizon restless bandit; one program per user (relaxation) + a conflict rule; Lagrangian bound: our rule within 0.26 dB of any causal scheduler at K 8 (mean; the clairvoyant optimum, +0.23, is the tighter cap), 2-5.5 points at the target. In the theory's regime (B = K/8 units per slot) the Lagrangian index's gap falls 0.21 -> 0.005 dB from K 8 to 128 (faster than 1/sqrt(K)), Whittle 0.27 -> 0.016, our closed form plateaus at 0.15, PF 0.52; at 30 dB the Lagrangian index is 1.4-2.2 points from the bound, 2.8 above the Whittle index at K 8. On stale reports the bound holds and the Lagrangian index stays best (the Whittle index degrades: 0.15 dB below our rule at r = 0.4). Conflict rules: value ranking (indices) for the mean; for the target ID beats random by +0.5-0.7 points but trails the Lagrangian index | 3, 6b, 6c | done in rmab_scale; on the main frames and decoded: COMMANDS 21 |
| | C14 | Delay: the same gains as latency | latency budget (`tools/latency.py`): our rule needs 9.3% [8.7, 9.9] / 18.6% [16.3, 20.7] fewer slots than PF for its mean PSNR at T 16 / 24, online 38.6% [34.9, 41.9] fewer than PADC for its share at 30 dB at T 16 (PADC never reaches its T 24 share). Time to 30 dB in the frame (`--delay`): the Whittle index is the earliest (38 / 60 / 69 / 73% of users by T/4 ... T; PADC 20 / 48 / 61 / 65), 2.9 [2.7, 3.1] slots before PADC on the same users; the Lagrangian index ends highest (32 / 56 / 70 / 77%: ahead of PADC and online at every slot, 1.25 slots behind Whittle on the same users); online trails PADC in the first half and wins at the deadline. Continuous traffic (`tools/stream_sim.py`, 30 dB, D 24, the frame's load): online 70-71% vs PADC 46%, capacity at 60% >= 0.45 requests per slot vs never; the Whittle index 68% with about half online's delay (4.2 vs 8.0 slots), faster than PADC on the same requests. The first coarse image comes later (slot 6.7 vs PF 4.5) | 3, 6c | done; the Lagrangian index's latency budget: COMMANDS 21 |
| | C7 | Which QoS policy when | plan_adv with known slots, under CQI errors (online loses 13 points at 3 dB error) and on stale reports (aware, with a 0.5-1 dB margin: 71.1% vs online 67.4% at r = 0.8), replan_adv with lost slots; the Lagrangian index for the most users at the target, the Whittle index when earliness within the deadline matters | 3, 5, 6b, 6c | done |
| 3. The scheduler-made channel | C8 | Training for it pays | +0.28 dB on the same slots over the same-budget control; +0.08-0.12 beyond length specialisation (two seeds); nothing at the target. A fresh-report effect: the lift (+5.0 / +4.5 / +3.6 dB, phases 1-3) keeps 65% of itself at r = 0.8 and 8% at 0.3, so the claim is for low mobility | 6, 6b | holds, small; scoped to fresh reports |
| | C9 | The physical channel is the wrong training channel | blind slots -0.04 dB vs generic training, -0.32 vs the scheduler's channel | 6 | done |
| | C10 | A code per scheduler; codes learn their scheduler; one round is the fixed point | PF-trained code wins under PF slots; kappa tracks the deep-fade rate, rho the favoured phase; channel unchanged after retraining | 6 | done |
| | C11 | It grows with the number of users | the lift grows like an order statistic; the decoded gain only +0.12 to +0.15 dB (K 8 to 16) | 6 | weak: a remark |
| 4. System verdicts | C6 | Encode once + a decoder per length is the JSCC system to use (the mechanism is DeepJSCC-l's multiple-decoder architecture; the claim is the verdict inside a scheduled downlink) | best on mean PSNR and with lost slots (+0.78 dB, +1.9 points decoded vs per-CBR), ties per-CBR at the target; at equal budget a decoder per length recovers about half of the nesting penalty | 4 | done |
| | C5 | Against digital under realistic CSI | on par with error-free CQI (+0.3 dB). Against HEIC with tuned link adaptation AND incremental-redundancy HARQ: +0.9 / +1.3 / +1.6 dB and +8.5 / +12.9 / +15.1 points at 1 / 2 / 3 dB CQI error; on stale reports (r = J0(2 pi f_D tau)) +1.3 / +1.7 / +1.7 dB and +11.8 / +15.5 / +14.2 points at r = 0.95 / 0.8 / 0.6, about +1.7 dB at 10 km/h with 5 ms-old reports; ahead of a 3 dB-gap code with HARQ from 2 dB of CQI error (+1.0) and from r = 0.95 (+0.9). HARQ took 0.3 dB (2 dB error) and 0.75 dB (r = 0.8) off the leads without it (+1.6, +2.5). The JSCC side holds decoded at r = 0.8 / 0.6 | 5 | done |

## Prior work: what is known and what is new (docs/SURVEY.md, F-F4)

Updated by the cross-field survey of 2026-10-20 (`reports/Related work for JSCC scheduling.md`):
the per-user-program + price + index recipe is prior art (RaDiO, Fu and van der Schaar,
Hosseini and Panwar, Yu-Xu-Tong, Hao-Xu-Tong, Shisher et al.); what no work has is our arm (a
learned prefix code graded by each slot's SNR), the validated abstraction for it, the numerical
certificate, the stale-report program, the HARQ digital baseline and scheduler-aware training.

Full texts, batch 1 (2026-10-20;
`research_notes/Related work for JSCC scheduling/fulltext_readings.md`) sharpen three points. (a) Our Lagrangian index is Hao, Xu and Tong's
Lagrangian priority policy (IEEE TAC; "the greatest tax under which it is optimal to activate
the arm", future prices at the relaxation's optimum): cite it as the source. Their theorem uses
exact dual prices (an LP) and ties randomized by the relaxation's optimal policy; ours uses a
subgradient from the best constant price and Whittle-index ties, and the paper must say so.
(b) A relaxation value plotted beside the policies is standard (Hosseini and Panwar at 20 users,
calling it "the optimum"; Yu, Xu and Tong; Hao, Xu and Tong), so "the numerical certificate" is
new evidence for a new arm (finite horizon, per-slot prices, decoded reward, CIs, a scaling
study), not a new kind of evidence. (c) The top-of-range convention for the Whittle index is ours
(Yu, Xu and Tong take the bottom; Hao, Xu and Tong's sup is over a single point): one sentence,
justified by the interpolation-noise result. Fu and van der Schaar's file was their single-user
paper (arXiv 0812.1244); the multi-user one (0903.0207) is still unread.

Full texts, batch 2 (2026-10-20; same file) add four points. (d) The nearest scheduler of a
learned code is McDowell, Panebianco, Chakraborty and Sun (arXiv 2608.28441): a meta-learned VIB
code gets a codelength per sensor and fading slot from a Lagrangian multi-action bandit with
LP-priority ties, rewarded by a decoded task value. Every slot carries a fresh, self-contained
codeword, so nothing accumulates across slots; the horizon is infinite, CSI is perfect, and the
slot values are learned by RL (3.5 GPU-hours) for lack of an abstraction. The arm claim must
therefore say "phases that accumulate across slots"; the abstraction is what makes our programs
exact on a CPU. Their physical layer is trained over -5 to 20 dB and frozen: C8-C10 untouched.
(e) Encode once with a decoder per length is DeepJSCC-l's multiple-decoder architecture
(Kurka and Gündüz, TWC 2021): C6 is the system verdict on their mechanism. DeepJSCC-l also
supplies C12's mechanism: refinements trained at one SNR are coded against the base
reconstruction expected there and lose value when the SNR differs; DRJSCC shows a
one-SNR-per-image code degrading under per-block SNR. Neither tests a composition rule.
(f) Shisher, Ji, Hou and Sun (JSAIT 2023) rank by a "net gain", the advantage at lambda* (the
Lagrangian index), and plot the relaxation's bound against system scale: our scaling study is
new evidence of a known kind. (g) McDowell et al., citing a 2026 manuscript, report that the
per-slot knapsack with arbitrary ties is not asymptotically optimal: ties decide, as in Hao, Xu
and Tong (owed item 9). A BUPT case study (arXiv 2609.02924) finds digital ahead of SwinJSCC
from 7 dB up on AWGN with the SNR known exactly, which fits C5's ideal-CQI near-tie.

Full texts, batch 3 (2026-10-20; same file) narrow one claim and sharpen three. (h) The bound on
stale reports is the standard belief-state relaxation: Ouyang, Eryilmaz and Shroff (arXiv
1108.3768, INFOCOM 2012) bound every policy for ARQ-revealed Markov channels the same way, and
plot the bound against N (Whittle's index policy meets it by N = 50). C13's stale part is
therefore an application, new in its observation model (every user reports every slot through
a Gaussian kernel, i.i.d. fading, no exploration value) and its finite horizon, not in the kind
of guarantee. Their memoryless-channel result (feedback worthless, the index policy uniform
random; arXiv 1009.3959) is the counterpart of our r = 0 limit. (i) Le Ny, Feron and Dahleh
prove a relaxation bound tight (periodic switching within O(eps)) for Kalman-filter scheduling,
a deterministic problem; our frame cannot time-share, so its gap is measured. Le Ny et al. and
Ouyang et al. also define the Whittle index as an infimum (the bottom of the range): our
top-of-range choice is a deliberate departure, to state and justify. (j) The PI index is
c mu_n / sum_{m>n} q_m (mu_m - mu_n), infinite in the best state (Ayesta, Erausquin and Jacko);
RB and PB are approximate relatives, not special cases. Our PI-type baseline would be g_k(s) /
E[(g_k(S') - g_k(s))^+] (owed item 10). (k) ABG (Ma et al.) fits single-SNR quality curves of
learned codes with an exponent near 1 (three of four networks), the MMSE shape of our per-slot
term at one SNR; one SNR per image throughout, so the composition rule (C1, C12) stays ours.

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
  paper found schedules a deep JSCC prefix whose phases accumulate over the slots of a fading
  multi-user downlink; the nearest (McDowell et al. 2026) schedules a learned code per fading
  slot, but each slot carries a fresh codeword.
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
  (0.005 dB at 128 users). Heuristics became analysed policies. The machinery is imported
  (RaDiO, Hosseini and Panwar, Hao, Xu and Tong); what is new is the arm it now applies to.
* **A practical verdict** (C5): under the CSI errors real systems have, JSCC with these
  schedulers beats digital with tuned link adaptation and incremental-redundancy HARQ by 1.3 dB
  and 13 points at 2 dB CQI error, and by 1.7 dB and 15.5 points with reports as old as 10 km/h
  and 5 ms make them: a concrete answer to "why JSCC in a cellular downlink?".

The reverse interface (C8-C10) completes the picture: modest in dB, clean in mechanism. The
evidence standard is high throughout: decoded by the real codec, paired CIs, three code
families, two seeds where it matters, digital with tuned backoff and lost blocks counted.

## Outline

1. Introduction: learned JSCC meets the MAC; the two interfaces; contributions 1-4.
2. Related work: DeepJSCC and variable rate (PADC, SwinJSCC, DeepJSCC-MIMO, JSCCformer-f);
   SNR-varying training (CFA-JSCC, DRJSCC, euJSCC); semantic resource allocation; opportunistic
   scheduling and restless bandits (PF, RaDiO, Hosseini-Panwar, Yu-Xu-Tong, Hao-Xu-Tong,
   Shisher et al., Borkar et al.); learned codes under a scheduler (McDowell et al.); layered
   codes and a decoder per length (DeepJSCC-l); partially observed downlinks (Ouyang et al.);
   flow-level index rules (PI); proved-tight bounds (Le Ny et al.); link abstraction (EESM,
   MIESM; single-SNR quality fits such as ABG). `docs/SURVEY.md`.
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
3. Done 2026-10-20: HARQ in the digital baseline (`schedule_sim.py --harq ir|chase`), measured
   in COMMANDS 20.2 (NOTES 5): C5 restated with it.
4. Optional: a Jacobian probe (the first-order mechanism behind section 4); a better codec
   (more data) for the absolute numbers.
5. Writing and figures.
6. Done 2026-10-19: COMMANDS 18 (stale reports, many users per slot; NOTES 5, 6b) and 19
   (delay; NOTES 6c).
7. Done 2026-10-20: COMMANDS 20 (HARQ, decoded stale-report schedules, the Lagrangian index's
   delay, margins; NOTES 5, 6b, 6c).
8. COMMANDS section 21 (built 2026-10-20): the Lagrangian index as `schedule_sim.py --policies
   lagrange` on the main frames (mean and target, CQI errors, stale reports, lost slots, a
   tight frame), its latency budget, and its schedules (and the Whittle index's) decoded by the
   codec on GPUs 1-4. If it holds decoded, it is the paper's scheduler and our closed form its
   approximation.
9. The 30 dB gap (the Lagrangian index ~1.5 points below the bound at every K with B = K/8),
   after COMMANDS 21. Hao, Xu and Tong's full text shows the same flat gap when ties are not
   broken by the relaxation's own randomized policy; and the price solver's subgradient step
   does not scale with B; at B = 4 and 8 it never improved on the constant price (checked on the
   smoke stand-in, 2026-10-20: 0.007 dB per user for the mean; the stand-in's target does not
   bind). No one-unit result changes (all of schedule_sim, COMMANDS 21). Plan: supply-scaled
   steps (more of them), a tie rule drawn from the relaxation, finer fade bins in the programs,
   and frames on the programs' own discrete channel, in `tools/rmab_scale.py`. CPU.
10. A PI-type ratio baseline (Ayesta, Erausquin and Jacko; formula known since 2026-10-20):
   serve the largest g_k(s) / E[(g_k(S') - g_k(s))^+], g_k(s) the predicted PSNR gain of user
   k's next phase at slot SNR s, S' from k's own fading; infinite at the top of the range, ties
   by the largest gain. In `schedule_sim.py` and `rmab_scale.py`, mean objective. Not built. CPU.

## Expected reviewer objections

| objection | answer |
|---|---|
| Kodak only, PSNR only | CLIC and MS-SSIM / LPIPS (owed, item 2) |
| The codec is ~0.8 dB behind the best published one | every claim is relative; the abstraction and the schedulers hold on three code families |
| Gains are moderate | the claims are the abstraction, certified optimality and the CSI verdict, not a single gain |
| The digital baseline is weak | HEIC with the LTE CQI table, the backoff (or, on stale reports, an offset on the rate of best expected goodput) tuned per policy, incremental-redundancy HARQ with up to 4 transmissions (Chase combining is worse), lost blocks counted, plus a 3 dB-gap code with the same HARQ and capacity; ideal-CQI results reported too. HARQ narrows the stale-report lead from +2.5 to +1.7 dB at r = 0.8; it does not close it |
| TDMA, one user per slot | a slot is any resource unit (a group of RBs over a TTI); with B = K/8 units per slot, the regime of the index-policy theory, the Lagrangian index's gap to the bound falls from 0.21 dB at K 8 to 0.005 at K 128, and the ordering of every policy is the same as with one unit (NOTES 6b) |
| Why an index and not the ID policy of the recent theory (Hong et al.)? | both are the same relaxation with a different conflict rule; their theorems are for average reward, ours is a finite horizon where index policies need no attractor condition; measured: for the mean the indices beat both guided policies at every K (0.005 vs 0.09-0.11 dB from the bound at K 128); at 30 dB ID beats its random-order baseline by +0.5-0.7 points (persistency) but stays behind the Lagrangian index (2.6 vs 1.5 points from the bound at K 128) |
| The Lagrangian index and the bound are prior art | yes, and the paper says so: the policy is Hao, Xu and Tong's Lagrangian priority policy (with Hu and Frazier's and Brown and Smith's indices), the bound is Hawkins's Lagrangian relaxation, a relaxation value beside the policies is in Hosseini and Panwar's SVC bandit, and Shisher, Ji, Hou and Sun plot one against system scale beside their "net gain" policy (the Lagrangian index in another name); RaDiO had per-unit programs under a price in 2001. What is new is the arm they apply to (a learned prefix code graded by each slot's SNR, two-number state from the validated abstraction), the bound kept valid on stale reports, and its measured gap against K on a decoded reward. Departures from their theorem's conditions (approximate prices, Whittle-index ties) are stated with their measured cost |
| McDowell et al. (2026) already schedule a learned code with a restless bandit | they give each sensor a codelength per fading slot, but every slot carries a fresh, self-contained codeword of the newest observation, so nothing accumulates across slots; their reward decays with age, there is no deadline, CSI is perfect, and slot values are learned by RL because no abstraction compresses the state. Our arm is a prefix whose phases accumulate across slots toward a terminal quality, and the abstraction makes each user's program exact (a two-number state, solved on a CPU). Their transceiver is trained over -5 to 20 dB and frozen, so it never sees its scheduler's channel (C8-C10) |
| Is the Whittle index well defined (indexability)? | the Lagrangian index (gain at the bound's prices) needs no indexability and is the best policy we run, with fresh and stale reports and at every K; on the programs' grid 1-1.7% of states flip back as the price rises for the mean, 6-13% for the target; the Whittle index takes the top of the indifference range where now and later are alike |
| Whittle indices and relaxation bounds under imperfect CSI already exist (Ouyang et al.) | yes: for channels whose state a scheduled user reveals at the end of the slot, with Markov memory, so scheduling buys information; their bound covers every policy with that information and their index policy meets it as N grows. Ours applies the same relaxation to a different information structure: every user reports every slot through a Gaussian report kernel over i.i.d. fading, so reports have no exploration value, and the frame has a deadline with a terminal quality reward. The claim is the model and what it predicts (the opportunistic part fades as r falls; stopping needs the posterior and a margin), not the kind of bound |
| Real CSI is stale, not just noisy | the outdated-CSI model h_rep = r h + sqrt(1 - r^2) e, r = J0(2 pi f_D tau): the programs decide on the report and the bound still holds (the standard belief-state relaxation). Measured: JSCC's lead over digital with HARQ grows with report age (+1.7 dB / +15.5 points at r = 0.8, about 10 km/h with 5 ms-old reports at 3.5 GHz); our gain over PF keeps 81% of itself at r = 0.8 (all of it with an aware transmitter; +0.50 dB decoded); quality-target stopping must use the posterior and a 0.5-1 dB margin; the scheduler-made channel is a low-mobility effect (NOTES 5, 6b) |
| A scheduler that waits for peaks must add delay | the deadline is the latency budget: the same mean PSNR with 9-19% fewer slots than PF, the same share at 30 dB with 39% fewer than PADC (C14). Within the frame both program indices bring users to 30 dB earlier than PADC at every slot: the Whittle index earliest (2.9 slots on the same users), the Lagrangian index with the highest final share (+2.8 points over Whittle, 1.25 slots later than it); with requests arriving over time the Whittle index keeps online's success with about half its delay. Our closed-form online rule does wait: it trails PADC early in the frame and wins at the deadline. Slower only for the first coarse image (slot 6.7 vs PF's 4.5), which the deadline objective does not reward; a per-slot reward in the same programs covers that if an application needs it |
| Why not learn the scheduler (bandits, RL)? | the model is known: every user reports its channel each slot (full, not bandit, feedback) and the quality model is calibrated from 12 decodes per image; learning restless bandits address a different information structure; and any learned causal policy is capped by our bound (+0.26 dB mean) |
| "The MMSE form is SoftCast's" | yes for linear analog codes, by construction; the finding is that a learned nonlinear code obeys it only when trained for it, and that this, not MIESM, is the abstraction for scheduling learned codes |
| Analog symbols are not NR-compatible | a 6G mode (as all DeepJSCC work assumes); the scheduler and abstraction need only per-slot noise, so they carry over to quantised (DeepJSCC-Q-style) codes, to measure |
