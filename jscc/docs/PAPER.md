# Paper plan (IEEE TWC): condensed 2026-10-15, story revised 2026-10-18 to -20, rewritten after the related-work survey 2026-10-21

Every number is in `docs/NOTES.md` (section numbers below). Earlier versions of this plan are in
git: `git show ec9cc68:jscc/docs/PAPER.md` (before the restless-bandit framing) and
`git show 5025d5a:jscc/docs/PAPER.md` (the last version before this rewrite, with the survey's
batch-by-batch notes). The survey itself: `reports/Related work for JSCC scheduling.md` and
`research_notes/Related work for JSCC scheduling/fulltext_readings.md` (23 full texts).

## Working title

**Scheduling Learned Image Codes over Fading Multi-User Downlinks: An MMSE Link Abstraction,
Restless-Bandit Scheduling, and Scheduler-Aware Training** (short form: "A Link Abstraction for
Scheduling Learned Image Codes")

Dropped from the old title: "Near-Optimal Index Scheduling". The index and the bound are imported,
and at a quality target the measured gap does not yet shrink with the number of users.

## The story

Deep JSCC codes are designed and evaluated on one link. In a cellular downlink they would be
delivered slot by slot, by a scheduler, to many users, before a deadline. The scheduling theory
for that already exists: per-unit programs under a rate price (RaDiO, 2001), restless bandits for
layered video (Hosseini and Panwar), per-user programs coordinated by prices (Fu and van der
Schaar), and Lagrangian priority indices whose gap to a relaxation bound vanishes as users grow
(Hao, Xu and Tong; Brown and Smith; Zhang and Frazier; Gast, Gaujal and Yan). None of it can be
applied to a learned code, because none of it knows what a slot is worth to one. Digital systems
solved the same problem for OFDM with an effective-SNR link abstraction (EESM, MIESM). The
paper's thesis: **a learned code needs a link abstraction of its own, and has one when it is
trained for it; with it, the existing theory applies to learned codes, with measurable
consequences for scheduling, for training, and for the JSCC-versus-digital question.**

1. **The abstraction, the contribution everything rests on.** A learned prefix code trained for
   slot-varying SNR composes slots as a sensitivity-weighted sum of per-slot MMSE terms,
   D = Dsrc + C sum_j rho_j phi(n_j). On held-out profiles it errs by 0.055 dB, against 0.49 for
   MIESM and 0.19 for EESM; to our knowledge this is the first test of effective-SNR mappings on a
   learned code. The transmitter calibrates it from 12 decodes per image, and it predicts the
   decoded quality of scheduled histories within 0.05-0.09 dB. It holds only when the code is
   trained for slot-varying SNR: a constant-SNR-trained code obeys no rule (0.22 dB), because
   such a code codes its refinements against the base reconstruction it expects at one SNR (the
   mechanism DeepJSCC-l observed for layers). The form is SoftCast's for linear analog codes; new
   is that a nonlinear learned code acquires it, with learned per-phase weights, through training.
2. **What the abstraction buys the scheduler: a two-number user state.** A user becomes (phases
   held, accumulated weighted noise), and deadline scheduling of many users becomes a
   finite-horizon restless bandit with a new kind of arm: phases that accumulate across slots,
   each graded by its own slot's SNR, toward a terminal quality. (The nearest scheduler of a
   learned code, McDowell et al. 2026, sends a fresh codeword in every slot and has to learn its
   slot values by reinforcement learning.) The standard machinery then applies exactly, on a CPU:
   one dynamic program per user under per-slot prices, the Lagrangian bound on every causal
   scheduler, and the Lagrangian (LP-priority) index of Hao, Xu and Tong and Brown and Smith.
   Measured against that bound, for mean PSNR the index's total gap stays bounded as the cell
   grows (1.66 -> 0.64 dB from 8 to 128 users with K/8 units per slot, 0.005 dB per user at
   128), the regime Zhang and Frazier's theorem predicts, and it stays the best policy on stale
   reports. A closed-form rule, the index's certainty-equivalent version (a slot's predicted gain
   against a typical mean-SNR slot), captures most of it at small scale (+0.46 dB over PF,
   decoded; within 0.26 dB of the bound at 8 users) and plateaus at scale (0.15 dB per user), as
   certainty equivalence predicts. For a quality target, admitting the users with the smallest
   remaining need and stopping each online adds +11 points over PADC's rule (decoded), a
   combination with no precedent across users. Read as latency, the gains are a 9-19% shorter
   frame for the same mean PSNR and a 39% shorter one for the same share at 30 dB. **Open:** at
   30 dB the index's per-user gap does not shrink with the number of users (~1.2-1.9 points),
   which the theory says a policy that follows the relaxation cannot do. The relaxation's tie
   rule, a better price solver, re-solving and finer grids do not close it (COMMANDS 22, NOTES
   6d); the suspect left is the transmitter's curve errors, which the bound does not have
   (COMMANDS 23.1). Until that is settled, the target result is reported as measured, not as
   near-optimal.
3. **The system verdict: JSCC beats digital under the CSI real systems have.** Against HEVC intra
   with CQI-table link adaptation and incremental-redundancy HARQ, under the same schedulers:
   on par with fresh, error-free CQI (+0.3 dB, as theory and single-link studies predict),
   +1.3 dB / +13 points at 2 dB CQI error, and +1.7 dB / +15.5 points at report correlation
   0.8 (about 10 km/h with 5 ms-old reports at 3.5 GHz); the JSCC side holds decoded. JSCC loses
   0.5 dB from fresh reports to r = 0.8, digital with HARQ 1.9 dB. Stale reports change the
   scheduler too: the opportunistic part of every policy fades, and stopping at a target must
   decide on each slot's posterior with a 0.5-1 dB margin, not on its report. Layered
   (superposition) HARQ, the theoretical digital competitor, is designed for no transmit CSI;
   our chain has CQI, retransmits in newly faded slots and allows four rounds, so it is argued
   out, with an optional check at r <= 0.6.
4. **The reverse interface: the scheduler makes the code's channel.** With fresh reports,
   selection lifts the first phases ~5 dB above the physical channel. Training on that channel
   gives +0.28 dB on the same slots (+0.08-0.12 beyond length specialisation, two seeds); the
   codes learn their scheduler (kappa tracks the deep-fade rate, rho the favoured phase); one
   retraining round is the fixed point. No prior work trains a code on its scheduler's channel;
   the effect is small and a low-mobility one (the lift keeps 65% of itself at r = 0.8 and 8% at
   0.3, close to r^2).

A system choice runs through all four: encode once, with a decoder per length (C6), is the JSCC
system to use in a scheduled downlink; the mechanism is DeepJSCC-l's, the verdict ours.

The literature also turns four of our measurements into confirmed predictions, which is stronger
evidence than the numbers alone: the bounded total gap of the Lagrangian index for the mean
(Zhang and Frazier's non-degenerate O(1)); the plateau of the certainty-equivalent rule at scale;
the fragility of a Whittle index when reports say little; and the small payoff of matched
training under Rayleigh fading.

## What the survey changed (2026-10-20 to -21)

| before the survey | after |
|---|---|
| Headline: "our simple index is certified near-optimal" (C13) | The index and the bound are imported: Hao, Xu and Tong's Lagrangian priority policy (= Brown and Smith's optimal Lagrangian index = Gast et al.'s LP index) and Hawkins's relaxation. A relaxation value beside the policies, and against N, is standard evidence (Hosseini and Panwar; Hao, Xu and Tong; Shisher et al.; Ouyang et al.; Tang et al.). New: the arm it is applied to and the measured gap on a decoded reward. "Near-optimal" only for the mean; at the target, open |
| The abstraction as one of three pillars | The pillar: it is what lets the existing theory reach learned codes (the EESM analogy), and its dependence on training now has a mechanism (DeepJSCC-l; DRJSCC) |
| "No paper schedules deep JSCC codes slot by slot" | False as worded (McDowell et al. schedule a learned code per fading slot). Now: no paper schedules a learned prefix whose phases accumulate across slots |
| "The bound still holds on stale reports" as a contribution | The standard belief-state relaxation (Ouyang et al.). The contribution is the report model inside deadline programs and what follows from it: the opportunistic gain fades, stopping needs the posterior and a margin |
| The closed form as "our rule" | The certainty-equivalent approximation of the imported index, which explains its plateau; a PI-type ratio baseline (Ayesta et al.) is built (`pi`) and runs in COMMANDS 22 |
| Top-of-range Whittle index as a refinement | A deliberate departure from the standard definition (an infimum in five papers), justified by the programs' interpolation noise |
| C6 as our design | DeepJSCC-l's multiple-decoder mechanism; our claim is the system verdict |
| C5 as "the first comparison with HARQ" | Worded "to our knowledge" (the closest study, Mohajer et al., could not be read); layered HARQ argued out with one caveat; He et al. reach the same qualitative verdict for linear analog video on one link |
| The flat 30 dB scaling gap as an anomaly | A specific diagnosis: a total gap growing linearly in K means the policy does not follow the relaxation; our tie order among indifferent users is the likely cause; the price solver is ruled out as the main cause on the stand-in |
| (no entry) | New limitations to state: per-frame admission could starve persistent hard users (per-user debts, Hou, Borkar and Kumar, are the fix); runs with lost slots are outside the scaling theorems; repeating a phase with MRC (He et al.) is an extension we do not evaluate |

## Contributions and evidence

Claim IDs are stable (COMMANDS.txt and the other docs cite them). All "decoded" results are the
codec itself decoding the schedules. The novelty column is the survey's verdict.

| contribution | ID | claim as it will be written | key evidence | novelty after the survey | NOTES | status |
|---|---|---|---|---|---|---|
| 1. Link abstraction | C12 | Learned codes need their own link abstraction, and training decides whether they have one | sensitivity-weighted MMSE 0.055 dB vs MIESM 0.49, EESM 0.19; a constant-SNR-trained code obeys no rule (0.22 dB). CLIC (60 images): MMSE 0.083 vs MIESM 0.44, EESM 0.16, noise 0.135; s1 again obeys no rule | to our knowledge the first test of EESM / MIESM on a learned code; the training dependence is new (mechanism: DeepJSCC-l) | 2, 6d | done, Kodak and CLIC: the anchor |
| | C1 | A piecewise-trained prefix code decodes slot-split prefixes, and its distortion is additive over slots: D = Dsrc + C sum_j rho_j phi(n_j) | held-out 0.03-0.07 dB per image and phase (CLIC 0.05-0.11); constant-SNR decoders do not obey it (2x worse on both sets) | the form is SoftCast's (linear codes) and ABG's single-SNR shape; new for a nonlinear learned code, with learned per-phase weights | 1, 2 | done |
| | C2 | It is a usable link-to-system model | scheduled histories decoded within 0.05-0.09 dB, no bias, including the Lagrangian index's schedules (0.062-0.095) and CLIC's (0.076-0.087); calibrated from 12 decodes per image (0.03-0.06 dB; CLIC 0.05-0.09); holds for prefix, per-length decoders and per-CBR codes | new (single-SNR predictors exist: OraNet, ABG) | 2 | done |
| 2. Scheduling on it | C13 | The abstraction makes deadline scheduling of learned codes a finite-horizon restless bandit with a two-number state, so the standard relaxation, bound and Lagrangian index apply exactly; for mean PSNR the index's gap to the bound behaves as the theory predicts | Lagrangian index, B = K/8: per-user gap 0.21 -> 0.005 dB from K 8 to 128 (total bounded, 1.66 -> 0.64 dB); Whittle 0.27 -> 0.016; closed form plateaus at 0.15; PF 0.52; at K 8 the closed form is within 0.26 dB of every causal scheduler (the clairvoyant optimum, +0.23, is the tighter cap); on stale reports the bound holds and the Lagrangian index stays best (the Whittle index degrades) | machinery imported (RaDiO; Hosseini and Panwar; Hao, Xu and Tong; Brown and Smith; Gast et al.; Zhang and Frazier); new: the arm (accumulating, SNR-graded phases toward a terminal quality) and the measured gap on a decoded reward | 3, 6b, 6c, 6d | mean: done (main frames: the best policy in every setting, +0.48-0.52 dB over PF; decoded +0.06 [0.02, 0.10] over the closed form on Kodak, +0.12 [0.08, 0.17] on CLIC). Target: open (1.2-1.9 points from the bound at every K; the tie rule, prices, re-solving and grids ruled out in COMMANDS 22; the transmitter's curve errors tested in 23.1) |
| | C3 | Mean PSNR: the closed-form advantage index, the certainty-equivalent version of the Lagrangian index | +0.465 dB [0.39, 0.53] decoded over PF, +0.46 for the 5th-percentile user, robust over K, T, curve errors, correlation; content-aware lengths only +0.05. Stale reports: +0.385 / +0.256 dB over PF at r = 0.8 / 0.6 (aware transmitter +0.475 / +0.342; decoded +0.500 [0.40, 0.61] / +0.361 [0.25, 0.48]); all of it is how slots are valued (the content part is 0.000 at K 8, T 24) | certainty-equivalent rule (ratio-form relative: the PI index, which loses to PF here: -0.48 dB simulated, -0.96 decoded against ours, and starves users); the difference form in terminal-PSNR units inside a deadline frame, and the diagnosis of its plateau, are ours | 3, 5, 6b, 6d | done; MS-SSIM +0.006, LPIPS -0.012 over PF; CLIC +0.51 decoded |
| | C4 | Quality target: admission by smallest remaining need + stopping on realised SNRs + index timing | +11.0 points [8.3, 13.8] decoded over PADC's rule, +14 over PF; sizing +5, stopping +4, timing +3, admission +2. Stale reports: stopping on the reports breaks (naive online -7 points at r = 0.95); deciding on the posterior, planning the spare slots (plan_adv, aware) and a 0.5-1 dB margin keep +8 points over PADC at r = 0.8 and +6 at 0.6 (decoded without margin +10.6 [7.5, 14.0] / +5.2 [1.7, 8.8]); PADC is best at r = 0 | no precedent across users (ProgressFTX is single-device, Hou, Borkar and Kumar long-run); limitation: per-frame, not per-user across frames | 3, 6b, 6d | done; CLIC: online +5.2, the Lagrangian index +7.7 points over PADC decoded; the share costs the users who miss (perceptual metrics, NOTES 6d) |
| | C14 | Delay: the same gains read as latency | latency budget: our rule needs 9.3% [8.7, 9.9] / 18.6% [16.3, 20.7] fewer slots than PF for its mean PSNR at T 16 / 24; online 38.6% [34.9, 41.9] fewer than PADC for its share at 30 dB at T 16. Time to 30 dB: the Whittle index earliest (2.9 [2.7, 3.1] slots before PADC on the same users), the Lagrangian index ends highest (ahead of PADC at every slot). Continuous traffic: online 70-71% vs PADC 46%; the Whittle index 68% with about half online's delay. The first coarse image comes later (slot 6.7 vs PF 4.5) | metric forms known; new for scheduled learned codes | 3, 6c, 6d | done; the Lagrangian index needs 12.7% / 19.9% fewer slots than PF at T 16 / 24, and online 25% more than it at 30 dB at T 24 |
| | C7 | Which QoS policy when | the Lagrangian index with fresh, exact reports (+11.1 points over PADC; decoded +11.5-12.3); plan_adv under CQI errors (+8.3 vs the index's +1.6) and on stale reports (aware, 0.5-1 dB margin: +9.7 vs +6.1); the Whittle index or replan_adv with lost slots and tight frames, and the Whittle index when earliness matters; online when the users who miss must still get a good image | engineering guidance | 3, 5, 6b, 6c, 6d | done |
| 3. System verdicts | C5 | Against digital under realistic CSI | on par with error-free CQI (+0.3 dB). Against HEIC with tuned link adaptation and IR-HARQ: +0.9 / +1.3 / +1.6 dB and +8.5 / +12.9 / +15.1 points at 1 / 2 / 3 dB CQI error; on stale reports +1.3 / +1.7 / +1.7 dB and +11.8 / +15.5 / +14.2 points at r = 0.95 / 0.8 / 0.6; ahead of a 3 dB-gap code with HARQ from 2 dB of CQI error and from r = 0.95; HARQ took 0.3 / 0.75 dB off the leads without it; the JSCC side holds decoded at r = 0.8 / 0.6 | to our knowledge the first deep-JSCC comparison with CQI-table link adaptation and IR-HARQ under CQI errors and aged reports; the near-tie with exact CSI is predicted (AM-GM; 2609.02924) | 5 | done; layered HARQ check optional (item 11) |
| | C6 | Encode once + a decoder per length is the JSCC system to use in a scheduled downlink | best on mean PSNR and with lost slots (+0.78 dB, +1.9 points decoded vs per-CBR), ties per-CBR at the target; at equal budget a decoder per length recovers about half of the nesting penalty | mechanism: DeepJSCC-l's multiple-decoder architecture; new: the verdict inside a scheduled downlink | 4 | done |
| 4. The scheduler-made channel | C8 | Training for it pays | +0.28 dB on the same slots over the same-budget control; +0.08-0.12 beyond length specialisation (two seeds); nothing at the target. A fresh-report effect: the lift (+5.0 / +4.5 / +3.6 dB, phases 1-3) keeps 65% of itself at r = 0.8 and 8% at 0.3 | no precedent (the nearest co-design papers freeze their learning side); small, as matched-training studies predict | 6, 6b | holds, small; scoped to low mobility |
| | C9 | The physical channel is the wrong training channel | blind slots -0.04 dB vs generic training, -0.32 vs the scheduler's channel | new | 6 | done |
| | C10 | A code per scheduler; codes learn their scheduler; one round is the fixed point | PF-trained code wins under PF slots; kappa tracks the deep-fade rate, rho the favoured phase; channel unchanged after retraining | new | 6 | done |
| | C11 | It grows with the number of users | the lift grows like an order statistic; the decoded gain only +0.12 to +0.15 dB (K 8 to 16) | remark | 6 | weak: a remark |

## Positioning (docs/SURVEY.md; the survey report)

* **Known, and cited as such:** the Gaussian MMSE and SoftCast's additive per-component error for
  linear analog codes; effective-SNR link abstraction for digital OFDM (EESM, MIESM); single-SNR
  quality curves for learned codes (OraNet, ABG); single-link adaptive DeepJSCC (PADC,
  JSCCformer-f, DRJSCC, DeepJSCC-l); the scheduling machinery (RaDiO; Fu and van der Schaar;
  Hosseini and Panwar; Yu, Xu and Tong; Hao, Xu and Tong; Brown and Smith; Hu and Frazier; Zhang
  and Frazier; Gast, Gaujal and Yan; Shisher et al.); belief-state bounds under partial channel
  observation (Ouyang et al.); flow-level index rules (PI); layered HARQ (Steiner and Shamai).
* **New:** a learned nonlinear code acquires the additive MMSE form only when trained for
  slot-varying SNR, and that, not MIESM, is its link abstraction (C12, C1, C2); the arm it creates
  (C13) and the measured scheduling results on a decoded reward (C3, C4, C14); the JSCC-versus-
  digital verdict with link adaptation and HARQ under aged reports (C5); training on the
  scheduler's channel (C8-C10).
* **Nearest neighbours to contrast explicitly:** McDowell et al. 2026 (learned code, Lagrangian
  bandit, fresh codeword per slot, RL); Hosseini and Panwar (layered digital video, LP value as a
  benchmark); He et al. 2017 (progressive analog video, one stream); Hao, Xu and Tong (the index).
* **Kind of restless bandit:** a planning one (known model, every user reports every slot), not a
  learning one (Grunewalder and Khaleghi; Gornet and Sinopoli).
* **Wording for claims resting on unread texts** (owed item 12): "to our knowledge" for C12's
  first test (two 2026 EESM preprints) and C5's HARQ chain (Mohajer et al.); Fu and van der
  Schaar (JSAC 2010) cited from its abstract for a decomposition we concede.
* **Baselines in the paper:** RR, max-SNR, PF (the LTE/NR default), PADC's per-image rule, the
  exact optimum of the static allocation problem, the Whittle index, ID-guided and random-guided
  conflict rules, fixed-rate codes per CBR, encode once with per-length decoders, HEIC with LTE CQI
  link adaptation, tuned backoff and IR-HARQ, a 3 dB-gap code and capacity; for the abstraction
  mean dB, noise, MIESM and EESM; above all of them the Lagrangian bound and the clairvoyant
  optimum; a PI-type ratio rule (`pi`, item 10: COMMANDS 22).

## Why it is a TWC paper

TWC wants a wireless problem that matters, a non-obvious contribution backed by analysis, and a
fair, thorough evaluation. The survey changed what the analysis contributes, not whether there is
one:

* **A general, falsifiable finding** (C12): learned codes obey an MMSE link abstraction, not
  MIESM, and only when trained for it. Anyone doing system-level design with learned codes can
  reuse it, the role EESM played for OFDM.
* **It connects learned codes to a mature theory** (C13): with the abstraction, decades of
  decomposition and index-policy results apply to a new class of codes, exactly and cheaply, and
  their predictions check out on decoded schedules (bounded total gap, the certainty-equivalent
  plateau). Importing the theory is the gain: the heuristics became analysed policies with a
  bound, and where the measurements disagree (the 30 dB gap), the theory says what to fix.
* **A practical verdict** (C5): under realistic CSI, JSCC with these schedulers beats digital with
  tuned link adaptation and IR-HARQ by 1.3 dB / 13 points at 2 dB CQI error and 1.7 dB / 15.5
  points with reports as old as 10 km/h and 5 ms make them: a concrete answer to "why JSCC in a
  cellular downlink?".

The reverse interface (C8-C10) completes the picture: small in dB, clean in mechanism, and new.
The evidence standard is high throughout: decoded by the real codec, paired CIs, three code
families, two seeds where it matters, digital with tuned backoff, HARQ and lost blocks counted.

## Outline

1. Introduction: learned JSCC meets the MAC; the missing piece is a link abstraction; the two
   interfaces (what a slot is worth to the code; which slots the code sees); contributions 1-4.
2. Related work: DeepJSCC and variable rate (PADC, SwinJSCC, DeepJSCC-MIMO, JSCCformer-f,
   DeepJSCC-l); SNR-varying training (CFA-JSCC, DRJSCC, euJSCC); link abstraction (EESM, MIESM;
   single-SNR quality fits such as OraNet and ABG; SoftCast's linear MMSE); scheduling of media
   and learned codes (RaDiO, Hosseini-Panwar, Fu-van der Schaar, He et al., McDowell et al.);
   restless bandits and index policies (Whittle; Hao-Xu-Tong; Brown-Smith; Zhang-Frazier;
   Gast-Gaujal-Yan; Ouyang et al. for partial observation; Tang et al.; Shisher et al.); deadline
   QoS and stopping (Hou-Borkar-Kumar, ProgressFTX); JSCC against separation and layered HARQ
   (Steiner-Shamai). `docs/SURVEY.md`.
3. System model: K users, T slots, deadline, block fading, CQI with errors and aged reports, lost
   slots, encode-once prefix codes with per-length decoders; mean PSNR and share at a target.
4. The link abstraction: the model and why MMSE, the training condition and its mechanism,
   transmitter calibration, accuracy against MIESM / EESM / noise rules (C12, C1, C2).
5. Scheduling with the abstraction: the arm and its two-number state; the restless bandit; the
   standard relaxation, bound and Lagrangian index (imported, with the departures stated: price
   solver, tie rule, top-of-range Whittle index); the closed form as the certainty-equivalent
   version; QoS admission, stopping and planning; stale reports (the programs decide on the
   report); many units per slot, where the theory's regime is checked (C13, C3, C4, C7).
6. The scheduler-made channel: the induced distribution and its order statistics,
   scheduler-aware training, what the codes learn (C8-C11).
7. Evaluation: policies against PF / PADC and the bound, codes against per-CBR codes (C6), JSCC
   against digital under CQI errors and stale reports (C5); lost slots, correlated fading, K, T;
   delay (C14); complexity.
8. Conclusion and limits: the 30 dB gap (unless fixed), per-frame admission and fairness, lost
   slots outside the theorems, phase repetition, Kodak and PSNR (unless CLIC and MS-SSIM land).

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
2. Second test set (CLIC) and a perceptual metric (MS-SSIM / LPIPS) for the main tables. Built
   2026-10-22, COMMANDS 22.1 and 22.3: CLIC's mixed-SNR measurements (snrc4 and the s1 control,
   the first 60 test images), its nested utility and belief, the link abstractions compared, the
   main frames simulated and two schedules decoded; every Kodak schedule of 22.1 decoded with
   MS-SSIM and LPIPS (`mixed_snr.py --replay-metrics`; the decoded PSNR is unchanged) and paired
   by `replay_report.py --metric`. GPUs 1-4, ~2-3 h, if CLIC is on the server. Done 2026-10-25
   (NOTES 6d): C12, C1, C2 hold on CLIC; decoded gains of the same sign and size; MS-SSIM and
   LPIPS follow PSNR for the mean, and show the target policies' cost to the users who miss.
3. Done 2026-10-20: HARQ in the digital baseline (`schedule_sim.py --harq ir|chase`), measured
   in COMMANDS 20.2 (NOTES 5): C5 restated with it.
4. Optional: a Jacobian probe (the first-order mechanism behind section 4); a better codec
   (more data) for the absolute numbers.
5. Writing and figures.
6. Done 2026-10-19: COMMANDS 18 (stale reports, many users per slot; NOTES 5, 6b) and 19
   (delay; NOTES 6c).
7. Done 2026-10-20: COMMANDS 20 (HARQ, decoded stale-report schedules, the Lagrangian index's
   delay, margins; NOTES 5, 6b, 6c).
8. Done 2026-10-25 (NOTES 6d). COMMANDS section 21: the Lagrangian index as `schedule_sim.py --policies
   lagrange` on the main frames (mean and target, CQI errors, stale reports, lost slots, a
   tight frame), its latency budget, and its schedules (and the Whittle index's) decoded by the
   codec on GPUs 1-4. If it holds decoded, it is the paper's scheduler and our closed form its
   approximation.
9. The 30 dB gap (the Lagrangian index ~1.5 points below the bound at every K with B = K/8),
   after COMMANDS 21. A per-user gap that does not shrink is a total gap linear in K: the policy
   is not LP-compatible (Gast, Gaujal and Yan, Thm 2) or the bound is not the LP of the problem
   the policy faces; degeneracy alone would still give 1/sqrt(K) per user. Evidence so far: Hao,
   Xu and Tong measure flat gaps when ties among indifferent arms ignore the relaxation's
   fractions, Gast et al. prove that such a rule cannot be LP-compatible when several arms are
   mixed, and McDowell et al. report the same for the multi-action knapsack; the price solver's step does
   not scale with B (at B = 4 and 8 it never improved on the constant price), but on the smoke
   stand-in, where a 29 dB target binds (T 10), proper per-slot prices lower the bound by only
   0.07 / 0.15 points per user at K 8 / 32 (2026-10-21), so prices are not the main cause. No
   one-unit result changes (all of schedule_sim, COMMANDS 21). Plan, in order, in
   `tools/rmab_scale.py` (CPU): (a) the relaxation's tie rule: users indifferent at lambda*_t
   (zero gain within the programs' precision) are activated with the probability their own
   program assigns at the kink, so that expected demand meets the supply (water-filling /
   Brown and Smith's proportional split / Hao, Xu and Tong's phi*), with `demand()` mixing at the
   kink too; (b) the bound's discretisation of the 0/1 reward: finer Y grid and fade bins;
   (c) supply-scaled price steps; (d) LP-update: re-solve the prices from the current state
   every few slots. Pass: the 30 dB per-user gap shrinks with K (at least like 1/sqrt(K)).
   Built 2026-10-22 (COMMANDS 22.2a-d, CPU): (a) `lagrange_relax` in both tools (sure gains
   first, then the indifferent users at random, which activates every indifferent state with
   the same probability, as the relaxation's mix does; `alloc/whittle.kink_demand` gives that
   mix); (b) `--ny`, `--ng`; (c) `--price-scale supply` and `--kink-grad` (the defaults
   reproduce every earlier bound bit for bit); (d) `lagrange_lpu` (`--resolve N`). Every run
   prints the sure demand and the indifferent mass at lambda* as shares of the supply. Smoke
   stand-in (T 10, 29 dB, 30 frames): the indifferent users hold 38-44% of the supply, yet
   `lagrange_relax` schedules like `lagrange` (gap 6.67 / 3.96 vs 6.67 / 3.86 points per user at
   K 8 / 32), and there the gap already falls like 1/sqrt(K); finer grids move the stand-in's
   bound by 0.004 per user. So the stand-in cannot reproduce the flat gap; the real utility's
   runs decide.
   Run 2026-10-25 (NOTES 6d): none of the four closes it. The indifferent mass at lambda* is ~0 from
   K 16 on; `lagrange_relax` costs 1.75 points at one unit per slot and gains 0.27 at K 128;
   prices and re-solving change no schedule; finer grids raise the bound by 0.15 and make the
   policies worse. Last suspect: the policies decide on the calibrated belief while the bound
   and the scores use the fitted curves, which at a 0/1 target costs a fixed amount per user
   (a user stopped at its predicted threshold misses when the belief is optimistic). COMMANDS
   23.1 gives the policies the true curves, and tries 0.25 / 0.5 dB margins on the belief.
10. A PI-type ratio baseline (Ayesta, Erausquin and Jacko; formula known since 2026-10-20):
   serve the largest g_k(s) / E[(g_k(S') - g_k(s))^+], g_k(s) the predicted PSNR gain of user
   k's next phase at slot SNR s, S' from k's own fading; infinite at the top of the range, ties
   by the largest gain. In `schedule_sim.py` and `rmab_scale.py`, mean objective. Built
   2026-10-22 (`--policies pi`; S' on 32 equiprobable fades, aware SNRs on stale reports; both
   tools schedule alike with one unit per slot, a smoke check): COMMANDS 22.1 (decoded),
   22.2e (at scale), 22.2f (main frames). CPU. Done 2026-10-25 (NOTES 6d): it loses to PF (-0.48
   dB simulated, 1.06-1.34 dB from the bound at scale) and to our closed form by 0.96 dB decoded,
   and starves users; the paper cites it as the ratio form and shows why the difference form is
   the right one.
11. Optional: a two-layer superposition digital baseline with IR per layer (Steiner and Shamai),
   at report correlation r <= 0.6, where our 0-15 dB users sit in its best range. An ideal
   successively refinable source (base layer at R1, the quality of R1 + R2 when the refinement
   also decodes) makes it an optimistic stand-in for a scalable codec; power split and rates
   tuned per policy like the single-layer backoff. In `alloc/digital.py`. Only if a referee asks.
   CPU.
12. Wording of the claims that rest on texts the survey could not obtain (survey report, end of
   the reading section): "to our knowledge" for the first validation of EESM / MIESM on a
   learned code (two 2026 EESM preprints unread) and for the digital chain with CQI-table link
   adaptation plus IR-HARQ (Mohajer et al. unread); Fu and van der Schaar (JSAC 2010) cited from
   its abstract for the multi-user decomposition we concede anyway.


## Expected reviewer objections

| objection | answer |
|---|---|
| Kodak only, PSNR only | CLIC (60 images) repeats the abstraction and the decoded gains; MS-SSIM and LPIPS reported for every decoded table (NOTES 6d) |
| The codec is ~0.8 dB behind the best published one | every claim is relative; the abstraction and the schedulers hold on three code families |
| Gains are moderate | the claims are the abstraction, what it lets the existing theory do for learned codes, and the CSI verdict, not a single gain |
| The target result is not near-optimal | agreed, and the paper says so: at 30 dB the Lagrangian index's per-user gap to the bound does not shrink with K, which the theory says a policy that follows the relaxation cannot show; the four candidate causes (the tie rule among users indifferent at the bound's prices, the price solver, the bound's grids, no re-solving) are measured and ruled out; the transmitter's curve errors are tested last (owed item 9, COMMANDS 23.1) |
| The digital baseline is weak | HEIC with the LTE CQI table, the backoff (or, on stale reports, an offset on the rate of best expected goodput) tuned per policy, incremental-redundancy HARQ with up to 4 transmissions (Chase combining is worse), lost blocks counted, plus a 3 dB-gap code with the same HARQ and capacity; ideal-CQI results reported too. HARQ narrows the stale-report lead from +2.5 to +1.7 dB at r = 0.8; it does not close it. Layered (superposition) HARQ, the theoretical competitor without transmit CSI (Steiner and Shamai), gains most with no CSIT, a channel fixed between rounds and one retransmission; ours has CQI, newly faded retransmission slots and 4 IR rounds, where its gain nearly vanishes; a two-layer check at r <= 0.6 is ready to run if asked (owed item 11) |
| TDMA, one user per slot | a slot is any resource unit (a group of RBs over a TTI); with B = K/8 units per slot, the regime of the index-policy theory, the Lagrangian index's gap to the bound falls from 0.21 dB at K 8 to 0.005 at K 128, and the ordering of every policy is the same as with one unit (NOTES 6b) |
| Why an index and not the ID policy of the recent theory (Hong et al.)? | both are the same relaxation with a different conflict rule; their theorems are for average reward, ours is a finite horizon where index policies need no attractor condition; measured: for the mean the indices beat both guided policies at every K (0.005 vs 0.09-0.11 dB from the bound at K 128); at 30 dB ID beats its random-order baseline by +0.5-0.7 points (persistency) but stays behind the Lagrangian index (2.6 vs 1.5 points from the bound at K 128). The conflict rule matters here because of the deadline: for age of information, random selection among the users that want the slot is provably asymptotically optimal (Tang et al.), since a deferred user loses little; under a terminal all-or-nothing target it can lose its last good slot |
| Admission by smallest need starves hard users | within a frame it maximises the share at the target, the objective; across frames, a user who is always hard (cell edge) could be refused frame after frame. Per-user debts carried across frames (Hou, Borkar and Kumar's largest-debt-first, feasibility optimal for long-run delivery ratios) are the standard remedy and fit on top of any of our rules; our simulations redraw users every frame, so they do not test this, and the paper says so |
| The Lagrangian index and the bound are prior art | yes, and the paper says so: the policy is Hao, Xu and Tong's Lagrangian priority policy (with Hu and Frazier's and Brown and Smith's indices), the bound is Hawkins's Lagrangian relaxation, a relaxation value beside the policies is in Hosseini and Panwar's SVC bandit, and Shisher, Ji, Hou and Sun plot one against system scale beside their "net gain" policy (the Lagrangian index in another name); RaDiO had per-unit programs under a price in 2001. What is new is the arm they apply to (a learned prefix code graded by each slot's SNR, two-number state from the validated abstraction), the bound kept valid on stale reports, and its measured gap against K on a decoded reward. Departures from their theorem's conditions (approximate prices, Whittle-index ties) are stated with their measured cost |
| McDowell et al. (2026) already schedule a learned code with a restless bandit | they give each sensor a codelength per fading slot, but every slot carries a fresh, self-contained codeword of the newest observation, so nothing accumulates across slots; their reward decays with age, there is no deadline, CSI is perfect, and slot values are learned by RL because no abstraction compresses the state. Our arm is a prefix whose phases accumulate across slots toward a terminal quality, and the abstraction makes each user's program exact (a two-number state, solved on a CPU). Their transceiver is trained over -5 to 20 dB and frozen, so it never sees its scheduler's channel (C8-C10) |
| Is the Whittle index well defined (indexability)? | the Lagrangian index (gain at the bound's prices) needs no indexability and is the best policy we run, with fresh and stale reports and at every K; on the programs' grid 1-1.7% of states flip back as the price rises for the mean, 6-13% for the target; the Whittle index takes the top of the indifference range where now and later are alike, a deliberate departure from the standard infimum definition, because the bottom lets the programs' interpolation noise pick the users |
| Whittle indices and relaxation bounds under imperfect CSI already exist (Ouyang et al.) | yes: for channels whose state a scheduled user reveals at the end of the slot, with Markov memory, so scheduling buys information; their bound covers every policy with that information and their index policy meets it as N grows. Ours applies the same relaxation to a different information structure: every user reports every slot through a Gaussian report kernel over i.i.d. fading, so reports have no exploration value, and the frame has a deadline with a terminal quality reward. The claim is the model and what it predicts (the opportunistic part fades as r falls; stopping needs the posterior and a margin), not the kind of bound |
| Real CSI is stale, not just noisy | the outdated-CSI model h_rep = r h + sqrt(1 - r^2) e, r = J0(2 pi f_D tau): the programs decide on the report and the bound still holds (the standard belief-state relaxation). Measured: JSCC's lead over digital with HARQ grows with report age (+1.7 dB / +15.5 points at r = 0.8, about 10 km/h with 5 ms-old reports at 3.5 GHz); our gain over PF keeps 81% of itself at r = 0.8 (all of it with an aware transmitter; +0.50 dB decoded); quality-target stopping must use the posterior and a 0.5-1 dB margin; the scheduler-made channel is a low-mobility effect (NOTES 5, 6b) |
| A scheduler that waits for peaks must add delay | the deadline is the latency budget: the same mean PSNR with 9-19% fewer slots than PF, the same share at 30 dB with 39% fewer than PADC (C14). Within the frame both program indices bring users to 30 dB earlier than PADC at every slot: the Whittle index earliest (2.9 slots on the same users), the Lagrangian index with the highest final share (+2.8 points over Whittle, 1.25 slots later than it); with requests arriving over time the Whittle index keeps online's success with about half its delay. Our closed-form online rule does wait: it trails PADC early in the frame and wins at the deadline. Slower only for the first coarse image (slot 6.7 vs PF's 4.5), which the deadline objective does not reward; a per-slot reward in the same programs covers that if an application needs it |
| Why not learn the scheduler (bandits, RL)? | the model is known: every user reports its channel each slot (full, not bandit, feedback) and the quality model is calibrated from 12 decodes per image; learning restless bandits address a different information structure; and any learned causal policy is capped by our bound (+0.26 dB mean) |
| Why not repeat a phase that met a deep fade (He et al. retransmit important chunks and combine copies by MRC)? | a natural extension of our arm: a repeat of phase j adds its SNRs, so the state stays two numbers (Y changes by rho_j (phi(n_combined) - phi(n_j))) and the programs, the bound and the indices carry over; it needs a decoder that accepts repeats, which the encode-once prefix code does not train for. Not evaluated; future work |
| "The MMSE form is SoftCast's" | yes for linear analog codes, by construction; the finding is that a learned nonlinear code obeys it only when trained for it, and that this, not MIESM, is the abstraction for scheduling learned codes |
| Analog symbols are not NR-compatible | a 6G mode (as all DeepJSCC work assumes); the scheduler and abstraction need only per-slot noise, so they carry over to quantised (DeepJSCC-Q-style) codes, to measure |
