# Paper plan (IEEE TWC)

The claims (C1-C14), the evidence behind each, and the answers to the objections a reviewer will
raise. The section plan, title and abstract are in `docs/PAPER_OUTLINE.md`; the Introduction,
sentence by sentence, in `docs/INTRO_OUTLINE.md`, and the source of each of its numbers in
`docs/INTRO_NUMBERS.md`; the figures and tables in `docs/FIGURES.md`; every other number in
`docs/NOTES.md` (section numbers below). The related-work survey: `docs/SURVEY.md` and
`reports/Related work for JSCC scheduling.md`.

**Shares of users.** At a quality target the result is a share: the percentage of users whose
decoded PSNR reaches 30 dB by the deadline. A difference between two shares is the difference of
the two percentages, in % of all users (70.6% against 54.8% is +15.8%). Differences are paired by
frame and computed from unrounded shares, so they can differ by 0.1 from the difference of the
rounded shares.

## Working title

**Scheduling Learned Image Codes over Fading Multi-User Downlinks: A Link Abstraction and a
Calibration-Aware Restless-Bandit Scheduler**

## The method: CLASP

**CLASP: Calibration-aware, Link-Abstraction-based Scheduling of Progressive codes.**

- *Link-abstraction-based*: every slot is valued for every user through the code's MMSE link
  abstraction, which makes each user a two-number state and the frame a finite-horizon restless
  bandit, solved with exact per-user programs, a Lagrangian bound and the Lagrangian index.
- *Calibration-aware*: at a quality target each program decides on a calibrated predictive
  distribution of the user's decoded quality, whose spread is the measured error of the
  prediction: the calibrated curves' (0.03-0.05 dB per prefix length) and, with noisy or aged
  reports, the reports' (in all 0.18-0.97 dB). On aged reports the programs decide on the
  posterior of the channel.
- *Progressive codes*: what it schedules, a learned code that decodes from any prefix of its
  phases.
- In the tools: `lagrange` for mean PSNR, `lagrange_soft --soft-sigma csi` at a target,
  `--cqi-aware` on aged reports.

Why this name. It carries the two ideas a reader must keep, the link abstraction that values
every slot and the calibration-awareness that makes the scheduler robust to the errors of its
own prediction, and it names what is scheduled. The word fits the contribution: the method
fastens a learned code to a cellular scheduler. No prior use of the name was found in wireless,
JSCC or semantic communications (searched 2026-11-01; the answer-set solver "clasp" belongs to
logic programming). Weighed against it:
- AURA (Abstraction-based, Uncertainty-aware Restless-bandit Allocation) names all three designs,
  but the word is used by many systems and means nothing here;
- CRISP (Calibrated Restless-bandit Index Scheduling of Progressive codes) puts the imported
  index in the name and leaves out the link abstraction the method rests on, and the word is
  crowded too;
- a JSCC-style name ("SchedJSCC") suggests a new code, while the contribution is the scheduler.

## The story

Deep JSCC codes are designed and evaluated on one link. In a cellular downlink they would be
delivered slot by slot, by a scheduler, to many users, before a deadline. The scheduling theory
for that exists: per-unit programs under a rate price (RaDiO), restless bandits for layered video
(Hosseini and Panwar), per-user programs coordinated by prices (Fu and van der Schaar), and
Lagrangian priority indices whose gap to a relaxation bound vanishes as users grow (Hao, Xu and
Tong; Brown and Smith; Zhang and Frazier; Gast, Gaujal and Yan). None of it can be applied to a
learned code, because none of it knows what a slot is worth to one. Digital systems solved the
same problem for OFDM with an effective-SNR link abstraction (EESM, MIESM). The thesis: **a
learned code needs a link abstraction of its own, and has one when it is trained for it; with
it, and with a reward that knows the error of its own prediction, the existing theory applies to
learned codes, with measurable consequences for scheduling, for training, and for the
JSCC-versus-digital question.**

1. **The abstraction, the contribution everything rests on.** A learned progressive code
   trained under slot-varying SNR composes slots as a sensitivity-weighted sum of per-slot MMSE
   terms, D = Dsrc + C Σ_j ρ_j φ(n_j). On held-out profiles it errs by 0.055 dB (CLIC 0.083),
   against 0.194 for EESM and 0.337 for MIESM given the same weights (CLIC 0.159, 0.290); to our
   knowledge this is the first test of effective-SNR mappings on a learned code. The transmitter
   calibrates it from 12 decodes per image, and it predicts the decoded quality of complete
   scheduled deliveries within 0.06-0.09 dB, bias within ±0.03 dB, with exact, noisy or aged
   reports. It holds only when the code is trained for slot-varying SNR: a code trained at one
   SNR per image obeys no rule (MMSE 0.222 dB, the best rule 0.150), because such a code codes
   its later phases against the base reconstruction it expects at that SNR (the mechanism
   DeepJSCC-l observed for layers). The form is SoftCast's for linear analog codes; new is that
   a nonlinear learned code acquires it, with learned per-phase weights, through training.
2. **What the abstraction buys the scheduler: CLASP.** A user becomes (phases held, accumulated
   weighted noise), and deadline scheduling of many users becomes a finite-horizon restless
   bandit with a new kind of arm: phases that accumulate across slots, each graded by its own
   slot's SNR, toward a terminal quality. (The nearest scheduler of a learned code, McDowell et
   al. 2026, sends a fresh codeword in every slot and learns its slot values by reinforcement
   learning.) The standard machinery then applies exactly, on a CPU: one dynamic program per
   user under per-slot prices, the Lagrangian bound on every causal scheduler, and the Lagrangian
   index. For mean PSNR CLASP's gap to the bound falls from 0.207 dB per user at K 8 to 0.005 at
   K 128 with K/8 units per slot (a total gap of 1.66 → 0.64 dB, bounded, the regime Zhang and
   Frazier's theorem predicts). At a quality target the abstraction's small calibration error
   becomes first-order: programs that take their calibrated curves as exact leave an extra
   1.7-2.0% of the users below 30 dB from K 16 to 128 (a uniform margin makes it worse).
   Programs that reward the probability of success under the prediction's measured error recover
   it (100% at K 8, 99% pooled over K 16-128): CLASP on the calibrated curves is as close to the
   bound as the hard index given the true curves (its share at 30 dB within ±0.3% of the bound
   from 16 users on, with the measured spread of 0.03-0.05 dB; with an assumed spread of 0.05,
   0.1 or 0.2 dB also within ±0.3% at K 32 and 128). Decoded, it brings 80.4% of the users to
   30 dB against the hard index's 78.8%, and misses no user it predicted there (0.0% against
   1.2%). With noisy or aged reports the measured spread includes the reports' error, and CLASP
   stays the best target policy at every CSI condition. Read as latency: for the same mean PSNR
   as CLASP in 24 slots, PF needs 6.0 more (C14).
3. **The system verdict: JSCC beats digital under the CSI real systems have.** Against HEVC intra
   with CQI-table link adaptation and incremental-redundancy HARQ, CLASP on the same frames,
   paired: near-even with exact CQI (+0.20 dB; 78.0% of the users at 30 dB against 74.4%), as
   theory and single-link studies predict; at 2 dB CQI error +1.19 dB and 75.5% against 62.7%;
   at report correlation 0.8 (about 9 km/h with 5 ms-old reports at 3.5 GHz) +1.56 dB and 70.6%
   against 54.8%; ahead at every CSI condition run. CLASP loses 0.57 dB from fresh reports to
   r = 0.8, digital 1.93 dB. Against an idealised code 3 dB from capacity with the same HARQ,
   CLASP is behind with exact reports (78.0% of the users at 30 dB against 80.4%) and ahead from
   1 dB CQI errors or r = 0.95 on (by 5.3-13.7% of the users). The JSCC side holds decoded
   (r = 0.8: +0.51 dB over PF, and 72.1% of the users at 30 dB against PADC's 59.6%; 2 dB CQI
   error: +0.51 dB, and 77.3% against 64.0%). Layered (superposition) HARQ, the theoretical digital competitor, is
   designed for no transmit CSI; our chain has CQI, retransmits in newly faded slots and allows
   four rounds, so it is argued out, with an optional check at r <= 0.6.
4. **The reverse interface: the scheduler makes the code's channel.** With fresh reports CLASP
   lifts the first three phases 4.2-5.6 dB above a blindly taken slot. Training on that channel
   gives +0.21 dB on the same slots under CLASP (two seeds; CLASP's lengths with blind slots gain
   nothing) and +0.28 dB under the closed form; a code per scheduler, and after co-design the
   closed form ties CLASP for the mean; the codes learn their scheduler (κ tracks the deep-fade
   rate, ρ the favoured phase); one retraining round is the fixed point. No prior work trains a
   code on its scheduler's channel; the effect is small and a low-mobility one (the lift keeps
   65% of itself at r = 0.8 and 34% at 0.6, close to r²).

A system choice runs through all four: encode once, with one decoder for every prefix length
(C6), is the JSCC system to use in a scheduled downlink, where a user's number of slots is not
known when its image is encoded; a decoder per length behind the shared encoder is an optional
~0.2 dB upgrade (DeepJSCC-l's mechanism; the verdict is ours).

The literature turns four of the measurements into confirmed predictions, which is stronger
evidence than the numbers alone: the bounded total gap of the Lagrangian index for the mean
(Zhang and Frazier's non-degenerate O(1)); the plateau of the certainty-equivalent rule at scale;
the fragility of a Whittle index when reports say little; and the small payoff of matched
training under Rayleigh fading.

## Contributions and evidence

Claim IDs are stable (COMMANDS.txt and the other docs cite them). "Decoded": the codec itself
decoded the schedules. The novelty column is the survey's verdict.

| contribution | ID | claim | key evidence | novelty | NOTES | status |
|---|---|---|---|---|---|---|
| 1. Link abstraction | C12 | Learned codes need their own link abstraction, and training decides whether they have one | sensitivity-weighted MMSE 0.055 dB against EESM 0.194 and MIESM 0.337 with the same weights (equal weights 0.367 / 0.373 / 0.487); CLIC (60 images) 0.083 against 0.159 and 0.290 (equal 0.407 / 0.374 / 0.439); the noise-variance rule 0.133 (CLIC 0.135); a code trained at one SNR per image obeys no rule (Kodak MMSE 0.222, the best rule 0.150; CLIC 0.203 / 0.147) | to our knowledge the first test of EESM / MIESM on a learned code; the training dependence is new (mechanism: DeepJSCC-l) | 2, 6d | done, Kodak and CLIC |
| | C1 | A piecewise-trained progressive code decodes slot-split prefixes, and its distortion is additive over slots: D = Dsrc + C Σ_j ρ_j φ(n_j) | held-out 0.03-0.07 dB per prefix length (CLIC 0.05-0.11); the constant-SNR code does not obey it (0.12-0.29 dB) | the form is SoftCast's (linear codes) and ABG's single-SNR shape; new for a nonlinear learned code, with learned per-phase weights | 1, 2 | done |
| | C2 | It is a usable link-to-system model | calibrated from 12 decodes per image: held-out mixed-SNR decodes within 0.03-0.06 dB per prefix length (CLIC 0.05-0.09); complete scheduled deliveries decoded within 0.06-0.09 dB, bias within ±0.03 dB, with exact, noisy or aged reports, Kodak and CLIC; per-length decoders and per-CBR codes obey it too (0.089 / 0.110 dB). The calibration error (0.03-0.05 dB) costs nothing measurable for the mean, and leaves an extra 1.7-2.0% of the users below a hard 30 dB target when a scheduler takes the curves as exact | new (single-SNR predictors exist: OraNet, ABG) | 2, 6e, 6f | done |
| 2. CLASP | C13 | The abstraction makes deadline scheduling of learned codes a finite-horizon restless bandit with a two-number state, so the relaxation, the bound and the Lagrangian index apply exactly; CLASP's gap to the bound behaves as the theory predicts, for both objectives | B = K/8: per-user gap 0.207 → 0.005 dB from K 8 to 128 (total 1.66 → 0.64 dB); at 30 dB, with the calibration-aware reward, a share 0.84% below the bound at K 8 and within ±0.3% of it from K 16 (the bound's own discretisation error: 0.15-0.17%), as the hard index given the true curves; the Whittle index 0.265 → 0.016 dB; the closed form plateaus at 0.15; PF 0.71 → 0.52. Main frames (Table III): +0.43 to +0.61 dB over PF simulated, +0.51 to +0.63 decoded | machinery imported (RaDiO; Hosseini and Panwar; Hao, Xu and Tong; Brown and Smith; Gast et al.; Zhang and Frazier); new: the arm (accumulating, SNR-graded phases toward a terminal quality) and the measured gap on a decoded reward | 3, 6b, 6c, 6d, 6e, 6f | done |
| | C7 | One target policy for every CSI condition: CLASP's calibration-aware reward with the measured spread | exact reports: the best target scheduler, users at 30 dB over PADC +12.7% (decoded +14.8%) with fresh reports and +9.5% (decoded +8.1%) on CLIC; within 0.4% of the calibration-aware Whittle index with 25% of slots lost (+11.2% against +11.6%) and at T 16 (+11.5% against +11.6%). Under CQI errors and aged reports (2000 frames, paired): +1.4% to +3.6% over the planner at every CSI condition, every interval above zero, as good as the best fixed spread; decoded +3.3% [1.7, 5.2] at 2 dB CQI error, +2.1% [-0.2, +4.4] at r = 0.8. The Whittle index when earliness matters | the reward is new; the spread is measured, not tuned | 6f, 6g, 6h, 6i, 6j | done |
| | C3 | The closed form, CLASP's index in certainty-equivalent form, captures most of it at small scale and plateaus at scale | +0.46 dB over PF simulated and decoded (Table IV), +0.46 for the 5th-percentile user (simulated); within 0.27 dB per user of the bound at K 8, 0.15 at K 128; at r = 0.8 +0.47 simulated, +0.51 decoded (Table IV) | certainty-equivalent rule (the ratio form, the PI index, loses to PF here: -0.48 dB simulated, -0.50 decoded); the diagnosis of its plateau is ours | 3, 5, 6b, 6d | supporting (Table IV) |
| | C4 | Quality target without programs: admission by smallest remaining need, stopping on realised SNRs, index timing | users at 30 dB over PADC: +8.0% simulated, +9.6% decoded (Table IV) | no precedent across users (ProgressFTX is single-device; Hou, Borkar and Kumar long-run); per-frame, not per-user across frames | 3, 6b, 6d | supporting (Table IV); CLASP replaces it |
| | C14 | Delay: the same gains read as latency | mean PSNR: CLASP needs 12.7% / 19.9% fewer slots than PF at T 16 / 24 (PF needs +6.0 slots at T 24); at 30 dB PADC needs more than 32 slots for CLASP's share at T 24, and 70% of users at 30 dB take CLASP 15.8 slots (online 20.4). Within a 24-slot frame CLASP brings 29.3 / 53.1 / 70.2 / 77.2% of users to 30 dB by slots 6 / 12 / 18 / 24, PADC 20.1 / 48.4 / 60.8 / 64.6%; the mean slot to 30 dB is 1.85 [1.72, 1.98] earlier (1.04 on the users both bring there). A user's first decodable phase comes later (slot 7.75 against 6.54) | metric forms known; new for scheduled learned codes | 6c, 6d, 6g, 6k | done |
| 3. System verdicts | C5 | Against digital under realistic CSI | CLASP, single decoder, paired over 2000 frames; mean PSNR, then the share of users at 30 dB. Exact CQI: +0.20 [0.17, 0.22] dB; 78.0% against 74.4% (+3.6% [3.1, 4.1]). Against HEIC with the LTE CQI table and IR-HARQ at 1 / 2 / 3 dB CQI error: +0.77 / +1.19 / +1.47 dB; 77.3 / 75.5 / 72.8% against 68.4 / 62.7 / 58.1% (+8.8 / +12.7 / +14.7%); at r = 0.95 / 0.8 / 0.6: +1.16 / +1.56 / +1.54 dB; 75.8 / 70.6 / 65.7% against 63.6 / 54.8 / 49.8% (+12.2 / +15.8 / +15.8%); ahead at every condition to r = 0; fresh → r = 0.8 costs CLASP 0.57 dB, digital 1.93. Against a 3 dB-gap code with the same HARQ: behind with exact CSI (78.0% against 80.4%, -2.4% [-2.9, -1.9]; 0.64 dB, unpaired), ahead from 1 dB CQI error or r = 0.95 (+5.3% to +13.7% of the users at 30 dB). A decoder per length adds about 0.2 dB and 1.4-2.5% of the users to the closed form's leads; the JSCC side holds decoded | to our knowledge the first deep-JSCC comparison with CQI-table link adaptation and IR-HARQ under CQI errors and aged reports; the near-tie with exact CSI is predicted (AM-GM) | 5, 6e, 6h, 6j, 6k | done; layered HARQ check optional |
| | C6 | Encode once, with one decoder for every prefix length, is the JSCC system to use in a scheduled downlink; a decoder per length is an optional upgrade | the single decoder carries every result. With known slots one model per CBR is 0.175 dB and 3.3% of the users at 30 dB ahead of it (decoded, 60 frames: 31.76 against 31.59 dB, 81.0% against 77.7%), but needs the length fixed before encoding: with 25% of slots lost the single decoder leads by 0.53 dB (simulated). A decoder per length behind the shared encoder adds 0.23 dB and 2.5-3.1% of the users (decoded: 80.2-80.8% against 77.7%) at 6x the decoder memory, and leads per-CBR by 0.78 dB and 1.9% of the users (decoded: 71.9% against 70.0%) with lost slots | mechanism: DeepJSCC-l's multiple decoders; new: the verdict inside a scheduled downlink | 4, 6e | done |
| 4. The scheduler-made channel | C8 | Training for it pays | under CLASP +0.211 [0.204, 0.219] dB on the same slots over the same-budget control, seed 2 +0.206; CLASP's lengths with blind slots -0.089; under the closed form +0.28 (+0.08-0.12 beyond length specialisation, two seeds); nothing at the target. A fresh-report effect: under CLASP the lift over a blind slot (+5.6 / +4.8 / +4.2 dB, phases 1-3) keeps 92 / 65 / 34 / 14% of itself at r = 0.95 / 0.8 / 0.6 / 0.4, close to r² | no precedent (the nearest co-design papers freeze their learning side); small, as matched-training studies predict | 6, 6b, 6h, 6k | holds, small; scoped to low mobility |
| | C9 | The physical channel is the wrong training channel | blind slots -0.04 dB against generic training and -0.32 against the scheduler's channel (closed form); under CLASP (its lengths, blind slots) -0.09 and -0.30 | new | 6, 6h | done |
| | C10 | A code per scheduler; codes learn their scheduler; one round is the fixed point | a PF-trained code wins under PF slots; under CLASP the CLASP-trained code wins (31.875 against the closed form's code 31.831), under the closed form the closed form's (31.887 against 31.838); after co-design the closed form ties CLASP for the mean (+0.013 [-0.013, +0.037] dB); κ tracks the deep-fade rate, ρ the favoured phase; the channel is unchanged after retraining | new | 6, 6h | done |
| | C11 | It grows with the number of users | the lift grows like an order statistic: phase 1 sits +0.7 / +2.4 / +3.8 / +4.9 / +5.4 dB above the user's mean SNR at K 2 / 4 / 8 / 16 / 24 under CLASP (the closed form +0.6 / +2.2 / +3.2 / +3.9 / +4.3); the decoded gain of training for it, beyond length specialisation, only +0.08-0.12 dB at K 8 and +0.15 at K 16 (closed form) | remark | 6, 6k | a remark |

## Positioning

* **Known, and cited as such:** the Gaussian MMSE and SoftCast's additive per-component error for
  linear analog codes; effective-SNR link abstraction for digital OFDM (EESM, MIESM); single-SNR
  quality curves for learned codes (OraNet, ABG); single-link adaptive DeepJSCC (PADC,
  JSCCformer-f, DRJSCC, DeepJSCC-l); the scheduling machinery (RaDiO; Fu and van der Schaar;
  Hosseini and Panwar; Yu, Xu and Tong; Hao, Xu and Tong; Brown and Smith; Hu and Frazier; Zhang
  and Frazier; Gast, Gaujal and Yan; Shisher et al.); belief-state bounds under partial channel
  observation (Ouyang et al.); flow-level index rules (PI); layered HARQ (Steiner and Shamai).
* **New:** a learned nonlinear code acquires the additive MMSE form only when trained for
  slot-varying SNR, and that, not MIESM, is its link abstraction (C12, C1, C2); the arm it creates
  and CLASP's calibration-aware reward, with the measured results on a decoded reward (C13, C7,
  C14); the JSCC-versus-digital verdict with link adaptation and HARQ under CQI errors and aged
  reports (C5); training on the scheduler's channel (C8-C10).
* **Nearest neighbours to contrast explicitly:** McDowell et al. 2026 (learned code, Lagrangian
  bandit, fresh codeword per slot, RL); Hosseini and Panwar (layered digital video, LP value as a
  benchmark); He et al. 2017 (progressive analog video, one stream); Hao, Xu and Tong (the index).
* **Kind of restless bandit:** a planning one (known model, every user reports every slot), not a
  learning one (Grunewalder and Khaleghi; Gornet and Sinopoli).
* **Wording for claims resting on texts the survey could not obtain:** "to our knowledge" for the
  first test of EESM / MIESM on a learned code (two 2026 EESM preprints unread) and for the
  digital chain with CQI-table link adaptation plus IR-HARQ (Mohajer et al. unread); Fu and van
  der Schaar (JSAC 2010) cited from its abstract for a decomposition we concede.
* **Baselines in the paper:** PF (the LTE/NR default) for mean PSNR and PADC's per-image rate
  selection for a target; the Lagrangian bound above every causal scheduler; HEIC with LTE CQI
  link adaptation, a tuned backoff and IR-HARQ, and a 3 dB-gap code with the same HARQ; fixed-rate
  codes per CBR and per-length decoders (C6); for the abstraction mean dB, noise, MIESM and EESM.
  The other schedulers built on the abstraction (the closed form, the planner, the Whittle index,
  a PI-type ratio rule) appear in one appendix table (Table IV).

## Why it is a TWC paper

TWC wants a wireless problem that matters, a non-obvious contribution backed by analysis, and a
fair, thorough evaluation:

* **A general, falsifiable finding** (C12): learned codes obey an MMSE link abstraction, not
  MIESM, and only when trained for it. Anyone doing system-level design with learned codes can
  reuse it, the role EESM played for OFDM.
* **It connects learned codes to a mature theory** (C13): with the abstraction, decades of
  decomposition and index-policy results apply to a new class of codes, exactly and cheaply, and
  their predictions check out on decoded schedules (the bounded total gap, the
  certainty-equivalent plateau). Where they disagreed (the 30 dB gap), the cause was the
  prediction's own error, and a reward that knows it closes the gap (C7).
* **A practical verdict** (C5): under realistic CSI, JSCC with CLASP beats digital with tuned link
  adaptation and IR-HARQ by 1.19 dB, and 75.5% of the users at 30 dB against 62.7%, at 2 dB CQI
  error, and by 1.56 dB, and 70.6% against 54.8%, with reports aged to r = 0.8, and beats an idealised 3 dB-gap code once reports are imperfect:
  a concrete answer to "why JSCC in a cellular downlink?".

The reverse interface (C8-C10) completes the picture: small in dB, clean in mechanism, and new.
The evidence standard: decoded by the real codec, paired intervals, three code families, two
seeds where it matters, digital with tuned backoff, HARQ and lost blocks counted.

## Open

1. Writing: the Introduction (paragraph 1 approved; paragraphs 2-7 under review), then Sections
   II-VI (`docs/PAPER_OUTLINE.md`).
2. Optional, only if a referee asks: a two-layer superposition digital baseline with IR per layer
   (Steiner and Shamai) at report correlation r <= 0.6; a Jacobian probe of the mechanism behind
   the abstraction; a better codec (more data) for the absolute numbers.

## Expected reviewer objections

| objection | answer |
|---|---|
| Kodak only, PSNR only | CLIC (60 images) repeats the abstraction and the decoded gains; MS-SSIM and LPIPS are reported for every decoded table (NOTES 6d) |
| The codec is ~0.8 dB behind the best published one | every claim is relative; the abstraction and the schedulers hold on three code families |
| Gains are moderate | the claims are the abstraction, what it lets the existing theory do for learned codes, and the CSI verdict, not a single gain |
| The target result is not near-optimal | it is: with the calibration-aware reward, CLASP on the transmitter's calibrated curves is within ±0.3% of the bound from 16 users on, as close as a hard target given the true curves. Taking the curves at face value leaves an extra 1.7-2.0% of the users below 30 dB from K 16 on (a uniform margin makes it worse), which the paper reports as a finding: a hard target makes the abstraction's calibration error first-order (NOTES 6e, 6f) |
| The digital baseline is weak | HEIC with the LTE CQI table, the backoff (or, on aged reports, an offset on the aware rate) tuned per policy, IR-HARQ with up to 4 transmissions (Chase combining is worse), lost blocks counted; ideal-CQI results reported too. With HARQ CLASP still leads at r = 0.8, by 1.56 dB and with 70.6% of the users at 30 dB against 54.8%. Against an idealised code 3 dB from capacity with the same HARQ, CLASP is behind with exact reports (78.0% of the users at 30 dB against 80.4%) and ahead by 5.3-13.7% of the users from 1 dB CQI error or r = 0.95 on. Layered (superposition) HARQ, the theoretical competitor without transmit CSI (Steiner and Shamai), gains most with no CSIT, a channel fixed between rounds and one retransmission; ours has CQI, newly faded retransmission slots and 4 IR rounds, where its gain nearly vanishes; a two-layer check at r <= 0.6 is ready if asked |
| TDMA, one user per slot | a slot is any resource unit (a group of RBs over a TTI); with B = K/8 units per slot, the regime of the index-policy theory, CLASP's gap to the bound falls from 0.21 dB at K 8 to 0.005 at K 128, and the ordering of every policy is the same as with one unit (NOTES 6b) |
| Why an index and not the ID policy of the recent theory (Hong et al.)? | both are the same relaxation with a different conflict rule; their theorems are for average reward, ours is a finite horizon where index policies need no attractor condition; measured: for the mean the indices beat both guided policies at every K (0.005 against 0.09-0.11 dB from the bound at K 128); at 30 dB (hard target) ID beats its random-order baseline at every K (by 0.14-0.75% of the users at 30 dB) but stays behind the Lagrangian index (2.6% against 1.5% below the bound at K 128). The conflict rule matters because of the deadline: under a terminal all-or-nothing target a deferred user can lose its last good slot |
| Admission by smallest need starves hard users | within a frame it maximises the share at the target, the objective; across frames a user who is always hard (cell edge) could be refused frame after frame. Per-user debts carried across frames (Hou, Borkar and Kumar) are the standard remedy and fit on top; our simulations redraw users every frame, so they do not test this, and the paper says so |
| The Lagrangian index and the bound are prior art | yes, and the paper says so: the policy is Hao, Xu and Tong's Lagrangian priority policy (with Hu and Frazier's and Brown and Smith's indices), the bound is Hawkins's Lagrangian relaxation, a relaxation value beside the policies is in Hosseini and Panwar's SVC bandit, and Shisher, Ji, Hou and Sun plot one against system scale; RaDiO had per-unit programs under a price in 2001. New are the arm they apply to (a learned progressive code graded by each slot's SNR, a two-number state from the validated abstraction), the calibration-aware reward, the bound kept valid on aged reports, and its measured gap on a decoded reward |
| McDowell et al. (2026) already schedule a learned code with a restless bandit | they give each sensor a codelength per fading slot, but every slot carries a fresh, self-contained codeword of the newest observation, so nothing accumulates across slots; their reward decays with age, there is no deadline, CSI is perfect, and slot values are learned by RL because no abstraction compresses the state. Our arm is a prefix whose phases accumulate across slots toward a terminal quality, and the abstraction makes each user's program exact. Their transceiver is trained over -5 to 20 dB and frozen, so it never sees its scheduler's channel (C8-C10) |
| Is the Whittle index well defined (indexability)? | CLASP uses the Lagrangian index (gain at the bound's prices), which needs no indexability; on the programs' grid 1-1.7% of states flip back as the price rises for the mean, 6-13% for the target; the Whittle index (an appendix comparison) takes the top of the indifference range, a deliberate departure from the standard infimum definition, because the bottom lets the programs' interpolation noise pick the users |
| Whittle indices and relaxation bounds under imperfect CSI already exist (Ouyang et al.) | yes: for channels whose state a scheduled user reveals at the end of the slot, with Markov memory, so scheduling buys information. Ours applies the same relaxation to a different information structure: every user reports every slot through a report kernel over i.i.d. fading, so reports have no exploration value, and the frame has a deadline with a terminal quality reward. The claim is the model and what follows from it (the opportunistic part fades as r falls; a target needs the posterior and the reports' error in the reward), not the kind of bound |
| Real CSI is aged, not just noisy | the outdated-CSI model h_rep = r h + sqrt(1 - r^2) e, r = J0(2π f_D τ): the programs decide on the posterior and the bound still holds. Measured: CLASP's lead over digital with HARQ grows with report age (at r = 0.8, about 9 km/h with 5 ms-old reports at 3.5 GHz: +1.56 dB, and 70.6% of the users at 30 dB against 54.8%); its gain over PF holds at r = 0.8 (+0.49 dB simulated, +0.51 decoded, against +0.52 fresh); at a target it decides on the posterior with the reports' error in its reward, no margin; the scheduler-made channel is a low-mobility effect (NOTES 5, 6b, 6j, 6k) |
| A scheduler that waits for good slots must add delay | the deadline is the latency budget: the same mean PSNR with 12.7-19.9% fewer slots than PF; PADC needs more than 32 slots for CLASP's share at 30 dB in 24. Within the frame CLASP is ahead of PADC at every slot and brings users to 30 dB 1.85 slots earlier on average. It is slower only for the first decodable phase (slot 7.75 against PADC's 6.54), which the deadline objective does not reward; a per-slot reward in the same programs covers that if an application needs it |
| Why not learn the scheduler (bandits, RL)? | the model is known: every user reports its channel each slot (full, not bandit, feedback) and the quality model is calibrated from 12 decodes per image; learning restless bandits address a different information structure (an unknown reward mean and transition kernel, learned online); and any causal policy, learned or not, is capped by the bound, 0.21 dB per user above CLASP at K 8 for the mean. A recent example: Zamir and Hou's Deep Index Policy (MOBIHOC '24) learns a multi-resource partial index by policy gradient when the kernel is unknown and needs ~1000-2000 steps of online interaction to reach a known-kernel baseline; our channel law and calibrated curves make that baseline available on step one |
| Why not repeat a phase that met a deep fade (He et al. retransmit important chunks and combine copies by MRC)? | a natural extension of the arm: a repeat of phase j adds its SNRs, so the state stays two numbers and the programs, the bound and the index carry over; it needs a decoder that accepts repeats, which the encode-once progressive code does not train for. Not evaluated; future work |
| "The MMSE form is SoftCast's" | yes for linear analog codes, by construction; the finding is that a learned nonlinear code obeys it only when trained for it, and that this, not MIESM, is the abstraction for scheduling learned codes |
| Analog symbols are not NR-compatible | a 6G mode (as all DeepJSCC work assumes); the scheduler and the abstraction need only per-slot noise, so they carry over to quantised (DeepJSCC-Q-style) codes, to measure |
