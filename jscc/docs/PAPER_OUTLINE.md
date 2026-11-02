# Paper outline (IEEE TWC): title, abstract, section summaries

Six sections: Introduction, Related Work, Preliminaries, Method, Experiments, Conclusion. The
Introduction is planned sentence by sentence in `docs/INTRO_OUTLINE.md`; the source of each of
its numbers is in `docs/INTRO_NUMBERS.md`; the claims behind the paper are in `docs/PAPER.md`
(C1-C14); every other number is in `docs/NOTES.md`; the figures and tables are listed in
`docs/FIGURES.md`. The system is the single progressive code: one encoder and one decoder for
every prefix length. Per-length decoders are an optional upgrade.

---

## Title

**Scheduling Learned Image Codes over Fading Multi-User Downlinks: A Link Abstraction and a
Calibration-Aware Restless-Bandit Scheduler**

Alternatives:
- *CLASP: Calibration-Aware Scheduling of Progressive Deep JSCC Codes with a Learned-Code Link
  Abstraction*
- *What Is a Slot Worth to a Learned Code? Link Abstraction and Deadline Scheduling for Deep JSCC*

## Abstract (TWC: one paragraph, at most 250 words, no citations)

Deep joint source-channel coding (JSCC) is designed and evaluated on single links, yet a cellular
downlink serves many users slot by slot, through a scheduler, before a deadline. Scheduling needs
each slot's value to each user; no model provides it for a learned code. We show that a progressive
learned code trained under slot-varying SNR obeys a link abstraction: its distortion is the source
distortion plus a sensitivity-weighted sum of per-slot minimum mean-square-error terms. On two test
sets it predicts held-out decodes within 0.055-0.083 dB, against 0.16-0.19 dB for EESM and 0.29-0.34
dB for MIESM given the same weights; a constant-SNR-trained code obeys no such rule. Calibrated from
12 decodes per image, the abstraction reduces each user to a two-number state, and deadline
scheduling becomes a finite-horizon restless bandit. We propose CLASP, a calibration-aware,
link-abstraction-based scheduler that solves it with exact per-user dynamic programs, a Lagrangian
bound on every causal scheduler and an index policy, and rewards the probability of reaching a
quality target under the measured error of its own prediction, from the abstraction and reports.
Decoded, CLASP gains 0.5-0.6 dB of mean PSNR over proportional fair and brings an extra 8-15% of
users to 30 dB over per-image rate selection, with fresh, noisy or aged reports. Against digital
transmission with CQI-table link adaptation and HARQ it leads by 1.2-1.6 dB and 13-16% of users at
30 dB under 2 dB CQI errors or aged reports. Training the code on the channel CLASP creates adds
0.21 dB.

**Index Terms:** deep joint source-channel coding, link abstraction, multi-user scheduling,
restless bandits, Lagrangian relaxation, index policies, imperfect CSI, HARQ, image
transmission.

---

## I. Introduction (~1.5 pages)

Seven paragraphs, in the order of the supervisor's template: `docs/INTRO_OUTLINE.md` (with
Figure 1, the references, and the source of every number in `docs/INTRO_NUMBERS.md`).

## II. Related Work (~1 page)

The same two categories as the Introduction's paragraph 2, then the scheduling theory CLASP
builds on.

- **A. Physical link design for multi-user media delivery.**
  - A standard codec over adaptive modulation and coding from CQI reports, with IR-HARQ, and a
    channel-aware scheduler (proportional fair).
  - Effective-SNR link abstraction for coded blocks over many resources: EESM, MIESM, and the
    NR physical-layer abstraction for system-level simulation.
  - Media-aware scheduling: RaDiO (per-unit programs under a rate price); content-aware packet
    scheduling (Pahalawatta et al.); price-coordinated per-user programs (Fu and van der
    Schaar); restless bandits for scalable video (Hosseini and Panwar).
  - Every one of these rests on a binary outcome per block. Under imperfect CSI: outdated-CSI
    models (Jakes correlation), HARQ with Chase and IR combining; layered HARQ (Steiner and
    Shamai) is designed for no transmit CSI, and we say why it is not the competitor here.
- **B. The JSCC way.**
  - DeepJSCC and its successors (SwinJSCC, ViT-based codes); single-link adaptation to the SNR
    and the budget: one model for all SNRs, DeepJSCC-l (successive refinement with several
    decoders), NTSCC, PADC (a quality predictor that picks each image's rate), JSCCformer-f
    (feedback), DeepJSCC-MIMO.
  - Multi-user allocation by each user's predicted quality at one SNR (semantic spectral
    efficiency; user association and bandwidth; Bayesian optimisation over a learned quality
    oracle); non-orthogonal access built into the code; schedulers learned by deep
    reinforcement learning; a restless bandit whose slot values are learned (McDowell et al.:
    every slot carries a fresh codeword, nothing accumulates across slots).
  - Linear analog codes (SoftCast) obey an additive per-component MMSE by construction; our
    form is theirs, and what is new is that a nonlinear learned code acquires it, only through
    training.
  - JSCC against separation on single links: near-parity with exact CSI, measured with an
    idealised digital side.
- **C. Restless bandits and Lagrangian index policies.**
  - Whittle; Hawkins's Lagrangian decomposition.
  - Finite-horizon index policies whose gap to the relaxation vanishes as users grow: Hu and
    Frazier; Brown and Smith; Zhang and Frazier; Gast, Gaujal and Yan; the Lagrangian priority
    policy of Hao, Xu and Tong; PI indices (Ayesta et al.); relaxations under partial channel
    observation (Ouyang et al.).
  - Learning restless bandits solve a different problem (unknown reward and transition kernel,
    learned online; e.g. Zamir and Hou's Deep Index Policy). CLASP is the planning kind: the
    channel law and the calibrated curves are known, so each user's program is solved exactly.
  - We import this machinery and say so. Ours are the arm it applies to (phases that accumulate
    across slots, each graded by its own slot's SNR), the calibration-aware reward, and the
    evidence on a decoded reward.
- A positioning table closes the section: columns are learned code, multi-user, slot-varying
  SNR, deadline, bound, imperfect CSI.

## III. Preliminaries: System Model and Problem Statement (~1.5 pages)

- **Downlink frame.**
  - K users, T slots, B resource units per slot (B = 1 is TDMA; B = K/8 in the scaling study).
  - A unit carries one *phase* of one user's code.
  - Block Rayleigh fading per user and slot; mean SNRs drawn from U[0, 15] dB.
- **CSI models.**
  - Exact CQI.
  - Gaussian CQI error with standard deviation σ dB.
  - Outdated reports h_rep = r h + sqrt(1-r^2) e, with r = J0(2π f_D τ).
  - Units lost to other traffic with probability p.
- **The progressive code.**
  - An encoder maps the image to 6 phases (CBR 1/48 each, up to 1/8).
  - Any prefix of L phases decodes. Phase j crosses the channel at its own slot's SNR.
  - Encode once, one decoder for every L.
- **Objectives.**
  - Mean PSNR over users at the deadline.
  - The share of users reaching a target Q (30 dB).
  - Causal policies only: the current slot's report, never a future one.
- **Digital baseline.**
  - HEVC intra (HEIC) at the rate the LTE CQI table allows, with a tuned backoff (or, on aged
    reports, an offset on the aware transmitter's rate).
  - IR-HARQ up to four transmissions; lost blocks are counted.
  - An idealised code 3 dB from capacity with the same HARQ.
- **Notation table.**

## IV. Method (~4 pages)

- **A. Training the progressive code for slot-varying SNR.**
  - ViT progressive code: patch 8, width 768, 10 blocks.
  - Multi-rate training over prefix lengths.
  - Piecewise-SNR training: each step splits the prefix into chunks at independent SNRs.
  - One encoder and one decoder serve every rate; the encoder and decoder never see the SNR.
- **B. The MMSE link abstraction.**
  - D_k(L) = Dsrc_k(L) + C_k(L) Σ_j ρ_j φ(n_j), with n_j = 1/SNR_j and φ(n) = n/(1+n/κ).
    PSNR = -10 log10 D.
  - A nested form: one master sensitivity profile ρ for every L.
  - Fit from decodes under constant, one-phase-at-a-time and random SNR profiles.
  - Transmitter calibration: κ and ρ pooled over other images; each image's own Dsrc and C from
    12 decodes (2 SNRs x 6 prefix lengths).
  - Comparison rules: mean dB, noise variance, capacity (MIESM-like) and EESM, each with equal
    or sensitivity weights.
  - Why the training decides it: a code trained at one SNR per image codes its later phases
    against the base reconstruction it expects at that SNR (the mechanism DeepJSCC-l observed
    for layers).
- **C. CLASP: deadline scheduling as a finite-horizon restless bandit.**
  - *State.* User k's state is (n, Y): phases held and accumulated weighted noise. Serving at
    slot SNR g gives Y ← Y + m_{n+1} φ(noise(g)). The terminal reward is the PSNR, or the
    calibration-aware reward of D at a target.
  - *Relaxation and bound.* Price each unit at λ_t. The frame decomposes into K single-user
    dynamic programs on a (Y, fading) grid. The Lagrangian bound (1-p)B Σ_t λ_t + Σ_k W_k(λ)
    caps every causal scheduler; per-slot prices are found by subgradient.
  - *Policy.* The Lagrangian index: each user's gain at λ*_t; equal gains (within the programs'
    precision) are ordered by the Whittle index.
- **D. Imperfect information: the calibration-aware reward and aged reports.**
  - At a target Q the terminal reward is Φ((q + μ_n − Q)/σ_n): the probability that the true
    PSNR reaches Q when the prediction q is off by N(μ_n, σ_n²) at n phases. The spread is the
    measured error of the transmitter's own prediction, measured offline over images and random
    slot histories: with exact reports its calibrated curves against the fitted ones (0.03-0.05
    dB per prefix length); under CQI errors or aged reports its curves read at the SNRs it
    decides on (the reports, or their posterior) against the truth at the true SNRs (0.18-0.97
    dB). Exact Bayes for a Gaussian prediction error: the decisions reveal nothing about it
    within the frame, so only the terminal reward changes. The bound keeps the hard reward on
    the true curves.
  - *Aged reports.* The programs decide on the posterior of the channel through a
    report-to-channel kernel, so the bound stays valid for every scheduler that sees only
    reports.
- **E. Training the code on the channel CLASP creates.**
  - Record the per-phase SNR histories CLASP delivers and retrain the code on them.
  - One retraining round is the fixed point.
- **Complexity.** Per-user programs cost O(T P n_y n_g) on a CPU. At K 8 the bound and every
  policy take 8.9 s per frame and worker process, 21.3 s with the calibration-aware reward
  (`tools/rmab_scale.py`).

## V. Experiments (~5 pages)

- **Setup.**
  - Training on DIV2K. Testing on Kodak (24 images) and CLIC (the first 60 test images).
  - Metrics: PSNR, plus MS-SSIM and LPIPS of every decoded schedule.
  - Main frames: K 8, T 24; 2000 simulated frames for the CSI study, 500 for the main tables.
  - Decoded: the codec decodes 60 frames of every policy's schedules (150 for scheduler-aware
    training).
  - Every comparison is paired by frame, with 95% bootstrap intervals.
  - Each figure shows a contribution: CLASP against the baseline it replaces (PF for mean
    PSNR, PADC for a target). How the schedulers built on the abstraction compare with each
    other is one appendix table (Table IV). `docs/FIGURES.md` lists every figure and table.
- **A. Link abstraction accuracy** (Table I: mean absolute error per rule and code; fig02: how
  much each phase's slot matters, and what piecewise training costs at constant SNR; fig03: the
  model's error on every decoded schedule).
  - Kodak: MMSE 0.055 dB, against EESM 0.194 and MIESM 0.337 with the same sensitivity
    weights (equal weights: 0.367, 0.373, 0.487); the best other rule is the noise variance,
    0.133. CLIC: MMSE 0.083, EESM 0.159, MIESM 0.290 (equal: 0.407, 0.374, 0.439).
  - A decoder per length and one model per CBR obey it too (MMSE 0.089 and 0.110 dB).
  - The constant-SNR-trained code obeys no rule: MMSE 0.222 dB, the best rule 0.150 (CLIC: MMSE
    0.203, the best 0.147).
  - Calibrated from 12 decodes per image, the curves predict held-out mixed-SNR decodes within
    0.03-0.06 dB per prefix length (CLIC 0.05-0.09).
  - Scheduled deliveries, decoded: mean absolute error 0.06-0.09 dB, bias within ±0.03 dB,
    with exact, noisy (2 dB CQI errors) or aged (r = 0.8) reports, Kodak and CLIC.
- **B. Mean PSNR: near the bound, and over PF** (fig04a: the gap to the bound per user against
  K, B = K/8; Table III: CLASP over PF under each condition, simulated and decoded, with MS-SSIM
  and LPIPS; fig08a: what it does to the users, decoded; Table IV, appendix: the other
  schedulers).
  - CLASP's gap to the bound goes 0.207 → 0.005 dB per user from K 8 to 128 (total 1.66 → 0.64
    dB). The Whittle index 0.265 → 0.016; the closed form plateaus at 0.15; PF 0.71 → 0.52.
  - Main frames (Table III, simulated, 500 frames): +0.52 dB over PF with fresh reports, +0.48
    at 2 dB CQI error, +0.49 at r = 0.8, +0.48 with 25% of slots lost, +0.43 at T 16, +0.61 on
    CLIC. Decoded: +0.52 (Kodak), +0.51 (2 dB), +0.51 (r = 0.8), +0.63 (CLIC); MS-SSIM and
    LPIPS move with PSNR.
  - Table IV: CLASP's index is the best mean-PSNR scheduler in every column (tied with the
    closed form decoded at r = 0.8). The closed form is its certainty-equivalent version, which
    explains its plateau at scale. The PI ratio index loses even to PF (-0.48 dB simulated,
    -0.50 decoded): the ratio form fails on a saturating quality.
- **C. Quality target and latency** (fig04b: the gap per user against K; Table III: users at
  30 dB over PADC; fig08b: the share over 30 dB and the quality of the users below it, decoded;
  fig12: the frame length each needs (a, b) and users at 30 dB by each slot (c)).
  - Table III, users at 30 dB over PADC, simulated: +12.7% with fresh reports, +10.5% at 2 dB
    CQI error, +10.3% at r = 0.8, +11.2% with 25% of slots lost, +11.5% at T 16, +9.5% on CLIC.
    Decoded: +14.8% (Kodak), +13.3% (2 dB), +12.5% (r = 0.8), +8.1% (CLIC).
  - Gap to the bound (fig04b): CLASP's share at 30 dB is 0.84% below the bound at K 8 and within
    ±0.3% of it from K 16 to 128, inside the bound's own discretisation error (0.15-0.17%); the
    calibration-aware Whittle index stays 1.1-4.3% below it.
  - Decoded, CLASP's schedules land on their prediction (80.4% decoded and predicted at 30 dB,
    0.0% of users predicted to reach it missed).
  - The share has a price for the users who miss: 0.26 dB of mean PSNR and +0.013 LPIPS
    against the hard-target index (F).
  - Latency: for the mean CLASP needs 12.7% / 19.9% fewer slots than PF at T 16 / 24 (PF needs
    +6.0 slots at T 24, fig12a); at 30 dB PADC needs more than 32 slots for CLASP's share at
    T 24 (fig12b). Within a 24-slot frame (fig12c) CLASP brings 29.3 / 53.1 / 70.2 / 77.2% of
    users to 30 dB by slots 6 / 12 / 18 / 24, PADC 20.1 / 48.4 / 60.8 / 64.6%; the mean slot to
    30 dB is 1.85 [1.72, 1.98] slots earlier (1.04 on the users both bring there). A user's
    first decodable phase comes later (slot 7.75 against 6.54): CLASP waits for good slots.
- **D. Imperfect CSI and the digital baseline** (fig14: CLASP, HEIC and the 3 dB-gap code, mean
  PSNR and users at 30 dB against σ and r; 2000 frames, paired).
  - Against HEIC with IR-HARQ (its best policy and transmitter at each point), mean PSNR and the
    share of users at 30 dB: +0.20 dB and 78.0% against 74.4% with exact reports; +1.19 dB and
    75.5% against 62.7% at σ = 2 dB; +1.56 dB and 70.6% against 54.8% at r = 0.8; ahead at every
    σ and r run. Fresh → r = 0.8 costs CLASP 0.57 dB, digital 1.93 dB.
  - Against the 3 dB-gap code with the same HARQ: it leads with exact reports (80.4% of the users
    at 30 dB against CLASP's 78.0%; 0.64 dB of mean PSNR, unpaired); from σ = 1 dB or r = 0.95
    on, CLASP leads it by 5.3-13.7% of the users at 30 dB.
  - With the reward's measured spread CLASP is the best JSCC target policy at every CSI
    condition (+1.4% to +3.6% of the users over the planner, every interval above zero; decoded
    +3.3% [1.7, 5.2] at 2 dB). On aged reports the measured spread gains 1.9-5.7% over the
    curves' spread alone (with its better of no margin and a 0.5 dB margin).
  - A decoder per length adds about 0.2 dB and 1.4-2.5% of the users to the JSCC leads (closed
    form).
- **E. Training the code on CLASP's channel** (fig15: the lift of the first phases against
  report age and K; the trained codes' κ and ρ: a table).
  - +0.21 dB on the same slots over the same-budget control, two seeds (+0.211, +0.206); CLASP's
    lengths with blindly taken slots gain nothing (-0.09 dB); under the closed form +0.28. A
    code per scheduler; after co-design the closed form ties CLASP for the mean.
  - CLASP puts the first three phases 3.8 / 3.0 / 2.4 dB above the user's mean SNR (5.6 / 4.8 /
    4.2 above a blindly taken slot). The lift keeps 92 / 65 / 34 / 14% of itself at r = 0.95 /
    0.8 / 0.6 / 0.4, close to r², and grows with K (phase 1: +0.7 dB at K 2, +5.4 at K 24).
- **F. Ablations and system choice** (fig05 and Table II: the calibration-aware reward; Table V,
  appendix: everything else tried on the 30 dB gap; fig16, appendix: what a decoder per length or
  a model per CBR adds, by prefix length, with the length known in advance).
  - The calibration-aware reward (fig05a): CLASP's index with a hard target on the same
    calibrated curves leaves an extra 1.7-2.0% of the users below 30 dB from K 16 to 128 (0.75%
    at K 8). With the calibration-aware reward and the measured spread it lands on what a hard
    target achieves on the true curves: gaps to the bound of 0.84%, -0.10%, -0.30%, +0.01% and
    -0.19% at K 8 to 128, against 0.84%, -0.23%, -0.27%, +0.06% and -0.19%. With an assumed
    spread of 0.05, 0.1 or 0.2 dB it is within ±0.3% too at K 32 and 128 (fig05b).
  - Decoded (Table II): on Kodak 80.4% of the users at 30 dB against the hard target's 78.8%
    (+1.67% [0.42, 2.92]), with 0.0% of the users missed against 1.2%; at r = 0.8, with the
    reports' error in the spread, 72.1% against 63.3% (+8.75% [6.04, 11.25]; with the curves'
    spread alone 64.0%, +0.62%).
  - Ruled out (Table V): margins (0.25 / 0.5 dB widen the gap to the bound to 2.97% / 6.47% at
    K 8), the relaxation's tie rule, the price solver, re-solving, finer grids.
  - Decoders (fig16, Table I): one model per CBR is 0.175 dB and 3.3% of the users at 30 dB ahead
    of the single decoder with known slots (decoded), but its length is fixed at encoding: with
    25% of slots lost the single decoder leads it by 0.53 dB (simulated), and a decoder per length
    by 0.78 dB and 1.9% of the users (decoded). A decoder per length adds 0.23 dB and 2.5-3.1% of
    the users at 6x the decoder memory.

## VI. Conclusion (~0.3 page)

- Learned codes become schedulable once they have a link abstraction. Training for slot-varying
  SNR gives them one: a sensitivity-weighted MMSE sum, accurate to a few hundredths of a dB.
- With it, CLASP brings the scheduling theory of restless bandits to learned codes: exact
  per-user programs, a computable bound, an index policy near it, and a reward that knows the
  error of its own prediction.
- Under the CSI real systems have, scheduled JSCC is ahead of digital with link adaptation and
  HARQ, and of an idealised 3 dB-gap code once reports are imperfect.
- **Limitations.**
  - With exact reports an idealised digital code 3 dB from capacity is ahead (80.4% of the users
    at 30 dB against CLASP's 78.0%).
  - At a hard target CLASP buys the share with the quality of the users who miss (0.26 dB of
    mean PSNR, +0.013 LPIPS against the hard-target index).
  - The reward's spread is measured offline for a given CSI condition: the transmitter must know
    its report error and age.
  - Per-frame admission ignores long-run fairness (per-user debts are the fix).
  - Lost-slot runs fall outside the scaling theorems.
  - Analog symbols are not NR-compatible.
- **Future work.**
  - Repeating phases with combining.
  - Quantised (digital-compatible) learned codes.
  - Perceptual objectives inside the programs.
