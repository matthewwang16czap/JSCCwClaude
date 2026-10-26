# Paper outline (IEEE TWC): title, abstract, section summaries

Drafted 2026-10-23 from `docs/PAPER.md` (claims C1-C14) and `docs/NOTES.md` (numbers); updated
2026-10-26 with COMMANDS 21-24 (NOTES 6d-6f). Every number is now measured: no assumption is
left (the list at the end records how each one came out). The main system is the single prefix
code: one encoder and one decoder, trained over every rate. Per-length decoders are an optional
upgrade.

---

## Title

**Scheduling Learned Image Codes over Fading Multi-User Downlinks: A Link Abstraction and a
Restless-Bandit Approach**

Alternatives:
- *A Link Abstraction for Scheduling Deep Joint Source-Channel Codes in Multi-User Downlinks*
- *What Is a Slot Worth to a Learned Code? Link Abstraction and Deadline Scheduling for Deep JSCC*

## Abstract (~250 words; TWC: one paragraph, at most 250 words, no citations)

Deep joint source-channel coding (JSCC) is evaluated on single links, yet a cellular downlink
delivers data slot by slot, through a scheduler, to many users before a deadline. Scheduling
theory needs to know what a slot is worth to each user, which no model provides for learned
codes. We show that a learned prefix code trained under slot-varying SNR obeys a simple link
abstraction: its distortion is the source distortion plus a sensitivity-weighted sum of per-slot
minimum mean-square-error terms. On two test sets it predicts quality within 0.06-0.08 dB,
against 0.44-0.49 dB for the mutual-information mapping of digital system simulation; calibrated
from 12 decodes per image, it holds only when the code is trained for slot-varying SNR. With it,
each user is a two-number state and deadline scheduling becomes a finite-horizon restless
bandit, so per-user dynamic programs, a Lagrangian bound on every causal scheduler and
Lagrangian index policies apply exactly. The index's gap to the bound vanishes as the cell
grows, for mean PSNR and for a quality target, once the programs reward the probability of
success under the abstraction's 0.04 dB calibration error, which, ignored, costs 1.6 points of
satisfied users. Decoded, the index gains 0.5-0.6 dB over proportional fair and brings 8-15
points more users to 30 dB than per-image rate selection. Under CQI errors and outdated reports,
scheduled JSCC outperforms a digital chain with CQI-table link adaptation and
incremental-redundancy HARQ by 1.1-1.5 dB, and training the code on the channel its scheduler
creates adds 0.28 dB.

**Index Terms:** deep joint source-channel coding, link abstraction, multi-user scheduling,
restless bandits, Whittle index, Lagrangian relaxation, imperfect CSI, HARQ, image transmission.

---

## I. Introduction (~1.5 pages)

- **Motivation.** Deep JSCC is a candidate for 6G semantic and image services. It matches or beats
  separate coding at low SNR and degrades gracefully. But every evaluation we know of is one
  link with a known or sampled SNR. In a real downlink:
  - a scheduler splits a frame of slots among K users;
  - each slot fades independently;
  - the base station knows only the current slot's SNR, often with errors or delay;
  - each image must be delivered by a deadline.
- **The gap.** Mature theory exists for deadline scheduling of progressive content:
  rate-distortion-optimised packet scheduling, restless bandits for layered video, and
  Lagrangian index policies with vanishing optimality gaps. It needs one thing a learned code
  does not provide: what a slot at SNR s is worth to user k, given what k already received.
  Digital OFDM met the same problem with effective-SNR link abstraction (EESM, MIESM).
- **Thesis.** A learned code needs a link abstraction of its own. It has one when trained for
  slot-varying SNR, and with it the existing scheduling theory applies to learned codes. That
  has measurable consequences for scheduling, for training, and for JSCC versus digital.
- **Contributions** (the bulleted list TWC expects):
  1. *Link abstraction for learned codes* (C12, C1, C2).
     - A prefix code trained under piecewise SNR composes slots as a sensitivity-weighted sum
       of per-slot MMSE terms.
     - Held-out error is 0.055 dB, against 0.49 for MIESM and 0.19 for EESM. To our knowledge
       this is the first test of effective-SNR mappings on a learned code.
     - A transmitter calibrates it from 12 decodes per image. It predicts scheduled, decoded
       quality within 0.05-0.09 dB. All of it repeats on CLIC (MMSE 0.083 dB, MIESM 0.44).
     - A constant-SNR-trained code obeys no such rule.
  2. *Deadline scheduling as a restless bandit with a two-number state* (C13, C3, C4, C14).
     - The abstraction turns each user into a small Markov decision process: phases that
       accumulate across slots, each graded by its slot's SNR, toward a terminal quality.
     - The Lagrangian bound and index apply exactly and run on a CPU.
     - The Lagrangian index's gap to the bound vanishes as the cell grows, for both
       objectives (mean: 0.005 dB per user at 128 users; 30 dB target: within ±0.3 points
       from 16 users on). A closed-form certainty-equivalent rule captures most of the mean
       gain at small scale.
     - A hard target makes the abstraction's calibration error first-order: a scheduler that
       trusts its calibrated curves loses ~1.6 points per user at every K (a uniform margin
       makes it worse; the mean is immune). Programs that reward the probability of success
       under the error's measured spread recover all of it, and their schedules decode
       exactly as predicted.
     - For a quality target, admission by smallest remaining need plus online stopping beats
       per-image rate selection. The gains also read as 9-39% shorter frames.
  3. *System verdict under realistic CSI* (C5, C6).
     - With CQI errors and outdated reports, scheduled JSCC beats HEVC-intra with CQI-table
       link adaptation and IR-HARQ (+1.1 dB at 2 dB CQI error, +1.5 dB at r = 0.8); with exact
       CSI they are on par.
     - One encoder and one multi-rate decoder suffice when the number of slots is decided
       after encoding. Per-length decoders add ~0.2 dB at six times the decoder memory.
  4. *The scheduler makes the code's channel* (C8-C10).
     - Opportunistic selection lifts the SNR of the first phases by ~5 dB.
     - Training on that channel gains 0.28 dB, and codes adapt to their scheduler.
     - The effect is small and fades with outdated CSI.
- **Organisation** paragraph.

## II. Related Work (~1 page)

- **A. Deep JSCC and rate-adaptive learned codes.**
  - DeepJSCC and its successors: SwinJSCC, ViT-based codes.
  - Rate and SNR adaptation: PADC, DRJSCC, JSCCformer-f, DeepJSCC-l with successive refinement
    and multiple decoders, DeepJSCC-MIMO, OraNet and ABG quality predictors.
  - All of these are single-link or single-SNR. Our code is a prefix (successive-refinement)
    code. What is new is how it behaves when its pieces arrive at different SNRs.
- **B. Link abstraction.**
  - EESM and MIESM in OFDM system simulation.
  - SoftCast's additive per-component MMSE for linear analog codes. Our form is SoftCast's;
    what is new is that a *nonlinear learned* code acquires it, and only through training.
- **C. Scheduling progressive and scalable media.**
  - RaDiO (per-unit programs under a rate price).
  - Restless bandits for SVC (Hosseini and Panwar).
  - Price-coordinated per-user programs (Fu and van der Schaar).
  - Multi-user progressive transmission with chunk repetition (He et al.).
  - McDowell et al. schedule a learned code per fading slot, but every slot carries a fresh
    codeword and slot values are learned by RL. In ours, phases accumulate and the values are
    exact.
- **D. Restless bandits and Lagrangian index policies.**
  - Whittle; Hawkins's Lagrangian decomposition.
  - Finite-horizon index policies with vanishing gaps: Hu and Frazier; Brown and Smith; Zhang
    and Frazier; Gast, Gaujal and Yan.
  - Lagrangian priority (Hao, Xu and Tong), PI indices (Ayesta et al.), imperfect-CSI
    relaxations (Ouyang et al.).
  - We import this machinery and say so. Our contributions are the arm it applies to and the
    evidence on a decoded reward.
- **E. JSCC versus digital under imperfect CSI.**
  - Single-link comparisons show near-parity with exact CSI.
  - Outdated-CSI models (Jakes correlation); HARQ with Chase and IR combining.
  - Layered HARQ (Steiner and Shamai) is designed for no transmit CSI; we argue why it is not
    the right competitor here.
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
- **The prefix code.**
  - An encoder maps the image to 6 phases (CBR 1/48 each, up to 1/8).
  - Any prefix of L phases decodes. Phase j crosses the channel at its own slot's SNR.
  - Encode once, one decoder for every L.
- **Objectives.**
  - Mean PSNR over users at the deadline.
  - The share of users reaching a target Q (30 dB).
  - Causal policies only: the current slot's report, never a future one.
- **Digital baseline.**
  - HEVC intra (HEIC) at the rate the CQI table allows, with a tuned backoff.
  - IR-HARQ up to four transmissions; lost blocks are counted.
  - Also a 3 dB-gap capacity code.
- **Notation table.**

## IV. Method (~4 pages)

- **A. Training the prefix code for slot-varying SNR.**
  - ViT prefix code: patch 8, d 768, 10 layers.
  - Multi-rate training over prefix lengths.
  - Piecewise-SNR training: each step splits the prefix into chunks at random SNRs.
  - One encoder and one decoder serve every rate. Per-length decoders on the frozen encoder
    are an optional refinement (Sec. V-F).
- **B. The MMSE link abstraction.**
  - D_k(L) = Dsrc_k(L) + C_k(L) Σ_j ρ_j φ(n_j), with n_j = 1/SNR_j and φ(n) = n/(1+n/κ).
    PSNR = -10 log10 D.
  - A nested form: one master sensitivity profile ρ for every L.
  - Fit from decodes under constant, one-phase-at-a-time and random SNR profiles.
  - Transmitter calibration: κ and ρ pooled over other images; each image's own Dsrc and C
    from 2 SNRs × 6 lengths.
  - Comparison mappings: mean dB, noise, capacity (MIESM-like) and EESM.
- **C. Deadline scheduling as a finite-horizon restless bandit.**
  - *State.* User k's state is (n, Y): phases held and accumulated weighted noise. Serving at
    slot SNR g gives Y ← Y + m_{n+1} φ(noise(g)). The terminal reward is the PSNR, or 1{PSNR ≥ Q}.
  - *Relaxation.* Price each unit at λ_t. The frame decomposes into K single-user dynamic
    programs on a (Y, fading) grid. The Lagrangian bound (1-p)B Σ_t λ_t + Σ_k W_k(λ) caps every
    causal scheduler, and per-slot prices are found by subgradient.
  - *Policies.*
    - The Lagrangian index: gain at λ*_t; equal gains (within the programs' precision) by the
      Whittle index. The relaxation's own random order among indifferent users was measured
      and is worse at one unit per slot (-1.75 points at 30 dB), and at scale it has almost
      no one to act on.
    - The Whittle index, taken at the top of the indifference range.
  - *Calibration-aware reward.* At a target the terminal reward is
    Φ((q + μ_n − Q)/σ_n), the probability that the true PSNR reaches Q when the calibrated
    curves are off by N(μ_n, σ_n²) at n phases; σ_n and μ_n are calibration statistics,
    measured offline over images (0.03-0.05 dB). Exact Bayes for a Gaussian curve error: the
    decisions reveal nothing about it within the frame, so only the terminal reward changes.
    The bound keeps the hard reward on the true curves.
  - *Outdated reports.* The programs decide on the report through a report-to-channel kernel,
    so the bound stays valid for every scheduler that sees only reports.
- **D. Closed-form schedulers.**
  - *Mean PSNR: the advantage index.* The predicted gain of this slot minus the gain of a
    typical (mean-SNR) slot, with equal-slope quotas. It is the certainty-equivalent version
    of the Lagrangian index, which explains its plateau at scale.
  - *Quality target: the online rule.* Admit by smallest remaining need, stop each user when
    its realised SNRs predict the target, time by the advantage index.
  - *Outdated reports.* Decide on each slot's posterior and stop with a 0.5-1 dB margin.
- **E. Scheduler-aware training (the reverse interface).**
  - Record the per-phase SNR histories a scheduler delivers and retrain the code on them.
  - Iterate to a fixed point (one round).
- **Complexity.** Per-user programs cost O(T P n_y n_g) on a CPU, about 2 s per frame at K 8.
  The closed forms cost O(K) per slot.

## V. Experiments (~5 pages)

- **Setup.**
  - Training on DIV2K. Testing on Kodak and CLIC (the first 60 test images).
  - Metrics: PSNR, plus MS-SSIM and LPIPS of every decoded schedule.
  - Main frame: K 8, T 24, 2000 simulated frames.
  - Decoded replays: the codec itself decodes 60 frames of every policy's schedules.
  - Every comparison is paired by frame, with 95% bootstrap confidence intervals.
  - Figures: `tools/paper_figs.py` (COMMANDS 25) draws them all from results/; the list, with
    each figure's claim and inputs, is in docs/FIGURES.md (fig02 ... fig16, table01-02 below).
- **A. Link abstraction accuracy** (Table I = table01: MAE per mapping and code; fig02: the two
  trainings compared, how much each phase's slot matters and what piecewise training costs at
  constant SNR; fig03: the model's error on every decoded schedule).
  - MMSE 0.055 dB vs EESM 0.19 and MIESM 0.49.
  - The constant-SNR-trained code obeys no rule (0.22).
  - Transmitter calibration 0.03-0.06 dB; scheduled histories 0.05-0.09 dB, the Lagrangian
    index's included.
  - CLIC: MMSE 0.083 dB vs noise 0.135, EESM 0.16, MIESM 0.44; the constant-SNR code again
    obeys no rule; decoded histories within 0.076-0.087 dB.
- **B. Mean PSNR: policies against the bound** (fig04a: gap per user vs K, B = K/8; fig08: the
  main frames under each impairment; fig11: decoded, with MS-SSIM and LPIPS).
  - The Lagrangian index goes 0.21 → 0.005 dB per user from K 8 to 128; total gap 1.66 → 0.64 dB.
  - Whittle 0.27 → 0.016. The closed form plateaus at 0.15; PF sits at 0.52.
  - Main frame (500 frames): the Lagrangian index is the best policy in every setting, +0.52
    dB over PF fresh, +0.49 at 2 dB CQI error, +0.48 with 25% of slots lost, +0.49 at r 0.8.
  - Decoded: the closed form +0.47 dB over PF (+0.46 for the 5th-percentile user); the
    Lagrangian index +0.06 [0.02, 0.10] dB above it on Kodak, +0.12 [0.08, 0.17] on CLIC.
  - The PI ratio index loses even to PF (-0.48 dB; -0.96 decoded against ours) and starves
    users: the ratio form fails on a saturating quality, the difference form is the right one.
  - MS-SSIM and LPIPS move with PSNR (closed form vs PF: +0.006 MS-SSIM, -0.012 LPIPS).
  - At K 8 the Lagrangian index is within 0.21 dB of the bound, the closed form within 0.27.
- **C. Quality target and latency** (fig04b: gap per user vs K; fig09: share at 30 dB under
  each impairment; fig10: decoded; fig12: the latency budget; fig13: the time to the target).
  The Whittle and Lagrangian indices here use the calibration-aware reward (IV-C); the
  hard-target versions are the ablation of F.
  - The online rule gains +11.0 points over PADC's rule, decoded (+14 over PF). The
    decomposition: sizing +5, stopping +4, timing +3, admission +2.
  - The Lagrangian index (calibration-aware) brings the most users to 30 dB with fresh reports:
    +12.7 points over PADC simulated, +14.8 decoded (Kodak; CLIC +8.1).
    Under CQI errors (+6.1) and stale reports (+7.1) the planner plan_adv stays best (+8.3,
    +9.7); with lost slots and tight frames the Whittle index leads. A table of which policy
    when (C7).
  - The share has a price for the users who miss: the index's mean PSNR is 0.25 dB below the
    hard-target index's and its LPIPS 0.013 worse (F); online gains +8-9 points with better
    MS-SSIM and the same LPIPS as PADC.
  - Gap to the bound at scale (fig04b): 0.84 points per user at K 8 and within ±0.3 from K 16
    to 128, inside the bound's own discretisation error (0.15-0.17); the Whittle index stays
    1.1-4.3 points away.
  - Decoded, its schedules land exactly on their prediction (80.4% decoded and predicted,
    0.0% of users missed): the hedge also covers the codec's own error, which the measured
    spread does not include.
  - Latency at equal quality: the Lagrangian index needs 12.7% / 19.9% fewer slots than PF at
    T 16 / 24; online 39% fewer than PADC; at T 24 online needs a third more slots than the
    Lagrangian index for its share at 30 dB.
- **D. Imperfect CSI and the digital baseline** (fig14: mean PSNR / share vs σ and vs r).
  - On par with exact CQI (+0.3 dB).
  - With HARQ, the single decoder: +1.1 dB / +10.7 points at σ = 2 dB and +1.5 dB / +13.5
    points at r = 0.8; ahead of HEIC and of a 3 dB-gap code from σ = 1 dB or r = 0.95 (0.7 dB
    behind the gap code with exact CSI). Per-length decoders add 0.17-0.21 dB and 1.4-2.5
    points to every lead.
  - On outdated reports the opportunistic gain fades, and stopping needs the posterior plus
    a margin.
- **E. Scheduler-aware training** (fig15: per-phase SNR lift; the trained code's κ and ρ: a
  table).
  - +0.28 dB on the same slots; codes learn their scheduler.
  - The lift keeps 65% of itself at r = 0.8 and 8% at 0.3.
- **F. System choice and ablations** (fig05 and Table II = table02: the calibration-aware
  reward; fig07: everything else tried on the 30 dB gap; fig16: what a decoder per length or a
  model per CBR adds, by prefix length).
  - The calibration-aware reward (fig05a): with a hard target on the same calibrated curves the
    Lagrangian index loses ~1.6 points per user at every K (0.75 at K 8); the calibration-aware
    reward lands on what a hard target achieves on the true curves (0.84; −0.10, −0.30, +0.01,
    −0.19): 99-100% of the curve errors' cost recovered, for any assumed spread from 0.03 to
    0.2 dB (fig05b). Main frames: +1.6 points fresh, +4.5 at 2 dB CQI error, +1.0 at r = 0.8
    over the hard target. Decoded (Table II): +1.7 [0.4, 2.9] points on Kodak, 0.0% of users
    missed against 1.2%; the price −0.25 dB mean PSNR and +0.013 LPIPS, from the users who
    miss. Ruled out on the way (fig07): margins (worse), the relaxation's tie rule, the price
    solver, re-solving, finer grids.
  - Single multi-rate decoder vs per-length decoders: +0.23 dB mean and 2.5-3 points at 30 dB
    (decoded) for per-length decoders, at 6× decoder memory.
  - One model per CBR is 0.17 dB / 3.3 points ahead of the single decoder with known slots, but
    its length is fixed at encoding: with 25% of slots lost the single decoder leads (+0.54 dB),
    and per-length decoders lead by +0.78 dB / +1.9 points (decoded).
  - Robustness over K, T, curve-prediction errors and correlated fading.

## VI. Conclusion (~0.3 page)

- Learned codes become schedulable once they have a link abstraction. Training for slot-varying
  SNR gives them one: a sensitivity-weighted MMSE sum, accurate to a few hundredths of a dB.
- With it, decades of scheduling theory apply as is: per-user programs, a computable bound,
  index policies near it.
- The system verdict: under the CSI real systems have, scheduled JSCC is ahead of digital with
  link adaptation and HARQ.
- **Limitations.**
  - At a hard target the calibration-aware index buys the share with the quality of the users
    who miss (−0.25 dB mean PSNR, +0.013 LPIPS against the hard index); where that matters,
    online is the balanced target policy.
  - The programs model outdated reports but take Gaussian CQI errors as exact; under such
    errors the closed-form planner (plan_adv) is the robust target policy.
  - Per-frame admission ignores long-run fairness (per-user debts are the fix).
  - Lost-slot runs fall outside the scaling theorems.
  - Analog symbols are not NR-compatible.
- **Future work.**
  - Repeating phases with combining.
  - Quantised (digital-compatible) learned codes.
  - Perceptual objectives inside the programs.

---

## How the assumptions came out (COMMANDS 21-24)

| assumed in the outline | run | if it comes out otherwise |
|---|---|---|
| ~~CLIC repeats Kodak~~ | 22 | confirmed (NOTES 6d) |
| ~~MS-SSIM / LPIPS move with PSNR~~ | 22 | confirmed for the mean; at the target they show the cost to the users who miss (now in the text) |
| ~~The relaxation's tie rule closes the 30 dB gap~~ | 22 | refuted, as were prices, re-solving and finer grids |
| ~~The 30 dB gap is the cost of the transmitter's curve errors~~ | 23.1 | confirmed: with the true curves the index meets the bound from K 16 on; the margin did not recover it (it made it worse), so the cost is reported as a finding and a limitation |
| ~~The advantage index is at or above the PI index~~ | 22 | confirmed, by ~1 dB |
| ~~The Lagrangian index holds decoded~~ | 21 | confirmed: it is the paper's scheduler for the mean and for the target with fresh reports |
| ~~JSCC's lead over digital with the single decoder is ~0.2 dB below the per-length decoders'~~ | 23.2 | confirmed: 0.17-0.21 dB and 1.4-2.5 points below; the verdict is unchanged |
| ~~A calibration-aware reward recovers the curve errors' cost~~ | 24 | confirmed: 99-100% at every K, insensitive to the assumed spread; decoded +1.7 points over the hard index and 0.0% missed; the target result is now near-optimal |
