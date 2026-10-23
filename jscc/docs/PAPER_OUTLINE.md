# Paper outline (IEEE TWC): title, abstract, section summaries

Drafted 2026-10-23 from `docs/PAPER.md` (claims C1-C14) and `docs/NOTES.md` (numbers).
**[A]** marks a result the runs have not given yet. Each one is written as we expect it,
and the list at the end says what changes if it comes out otherwise. The main system is
the single prefix code: one encoder and one decoder, trained over every rate. Per-length
decoders are an optional upgrade.

---

## Title

**Scheduling Learned Image Codes over Fading Multi-User Downlinks: A Link Abstraction and a
Restless-Bandit Approach**

Alternatives:
- *A Link Abstraction for Scheduling Deep Joint Source-Channel Codes in Multi-User Downlinks*
- *What Is a Slot Worth to a Learned Code? Link Abstraction and Deadline Scheduling for Deep JSCC*

## Abstract (~230 words; TWC: one paragraph, no citations, no undefined acronyms)

Deep joint source-channel coding (JSCC) is designed and evaluated on a single link, yet a
cellular downlink delivers data slot by slot, through a scheduler, to many users before a
deadline. Scheduling theory for such problems exists, but it needs to know what a slot is worth
to each user, and no such model exists for learned codes. We show that a learned prefix code
trained under slot-varying SNR obeys a simple link abstraction: its distortion is the source
distortion plus a sensitivity-weighted sum of per-slot minimum mean-square-error terms. The
abstraction predicts decoded quality within 0.06 dB, against 0.49 dB for the mutual-information
mapping used in digital system simulation. A transmitter can calibrate it from 12 decodes per
image. It holds only when the code is trained for slot-varying SNR. With it, each user is a
two-number state, and multi-user deadline scheduling becomes a finite-horizon restless bandit.
Per-user dynamic programs, a Lagrangian upper bound on every causal scheduler, and Lagrangian
index policies therefore apply exactly. For mean PSNR, the index's total gap to the bound stays
bounded as the cell grows (0.005 dB per user at 128 users). A closed-form approximation gains
0.46 dB over proportional fair, decoded. For a quality target, admission by remaining need with
online stopping serves 11 points more users than per-image rate selection. Under CQI errors and
outdated reports, the scheduled JSCC system outperforms a digital chain with CQI-table link
adaptation and incremental-redundancy HARQ by up to 1.5 dB **[A: single-decoder re-run]**.
Finally, the scheduler reshapes the channel the code sees, and training on it adds a further
0.28 dB.

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
       quality within 0.05-0.09 dB, on Kodak and CLIC **[A: CLIC]**.
     - A constant-SNR-trained code obeys no such rule.
  2. *Deadline scheduling as a restless bandit with a two-number state* (C13, C3, C4, C14).
     - The abstraction turns each user into a small Markov decision process: phases that
       accumulate across slots, each graded by its slot's SNR, toward a terminal quality.
     - The Lagrangian bound and index apply exactly and run on a CPU.
     - For mean PSNR, the gap to the bound behaves as the theory predicts. A closed-form
       certainty-equivalent rule captures most of the gain at small scale.
     - For a quality target, admission by smallest remaining need plus online stopping beats
       per-image rate selection. The gains also read as 9-39% shorter frames.
  3. *System verdict under realistic CSI* (C5, C6).
     - With CQI errors and outdated reports, scheduled JSCC beats HEVC-intra with CQI-table
       link adaptation and IR-HARQ; with exact CSI they are on par.
     - One encoder and one multi-rate decoder suffice. Per-length decoders add ~0.2 dB at six
       times the decoder memory.
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
    - The Lagrangian index: gain at λ*_t. Users indifferent at the price are activated as the
      relaxation's own mix does **[A: this tie rule closes the 30 dB gap]**.
    - The Whittle index, taken at the top of the indifference range.
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
  - Training on DIV2K. Testing on Kodak and CLIC (first 60 test images) **[A: CLIC]**.
  - Metrics: PSNR, plus MS-SSIM and LPIPS **[A]**.
  - Main frame: K 8, T 24, 2000 simulated frames.
  - Decoded replays: the codec itself decodes 60 frames of every policy's schedules.
  - Every comparison is paired by frame, with 95% bootstrap confidence intervals.
- **A. Link abstraction accuracy** (Table: MAE per mapping and code).
  - MMSE 0.055 dB vs EESM 0.19 and MIESM 0.49.
  - The constant-SNR-trained code obeys no rule (0.22).
  - Transmitter calibration 0.03-0.06 dB; scheduled histories 0.05-0.09 dB.
  - The same picture on CLIC **[A]**.
- **B. Mean PSNR: policies against the bound** (Fig.: gap vs K, B = K/8).
  - The Lagrangian index goes 0.21 → 0.005 dB per user from K 8 to 128; total gap 1.66 → 0.64 dB.
  - Whittle 0.27 → 0.016. The closed form plateaus at 0.15; PF sits at 0.52.
  - Decoded on the main frame, the closed form gains +0.47 dB over PF (+0.46 for the
    5th-percentile user). The Lagrangian index is as good or better **[A: COMMANDS 21]**.
  - The PI ratio index sits below the advantage index **[A]**.
  - Within 0.26 dB of the bound at K 8.
- **C. Quality target and latency** (Fig.: share at 30 dB; CDF of the time to the target).
  - The online rule gains +11.0 points over PADC's rule, decoded (+14 over PF). The
    decomposition: sizing +5, stopping +4, timing +3, admission +2.
  - The Lagrangian index brings the most users to 30 dB. Its gap to the bound shrinks with K
    under the relaxation's tie rule **[A]**.
  - Latency at equal quality: 9-19% fewer slots than PF, 39% fewer than PADC.
- **D. Imperfect CSI and the digital baseline** (Fig.: mean PSNR / share vs σ and vs r).
  - On par with exact CQI (+0.3 dB).
  - With HARQ: +1.3 dB / +13 points at σ = 2 dB, and +1.7 dB / +15.5 points at r = 0.8
    (measured with per-length decoders; the single decoder is expected ~0.2 dB lower **[A]**).
  - On outdated reports the opportunistic gain fades, and stopping needs the posterior plus
    a margin.
- **E. Scheduler-aware training** (Fig.: per-phase SNR lift; the trained code's κ and ρ).
  - +0.28 dB on the same slots; codes learn their scheduler.
  - The lift keeps 65% of itself at r = 0.8 and 8% at 0.3.
- **F. System choice and ablations.**
  - Single multi-rate decoder vs per-length decoders: +0.23 dB mean for per-length decoders,
    at 6× decoder memory.
  - Fixed per-CBR codes lose most when slots are lost: −0.78 dB, −1.9 points.
  - Robustness over K, T, curve-prediction errors and correlated fading.

## VI. Conclusion (~0.3 page)

- Learned codes become schedulable once they have a link abstraction. Training for slot-varying
  SNR gives them one: a sensitivity-weighted MMSE sum, accurate to a few hundredths of a dB.
- With it, decades of scheduling theory apply as is: per-user programs, a computable bound,
  index policies near it.
- The system verdict: under the CSI real systems have, scheduled JSCC is ahead of digital with
  link adaptation and HARQ.
- **Limitations.**
  - Per-frame admission ignores long-run fairness (per-user debts are the fix).
  - Lost-slot runs fall outside the scaling theorems.
  - Analog symbols are not NR-compatible.
- **Future work.**
  - Repeating phases with combining.
  - Quantised (digital-compatible) learned codes.
  - Perceptual objectives inside the programs.

---

## Assumptions to confirm (COMMANDS 21-22, later)

| assumed in the outline | run | if it comes out otherwise |
|---|---|---|
| CLIC repeats Kodak's abstraction accuracy and decoded gains | 22.1, 22.3 | report per dataset; soften "holds on two test sets" |
| MS-SSIM / LPIPS move with PSNR | 22.1, 22.4 | report both; say the scheduler optimises the distortion it is given |
| The relaxation's tie rule (or re-solving, or a tighter bound) makes the 30 dB gap shrink with K | 22.2a-d | "near-optimal" for the mean only; the target gap stays a stated limitation |
| The advantage index is at or above the PI index | 22.1, 22.2e-f | restate C3: the PI index is the closed form; recommend the Lagrangian index |
| The Lagrangian index holds decoded on the main frames | 21 | present the closed form as the scheduler, the index as the certificate |
| JSCC's lead over digital with the single decoder is ~0.2 dB below the per-length-decoder numbers | not yet in the sheet | quote the measured lead; the qualitative verdict is unaffected unless it falls below ~0.5 dB |
