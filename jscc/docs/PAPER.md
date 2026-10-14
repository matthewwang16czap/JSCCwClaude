# Paper plan (IEEE TWC)

2026-10-12, story candidate added 2026-10-13. What the measurements so far support, how the
paper is laid out, and what is still owed. Numbers are Kodak (24 images), the snrc4 ViT (p8, 768 x 10, 4 symbols per
token, 6 phases of CBR 1/48 at 256 px), 8 users, 24 slots, mean SNRs U[0, 15] dB, Rayleigh
block fading, unless stated; "decoded" means the schedules were decoded by the codec
itself (`docs/NOTES.md`, P1 steps 1-7).

## Working title

Quality-Aware Opportunistic Scheduling of Deep JSCC Images over Fading Multi-User Downlinks
(framing B since P1 step 6: the abstraction and the schedulers are the contribution, shown on a
prefix code, on one model per CBR, and on the system P1 step 7 recommends: one encode-once
prefix encoder with a decoder per received length. The prefix-centred title "Scheduling
Encode-Once Deep JSCC Prefixes ..." stays off: per-length decoders close most of the nesting
penalty at 2-4 phases, not at the ends, and per-CBR codes still win the target with known slots)

## The story, if section 14 passes: the scheduler makes the code's channel (2026-10-13)

The leading candidate; the two framings below it (abstraction; dynamic program) become its
supporting sections if it passes, and stay the fallback if it fails. Candidate title:
"Scheduling Makes the Channel: Co-Designing Deep JSCC Codes and Opportunistic Multi-User
Schedulers".

1. Observation. Learned JSCC codes are trained for a channel: a random SNR, piecewise SNRs,
   block fading. In a multi-user downlink that channel does not exist: every phase of a
   user's code goes out in the slot the scheduler picked for it, and how many phases the
   user gets is the scheduler's decision too. The channel a code sees is ENDOGENOUS, made by
   the MAC. Measured (`tools/codesign.py`, COMMANDS 14.1): an opportunistic, quality-aware
   scheduler gives the first phases fading peaks (stand-in: +3.3 dB over the user's mean SNR
   at phase 1, ~0 at the tail; blind slots -1.9 dB), shapes the lengths, and makes the
   received length informative about the channel -- none of it in the i.i.d. channel snrc4
   trained on.
2. Claim. The code should be trained for the channel its scheduler makes, and the scheduler
   re-planned for the code it gets: two designers with one objective, a common-payoff game;
   alternating best responses cannot lower the objective (up to the measured approximation
   gaps of each step), and a fixed point is a code trained for its own scheduler's channel
   (docs/PROBLEM.md, section 7).
3. Enabler. Each step needs the other side in a cheap form: the scheduler needs the code as a
   utility (our additive MMSE model, calibrated from 2P decodes: the abstraction framing below
   becomes this section), the code needs the scheduler as a channel (a bank of histories,
   generated on the model with no decoding). The scheduler's near-optimality (the restless
   bandit bounds) makes its step a near-best response.
4. Evidence (section 14, decoded): on the SAME slots, the code trained on the scheduler's
   channel against the same-budget control (C8), the blind control that separates slot
   selection from lengths (C9), scheduler specificity (C10), the loop closed once against the
   original system, convergence of the induced channel, and the price at constant SNR.
5. Prior work it must be distinguished from: SNR-varying DeepJSCC training (CFA-JSCC, DRJSCC,
   euJSCC: an exogenous channel); multi-user JSCC resource allocation (the code fixed, the
   allocation optimised); training on the deployment SNR distribution (matched training: our
   blind control is exactly that); end-to-end learned MAC (not with a fixed, analysable
   scheduler). Searches of 2026-10-13 found no work training a learned code on the channel a
   scheduler induces (docs/SURVEY.md, F3).

If it fails (C8 < +0.1 dB and < +1 point), docs/ASSESSMENT.md 13 lists the fallbacks.

Section 14 (2026-10-14, docs/NOTES.md "P2, round 1"): the mean-PSNR half holds (+0.28 dB on
the same slots over the same-budget control, +0.32 over training on the physical channel, which
loses to generic training; specific to the scheduler; +0.35 dB over the original system with
its own scheduler), the QoS half does not (+0.2 points), and the loop converges in one round,
so "a game" overstates it: the honest name is MAC-aware training of learned codes. Whether the
mean-PSNR gain is the scheduler's channel or the single length its quotas give is section 15's
question; if it is the channel and it grows with the number of users, the story is "multi-user
selection reshapes the channel a learned code sees: train for the selection-biased channel"
(with the order-statistics form of the lift as its theory), not a co-design game.

## The story, if 13.3f / the Jacobian probe confirm it (proposed 2026-10-13)

"A slot's SNR buys bits in a digital link. What does it buy a learned JSCC code?" Both
systems allocate bandwidth: how many slots, i.e. channel uses, each user gets (for us the CBR,
1/48 per phase). They differ in what a slot's SNR does. In a digital link it sets the BIT
RATE of the slot (link adaptation: a 13 dB slot carries ~4.4 bits per channel use, a 1 dB slot
~1.2), bits add, quality is a function of the total (and system simulation maps a codeword's
SNRs to one effective SNR by averaging mutual information, MIESM). Order does not matter. In
a learned JSCC link every slot carries the same symbols (one phase) whatever its SNR; the SNR
sets how noisy they arrive, and what that costs depends on WHICH phase the slot carried: two
slots at 13 and 1 dB give ~29.2 dB PSNR at 1/24 when the 13 dB slot carries phase 1, ~28.1
when it carries phase 2 (our model; the decoded order effect is +0.55 dB on average at 1/24).
The paper's claim: such a code still admits a link abstraction, a different one -- the slots'
contributions add as MMSEs (Gaussian MMSE of variance kappa at the slot's noise), weighted by
the decoder's sensitivity to the phase they carry -- but only if the code is trained for
slot-varying channels. Then:

1. Abstraction (theory + measurement): first-order analysis of the decoder (distortion =
   noise-free distortion + sum_j n_j x Jacobian energy of chunk j), the MMSE saturation, the
   training condition (piecewise SNR), calibration from 2P decodes at the transmitter; MMSE
   averaging vs MIESM / EESM / mean dB, equal vs sensitivity weights (13.3f).
2. What it does to scheduling: no SNR-dependent bit rate per slot (the scheduler allocates
   bandwidth only), so a quality ceiling per channel use; the scheduler's currency becomes
   "weighted MMSE removed", not bits; a user's state is two numbers; the frame is a
   restless bandit (Whittle index; the advantage index as its certainty-equivalent form;
   bounds certifying near-optimality, 13.3e); QoS deadlines and lost slots exploit encode-once.
3. Evidence: decoded gains over PF (+0.46 dB) and PADC (+11 points); certified gap to any
   scheduler; per-length decoders; digital under ideal and noisy CSI.

The schedulers stop being the novelty and become the demonstration of what the abstraction
buys. Everything already measured still serves; plan_adv / replan_adv details and most
baseline variants move to an appendix.

## The story as a dynamic program (proposed 2026-10-13)

The learned link-to-system model turns the system into a reward function: a user's PSNR at
the deadline is a closed-form function of the phases it got and their slot SNRs. The paper
can then be stated as one stochastic control problem and its solution:

1. Problem: K users, T slots before a deadline, one user per slot, i.i.d. block fading with
   per-slot CQI (slots possibly lost to other traffic); maximise the expected mean PSNR, or
   the expected number of users reaching a target. A finite-horizon MDP whose joint state is
   every user's history: intractable (2K continuous dimensions for the nested model).
2. Structure: with the nested model each user's state is (n, Y), two numbers; the users are
   coupled only by "one per slot", so the problem is a finite-horizon restless bandit
   (weakly coupled DP).
3. Solution: the Lagrangian (Whittle) decomposition into K single-user DPs on (t, n, Y):
   an upper bound on every causal scheduler; the Whittle index policy; the advantage index as
   its certainty-equivalent approximation (why the simple rule works); exact results for the
   static parts (slot counts by a DP, shortest-first admission for the count) and for
   perfect foresight (the clairvoyant optimum by column generation, certified).
4. Results: decoded gains over PF / PADC (unchanged), and the certified optimality gap:
   our rule against the causal bound and the clairvoyant optimum (stand-in: within 0.2 dB
   for the mean; 3-6 points left at a 30 dB target, where a better causal policy has room).
5. Extensions inside the same frame: lost slots, CQI errors (a belief state), per-length
   decoders, the digital baseline (a step reward), phase order (Derman-Lieberman-Ross).

Not a game: the base station is the only decision maker. The Lagrangian prices read as
market-clearing slot prices and the index as each user's bid (an auction view of the same
mathematics); a genuine dynamic game needs strategic users (uplink contention, users
reporting their own curves) and is a different system, a possible follow-up. Closest prior
work to distinguish: Hosseini and Panwar's restless bandits for scalable video (digital
layers, no learned quality model); Yu, Xu and Tong's deadline restless bandits (docs/SURVEY.md
F2). Gate: COMMANDS 13.3e (the nested model on the real codec, then every bound on it).

## Claims and their evidence

| # | claim | evidence | status |
|---|---|---|---|
| C1 | A prefix (nested) ViT JSCC trained on piecewise SNR profiles decodes a prefix whose slots had different SNRs, and its distortion is ADDITIVE over slots: D = Dsrc(L) + C(L) sum_j rho_j phi(n_j). Symmetric effective-SNR rules (EESM-like) fail. | P1: additive within 0.06 dB (2-chunk means); symmetric rules 0.23-0.71 dB MAE; constant-SNR decoders do not obey it (s1: 2-4x worse) | done |
| C2 | The model is a link-to-system abstraction: per image and phase it predicts held-out profiles within 0.03-0.07 dB and real schedules within 0.05-0.09 dB, unbiased; the transmitter can calibrate it from 2P constant-SNR decodes of its own image (kappa, rho transfer across images). Codec-agnostic: the same holds, decoded, for six per-CBR codes and for per-length decoders | P1 step 2 (fit), step 4 (calibration 0.03-0.06 dB), step 7 (per-CBR decoded MAE 0.06-0.08; per-length decoders 0.04-0.07) | done |
| C3 | Mean quality: PF on rate leaves ~0.5 dB on the table; the ADVANTAGE index (this slot's predicted PSNR gain minus a typical slot's) gains +0.46 dB decoded, +0.46 dB at the 5th-percentile user, robust over K, T, curve errors and slot correlation; content-aware lengths add only +0.05 | P1 steps 2-4 | done (theory owed) |
| C4 | Quality target (QoS): online admission by remaining need + stopping on the realised slot SNRs + advantage timing raises the share of users reaching 30 dB by +11 points decoded over PADC's per-image rule (given shortest-first admission and PF timing), +14 over PF; decomposition: sizing +5, stopping +4, timing +3, admission +2 (+11 in tight frames) | P1 steps 3-4 | done |
| C5 | Against separation (HEIC intra + error-free CQI link adaptation, the same schedulers, the same 6-slot cap per user): ahead by small margins -- mean PSNR +0.14 dB (prefix) / +0.34 (a decoder per length) / +0.29 (per CBR); target +0.5 to +10.5 points with per-length decoders (largest in tight frames), the prefix code alone -1.6 at 32 dB; a 3 dB-gap code is 0.5-0.7 dB ahead. Under CQI errors (sd 1-3 dB; digital with its best backoff per policy) the gap is expected to open: to measure | P1 steps 6-7; section 13 (`--cqi-error`) | done at ideal CQI; CQI errors owed |
| C6 | Against one model per CBR (six piecewise-trained specialists): nesting penalty 0.17-0.47 dB; a decoder per received length behind the one encoder recovers 79-101% of it at 2-4 phases (49-65% at 1, 5, 6). Encode once + per-length decoders is the best JSCC system on mean PSNR (31.63 vs 31.58 dB) and when slots can be lost (+0.73 dB, +2.8 points at 25%); per-CBR codes keep +0.9 to +1.4 points at the target with known slots. The model and the schedulers work on all three (codec-agnostic) | P1 steps 5-7; section 13 (equal training budget) | done, equal-budget check owed |
| C7 | Which QoS policy when: plan_adv (shortest-first admission + slots planned before the frame + advantage timing) with known slots; replan_adv (the same re-planned every slot from what users hold, with stopping) when slots are tight, lost or (to measure) the CQI is noisy; online when mean PSNR matters too | P1 step 7 (decoded: replan_adv 77.7, plan_adv 76.7, online 76.5, padc_adv 67.7%) | done; CQI errors owed |
| C8 | Trained on the channel its scheduler makes, the code beats the same-budget fine-tune on the usual i.i.d. channel ON THE SAME SLOTS, decoded | section 14: +0.28 dB [0.27, 0.28] under slope_adv (bar +0.3); +0.2 points at 30 dB (nothing). Confound: slope_adv gives everyone 3 phases, so the sa code is also a 1/16 specialist; section 15's fs16 separates the two | mean: supporting, confounded; QoS: fails |
| C9 | It is the scheduler's slot SELECTION the code adapts to: training on the physical channel at the same lengths does worse, even worse than generic training | section 14: sa - blind +0.32 dB, blind - hf -0.04 (same 3-phase lengths). Section 15: which phase gets the good slot (shuffle) | holds at equal lengths |
| C10 | Specific to its scheduler, and one round reaches the fixed point | section 14: sa's lead +0.09 under PF slots (+0.28 under slope_adv); closed loop +0.35 dB over the original system, +0.25 over hf's; re-banked on sa's utility 99.7% of users keep their length, lifts move <= 0.14 dB. Section 15: a PF-trained code | holds; the loop is shallow |
| C11 | The gain grows with multi-user diversity (the selection lift grows with K: stand-in +2.2 / +3.3 / +4.0 dB at K 4 / 8 / 16) | section 15 (k16) | to measure |

## Outline

1. Introduction: semantic / JSCC image delivery to many users under deadlines; why
   per-rate codes and feedback schemes (JSCCformer-f) do not fit opportunistic scheduling;
   contributions C1-C7.
2. Related work (`docs/SURVEY.md`): DeepJSCC and variable rate (DeepJSCC-V/PADC, SwinJSCC,
   DeepJSCC-MIMO, JSCCformer-f, DRJSCC), scalable multi-user DeepJSCC, semantic resource
   allocation (static, mean-SNR), opportunistic and gradient scheduling (PF, Stolyar,
   Agrawal-Subramanian), link abstraction (EESM / MIESM).
3. System model: frame of T slots, K users, block fading with (possibly erroneous) CQI,
   slots that other traffic may take, encode-once prefix, one phase per slot, objectives (mean
   PSNR; share at a target).
4. The codec and its abstraction: architecture, piecewise-SNR training, encode once with a
   decoder per received length (where the nesting penalty lives), the additive model, fitting,
   transmitter-side calibration; accuracy on all three codes (C1, C2, C6).
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
* Fig. "vs digital" and "vs per-CBR codes" (s11_*, s12_*), "vs CQI error" (s13_cqi)
* Table "where the nesting penalty lives" (s11_heads, s12_heads, s13_budget)

## Still owed, in order

0. Section 15 (GPUs 2-5, ~5 h): is section 14's +0.28 dB the scheduler's channel or the one
   length slope_adv gives (fs16), does the code use the phase order (shuffle), does it grow
   with K (k16), a code per scheduler (pf). Decides whether C8-C11 carry the paper.
1. CQI errors (section 13, CPU): JSCC (prefix, per-length decoders, per-CBR) against HEIC with
   its best backoff, sd 1-3 dB. Decides whether C5 is "on par" or "ahead under realistic CSI".
2. Equal training budget (section 13, GPUs): the prefix code and per-CBR codes with the same
   400 extra epochs as the per-length decoders (HF; FS at 1/24, 1/16, 1/12), paired; HF also as
   a utility (the fair prefix baseline for C6).
3. Decoded per-length-decoder schedules (section 13, GPU 5 first) and the paired cross-codec
   replay (per-CBR vs prefix on the same frames).
4. The codec gap: about 0.8-0.9 dB below JSCCformer-f's codec after discounting feedback; more
   training (HF tells how much 400 epochs give) or data. Relative claims do not depend on it.
5. Theory for section 5, now layered (docs/PROBLEM.md, section 3, "Optimality"): exact slot
   counts (dynamic program; greedy exact for concave curves), exact count admission
   (shortest first), the clairvoyant frame problem solved with a certificate (column
   generation + integer program, `tools/headroom.py`), the online problem as a stochastic DP
   with the index as a one-step approximation and rollout as one improvement step, and the
   phase-order result (rearrangement; Derman-Lieberman-Ross thresholds online). Section
   13.3d measures every gap on the real curves (stand-in: rollout +0.06 dB, clairvoyant
   optimum +0.19, any phase order +0.27; at 30 dB +1.6 / +4.7 / +4.7 points). Small rollout
   gains -> "the rules are near the best causal policy for the model"; large ones -> an RL
   scheduler trained on the model, with our rules as baselines. A large phase-order gain ->
   a DLR phase-mapping rule for fixed-quota frames (a new, provable component).
6. Robustness a reviewer may ask for: a second test set (CLIC), MS-SSIM targets, OFDMA
   (several users per slot).
