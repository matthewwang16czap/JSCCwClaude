# The problem: scheduling encode-once token prefixes over a fading multi-user downlink

2026-10-09. Chosen after reading PADC (IEEE TWC 2023, with its code) and JSCCformer-f (IEEE
TWC 2024) in full: `docs/ASSESSMENT.md` 1f explains why the earlier framing ("one model for
every rate + per-image allocation") is no longer new. This note states a problem in which the
properties of this tree's token codec are REQUIRED, not merely convenient, and the experiments
that decide whether it carries a TWC paper.

## 1. System

* A base station serves K users in a frame of T slots (the deadline). User k is owed one image
  x_k (a camera frame for remote monitoring, a map tile, an AR asset): all are due at the end
  of the frame.
* Each slot carries S channel uses and is given to one user (TDMA; the OFDMA version gives
  resource blocks).
* Block fading: user k's slot-t gain h_k(t) ~ CN(0, beta_k), independent over slots, with mean
  SNRs that differ per user (path loss). The scheduled receiver knows h and equalises, so the
  slot is an AWGN channel at SNR g_k(t) = |h_k(t)|^2 SNR_k. The base station learns g_k(t) at
  the start of slot t (CQI, as in every opportunistic scheduler); future slots are unknown
  beyond their statistics.
* Codec: every image is encoded ONCE, before the frame, into its token sequence (SNR- and
  rate-agnostic encoder). In a slot the scheduled user receives the next chunk of its prefix:
  the next S / (sym x tiles) tokens of every tile. At the deadline user k holds L_k tokens per
  tile that arrived over its slots' SNRs, and decodes once.
* Objective: maximise the mean (or an alpha-fair function) of the users' quality at the
  deadline; or, the QoS form (PADC's own objective, made multi-user): the share of users whose
  image reaches a target quality by the deadline; variant: the slots needed to bring every
  user to a target (makespan).

## 2. Why this codec, and not the published ones

| requirement of the problem | this tree | PADC (TWC 2023) | JSCCformer-f (TWC 2024) | DeepJSCC-MIMO / SwinJSCC per CBR |
|---|---|---|---|---|
| the code exists before the slot SNRs are known | yes: SNR- and rate-agnostic encoder | no: the encoder's attention blocks take the SNR | no: the encoder re-encodes every block from feedback | per-rate models trained over random SNR: yes |
| the length is decided online, slot by slot | yes: any prefix decodes | prefix code (channel prefix), but the length is chosen before sending | yes, block by block | no: one length per model |
| a prefix whose chunks had different SNRs decodes | yes after `--snr-chunks` training (P-A); predicted per image within 0.08 dB (P-B) | decoder conditioned on ONE SNR for the whole code | yes (feedback carries it) | n/a |
| no feedback beyond CQI | yes | yes | no: channel-output feedback of every symbol, a decoder copy per user at the transmitter | yes |
| content-aware marginal value of a slot, before sending | per-image curves (alloc/) | OraNet (per image, per SNR, per rate) | from the decoder copy | no |
| many users | yes | per-image rule, no shared budget | single link (and a 2-user broadcast) | n/a |

The properties that make the problem work, together: encode-once tokens, any-length prefixes,
decoding of mixed-SNR prefixes, transmitter-side utility curves, and only CQI feedback. No
published scheme has all of them (searches on 2026-10-09 found no opportunistic multi-user
scheduler over progressive or prefix DeepJSCC streams with a semantic utility; the closest are
DRJSCC (single link, re-encodes the remaining blocks when the channel changes, arXiv
2311.15309) and a scalable multi-user DeepJSCC for diverse bandwidths (multicast layers)).

## 3. The scheduler

* Utility model (gate P-B, passed 2026-10-10, `docs/NOTES.md` P1 and P1 step 2): NOT an
  effective SNR of the chunks -- the code is ordered and early tokens are 2-20x more
  noise-sensitive -- but an additive noise-sensitivity model, valid once the decoder is trained
  on piecewise SNRs (`--snr-chunks`):
  D_k(L; n_1..n_L) = Dsrc_k(L) + C_k(L) sum_j rho_L[j] phi(n_j), phi(n) = n / (1 + n / kappa)
  (`alloc/utility.py`, fitted by `tools/utility_fit.py`): per image and phase, held-out
  profiles within 0.03-0.07 dB and decoded schedules within 0.08 dB (p95 0.29), no bias.
  Separable over slots, position- and SNR-aware: the value of a slot to a user is a
  closed-form function of what it already holds.
* Mean quality. On the real Kodak utility PF takes the multi-user diversity gain (+1.1 dB
  over round robin) and content-aware LENGTHS add only +0.05 dB (equal-slope, fixed or
  re-planned); the myopic marginal-utility greedy loses 0.23 dB. What PF misses is how much a
  slot is worth to the user in QUALITY: the ADVANTAGE index serves the user whose predicted
  PSNR gain from this slot most exceeds its gain from a slot at its mean SNR (`slope_adv`):
  +0.476 dB over PF, +0.465 decoded by the codec (`docs/NOTES.md`, P1 step 3), +0.46 dB on the
  5th-percentile user. On a stand-in about 0.3 dB of it is valuing slots in PSNR (users near
  their noise-free quality take the bad slots) and 0.2 dB the position (sensitive head phases
  wait for peaks).
* Quality target. The share of users reaching Q dB. `online`: every slot, admit the users with
  the smallest remaining need (phases at the mean SNR, from what they already hold) that fit
  in the slots left, open spare slots to all, serve by the advantage index, and stop a user as
  soon as the model says its realised SNRs got it there. Against PADC's rule (per-image length
  at the mean SNR, fixed; given shortest-first admission and PF timing): +9.4 points of users
  at 30 dB (K = 8, T = 24), +10.0 decoded, +6.9 to +10.4 over targets, frame lengths, K = 16
  and identical images. Its parts: per-image sizing (+5.0 points; the curves), online stopping
  (+3.9; encode-once prefix + mixed-SNR utility, no feedback beyond CQI), advantage timing
  (+3.2 to +3.4; the utility), admission (+2.0, +11 in tight frames).
* Optimality, layer by layer (what is exact, what is not; `tools/headroom.py` measures each):
  1. Slot COUNTS with a channel that does not change within the frame (every slot alike for
     a user): max sum_k q_k(L_k) s.t. sum_k L_k = T, L_k <= P is a separable allocation, exact
     by a dynamic program over users (O(K T P)); the greedy equal-slope rule (our quotas) is
     exact when every q_k is concave (marginal analysis: Gross 1956, Fox 1966). For a target
     with known needs, shortest-first admission maximises the number served (cardinality
     knapsack; Moore-Hodgson with one deadline).
  2. WHICH slots, with every SNR of the frame known (clairvoyant): a combinatorial
     assignment (8^24 ~ 5e21 raw assignments at K 8, T 24) whose user values are non-linear in
     the set of slots and their order. Our policies do not solve it. It is solved offline for
     evaluation: column generation on the configuration LP (each user one set of up to P
     slots, each slot to at most one user; every set is priced, so its value bounds every
     schedule from above, causal or not) and an integer program over the generated columns;
     equal to brute force on small frames, bound - optimum ~0.001 dB at K 8, T 24 on the
     stand-in (certified).
  3. ONLINE (the real problem: a slot's SNR is known when it starts): a finite-horizon
     restless multi-armed bandit (`alloc/whittle.py`). With the nested model (one master
     sensitivity profile m for every prefix length, `tools/utility_fit.py --nested`) a
     user's state is two numbers, n phases held and Y = sum_j m_j phi(n_j), and its quality
     at the deadline is -10 log10(Dsrc(n) + C(n) Y / sum(m[:n])). The joint program (2K
     continuous states) is intractable, but the users are coupled only by one user per
     slot: pricing a slot at lam decomposes it into K single-user programs on (t, n, Y)
     (Whittle; Hawkins; Adelman and Mersereau), which gives (a) an upper bound on EVERY
     causal scheduler, (1 - p) T lam + sum_k W_k(lam), minimised over the price; (b) the
     Whittle index, user k's indifference price for this slot, and the index policy (serve
     the largest); (c) the advantage index as its certainty-equivalent approximation (the
     continuation valued at a typical, mean-SNR slot instead of the program's). Checked
     against exact expectimax on small frames: bound >= best causal >= Whittle. Stand-in,
     K 8, T 24: the Whittle policy ties our rules (-0.02 dB [-0.04, +0.01] mean PSNR; -0.4
     points at 30 dB), rollout +0.05 dB / +3.1 points, causal bound +0.20 dB / +5.6 points,
     clairvoyant optimum +0.19 / +6.3: best causal - our rule is within [+0.05, +0.19] dB
     for the mean (certified), [+3.1, +5.6] points at 30 dB, where a better causal policy
     has room. The gradient-scheduler link (Stolyar; Agrawal and Subramanian) stays as the
     long-horizon motivation.
  4. PHASE ORDER: when a user's slot count is fixed before the frame (fixed quotas, no lost
     slots) the phases need not go in prefix order. For a fixed set of slots the best mapping
     puts the most sensitive phase in the best slot (rearrangement inequality); online, with
     the user's remaining slots i.i.d., the optimal causal mapping for the expected
     distortion is the threshold rule of the sequential stochastic assignment problem
     (Derman, Lieberman and Ross 1972), which applies exactly because the distortion is
     linear in the per-phase noise terms. Stand-in: +0.12 dB ex post on our rule's own slots,
     +0.27 for the clairvoyant optimum with any order (prefix order +0.19). With online
     lengths or lost slots the prefix order is what keeps a cut-short code decodable.
  5. The price of prefix coding is the nesting penalty (0.17-0.47 dB, mostly recoverable at
     2-4 phases by a decoder per length), which the scheduling gain has to exceed.

## 4. Baselines

1. Round robin with the same prefix codes (channel- and content-blind).
2. Max-SNR (opportunistic, content-blind) and proportional fair on log2(1 + g) (the LTE / NR
   scheduler).
3. PADC-style: each user's length fixed before the frame from predicted curves at its mean SNR
   (PADC's per-image rule, or the equal-slope rule), delivered with PF timing.
4. Fixed-length per-user codes (one model per CBR, as DeepJSCC-MIMO / SwinJSCC) with
   opportunistic timing: isolates what online LENGTH adds over online TIMING.
5. Digital: BPG (and a progressive codec) with capacity-achieving or LDPC codes, AMC per slot,
   the same schedulers.
6. JSCCformer-f's setting (K = 1, a quality target, feedback): the single-user special case,
   where this scheme stops on predicted quality with CQI only.

Metrics: mean PSNR / MS-SSIM / LPIPS at the deadline, 5th-percentile user, Jain's fairness,
slots to bring all users to a target. Kodak and CLIC; K in {4, 8, 16}; mean SNRs spread over
0-20 dB; T from tight to loose.

## 5. Gates

| gate | test | pass |
|---|---|---|
| P-A mixed-SNR decoding | `tools/mixed_snr.py` on the w1 ViT, then on a `--snr-chunks 4` fine-tune vs its constant-SNR control (COMMANDS.txt section 5) | a mixed prefix within ~0.2 dB of the model's prediction; constant-SNR quality within 0.1 dB of the control. **2026-10-10: passes with `--snr-chunks` training (additive within 0.06 dB on 2-chunk means; constant-SNR cost 0.06-0.10 dB); fails for constant-SNR decoders** |
| P-B utility model | the same runs, then per phase (COMMANDS.txt section 6) | MAE <= 0.2 dB per image on held-out profiles. **2026-10-10: passes for snrc4 (held-out 0.03-0.07 dB per image and phase, replayed schedules 0.08 dB); s1 0.12-0.29** |
| P-C1 mean quality | simulator on the real utility, then the codec on the schedules (section 7) | >= 0.3 dB mean PSNR over PF, decoded, K = 8, Rayleigh. **2026-10-10: content-aware lengths +0.05 (the first form fails); the advantage index passes: +0.476 simulated, +0.465 [0.39, 0.53] decoded; +0.34 to +0.49 over every scenario** |
| P-C2 quality target | the same, with `--target` | >= 5 points more users at the target than PADC's rule (with shortest-first admission and PF timing), decoded, K = 8, T = 24, 30 dB. **2026-10-10: passes: +9.4 simulated, +10.0 [7.3, 12.9] decoded (+9.6 with a 0.2 dB margin, 0.2% missed); +6.9 to +10.4 over the scenarios, +4.2 with 10% curve errors** |
| P-C3 realistic transmitter | the same with `--belief` (each image calibrated from 2P constant decodes; kappa, rho from other images), correlated fading (section 8) | P-C1 and P-C2 still pass. **2026-10-11: passes: decoded +0.464 dB [0.39, 0.53] and +11.0 points [8.3, 13.8]; calibration error 0.03-0.06 dB, as good as the full fit; under slot correlation up to 0.98 the leads are >= +0.34 dB and >= +5.6 points** |
| P-D separation | `tools/digital_rd.py` + the same schedulers on a digital utility (alloc/digital.py: HEIC, error-free CQI link adaptation, free file switching, up to 12 slots per user) | ahead of HEVC-intra + CQI link adaptation at low and mid SNR; honest against the capacity bound. **2026-10-11: ON PAR, not ahead: mean PSNR +0.13 dB (prefix) / +0.28 (per-CBR) at each one's best policy; target: +6 to +9 points in tight frames, -6 to +0.2 at 32 dB; a 3 dB-gap code is 0.5-0.7 dB ahead, capacity 1.7-1.9. At the equal per-user cap (6 slots, 2026-10-12): mean unchanged; target +0.5 / +5.0 / +8.4 points at 30 / 28 / T 16, -1.6 at 32 (prefix code), +0.8 to +10.5 with a decoder per length. Error-free CQI is the assumption left that favours digital: section 13 (`--cqi-error`)** |
| P-E one model per CBR | six specialists (1/48 ... 1/8), each trained like snrc4_s1 plus `--fixed-cbr`, one per-CBR utility; scheduled with `--fixed-codes` (spare slots planned before the frame) | the prefix code's best policy ahead of the specialists' best. **2026-10-11: with known slots per-CBR wins: mean +0.15 dB, target +2.0 to +6.0 points against `online` (much of it the slot planner, which the prefix code can use too: `plan_adv` / `replan_adv`, section 12). With lost slots the prefix code wins mean PSNR by +0.36 / +0.53 dB (10% / 25% lost) against per-CBR's best hedge. 2026-10-12, the same planner on both: per-CBR still +1.0 to +4.4 points with known slots (the nesting penalty itself); with 25% lost `replan_adv` +0.5; with a decoder per length (P-F) -0.9 to +1.2 known, +2.8 points / +0.73 dB at 25% lost. Decoded: the model and both schedulers hold on the per-CBR codes (MAE 0.06-0.08 dB)** |
| P-F decoder heads | decoder-only specialists and LoRA budget heads on the frozen snrc4 encoder | how much of the penalty a per-length decoder recovers. **2026-10-11: LoRA heads none (= more training alone); a decoder of its own ~40% at 1 and 6 phases. 2026-10-12, all six lengths: 79-101% of the penalty at 2-4 phases (57-100% beyond more training), 49-65% at 1, 5, 6: where schedules live the penalty is mostly a decoder matter, paid at the receiver. Encode once + a decoder per length is the recommended JSCC system; equal-budget check: section 13** |

G-A, G-B and G-C of `docs/ASSESSMENT.md` stay relevant: G-A picks the codec, G-B says how much
the content part of the utility can add, G-C prices the prefix.

## 6. Risks

* PF with prefix codes takes most of the mean-quality gain, and content-aware lengths add
  +0.05 dB on Kodak (measured): the mean-quality claim rests on the advantage index, the
  content claim on the quality target. Report the parts separately.
* For mean quality the prefix code itself is not required: equal shares with PF timing tie PF
  (what one model per CBR allows), and the advantage index works on any code split over slots
  once the decoder is trained piecewise (with uniform rho: about +0.3 of its +0.5 dB on the
  stand-in), so a per-CBR codec without the nesting penalty is the fair baseline there. What
  is prefix-specific: the position part of the index and, for the target, online stopping.
* The decoder may need a per-token noise map (DeepJSCC-MIMO's heatmap idea) for strongly mixed
  profiles; that is the next code change if P-A fails after `--snr-chunks` training.
* Correlated fading (Jakes) and OFDMA change the numbers, not the method; both belong in the
  paper's evaluation (`--fade-corr`: Gauss-Markov slot gains; on the stand-in a correlation of
  0.9 widens online's lead over padc, whose fixed lengths cannot react to a long fade).
* The transmitter's knowledge of each image's curves: 10% log-MSE errors halve the target
  lead (+4.2). The base station has the image and the codec, so it can calibrate each image
  from 2P constant-SNR decodes of its own (`tools/utility_fit.py --calib`), with kappa and rho
  measured once on other images: gate P-C3.
* The codec's competitiveness: at CBR 1/12 and 1 / 4 / 7 dB on Kodak this tree's w1 ViT gives
  28.94 / 30.26 / 31.35 dB, JSCCformer-f 30.84 / 32.05 / 33.16 with two-block feedback; after
  discounting ~1 dB of feedback gain, still ~0.8-0.9 dB behind, and about on par with PADC's
  CNN. The scheduling claims are relative (same codec under every policy), but a TWC reviewer
  will ask; the likely fix is data (JSCCformer-f and PADC train on ImageNet, this tree on the
  800 DIV2K images; `--trainset DIV2K Flickr2K CLIC` is the cheap first step).

## 7. The code's channel is made by the scheduler: code-scheduler co-design (2026-10-13)

Everything above designs the scheduler for a GIVEN code. But under a scheduler a code's phases
never see "the channel": each phase goes out in the slot the scheduler picked for it, so the SNR
every phase sees, and how many phases every user receives, are decisions of the MAC. The code's
channel is endogenous. Our index puts the first, most sensitive phases into fading peaks and lets
the tail take ordinary slots; snrc4 was trained on chunks at i.i.d. SNRs uniform over [-2, 22]
dB with random cuts and uniform lengths: another channel.

* The object (`tools/codesign.py bank`): the distribution p_pi(L, s_1..s_L) of the number of
  phases a user receives and the SNR of each, in prefix order, under policy pi, over the frames
  (users' mean SNRs, fades, images). On the stand-in utility (K 8, T 24, U[0, 15] dB, Rayleigh):
  slope_adv's first phase sits +3.3 dB above the user's mean SNR, its third +1.8, the tail ~0;
  blindly taken slots -1.9 dB (the fade's mean, -2.5 dB, less clipping at -2 dB); the per-phase
  mean SNR falls from 10.7 dB at phase 1 to 1.1 dB at phase 6 (the users who get long prefixes
  are the weak ones); replan_adv at 30 dB 10.8 -> 4.8 dB. And the profile differs by length:
  the receiver KNOWS how many phases arrived, so under an opportunistic scheduler the length is
  side information about the channel (stand-in: users with 2 phases saw ~11 dB on both, users
  with 6 saw 3.9 dB on the first and 1.1 on the last), which a length-conditioned decoder (FiLM
  on L) can learn and the transmitter never has to send.
* Two designers, one objective. The code designer picks the weights theta, the MAC designer the
  policy pi; both want J(theta, pi) = E[quality at the deadline] (mean PSNR, or the share at a
  target). The scheduler needs the code only through its utility u_theta (the additive model,
  measured and calibrated from 2P decodes per image: sections 3 and 5); the code needs the
  scheduler only through p_pi. Alternating best responses:
    code:      theta_{r+1} = argmin_theta E_{(L, s) ~ p_{pi_r}} E_x D_theta(x; L, s)
               (fine-tuning on a bank of pi_r's histories: `main.py --snr-profiles`)
    scheduler: pi_{r+1} = our policy on u_{theta_{r+1}} (measure, fit, calibrate, re-plan)
  This is a common-payoff (identical-interest) game; with exact best responses J cannot
  decrease, J(theta_{r+1}, pi_{r+1}) >= J(theta_{r+1}, pi_r) >= J(theta_r, pi_r), and its fixed
  points are codes trained for the channel their own scheduler makes. Ours are approximate:
  the scheduler step is within the certified gaps of section 3 (stand-in: within 0.05-0.19 dB
  of the best causal policy for the mean) and the model's error (0.03-0.09 dB); the code step
  is SGD on MSE while J is PSNR (a concave transform: E[PSNR] and -E[MSE] rank codes alike
  only approximately); and the bank draws histories independently of the training image,
  where pi decides with the image's own curve (a content-conditioned bank needs a quality
  predictor for training crops: not built).
* Why only this tree can close the loop: a round of the loop is cheap because the scheduler runs
  on the model (a bank of 32,000 histories is a CPU minute, no decoding), and the scheduler can
  re-plan for a new code from 2P decodes per image. Without a link abstraction every round
  would need the codec inside the scheduler.
* The test, round 1 (COMMANDS.txt section 14): four fine-tunes of snrc4 with the same recipe and
  budget (400 epochs), differing only in the channel: hf (the recipe so far, the control),
  blind (slope_adv's lengths, every phase in a blindly taken slot: the physical channel at the
  scheduler's lengths), sa (what slope_adv delivers), sat (what replan_adv delivers at 30 dB).
  Each decodes the ORIGINAL system's schedules (same users, images and slot SNRs: the code's
  best response alone), then its own re-planned schedules (the loop closed once). sa - blind
  isolates the scheduler's slot selection (the claim), blind - hf the lengths and the users'
  mean SNRs (matched training, the older idea). The scheduler's channel for the retrained
  codes against the one they trained on (`codesign.py stats --against`) says whether one round
  is near the fixed point.
* Round 1 measured (2026-10-14, docs/NOTES.md "P2, round 1"): on the real utility slope_adv's
  quotas give every user 3 phases at +3.2 / +2.7 / +1.8 dB over its mean SNR (blind slots
  -1.8); the code trained on that channel decodes the same slots +0.28 dB better than the
  same-budget control and +0.32 better than the code trained on blind slots, +0.09 under PF
  slots; at 30 dB nothing. Re-banked on the retrained code's utility the channel barely moves
  (99.7% of users keep their length): the fixed point is reached in one round, so the loop's
  second step adds nothing measurable. Round 2 (COMMANDS 15): a 3-phase specialist on the
  generic channel, a shuffled bank, 16 users, a PF-trained code.
* Theory owed if it passes: (a) the game and its monotone improvement (above), with the two
  approximation gaps bounded by what section 3 measures; (b) the linear-Gaussian version
  (analog prefix code of a Gaussian source, per-phase power allocation under a per-phase SNR
  distribution): the matched allocation, and the loss of the i.i.d.-trained one, in closed form;
  (c) the information the received length carries about the channel under a given policy,
  and what an L-conditioned decoder gains from it.
