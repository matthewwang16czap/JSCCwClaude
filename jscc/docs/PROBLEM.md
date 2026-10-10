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
* Clairvoyant bound: all slot SNRs known in advance; the greedy on marginal utilities over
  (user, slot) pairs (the equal-slope rule of `alloc/core.py` generalised to slots).
* Theory to state: (i) with the clairvoyant SNRs and concave utilities, the greedy is optimal
  for the relaxed problem (equal slopes); (ii) for long frames and i.i.d. fading, the
  marginal-utility rule is the gradient scheduler, asymptotically optimal for sums of concave
  utilities (Stolyar; Agrawal and Subramanian) -- the advantage index is its finite-deadline,
  fixed-quota form; (iii) for the target, shortest-remaining-need first is the classic rule
  for the number of jobs meeting a deadline (Moore-Hodgson with known sizes); (iv) the price
  of prefix coding is the nesting penalty (measured: 0.2-0.5 dB), which the scheduling gain
  has to exceed.

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
| P-C3 realistic transmitter | the same with `--belief` (each image calibrated from 2P constant decodes; kappa, rho from other images), correlated fading (section 8) | P-C1 and P-C2 still pass |
| P-D separation | `tools/digital_rd.py` + the same schedulers on a digital utility (alloc/digital.py: BPG, ideal CQI link adaptation, free file switching) | ahead of BPG + CQI link adaptation at low and mid SNR; honest against the capacity bound |

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
