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
  deadline; variants: the slots needed to bring every user to a target (makespan), the share
  of users above a target.

## 2. Why this codec, and not the published ones

| requirement of the problem | this tree | PADC (TWC 2023) | JSCCformer-f (TWC 2024) | DeepJSCC-MIMO / SwinJSCC per CBR |
|---|---|---|---|---|
| the code exists before the slot SNRs are known | yes: SNR- and rate-agnostic encoder | no: the encoder's attention blocks take the SNR | no: the encoder re-encodes every block from feedback | per-rate models trained over random SNR: yes |
| the length is decided online, slot by slot | yes: any prefix decodes | prefix code (channel prefix), but the length is chosen before sending | yes, block by block | no: one length per model |
| a prefix whose chunks had different SNRs decodes | to be shown (gate P-A; `--snr-chunks` training) | decoder conditioned on ONE SNR for the whole code | yes (feedback carries it) | n/a |
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

* Utility model (gate P-B), as measured on 2026-10-10 (`docs/NOTES.md`, Problem P1): NOT an
  effective SNR of the chunks -- the code is ordered and early tokens are 2-20x more
  noise-sensitive -- but an additive noise-sensitivity model, valid once the decoder is trained
  on piecewise SNRs (`--snr-chunks`):
  D_k(L; n_1..n_L) = Dsrc_k(L) + C_k(L) sum_j rho_L[j] phi(n_j), phi(n) = n / (1 + n / kappa)
  (`alloc/utility.py`, fitted by `tools/utility_fit.py`). Separable over slots, position- and
  SNR-aware: the value of a slot to a user is a closed-form function of what it already holds.
* Online policy: in slot t, serve argmax_k w_k [q_k(L_k + D, history + g_k(t)) - q_k(L_k,
  history)] -- the marginal utility of the slot, which grows with the slot's SNR (multi-user
  diversity) and with the image's need (content). w_k = U'(q_k) gives alpha-fairness (w = 1:
  mean quality). A threshold version (serve only above an opportunity cost lambda_k) handles
  short deadlines; dynamic programming for small K is the benchmark.
* First simulations (`tools/schedule_sim.py`, on an illustrative utility, `docs/NOTES.md`): the
  myopic greedy is beaten by a TWO-TIMESCALE policy: equal-slope lengths from the content (at
  the users' mean SNRs) plus PF timing from the CQI. With identical images PF alone already
  takes the multi-user diversity gain (+1.1 dB over round robin) and nothing content-aware adds
  to it; with a synthetic content spread the two-timescale policy adds +0.47 dB over PF. The
  content part on real per-image curves decides P-C.
* Clairvoyant bound: all slot SNRs known in advance; the greedy on marginal utilities over
  (user, slot) pairs (the equal-slope rule of `alloc/core.py` generalised to slots).
* Theory to state: (i) with the clairvoyant SNRs and concave utilities, the greedy is optimal
  for the relaxed problem (equal slopes); (ii) for long frames and i.i.d. fading, the
  marginal-utility rule is the gradient scheduler, asymptotically optimal for sums of concave
  utilities (Stolyar; Agrawal and Subramanian); (iii) the price of prefix coding is the
  nesting penalty (measured: 0.2-0.5 dB), which the scheduling gain has to exceed.

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
| P-B utility model | the same runs, then per phase (COMMANDS.txt section 6) | MAE <= 0.2 dB per image on held-out profiles. **2026-10-10: every symmetric rule fails (0.23-0.71 dB); the additive model fits the means within 0.09 dB; per image and per phase: section 6** |
| P-C scheduling gain | simulator on curve files (next turn), then the codec in the loop | >= 0.5 dB mean PSNR over PF with the same codes, and >= 0.3 dB over PADC-style fixed lengths, K = 8, Rayleigh |
| P-D separation | the digital baselines | ahead at low and mid SNR, honest about high SNR |

G-A, G-B and G-C of `docs/ASSESSMENT.md` stay relevant: G-A picks the codec, G-B says how much
the content part of the utility can add, G-C prices the prefix.

## 6. Risks

* PF with prefix codes may take most of the gain (multi-user diversity is content-blind); the
  semantic part then rests on G-B's oracle size. Report both parts separately.
* The decoder may need a per-token noise map (DeepJSCC-MIMO's heatmap idea) for strongly mixed
  profiles; that is the next code change if P-A fails after `--snr-chunks` training.
* Correlated fading (Jakes) and OFDMA change the numbers, not the method; both belong in the
  paper's evaluation.
* The codec's competitiveness: at CBR 1/12 and 1 / 4 / 7 dB on Kodak this tree's w1 ViT gives
  28.94 / 30.26 / 31.35 dB, JSCCformer-f 30.84 / 32.05 / 33.16 with two-block feedback; after
  discounting ~1 dB of feedback gain, still ~0.8-0.9 dB behind, and about on par with PADC's
  CNN. The scheduling claims are relative (same codec under every policy), but a TWC reviewer
  will ask; the likely fix is data (JSCCformer-f and PADC train on ImageNet, this tree on the
  800 DIV2K images; `--trainset DIV2K Flickr2K CLIC` is the cheap first step).
