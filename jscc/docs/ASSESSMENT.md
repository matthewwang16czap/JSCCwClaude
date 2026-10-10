# Assessment v2: adaptive-length token communication, aimed at IEEE TWC

2026-10-08, after cascade wave c1; updated 2026-10-09 (sections 0.8, 1f; the target problem
moved to `docs/PROBLEM.md`). Supersedes v1 (in git history at commit 0af2e1f). The
literature behind it is in `docs/SURVEY.md`; measurements in `docs/NOTES.md`. "TWS" is read as
IEEE Transactions on Wireless Communications (TWC).

## 0. Verdict

1. **On the yardstick, you are right.** If the multi-rate (nesting) penalty is 0.2-0.5 dB, the
   c1 exits' +0.07 dB at 1/8 on the ViT is 15-35% of it, and my +0.3 dB gate asked for more than
   the whole headroom. Keep budget-specialised decoding in the paper. But keep the LINEAR exits
   only (the stacked modules added nothing), measure the gap under this protocol, and
   replicate over seeds: the effect is ~0.05 dB on average.
2. **The headline is not yet a fair claim.** The ViT that beats Swin uses 27x the parameters and
   16x the FLOPs. At matched compute it was never run, and the two near-matched token models you
   do have both lost to Swin at 1/16 and above. A TWC reviewer checks this first. It is this
   turn's main experiment.
3. **"AdaTok's flow in JSCC" will not survive review as the novelty.** AdaTok itself (June 2026)
   is new, but each of its ingredients has a JSCC precedent: a per-image policy over a prefix
   mask (2022), an RL policy choosing how many leading tokens to send (2025), and a single-model,
   tail-dropped, content-adaptive SwinJSCC at a fixed mean rate (TS-JSCC, August 2026).
   Per-budget LoRA heads inside a JSCC decoder look untried; they are in this turn's wave.
4. **Stacking the cascade does not make the story more solid by itself**, and it fights the
   flow: the cascade has five discrete levels, while per-image allocation needs a code that
   decodes at any length. Use exits for a coarse rate menu with a prefix inside each exit, or
   only for fixed-rate operation.
5. **A TWC-shaped paper is reachable**, as a system paper: a prefix-decodable token codec,
   side-information-free content- and channel-aware bandwidth allocation across images or
   users with an optimality argument, and budget-specialised decoding that closes a measured
   share of the nesting penalty. It needs gates G-A and G-B (section 3) to pass.
6. If G-A fails (Swin large beats the ViT at matched compute), the ViT is a choice paid for in
   compute; the rate-adaptive codec, the allocation and the nesting analysis carry over, on
   whichever token backbone wins (the hybrid is the Swin-based one).
7. **Against DeepJSCC-MIMO (read in full, section 1e):** it already has the ViT JSCC with patch
   tokens mapped straight to symbols, the "ViT beats Swin at lower cost" result, and
   truncate-and-zero-pad adaptation of one model (over antennas). It has no rate adaptivity and
   names variable-length JSCC as future work. So the novelty beyond it is the rate dimension,
   and that must in turn be defended against PADC (IEEE TWC 2023: predicted per-image PSNR picks
   the rate) and JSCCformer-f (IEEE TWC 2024: feedback-driven stopping). Enough for TWC only as
   the combination: one ViT JSCC for every rate + an open-loop joint allocation that beats
   PADC's rule + the measured nesting penalty.
8. **After PADC and JSCCformer-f in full (section 1f), 7 no longer holds:** a single model
   that decodes any length, per-image PSNR prediction and per-image rate choice are all in
   PADC; content-adaptive stopping with a ViT is in JSCCformer-f. The paper moves to the
   problem these designs cannot serve: opportunistic scheduling of encode-once token prefixes
   over a fading multi-user downlink (`docs/PROBLEM.md`, gates P-A to P-D).
9. **After the first measurements (2026-10-10, `docs/NOTES.md` P1 and P1 step 2):** P-A and
   P-B pass (piecewise-SNR training makes the distortion additive over slots; the model
   predicts the codec on real schedules within 0.08 dB). Content-aware LENGTHS for mean
   quality do not carry a paper: +0.05 dB over PF on Kodak. The candidates are
   quality-/position-aware timing (the advantage index, P-C1) and the QoS objective, the share
   of users reaching a target, where PADC's own rule is the baseline (P-C2). Both were designed
   on a stand-in utility; COMMANDS.txt section 7 decides them on the real utility and by
   decoding. If both fail, the scheduling paper is off.
10. **Both pass, decoded by the codec (section 7):** the advantage index +0.465 dB mean PSNR
   over PF; online admission + stopping +10.0 points of users at a 30 dB target over PADC's
   per-image rule. The paper: a link-to-system abstraction for prefix DeepJSCC under
   slot-varying SNR (piecewise training + the additive utility, 0.06-0.09 dB on real
   schedules) and the two schedulers it enables. Still owed before writing: a realistic
   transmitter (P-C3), the digital baseline (P-D), fixed-length per-CBR codes with the G-C
   nesting penalty (section 4's results), correlated fading, and the codec gap.

## 1. Your four points, checked

### 1a. "The specialist gap is 0.2-0.5 dB, so 0.1 dB is large"

Share of a 0.2-0.5 dB gap closed by the c1 arms (Kodak, paired against the grid-trained prefix
control):

| arm | at 1/8 | at 1/48 | mean over CBRs |
|---|---|---|---|
| vit, linear exits | +0.069 dB = 14-35% | +0.028 | +0.048 |
| vit, mixed modules | +0.076 = 15-38% | +0.031 | +0.049 |
| vit, mixed modules, grad share 0.5 | +0.142 = 28-71% | **-0.211** | +0.005 |
| swin, linear exits | +0.007 = 1-4% | +0.022 | +0.019 |
| swin, mixed modules, grad share 0.5 | +0.074 = 15-37% | -0.013 | +0.035 |

Three conditions before it goes in a paper:
* **The gap under this protocol.** Your 0.2-0.5 dB comes from an older project; the tail probes
  gave 0.29-0.71 dB at 1/8 under another recipe. The share above means nothing until the
  specialists are trained the c1 way. This wave does it (1/8 and 1/48, both backbones).
* **Seeds.** All six ViT arms gained (+0.039 to +0.049 dB mean), and they very likely trained on
  different random draws (the extra modules consume random numbers at construction, which
  shifts data order and budgets). So the effect is probably real, but at ~0.05 dB a paper needs
  three seeds per arm. This wave adds a second seed of the control and of the linear exits.
* **The part that worked is the linear exit.** "Smaller channels need deeper networks" was
  tested and failed (bodies, spatial context and depth schedules within 0.01-0.02 dB). Report
  it as an ablation; it makes the rest more credible.

### 1b. "ViT with token communication beats Swin by a large margin"

Measured cost per 256x256 image, encoder + decoder, and the wave-1 differences to Swin base:

| model | params | GFLOPs | vs Swin base, valid crops, 1/48 ... 1/8 (dB) | Kodak |
|---|---|---|---|---|
| Swin base (baseline) | 5.3 M | 18.5 | - | - |
| ViT p8 768x10, token | 142.4 M | 290.7 | +1.05 +1.06 +1.03 +0.81 +0.74 | +0.86 +0.90 +0.78 +0.51 +0.46 |
| ViT p16 384x6, token | 22.1 M | 11.2 | +0.28 -0.09 -0.31 -0.64 -0.79 | (retested, not recorded in the tree) |
| hybrid: Swin + token trunk | 12.5 M | 41.1 | +0.16 +0.15 -0.08 -0.45 -0.57 | +0.16 +0.13 -0.26 -0.75 -0.89 |
| Swin large | 12.7 M | 41.8 | never trained | |
| ViT p8 384x6, token | 21.6 M | 43.9 | never trained | |

* The two token models closest to Swin's compute lost to it from 1/16 up. The big ViT's lead
  can be its size, its patch 8, or the token interface; the data cannot tell them apart.
* The matched pair exists in the code today: **ViT p8 384x6 (43.9 GFLOPs) against Swin large
  (41.8 GFLOPs)**, both from scratch on the wave-1 protocol. This wave runs it. Swin large is
  also the published SwinJSCC size class (12-33 M params), which your 5.3 M baseline is not.
* A ViT JSCC is already in TWC (DeepJSCC-MIMO, 2024). So the claim has to be about the token
  interface and its efficiency, not about using a ViT.
* "Token communications" in the literature means discrete tokens from a foundation-model
  tokenizer (TokCom, ToDMA). Yours are analog JSCC tokens. Name them so, and say how they
  differ.
* Claiming that the token INTERFACE (per-token gain, DC removal, phase identity, budget FiLM)
  matters, rather than the backbone, needs a same-backbone ablation: a ViT with
  feature-style transmission. That is next turn's code if G-A passes.

### 1c. "Nobody has applied AdaTok's flow to JSCC"

AdaTok (Lu et al., arXiv 2606.07185) = tail masking + a LoRA decoder head per budget + a GRPO
policy that picks each image's token count. In JSCC:

| AdaTok ingredient | JSCC precedent (docs/SURVEY.md) |
|---|---|
| nested, tail-droppable code | DeepJSCC-l (TWC 2021), Yang and Kim (thermometer prefix mask, 2022), SwinJSCC channel selection, timeliness-aware JSCC (leading tokens), **TS-JSCC (tail-structured sparsity + active prefix, Aug 2026)** |
| budget-conditioned decoder | DeepJSCC-l++ (ratio as input), SwinJSCC rate modulation, rate tokens (NTSCC family), FiLM in this tree. **Per-budget LoRA heads in a JSCC decoder: none found** |
| per-image length policy | Yang and Kim (Gumbel-softmax policy), **timeliness-aware JSCC (PPO picks the length)**, TS-JSCC (sparsity), NTSCC (entropy model), DiT-JSCC (complexity score) |

What still looks unclaimed, and is defensible as a combination:
* a position-tied token code decodable at ANY length (1 token = 4 complex symbols per 256 px
  tile), so per-image allocation at a fixed mean rate is exact with time-sharing and needs no
  side information (the receiver counts tokens);
* allocation that is Lagrangian-optimal on predicted per-image, per-SNR curves, against a
  cross-fitted oracle (`alloc/`), instead of a learned black-box policy;
* budget-specialised decoding (FiLM, AdaTok-style LoRA heads, per-rate exits) with a measured
  share of the nesting penalty recovered.

Read TS-JSCC and the timeliness-aware paper in full before writing the related-work section;
they are the closest.

### 1d. "Stacking the cascade pushes the story"

As measured, the stack's contribution is the linear exit: +0.05 dB on average. It also turns
the codec into five discrete rates, which removes what makes the allocation story work. Two
ways out:
* exits for a coarse rate menu, and a prefix within each exit: a code of level k may be cut
  anywhere between its rate and the next level's. This keeps continuous rates and per-rate
  specialisation, and is a small code change once wanted;
* budget specialisation without discrete exits: the LoRA heads in this wave are continuous in
  the budget (hat-interpolated anchors), so they keep the any-length property.

### 1e. Against DeepJSCC-MIMO (read in full, 2026-10-08)

Wu, Shao, Bian, Mikolajczyk and Gündüz, IEEE TWC 2024 ([2309.00470](https://arxiv.org/abs/2309.00470)).
Details in `docs/SURVEY.md`, section B.

What it already has, so this paper cannot claim it:

| this tree's claim | DeepJSCC-MIMO |
|---|---|
| ViT JSCC whose patch tokens map straight to channel symbols | the same design: one linear layer per token -> a fixed number of symbols. This tree's ViT at a fixed rate (`--fixed-cbr`) is nearly that architecture: per-position symbol heads after a ViT trunk, a linear patch head. Left: fixed vs learned positions, per-token gain and DC removal vs one global normalisation, no Siamese input layer, no SNR / CSI heatmap (this decoder is SNR-blind) |
| "the ViT beats Swin" | its complexity table: ViT above CNN and Swin DeepJSCC with fewer parameters and FLOPs (CIFAR10, 2x2 MIMO, R 1/24, 5 dB: 24.41 vs 23.54 / 23.61 dB) |
| one model adapts by sending part of its code, the receiver zero-pads | "adaptive-M": the first M_i antenna rows are sent and zero-padded, <= 0.6 dB below per-M models. A nested code over antennas with three levels; the same mechanism as the prefix scheme on another resource |

What it does not have:
* any rate adaptivity: every bandwidth ratio is a separately trained model;
* per-image or per-user length, allocation of a shared budget;
* prefix decodability at token granularity, budget-specialised decoding, a measured nesting
  penalty in bandwidth;
* SISO results at high resolution and matched compute (its Kodak models train on 128 px crops;
  its complexity table is for 32 px images).

Its conclusion names exactly this as open: "variable length JSCC ... where the channel
resources are judiciously exploited depending on the channel state as well as the input
signal". That sentence is the motivation the paper can quote. The same group's JSCCformer-f
(IEEE TWC 2024) already varies the length per image, but closed-loop: it needs channel
feedback and stops block by block when the decoder (seen through feedback) reaches a target.
PADC (IEEE TWC 2023) already predicts each image's PSNR and picks its rate, by a per-image
quality floor (`docs/SURVEY.md`, section A).

Consequences:
1. **The backbone is not a contribution.** "Token communication on a ViT beats Swin" would be
   read as DeepJSCC-MIMO's result at a larger size. G-A stays, but as the choice of backbone
   at matched compute (and the reason to keep the ViT), not as a claim.
2. **The codec's novelty is rate adaptivity:** one ViT JSCC that decodes any prefix (one token =
   4 symbols per 256 px tile), with a phase-major layout that makes every prefix a code for
   every position (DeepJSCC-MIMO's codeword has one fixed length per model and is never cut
   along the channel uses), budget-specialised decoding, and its cost measured against
   per-rate models. The per-rate models are the DeepJSCC-MIMO baseline; this wave's
   specialists are close to it architecturally, and a faithful SISO DeepJSCC-MIMO (learned
   positions, global power normalisation, Siamese input, SNR heatmap) is a later turn's code,
   needed for the paper.
3. **The allocation's novelty is the rule and the setting, not the idea:** a joint split of a
   shared budget across images or users (equal slope, Lagrangian-optimal), exact average rate
   by time-sharing, no side information and no feedback, against PADC's per-image rule.
   `tools/alloc_policy.py oracle` now prints PADC's rule (`equal_q`) beside the equal-slope
   oracle on the same curves, so G-B also asks whether the rule matters.
4. **The adaptive-M number (<= 0.6 dB) is a second measurement of a nesting penalty** in a ViT
   JSCC; C3 can put it next to this tree's 0.2-0.5 dB and the linear-Gaussian <= 0.2 dB.
5. **Wireless depth is where DeepJSCC-MIMO is strong and this tree is thin** (SISO AWGN only).
   For TWC, the minimum is Rayleigh block fading and SNR mismatch; per-user SNRs in the
   allocation make it channel-aware. A MIMO version is a natural extension later (ordered tokens
   on eigenmodes sorted by gain), not a requirement.

### 1f. After reading PADC and JSCCformer-f in full (2026-10-09)

PADC (Zhang et al., IEEE TWC 2023; paper and code, github.com/wyzhang-ustb): DeepJSCC-V is an
ADJSCC CNN whose encoder AND decoder take the SNR (five attention blocks each); its latent is
H/4 x W/4 x 48 and rate R keeps the first round(96 R) of the 48 channels (a channel prefix in
~1/96 steps; R counts complex symbols per real source value, like this tree's CBR); it is
trained with R ~ U(0.05, 0.5) and SNR ~ U{0..27} dB, so ONE model serves every rate and SNR;
its decoder is not told R (zero padding). Against four per-rate ADJSCC models it is on par,
slightly below at R = 1/16. OraNet, an MLP on the code's channel-wise mean and std plus SNR
and R, predicts each image's PSNR with 0.07-0.42 dB mean absolute error on Kodak; a data-level
rule picks one R for all images and an instance-level rule each image's least R meeting a
PSNR floor.

JSCCformer-f (Wu et al., IEEE TWC 2024): DeepJSCC-MIMO's ViT over m blocks with channel-output
feedback (or a transmitter-side copy of the decoder); one encoder and decoder for every block;
in its variable-rate mode the transmitter stops when the decoder's estimate reaches a target
PSNR (CIFAR10, perfect feedback, m = 8). Training every block prefix to be good costs it
0.58-1.73 dB at the last block (Table VIII): a nesting penalty. Kodak, CBR 1/12, m = 2:
30.84 / 32.05 / 33.16 dB at 1 / 4 / 7 dB.

Status of the claims of 1e:
* "one ViT JSCC that decodes any prefix, its cost against per-rate models" -- DeepJSCC-V is one
  model that decodes any channel prefix, trained over random rates, with that cost measured.
  What is left is the ViT and token form: **not a contribution**.
* "predicted per-image curves" -- OraNet does it from code statistics, SNR and rate:
  **not a contribution**.
* "joint allocation of a shared budget" -- PADC's rules are per image or uniform, so the
  equal-slope rule is still different, but it is classical: **a section, not a paper**.
* "no side information" -- PADC's receiver also counts symbols: **not distinctive**.
* "nesting-penalty analysis" -- PADC (Fig. 4) and JSCCformer-f (Table VIII) measure it too;
  the encoder / decoder split and the linear-Gaussian bound remain: **supporting analysis**.
* per-budget LoRA heads in a JSCC decoder: still unseen, small.

So the earlier framing does not carry a TWC paper. What these two papers do NOT have is the
combination this codec has by construction: a code produced once, before any SNR or rate is
known, that decodes at any length, from tokens that may arrive at different SNRs, with only
CQI feedback. That is exactly what an opportunistic multi-user scheduler over a fading
channel needs, and the problem the paper should be about: `docs/PROBLEM.md`.

## 2. The paper to aim at

(Superseded on 2026-10-09 by `docs/PROBLEM.md`; kept as the record of the previous framing.)

**Working title**: prefix-decodable token communication with content- and channel-aware
bandwidth allocation.

**Problem**: K images (or users) share a frame of N channel uses; each has its own content and
SNR. Choose the channel uses per image to maximise mean (or worst) quality, with one codec, no
retraining per rate, and no side information.

**Contributions** (each conditional on the gates):
* C1 codec: DeepJSCC-MIMO's architecture class made rate-adaptive: one ViT JSCC that decodes
  any prefix (phase-major position-tied tokens), token-level power handling (gain, DC removal,
  prenorm), budget-specialised decoding (FiLM + LoRA heads + exits); its cost against per-rate
  (DeepJSCC-MIMO-style) models, and its standing against SwinJSCC / MambaJSCC at matched
  compute (G-A, G-D).
* C2 allocation: concave-hull curves, equal-slope allocation optimal for the Lagrangian
  relaxation, exact average rate by time-sharing, predicted curves from transmitter-side
  features and fed-back SNR, a multi-user formulation; open-loop (no feedback, unlike
  JSCCformer-f) and joint across images (unlike PADC's per-image quality floor) (G-B).
* C3 analysis: nesting is nearly free for a Gaussian source under linear analog coding
  (`tools/toy_nesting.py`: <= 0.2 dB); the learned code's measured penalty and the share that
  budget specialisation recovers (G-C).
* C4 evaluation to TWC standard: Kodak and CLIC; AWGN and Rayleigh (block and fast); SNR
  mismatch; PSNR, MS-SSIM, LPIPS; DeepJSCC-MIMO (SISO, per rate), SwinJSCC at its published
  size, MambaJSCC, NTSCC, DeepJSCC-l++, PADC's allocation rule, BPG + LDPC; parameters, FLOPs,
  latency; three seeds for every small effect.

If G-B fails (allocation gains are small), the fallback is an architecture paper (C1 + C3).
That only works if G-A shows a clear matched-compute win, and then IEEE TCCN or a Globecom/ICC
version fits better than TWC.

## 3. Gates

| gate | test | pass | if it fails |
|---|---|---|---|
| G-A matched compute | ViT p8 384x6 vs Swin large, 2000 epochs, same protocol (this wave) | ViT ahead by >= 0.3 dB mean over CBRs on Kodak: keep the ViT (context, not a claim: DeepJSCC-MIMO showed ViT > Swin on CIFAR) | report the ViT's cost, or move the token design to the hybrid; C1-C3 carry over |
| G-B allocation | cross-fitted oracle on Kodak (this wave), then CLIC | >= +0.2 dB or >= 10% bandwidth saving at 1/24-1/12, CI clear of 0, AND ahead of PADC's rule (`equal_q`) on the same curves; a policy recovering >= half of it | C2 shrinks to a section (or to PADC's rule on a new codec); architecture paper only |
| G-C budget specialisation | specialists (this wave), LoRA heads / exits, seeds | closes >= 30% of the matched-protocol gap at 1/8 with 2+ seeds, without losing at 1/48 | report as an ablation, not a contribution |
| G-D baselines | DeepJSCC-MIMO (SISO, one model per rate), official-size SwinJSCC, MambaJSCC, NTSCC, PADC's rule (later turns) | on par or better at matched compute; the single model within the nesting penalty of the per-rate DeepJSCC-MIMO models | reconsider the venue |

## 4. This turn's wave (`COMMANDS.txt`)

| run | answers |
|---|---|
| ViT p8 384x6 and Swin large, from scratch, 2000 epochs | G-A |
| allocation sweeps + oracle on the wave-1 ViT and Swin (Kodak) | G-B (skip if you already have these curves) |
| 1/8 and 1/48 specialists, ViT and Swin (c1 protocol) | the gap that G-C is measured against |
| ViT 1/8 with the encoder frozen (only the decoder specialises) | the decoder's share of the gap: the ceiling for any decoder-side head; if it is small, the penalty sits in the shared encoder and the next step conditions the encoder on the budget (the transmitter knows it) |
| ViT + AdaTok-style LoRA heads; + linear exits | budget-specialised decoding inside the decoder trunk; the stack you asked for |
| second seed of the c1 control and of the linear exits | whether the c1 effect is real (`--eval-seed` keeps the test's channel draws fixed across seeds, so the pairs stay paired) |

## 5. Risks

* **Concurrent work is moving monthly** (AdaTok Jun 2026, ATS-ToDMA Jul 2026, TS-JSCC Aug
  2026). A TWC review takes months; an arXiv preprint or a conference version early protects
  priority.
* **Complexity**: a 142 M-parameter, 291-GFLOP codec is hard to sell for devices. G-A also
  decides whether there is a smaller operating point worth reporting.
* **Tiles**: the ViT codes 256 px tiles independently; native-resolution results may show seams,
  and reviewers will ask. Show crops.
* **Kodak has 24 images**: every decision here uses paired intervals, but the paper needs CLIC.
* **SNR-blind decoders** are a design choice (SNR conditioning bought nothing in the old tree);
  compare against SNR-adaptive baselines on their terms and show robustness to SNR mismatch.
