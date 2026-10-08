# Assessment v2: adaptive-length token communication, aimed at IEEE TWC

2026-10-08, after cascade wave c1. Supersedes v1 (in git history at commit 0af2e1f). The
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
6. If G-A fails (the ViT loses at matched compute), the codec stops being the contribution; the
   allocation and the nesting analysis can still be, on whichever backbone wins.

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

## 2. The paper to aim at

**Working title**: prefix-decodable token communication with content- and channel-aware
bandwidth allocation.

**Problem**: K images (or users) share a frame of N channel uses; each has its own content and
SNR. Choose the channel uses per image to maximise mean (or worst) quality, with one codec, no
retraining per rate, and no side information.

**Contributions** (each conditional on the gates):
* C1 codec: position-tied ViT phase tokens, any prefix decodable, token-level power handling
  (gain, DC removal, prenorm), budget-specialised decoding (FiLM + LoRA heads + exits),
  competitive with SwinJSCC / MambaJSCC at matched compute (G-A).
* C2 allocation: concave-hull curves, equal-slope allocation optimal for the Lagrangian
  relaxation, exact average rate by time-sharing, predicted curves from transmitter-side
  features and fed-back SNR, a multi-user formulation (G-B).
* C3 analysis: nesting is nearly free for a Gaussian source under linear analog coding
  (`tools/toy_nesting.py`: <= 0.2 dB); the learned code's measured penalty and the share that
  budget specialisation recovers (G-C).
* C4 evaluation to TWC standard: Kodak and CLIC; AWGN and Rayleigh (block and fast); SNR
  mismatch; PSNR, MS-SSIM, LPIPS; SwinJSCC at its published size, MambaJSCC, NTSCC,
  DeepJSCC-l++, BPG + LDPC; parameters, FLOPs, latency; three seeds for every small effect.

If G-B fails (allocation gains are small), the fallback is an architecture paper (C1 + C3).
That only works if G-A shows a clear matched-compute win, and then IEEE TCCN or a Globecom/ICC
version fits better than TWC.

## 3. Gates

| gate | test | pass | if it fails |
|---|---|---|---|
| G-A matched compute | ViT p8 384x6 vs Swin large, 2000 epochs, same protocol (this wave) | ViT ahead by >= 0.3 dB mean over CBRs on Kodak | no backbone or interface claim; the paper rests on C2 / C3 with the better backbone |
| G-B allocation | cross-fitted oracle on Kodak (this wave), then CLIC | >= +0.2 dB or >= 10% bandwidth saving at 1/24-1/12, CI clear of 0; a policy recovering >= half of it | C2 shrinks to a section; architecture paper only |
| G-C budget specialisation | specialists (this wave), LoRA heads / exits, seeds | closes >= 30% of the matched-protocol gap at 1/8 with 2+ seeds, without losing at 1/48 | report as an ablation, not a contribution |
| G-D baselines | official-size SwinJSCC, MambaJSCC, NTSCC numbers (later turns) | on par or better at matched compute | reconsider the venue |

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
