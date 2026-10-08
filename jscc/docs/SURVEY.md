# Related work: adaptive-length token communication, rate-adaptive deep JSCC

Survey for `docs/ASSESSMENT.md` (2026-10-08). The sandbox that wrote it cannot open
arxiv.org, so most entries come from abstracts and search summaries, not full texts. Read the
entries marked **read in full** before claiming novelty against them. DeepJSCC-MIMO was read in
full on 2026-10-08 (the PDF you uploaded); its entry and `docs/ASSESSMENT.md` section 1e
reflect the full text. "Threat" is how much the
work overlaps what this tree can claim: HIGH = a reviewer will cite it against you.

## A. Rate- and bandwidth-adaptive deep JSCC (one model, many rates)

| work | rate mechanism | relation to this tree | threat |
|---|---|---|---|
| DeepJSCC, Bourtsoulatze, Kurka, Gündüz ([1809.01733](https://arxiv.org/abs/1809.01733), IEEE TCCN 2019) | one model per bandwidth ratio | the field's baseline | - |
| DeepJSCC-l, Kurka and Gündüz ([2009.12480](https://arxiv.org/abs/2009.12480), IEEE TWC 2021, DOI 10.1109/TWC.2021.3090048) | layered successive refinement and multiple descriptions, CNN | prefix-style progressive code in TWC; reports layering costs little | MED |
| Yang and Kim ([2110.04456](https://arxiv.org/abs/2110.04456), ICASSP 2022; code: Dynamic_JSCC) | a policy network picks a thermometer (prefix) mask per image from content and SNR; Gumbel-softmax | **per-image adaptive length over a prefix code, 4 years ago**: the core of "AdaTok's flow" in JSCC | HIGH |
| Entropy-aware adaptive rate control ([2306.02825](https://arxiv.org/abs/2306.02825)) | one network, feature maps activated by entropy and channel state | content-adaptive rate | MED |
| SwinJSCC, Yang et al. ([2308.09361](https://arxiv.org/abs/2308.09361)) | SNR and rate modulation modules scale and select latent channels | the feature-transmission baseline; its official size is 12-33 M params, not 5.3 M (section E) | HIGH (baseline) |
| DeepJSCC-l++ ([2305.13161](https://arxiv.org/abs/2305.13161), Globecom 2023) | Swin/ViT; bandwidth ratio and SNR fed as side information; loss weights set per ratio | transformer "one model, many ratios" | MED |
| **TS-JSCC**, "Single-model adaptive wireless image transmission via feature sparsity regularization" ([2608.21743](https://arxiv.org/abs/2608.21743), Aug 2026) | SwinJSCC + tail-structured L1 sparsity + an active-prefix transmission rule: per-image channel use varies under a shared mean CBR; a λ-controlled variant covers a range of rates | **the closest concurrent work to "AdaTok's flow in JSCC"**: tail-dropped prefix code, content-adaptive length, one model. **Read in full.** | HIGH |
| **Timeliness-aware JSCC** ([2509.19754](https://arxiv.org/abs/2509.19754), Tsinghua) | a PPO policy picks the code length; the leading tokens are sent, the receiver zero-pads | **an RL policy choosing how many leading tokens to send**: AdaTok's GRPO step in JSCC form. **Read in full.** | HIGH |
| **PADC**, Zhang, Zhang, Ma, Shao, Wang, Leung, "Predictive and adaptive deep coding for wireless image transmission in semantic communication" (IEEE TWC 2023, vol. 22 no. 8, pp. 5486-5501, DOI 10.1109/TWC.2023.3234408) | DeepJSCC-V, a variable-code-length DeepJSCC; an oracle network predicts each image's PSNR from its content, the SNR and the compression ratio (mean error < 0.5 dB); a rate optimiser gives each image the least ratio that meets a PSNR floor | **"predict per-image quality, then choose the rate" is in TWC already.** Its rule is per-image (a quality target), not a joint split of a shared budget; `tools/alloc_policy.py oracle` now reports it as `equal_q` next to the equal-slope rule, on the same curves | HIGH (for the allocation) |
| **JSCCformer-f**, Wu, Shao, Ozfatura, Mikolajczyk, Gündüz ([2306.09101](https://arxiv.org/abs/2306.09101), IEEE TWC 2024, vol. 23 no. 9, pp. 11904-11919) | transformer JSCC over several blocks with channel-output feedback; in its variable-rate mode the transmitter stops once the decoder's estimate (known through feedback) reaches a target PSNR | **content- and channel-adaptive length with a transformer, in TWC, from the DeepJSCC-MIMO group** -- but closed-loop: it needs a feedback link and sends block by block; the allocation here is open-loop and decided before sending | HIGH-MED |
| NTSCC, Dai et al. ([2112.10961](https://arxiv.org/abs/2112.10961), IEEE JSAC), improved NTSCC ([2303.14637](https://arxiv.org/abs/2303.14637)), MR-NTSCC ([2505.12740](https://arxiv.org/abs/2505.12740)) | a learned entropy model (hyperprior) sets how many channel symbols each latent element gets | content-adaptive variable length per element; strongest-PSNR family; needs side information for the lengths | HIGH (baseline) |
| Semantic comm. with entropy- and channel-adaptive rate control ([2501.15414](https://arxiv.org/abs/2501.15414)) | rate from feature entropy, CSI and SNR, multi-user MIMO | adaptive rate in a wireless system setting | MED |
| SA-OOSC ([2509.07436](https://arxiv.org/abs/2509.07436)); distributed NTSCC ([2503.21249](https://arxiv.org/abs/2503.21249), [2506.07391](https://arxiv.org/abs/2506.07391)) | variable length per patch from (MLLM-distilled) importance; learnable rate tokens | per-patch variable length, rate tokens as decoder conditioning | LOW-MED |
| E2EC variable-length digital JSCC ([2511.07826](https://arxiv.org/abs/2511.07826)); VL-SCC; Rateless DeepJSCC / NTRSCC ([2603.21616](https://arxiv.org/abs/2603.21616)) | variable-length digital codewords via policy gradients; rateless LT-coded JSCC | variable length and rateless operation, digital | LOW-MED |
| SA-RA-JSCC ([2606.17940](https://arxiv.org/abs/2606.17940), Jun 2026) | the SNR becomes one shared semantic vector that rescales all encoded features; a component aware of the rate budget | SNR and budget conditioning of a single model (like the FiLM here), no per-image length | LOW-MED |
| HJSCC, hierarchical VAE ([2408.16340](https://arxiv.org/abs/2408.16340), AAAI 2025) | coarse-to-fine latents, bandwidth by the levels sent | depth-nested bandwidth adaptation | MED (for the cascade) |
| Multi-exit JSCC, Zhang et al. ([2403.11693](https://arxiv.org/abs/2403.11693)) | CNN encoder with several downsampling modules, early exit, module-by-module training | the cascade's mechanism (depth exits) | HIGH (for the cascade) |
| DiT-JSCC ([2601.03112](https://arxiv.org/abs/2601.03112)) | diffusion-transformer decoder; per-image split of a fixed budget by a complexity score | per-image budget heuristic | LOW |

## B. Transformer / state-space JSCC backbones (the "ViT beats Swin" claim)

| work | backbone | relation | threat |
|---|---|---|---|
| DeepJSCC-MIMO, Wu, Shao, Bian, Mikolajczyk, Gündüz ([2309.00470](https://arxiv.org/abs/2309.00470), IEEE TWC 2024, vol. 23 no. 10, DOI 10.1109/TWC.2024.3422794; ICC 2023 version [2210.15347](https://arxiv.org/abs/2210.15347)). **Read in full** | ViT (d 256, 8 blocks, 8 heads per side; 8x8 patch grid on CIFAR); every patch token is mapped by ONE linear layer to a fixed number of channel symbols, the whole codeword power-normalised; a "channel heatmap" (each symbol's noise variance after ZF / SVD equalisation) goes to the decoder (and to the encoder with CSIT), which learns power allocation over eigenmodes; Siamese decoder input layer; MSE (+0.1 LPIPS). One model per bandwidth ratio (SNR-universal over 0-22 dB). "Adaptive-M": one model sends only the first M_i antenna rows and the receiver zero-pads (<= 0.6 dB below per-M models). Complexity table: ViT beats CNN and Swin DeepJSCC with fewer parameters and FLOPs (CIFAR10, R 1/24, 5 dB: 24.41 vs 23.54 / 23.61 dB at 8 blocks). Kodak results from ImageNet 128 px training. Its conclusion names variable-length JSCC "depending on the channel state as well as the input signal" as future work | **already has: a ViT JSCC with patch tokens mapped straight to symbols (this tree's ViT at a fixed rate is nearly the same architecture), "ViT beats Swin at lower cost", and truncate-and-zero-pad adaptation of one model (over antennas).** Does not have: any rate adaptivity (one model per ratio), per-image length, allocation, SISO high-resolution matched-compute results. See `docs/ASSESSMENT.md` 1e | HIGH |
| MambaJSCC ([2405.03125](https://arxiv.org/abs/2405.03125)), adaptive MambaJSCC ([2409.16592](https://arxiv.org/abs/2409.16592)) | visual state-space blocks | reports +0.1 dB (AWGN) to +0.5 dB (Rayleigh) over SwinJSCC at ~half the MACs; a current SOTA baseline | HIGH (baseline) |
| FAJSCC ([2504.04758](https://arxiv.org/abs/2504.04758)) | importance-aware attention, adjustable encoder / decoder complexity | complexity-adaptive single model | LOW |
| Implicit-JSCC ([2606.29737](https://arxiv.org/abs/2606.29737), Jun 2026) | encoder and decoder as fixed-point equilibria; the solver's convergence sets the depth, which theory ties to the SNR | "harder conditions need more depth", argued for the SNR; the cascade tested it for the rate and found nothing (`docs/NOTES.md`) | LOW |

## C. Token selection and token communications

| work | what | relation | threat |
|---|---|---|---|
| Devoto et al., adaptive semantic token selection ([2405.02330](https://arxiv.org/abs/2405.02330)); adaptive semantic token communication for edge inference ([2505.17604](https://arxiv.org/abs/2505.17604)) | ViT JSCC; attention-based selection of which patch tokens to send at a user-set rate; token count and embedding width adapted | token count as the rate knob in ViT JSCC (selection, not prefix order; task-oriented) | MED |
| Video DeepJSCC with dynamic token selection ([2411.09936](https://arxiv.org/abs/2411.09936)) | drops less important tokens, keep ratio sets the length | content-adaptive token length (video) | MED |
| TokCom, Qiao et al. ([2502.12096](https://arxiv.org/abs/2502.12096), IEEE Wireless Commun. Mag.); ToDMA ([2502.06118](https://arxiv.org/abs/2502.06118), [2505.10946](https://arxiv.org/abs/2505.10946)); ATS-ToDMA ([2607.03520](https://arxiv.org/abs/2607.03520), Jul 2026) | discrete tokens of foundation-model tokenizers as the unit; token-domain multiple access; adaptive token selection | owns the phrase "token communications" (discrete, generative); yours are analog JSCC tokens: say so explicitly | MED |
| TokCom-UEP ([2511.22859](https://arxiv.org/abs/2511.22859)); ISFR, importance-ordered restructuring with nested dropout for UEP ([2604.00595](https://arxiv.org/abs/2604.00595)); TONIC ([2605.21553](https://arxiv.org/abs/2605.21553)) | protection matched to token importance; nested windows; importance ordering | ordered tokens + unequal protection | LOW-MED |
| Text-guided token communication ([2507.05781](https://arxiv.org/abs/2507.05781)); joint semantic-channel coding and modulation for token comm. ([2511.15699](https://arxiv.org/abs/2511.15699)) | discrete tokens with channel codes, generative recovery | digital token pipelines | LOW |

## D. Adaptive-length visual tokenizers (the ML side of "AdaTok's flow")

| work | what |
|---|---|
| **AdaTok**, Lu et al., "Self-budgeting image tokenization with quality-preserving dynamic tokens" ([2606.07185](https://arxiv.org/abs/2606.07185), Jun 2026) | tail masking + a per-budget LoRA decoder head (rank 16; rFID 2.04 -> 1.63 over tail masking alone) + a GRPO policy that picks each image's token count (rFID 1.50 at ~118 tokens vs 1.31 at 256). Not to be confused with AdaTok for MLLM token compression ([2511.14169](https://arxiv.org/abs/2511.14169)). |
| FlexTok ([2502.13967](https://arxiv.org/abs/2502.13967), ICML 2025) | 1D ordered tokens, nested dropout, rectified-flow decoder, 1-256 tokens from one model |
| One-D-Piece ([2501.10064](https://arxiv.org/abs/2501.10064)) | TiTok + tail token drop; quality-controllable compression vs JPEG / WebP |
| ElasticTok ([2410.08368](https://arxiv.org/abs/2410.08368), ICLR 2025); ALIT ([2411.02393](https://arxiv.org/abs/2411.02393), ICLR 2025); STAT soft tail-dropping ([2601.14246](https://arxiv.org/abs/2601.14246)) | adaptive token count per image / frame |
| Fu et al. ([2601.01535](https://arxiv.org/abs/2601.01535)) | tail truncation crowds information into early tokens (a known cost of nesting) |
| Nested dropout, Rippel et al. ([1402.0915](https://arxiv.org/abs/1402.0915), ICML 2014); Matryoshka representation learning | ordered / nested representations, the root of all prefix codes |

This tree's own wave 1 is evidence on this side too: an AdaTok/TiTok-style 1D latent tokenizer
trained with MSE on DIV2K reached ~21-22 dB on Kodak and did not turn tokens into detail, while
position-tied phase tokens worked. AdaTok's own recipe (proxy codes, perceptual and adversarial
losses, ImageNet-scale data) was not reproduced.

## E. Model size of the baselines (fairness)

* MambaJSCC's complexity table (setting not stated in the excerpt): SwinJSCC 12.0-18.4 M params,
  26-33 GMACs; MambaJSCC 9.9-19.5 M params, 17-27 GMACs.
* A remote-sensing paper ([2609.20150](https://arxiv.org/abs/2609.20150)) counts SwinJSCC's transmission
  modules at 33 M params and 34 GFLOPs.
* Measured here (`docs/NOTES.md`, per 256x256 image, encoder + decoder): this tree's Swin base
  5.3 M / 18.5 GFLOPs, Swin large 12.7 M / 41.8 GFLOPs, ViT p8 768x10 142 M / 291 GFLOPs.

So this tree's Swin baseline is smaller than the published SwinJSCC, and the ViT that beats it is
16x its compute. Swin large is the published size class.

## F. Multi-user bandwidth allocation with learned JSCC

| work | what | relation |
|---|---|---|
| Reliable online resource allocation for multi-user semantic comm. ([2604.10931](https://arxiv.org/abs/2604.10931)) | an oracle network predicts PSNR, constrained Bayesian optimisation picks compression ratios | **predict quality, then allocate**: the same pattern as `alloc/`. **Read in full.** |
| MU-ASCC ([2509.24247](https://arxiv.org/abs/2509.24247)); SFMA pairing and bandwidth allocation ([2604.09261](https://arxiv.org/abs/2604.09261)); JSCC capacity resource allocation ([2211.11412](https://arxiv.org/abs/2211.11412)); MU-MIMO MAC image transmission ([2504.07969](https://arxiv.org/abs/2504.07969)); distributed DeepJSCC over a MAC ([2211.09920](https://arxiv.org/abs/2211.09920)) | multi-user bandwidth / power / pairing with learned JSCC over a few discrete rates | the wireless framing a TWC paper on allocation joins |

No paper found derives an equal-slope (Lagrangian) allocation across images for a learned JSCC
at a fixed mean CBR, but that is classical rate allocation (Shoham and Gersho 1988; Ortega and
Ramchandran 1998) applied to a new codec, and reviewers will say so. The closest learned-JSCC
rule is PADC's (section A, IEEE TWC 2023): predicted per-image PSNR, then each image's least
rate meeting a quality floor. At a fixed total that is the equal-quality (max-min) rule;
`tools/alloc_policy.py oracle` prints it as `equal_q` beside the equal-slope oracle on the same
curves, so the margin of the rule itself is measured. What is less common: a
single prefix code fine enough (one token = 4 complex symbols per 256 px tile) that the
allocation is exact with time-sharing and needs no side information.

## G. Theory anchors

* Successive refinement (Equitz and Cover 1991; Rimoldi 1994): a Gaussian source under MSE is
  successively refinable, so an embedded (prefix) code can be optimal at every rate at once.
  `tools/toy_nesting.py` shows the analog-linear version: one shared power shape loses <= 0.2 dB.
  A learned code's 0.2-0.5 dB penalty is therefore a capacity / optimisation effect, not a limit.
* Lagrangian rate allocation over independent operational R-D curves (convex hull, equal slopes,
  time-sharing): the optimality argument for `alloc/core.py`.

## H. Novelty matrix (Y = has it, per the summaries)

| | prefix code, any length | position-tied tokens | per-budget decoder specialisation | per-image length policy | zero side info | channel-aware multi-user allocation | analog JSCC |
|---|---|---|---|---|---|---|---|
| Yang and Kim 2022 | Y (few rates) | - | - | Y (Gumbel) | Y | - | Y |
| Timeliness-aware JSCC | Y (tokens) | ? | (CBR as input) | Y (PPO) | ? | - | Y |
| TS-JSCC 2026 | Y (prefix) | - (Swin channels) | (λ modules) | Y (sparsity) | ? | - | Y |
| DeepJSCC-l++ | discrete ratios | - | (ratio as input) | - | - | - | Y |
| NTSCC family | - | - | (rate tokens) | Y (entropy) | - (lengths sent) | some | Y |
| Devoto et al. | selection, not prefix | Y | - | user-set | - | - | Y |
| AdaTok (tokenizer) | Y | - (1D) | Y (LoRA heads) | Y (GRPO) | n/a | n/a | n/a (discrete) |
| 2604.10931 | - | - | - | - | - | Y | Y |
| DeepJSCC-MIMO (TWC 2024) | - (one model per rate; antenna rows truncated, 3 levels) | Y (patch -> symbols) | (CSI heatmap) | - (named as future work) | Y (fixed length) | - (single link; power over eigenmodes) | Y |
| PADC (TWC 2023) | Y (DeepJSCC-V, granularity ?) | ? | ? | Y (predicted PSNR + quality floor) | ? | - (per image) | Y |
| JSCCformer-f (TWC 2024) | Y (block by block) | ? | (refines per block) | Y (stops at a target, via feedback) | - (feedback link) | - | Y |
| **this tree** | **Y, 1-token steps** | **Y** | **Y: FiLM + LoRA heads + per-rate exits** | **Y: predicted curves + equal slope** | **Y (counts tokens)** | **possible (curves per SNR)** | **Y** |

No single row matches this tree's, but every column is taken by someone. Novelty has to come from
the combination, the evidence, and a wireless problem it solves better (docs/ASSESSMENT.md).
