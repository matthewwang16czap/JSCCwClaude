# Assessment: depth-cascade rate adaptation for deep JSCC

Written before any cascade run existed; wave c1 has now run (`docs/NOTES.md`, "Cascade
wave c1: results"). "TWS" is read as IEEE Transactions on Wireless Communications (TWC).

## Update after wave c1 (2026-10-08): the bottleneck cascade fails its gates

| gate | outcome |
|---|---|
| G1 any arm >= +0.3 dB mean | **fails**: best +0.049 dB (vit), +0.035 dB (swin) |
| G2 the linear arm helps | it takes ALL of the gain (vit +0.048, swin +0.019); bodies add <= 0.01 dB |
| G3 same sign on both interfaces | holds, at +0.02 to +0.05 dB |
| G4 1/8 gain >= 0.25 dB | **fails**: +0.08 (default), +0.14 with `--cascade-grad 0.5`, paid with -0.21 at 1/48 |
| G5 from scratch | not run; low prior (below) |
| G6 overhead | linear: +0.02 M (vit), +0.09 M (swin) |

**Verdict.** The depth cascade at the bottleneck, as proposed, does not remove the nesting
penalty and is not a paper method, for TWC or anywhere. The gain is real on Kodak but at
the level of seed noise, and a per-rate LINEAR map gets all of it, so the extra depth,
spatial context and depth schedule (the substance of the proposal) do nothing.

**What it does establish**, which is worth keeping: the penalty is not in the code layout at
the bottleneck. The one knob with an effect, gradient sharing, acts on the shared encoder
and only moves quality between rates. So the rates compete for backbone capacity.

**G5 (from scratch) has a low prior**: starting from scratch changes how the top code is
organised, not the two findings that bottleneck capacity is worth nothing and that the
rates compete in the backbone. Run it only if wave c2 puts the gap in the code layout.

**Where the idea can still live.** "Each rate its own network, sharing the backbones" may
still be right with the per-rate part moved to where the rates compete. Wave c2
(`COMMANDS.txt`) measures the fixed-rate specialist gap under this protocol and splits it
between encoder and decoder:
* gap mostly in the DECODER: per-rate low-rank adapters inside the decoder backbone. The
  decoder already runs once per rate, so they cost no compute; this is the cascade idea
  moved into the backbone, and a TWC-sized study if it closes most of the gap;
* gap mostly in the ENCODER: per-rate branches of the last encoder blocks (the original
  U-Net picture with the split moved earlier; costs an encoder pass per rate when several
  rates are trained), or report `--cascade-grad` as a rate-priority knob;
* gap small under this protocol: the nesting penalty is a protocol artefact of the tail
  probes, and the paper has to be about something else.

The sections below are the pre-registration as written before c1; they stand as the record.

## Verdict (before c1)

1. **The mechanism is not new; the study around it can be.** Rate adaptation by exiting
   at different depths of a JSCC autoencoder (the encoder has several downsampling
   modules, the decoder can exit early, the latent size follows the exit) is published
   (Zhang et al., arXiv 2403.11693, CNN-based, inside a beamforming paper), and
   hierarchical-VAE JSCC (AAAI 2025) varies bandwidth by the number of
   hierarchical levels sent. Pitched as "a stack of modules with early exits", this
   would be read as an incremental variant. I could read these two papers only through
   search summaries (the sandbox blocks arXiv): read them in full before writing.
2. **As an architecture trick alone, TWC is unlikely.** It becomes a plausible TWC
   submission only as a complete study with analysis, wireless-system content and
   strong, well-controlled evidence (section 4). Otherwise IEEE TCCN, the JSAC / WCL
   routes, or ICC / Globecom first.
3. **The strongest asset is not the architecture but a measured problem**: the nested
   (prefix) code wastes its tail, which this tree has already shown with probes
   (`docs/NOTES.md`, tail probes): -0.02 dB from 1/12 to 1/8 without noise against +1.00 dB for a
   fixed-rate code, and a 1/8 specialist beating the nested ViT by 0.29-0.71 dB. A paper
   that characterises this "nesting penalty", shows what causes it, and removes it for
   ~2% extra parameters (ViT) would have a story, if the cascade does remove it. If the cascade recovers little of it, there is
   no paper in this direction.
4. **The honest weak points**: the penalty is not explained by linear theory
   (`tools/toy_nesting.py`: <= 0.2 dB at 1 dB SNR, ~0 at 7-13 dB), the cascade loses
   incremental refinement, rates become discrete, and Swin pays +33% parameters.

## 1. What is being proposed, in the field's terms

Rate adaptivity in deep JSCC splits into two families:

* **Width-nested (prefix, mask, ordering).** One code; the rate is how much of it is sent.
  The encoder is pushed to order its features by importance.
* **Depth-nested (exits, hierarchy).** The code has stages; the rate is the stage you send.
  Smaller codes are produced by more processing. This proposal.

## 2. Related work

Width-nested / conditioned (what the cascade replaces):

| work | rate mechanism | note |
|---|---|---|
| DeepJSCC-l, Kurka and Gündüz, [arXiv 2009.12480](https://arxiv.org/abs/2009.12480) (IEEE TWC 2021) | layered successive refinement and multiple descriptions, CNN | incremental delivery; negligible loss vs single transmission is claimed |
| DeepJSCC-l++, [arXiv 2305.13161](https://arxiv.org/abs/2305.13161) | one Swin/ViT model fed the bandwidth ratio and SNR; loss weights set from per-rate quality | the closest transformer baseline for "one model, many rates" |
| Yang and Kim, [arXiv 2110.04456](https://arxiv.org/abs/2110.04456) (ICASSP 2022) | thermometer-coded channel mask from a policy network, Gumbel-softmax | content-adaptive rate, one network |
| SwinJSCC, [arXiv 2308.09361](https://arxiv.org/abs/2308.09361) | spatial modulation modules scale the latent by SNR and rate | the feature-transmission baseline of this tree |
| Token communications (TokCom), [arXiv 2502.12096](https://arxiv.org/abs/2502.12096) and adaptive semantic token communication, [arXiv 2505.17604](https://arxiv.org/abs/2505.17604) | tokens as the unit; token selection and embedding dimension set the rate | the token-transmission framing |
| nested dropout ([arXiv 1402.0915](https://arxiv.org/abs/1402.0915)), PLONQ ([2102.02913](https://arxiv.org/abs/2102.02913)), ProgDTD, Matryoshka / FlexTok ([2502.13967](https://arxiv.org/abs/2502.13967)) | ordered / nested representations by tail dropping | the prefix scheme's lineage; they all pay for ordering |

Depth-nested (what the cascade is):

| work | mechanism | difference to this proposal |
|---|---|---|
| Zhang et al., Beamforming Design for Semantic-Bit Coexisting Communication System, [arXiv 2403.11693](https://arxiv.org/abs/2403.11693) | multi-exit JSCC: several downsampling modules (residual block + conv, halving the image, fixed channels) in the encoder, early exit in the decoder, module-by-module training | CNN, spatial halving, not benchmarked against a width-nested model as far as the summaries show; no token / feature split |
| Learned image transmission with a hierarchical VAE, [arXiv 2408.16340](https://arxiv.org/abs/2408.16340) (AAAI 2025) | latents of smaller dimension at coarser levels; bandwidth by levels sent, with a rate-attention module | top-down hierarchy, successive refinement, different mechanism |
| FAJSCC, [arXiv 2504.04758](https://arxiv.org/abs/2504.04758) | adjustable encoder / decoder complexity, not rate | related only in sharing one trained model across operating points |

Training of shared multi-operating-point networks (borrowed, not new): universally
slimmable networks, sandwich rule and in-place distillation ([arXiv 1903.05134](https://arxiv.org/abs/1903.05134));
slimmable compressive autoencoders ([arXiv 2103.15726](https://arxiv.org/abs/2103.15726), rate by width);
multi-exit networks (BranchyNet, MSDNet).

## 3. What is and is not new here

Not new: exits at depth as the rate control; sandwich-style sampling; shared backbones with
per-operating-point heads.

Defensible as new, if the experiments support it:

1. **A strict superset of the prefix scheme.** Zero-initialised residual stages over a
   channel-truncation skip make the cascade bit-exact to a prefix model at step 0
   (checked), so it can be warm-started from any prefix checkpoint and cannot start worse.
   Earlier multi-exit JSCC trains from scratch with fresh heads.
2. **The nesting penalty, measured and attacked.** Tail probes plus a cascade that targets
   them, on the same backbone and the same training budgets, with a control trained on
   the same five rates.
3. **One construction on two interfaces.** The same stages on feature transmission (Swin
   grid, real channels) and token transmission (ViT phase tokens, per-level power profile
   and DC handling), with a parameter-matched module bake-off per rate.
4. **A training knob with a Pareto reading** (`--cascade-grad`, joint -> greedy): how much
   the top rate may specialise against the low rates.
5. **Free rate signalling and a hook for per-image level choice** (the receiver counts
   symbols; `alloc/` has the equal-slope allocator).

## 4. What a TWC paper would need

TWC reviewers will ask, in this order:

1. *What is new beyond multi-exit JSCC with a Transformer?* Section 3, items 1-4, backed
   by controls below.
2. *Is the gain structure or just parameters?* Required controls: prefix + same-size
   shared adapter (parameter-matched); linear-only cascade; module kinds matched per block
   (built in). Swin's +33% makes this non-optional.
3. *Against what?* Fixed-rate specialists at every rate (the ceiling), the prefix control
   on the same rates, SwinJSCC with its own rate modulation, DeepJSCC-l++, a digital
   BPG + LDPC reference at the same CBR, several seeds, Kodak and CLIC, PSNR + MS-SSIM +
   LPIPS (the tree already scores all, fp32).
4. *What does it do in a wireless system?* AWGN and Rayleigh (the code supports both), SNR
   mismatch, rate switching under fluctuating bandwidth, and per-image level choice
   with the existing allocator. Without a system section it reads as an ML paper.
5. *Theory.* The linear-Gaussian toy says nesting is nearly free, so a theory of why the
   cascade wins cannot be a power-allocation argument. Options with substance: a
   capacity-ordering argument for finite-width networks, or a bound on tandem loss
   (the level codes form a Markov chain X -> Z1 -> ... -> Z4, so a deeper code is a
   function of a shallower one) and when that loss is small. Without any theory, aim at
   TCCN-style venues.
6. *The cost.* No incremental refinement; discrete rates; K decoder-side stage chains to
   store; K decoder passes per step if all levels are trained each step.

Go / no-go from the screening wave (Kodak, paired intervals over images, last checkpoint):

| gate | pass | if it fails |
|---|---|---|
| G1 any module arm beats the grid-trained prefix control | mean >= +0.3 dB over the five CBRs, CI above 0 at >= 3 of them | < +0.1 dB: stop; the nesting penalty is not what the cascade removes |
| G2 the linear arm already helps | > 0 at the low CBRs | the gain, if any, is capacity: run the parameter-matched adapter control before anything else |
| G3 same sign in both interfaces | swin and vit both positive | a one-interface result is a workshop paper |
| G4 top rate | 1/8 gain >= 0.25 dB (the penalty is 0.3-0.7) with the default joint training or with `--cascade-grad` < 1 | no top-rate gain: the nested tail was not the bottleneck |
| G5 from scratch | the gain survives 2000 epochs from scratch, not only fine-tuning | a fine-tune-only effect is an optimisation artefact of warm starting |
| G6 overhead | vit +2%, swin <= 15% with width 64 holding most of the gain | a Swin result that needs +33% is a capacity result |

## 5. Risks, in order of how likely they are to hurt

1. **The gain is within noise.** Differences between identical-seed runs were 0.67 dB per
   validation epoch; screening at 400 epochs from a shared checkpoint reduces but does not
   remove it. Use paired Kodak tables and repeat the best arm with another seed.
2. **Fine-tune screening biases toward the prefix layout.** The warm start keeps `z0`
   ordered; a from-scratch run with `--mod-skip none` or `--cascade-grad` < 1 is the fair
   test of the top rate (wave 2, then the headline).
3. **The control is too weak.** It must be trained on the same five budgets
   (`--rate-sampling grid`), not the continuous prefix model of wave 1; the commands do.
4. **The multi-exit prior art is closer than the summaries suggest.** Read it in full.
5. **Tandem loss.** `z_k` is a function of `z_(k-1)`; if `z0` cannot be both a good top code
   and a good source for compression, the low rates lose. `--cascade-grad` measures it.

## 6. Contribution framings, from safest to boldest

* **Characterisation:** the nesting penalty of prefix JSCC and how much a cascade removes
  (needs G1-G5; fits a letter or a TCCN-style paper).
* **Method + system:** the cascade plus per-image level choice and rate switching over a
  fading channel, with the toy and a tandem-loss bound as the theory (the TWC route).
* **Unified view:** width- vs depth-nesting for feature and token transmission with the
  module taxonomy (needs more backbones than two).
