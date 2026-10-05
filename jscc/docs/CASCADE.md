# Depth cascade: one module path per rate, shared backbones

Design note for `--cascade` (`net/cascade.py`). What it is, why, which module goes where
and why, how it is trained, what it cannot do, and how to read the experiments in
`COMMANDS_CASCADE.txt`. Nothing here has been trained yet: the code is checked on CPU for
invariants (`tools/smoke.py`, `[cascade]`), the choices below are hypotheses with a test
each.

## 1. The idea

The prefix scheme makes the rate adaptive by sending the head of one ordered codeword:
the first *k* channels (feature transmission) or the first *l* tokens (token
transmission). One decoder path serves every rate, and the encoder has to arrange its
information so that every head is a good code of its own length.

The cascade gives each rate its own physical path and shares the backbones:

```
 image -> backbone encoder -> z0 ---------------------------------> channel -> (none) ---------+
                               \                                                               |
                                e1 -> z1 ------------------------> channel -> d1 --------------+
                                       \                                                       |
                                        e2 -> z2 ----------------> channel -> d2 -> d1 --------+--> backbone decoder -> image
                                               \                                               |
                                                e3 -> z3 --------> channel -> d3 -> d2 -> d1 --+
                                                       \                                       |
                                                        e4 -> z4 > channel -> d4 -> d3 -> d2 -> d1
```

* `z0` is the top rate (CBR 1/8), `z1` 1/12, `z2` 1/16, `z3` 1/24, `z4` 1/48. Each code has
  fewer channels than the one before: the stack narrows like a U-Net's encoder.
* Rate *k* transmits all of `z_k`. The receiver runs the decoder modules `d_k ... d_1`
  (its own, per rate) and then the unchanged backbone decoder. Exits are "early" on the
  encoder side (the deeper modules are skipped) and the decoder side mirrors them.
* A module sees the CLEAN code of the level above, held by the transmitter. No noise
  accumulates along the stack.
* The receiver counts the symbols that arrive: that number is the level. Nothing is
  signalled.

Intuition behind it: a smaller code needs more nonlinear work to compress into and
decompress out of, so the deeper the exit, the more computation (depth) its path gets,
while the backbones, which do the heavy lifting, are paid for once.

## 2. Why this should beat the prefix scheme (and what the evidence is)

Evidence from this tree (`COMMANDS_TAIL.txt`, ViT, Kodak):

| | nested (prefix) code | fixed-rate 1/8 code |
|---|---|---|
| tail gain 1/12 -> 1/8, no noise | -0.02 dB | +1.00 dB |
| tail gain, trained -2..22 dB, at 13 dB | +0.21 dB | +1.01 dB |

The nested code's last third carries almost nothing, and a 1/8 specialist beats the
nested model at 1/8 by +0.29 dB (1 dB SNR) to +0.71 dB (22 dB). The cascade attacks
exactly this: the top code `z0` is no longer a prefix of anything it must also serve
short, and each lower rate has a code trained for it, with its own power profile.

What does NOT explain it: `tools/toy_nesting.py` computes the nesting penalty of a
linear-Gaussian code (one shared power shape vs the best shape per rate). It is at most
0.2 dB at 1 dB SNR with a steep spectrum and ~0.00 dB at 7-13 dB. Successive refinement
is almost free for Gaussians under MSE. So the 0.3-0.7 dB the nested ViT loses lives in
the nonlinear, finite-capacity regime (ordering constraints on learned features,
optimisation, sampling), and any argument for the cascade has to be made there,
empirically. Do not claim a linear-theory reason.

## 3. A stage

```
 out = skip(in) + body(in)          body = in_proj -> depth x block -> out_proj (ZERO-init)
```

* **skip** (`--mod-skip`): `proj` (default) a learned linear map initialised to channel
  truncation (encoder) / zero padding (decoder); `trunc` the fixed truncation; `none`.
* **zero-init `out_proj`**: at step 0 every level is exactly the prefix scheme's
  truncation, whatever the body. A cascade loaded from a PREFIX checkpoint (`--pretrained`
  of a swin or vit run) therefore reproduces that model at all five CBRs before any
  training (max difference 0.0 for swin, 3e-7 for vit; checked in `tools/smoke.py`).
  Training can only add to the nested code. A stage with depth 0 and skip `trunc` is the
  prefix scheme inside this framework.
* **per-level token post-processing** (vit): each level has its own gain, DC estimate and
  prenorm (`TokenPost`), seeded from the prefix tables on load. A per-level gain is a
  per-rate power profile, the one thing a shared head cannot give.
* **fp32**: the modules run with autocast off; their outputs face the power constraint
  and the DC estimate (the collapse of the first token build came from exactly that
  interface).

## 4. Which module where, and why

Three body kinds, parameter-matched per block (8 w^2, checked by the smoke test):

| kind | mixes | notes |
|---|---|---|
| `mlp` | channels of one position | LayerNorm -> Linear(w, 4w) -> GELU -> Linear(4w, w) |
| `attn` | channels + all positions of a tile / image | global attention + MLP(2w); fixed sinusoid positions (scale 0.25) |
| `swin` | channels + a shifted 8x8-position window | Swin block + MLP(2w); relative positions, any image size |

Defaults (`MODULE_DEFAULTS` in `configs/config.py`), one entry per stage in the order
1/8 -> 1/12 -> 1/16 -> 1/24 -> 1/48:

| | feature transmission (`swin`) | token transmission (`vit`) |
|---|---|---|
| code `z_k` per position | 192 / 128 / 96 / 64 / 32 real channels of a 16x16-px position | 48 / 32 / 24 / 16 / 8 reals of an 8x8-px patch = 6 / 4 / 3 / 2 / 1 tokens of 4 complex symbols |
| kinds | mlp, mlp, **swin**, **swin** | mlp, mlp, **attn**, **attn** |
| depths | 1, 2, 3, 4 | 1, 2, 3, 4 |
| hidden width | 96 | 128 |
| extra parameters | 1.76 M (+33% of 5.3 M) | 2.71 M (+1.9% of 142 M) |
| level seen by the decoder | zero-padded estimate of `z0` -> the baseline's linear expand | estimate of `z0` -> the per-phase embeddings (bias and phase identity only for phases that were sent) -> fold |

Reasoning, each a hypothesis with a test:

1. **Rate-specific stages beat truncation** because they remove the ordering constraint
   and give each rate its own power profile. Test: linear (depth 0) vs the prefix control.
2. **Capacity beyond a linear map helps** because compressing 64 -> 32 channels needs a
   nonlinear fold of the dropped channels into the kept ones. Test: mlp vs linear.
3. **Spatial context helps at low rates, not at high ones.** The first stages (1/8 -> 1/12
   -> 1/16) drop 64 / 32 channels per position and only have to fold them into the kept
   ones, which a per-position map can do. The deep stages (-> 1/24 -> 1/48) must decide
   WHERE to spend a code too small for per-position detail, so a position needs its
   neighbours. Test: the factorial mmmm / mmss(aa) / ssss(aaaa), read per level.
4. **Depth grows as the rate falls**: narrower code, more nonlinear work, and a narrower
   input makes the extra depth cheap. Test: depths 1 2 3 4 vs 4 3 2 1 (same parameters).
5. **Position-tied codes, channels reduced**: stages keep every position and shrink the
   channels, never relabel or merge positions. Wave 1 showed untied 1D latents (AdaTok)
   losing to position-tied phase tokens; the multi-exit CNN JSCC that halves the spatial
   size per stage is a different design point that is not tried here.
6. **Width** is a free knob that scales the overhead (swin: width 192 would be +120%).
   The default is a guess; it is the first thing to shrink if Swin's +33% matters.

Considered and not built: separate full decoders per rate (K x parameters, kills the
point), conv bodies (the swin window covers local mixing), entropy / hyperprior models
(analog symbols have no entropy coder), spatially downsampling stages (point 5).

## 5. Training

* **Levels, not budgets.** `--cascade` forces grid sampling (default) or `sandwich`
  (always the smallest and the largest level, plus random ones). Compare it with a prefix
  control trained on the same five budgets (`--rate-sampling grid`): the continuous-rate
  prefix model is not the fair control, it is evaluated off its training distribution.
* **Compute.** The encoder backbone runs once; the module chain up to the deepest sampled
  level once; the decoder backbone once per sampled level. With `--rates-per-step 1` the
  cost per step equals the prefix model's plus the modules. A deep module only gets a
  gradient on steps that sample a level at least as deep (about 1/5 of steps for the
  deepest at k = 1); `--rates-per-step 2` or `sandwich` raises that.
* **Gradient sharing** (`--cascade-grad a`): the gradient a deeper exit sends into the
  shared shallower code is multiplied by `a` at every stage boundary (so by `a^j`, j
  stages down). `a = 1` joint training (default). `a = 0` greedy: every stage is trained
  on a detached input, the backbone only by the top level. In between trades the top
  rate's freedom to specialise against the low rates' ability to shape the code they
  compress. This is the knob for the paper's Pareto figure.
* **Warm start** (fine-tune screening, `COMMANDS_CASCADE.txt`): `--pretrained` a prefix
  checkpoint, `--mod-lr-mult 4` (new, randomly initialised parameters learn faster than
  the converged backbone). The control is the same checkpoint fine-tuned identically with
  `--rate-sampling grid` and no modules, so the only difference is the modules. The
  from-scratch comparison is the headline and comes after the screening.
* **Other knobs:** `--rate-mod none` removes the decoder backbone's FiLM (keeps the d-chains
  as the only rate-specific part); `--mod-skip trunc|none`; `--mod-width`.

## 6. What it does not do

* **Discrete rates only.** The five predefined CBRs. Between them: time-share adjacent
  levels, or a per-image level choice (the `alloc/` machinery's equal-slope rule applies,
  needing curves per level instead of per prefix; the sweep refuses cascade models today).
* **No incremental refinement.** A prefix code can be extended later (send more symbols
  and the earlier ones still count); a cascade's levels are different codes. It serves
  "the bandwidth is known when the image is sent", not progressive delivery or
  incremental redundancy. This is the price of per-rate optimality, and the first thing a
  reviewer will raise.
* `hybrid` is prefix-only; `--cascade` supports `swin` and `vit`.
* Per-sample budgets (a `(B,)` tensor of units) are a prefix feature.
* Not trained or evaluated for Rayleigh fading yet (the channel code supports it).

## 7. Reading the results (`COMMANDS_CASCADE.txt`)

Per CBR, differences against the control, mean over the last 5 validations
(`tools/compare_runs.py`) and on Kodak with paired intervals over images
(`tools/paired.py`). Single validations of identical runs differed by ~0.7 dB in an
earlier tree: believe the paired Kodak table and the 5-validation windows, not one epoch.

| question | comparison | reading |
|---|---|---|
| does a rate-specific path help at all? | linear (d0) - control | > 0 at several CBRs, mostly low ones |
| does capacity help? | mmmm - linear | |
| does spatial context help, and where? | late stages: mmss - mmmm at 1/24, 1/48; early stages: ssss - mmss at 1/12, 1/16 | spatial wins a stage if +0.05 dB with a CI clear of 0 at its level |
| top rate | every arm - control at 1/8 | the nested tail penalty (0.3-0.7 dB) is the target |
| gradient sharing | cg0.5 - default | top rate up and low rates down is the expected trade |
| depth schedule | d1234 - d4321 | d1234 ahead at 1/24, 1/48 supports "deeper for fewer channels" |

Kill criteria are in `docs/ASSESSMENT.md`.
