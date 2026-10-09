# JSCCwClaude

Deep joint source-channel coding experiments (feature transmission with a Swin backbone,
token transmission with a ViT backbone) and rate adaptation by a **depth cascade**: a stack
of small modules between the backbone encoder and decoder, one exit per rate.

Everything lives in [`jscc/`](jscc/):

* [`jscc/README.md`](jscc/README.md): the code, flags, conventions.
* [`jscc/docs/ASSESSMENT.md`](jscc/docs/ASSESSMENT.md): assessment v2 after wave c1: what the
  results support, the size confound, novelty against the literature, an IEEE TWC paper plan,
  go / no-go gates.
* [`jscc/docs/SURVEY.md`](jscc/docs/SURVEY.md): related work (rate-adaptive JSCC, token
  communication, adaptive tokenizers such as AdaTok, bandwidth allocation), by threat level.
* [`jscc/docs/PROBLEM.md`](jscc/docs/PROBLEM.md): the target problem since 2026-10-09:
  opportunistic scheduling of encode-once token prefixes over a fading multi-user downlink,
  why this codec fits it and the published ones do not, baselines, gates.
* [`jscc/docs/CASCADE.md`](jscc/docs/CASCADE.md): design, module choice per rate and per
  interface, training strategy, what it cannot do.
* [`jscc/docs/NOTES.md`](jscc/docs/NOTES.md): what every wave measured, wave c1 (the cascade) included.
* [`jscc/COMMANDS.txt`](jscc/COMMANDS.txt): the current wave (matched compute, allocation
  oracle, specialists, LoRA budget heads, seeds; section 5: mixed-SNR prefixes for the target
  problem), copy-paste ready (GPUs 2-5 only).
