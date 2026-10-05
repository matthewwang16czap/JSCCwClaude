# JSCCwClaude

Deep joint source-channel coding experiments (feature transmission with a Swin backbone,
token transmission with a ViT backbone) and rate adaptation by a **depth cascade**: a stack
of small modules between the backbone encoder and decoder, one exit per rate.

Everything lives in [`jscc/`](jscc/):

* [`jscc/README.md`](jscc/README.md): the code, flags, conventions.
* [`jscc/docs/ASSESSMENT.md`](jscc/docs/ASSESSMENT.md): the cascade idea judged against the
  literature, novelty, IEEE TWC fit, go / no-go gates.
* [`jscc/docs/CASCADE.md`](jscc/docs/CASCADE.md): design, module choice per rate and per
  interface, training strategy, what it cannot do.
* [`jscc/COMMANDS_CASCADE.txt`](jscc/COMMANDS_CASCADE.txt): the current wave, copy-paste ready.
