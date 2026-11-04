# Writing rules for the paper

The authors' rules for every sentence of the paper, set on 2026-11-04. `paper/style_check.py`
checks rules 1 to 3 mechanically (`python3 style_check.py sections/introduction.tex 18 45` checks
lines 18 to 45 of that file).

1. No colon.
2. No semicolon.
3. Every sentence is complete and grammatical, with at most two commas. Each comma-separated part
   is neither too short nor too long.
4. The register is that of an IEEE TWC paper. No jokes, idioms or casual phrasing.
5. No sentence repeats one that appears earlier in the paper, for example in paragraph 1 of
   Section I. When a point has to come back, it is expressed differently.

Also from the authors and the supervisor:

- A result widely known in the field, such as deep JSCC mapping an image directly to channel
  symbols, needs no citation.
- Cited works are from top venues of the last three years or highly cited, with the exceptions
  listed in `docs/RELATED_SOURCES.md`.
- Every statement about a cited work matches the work's own text. Nothing is generated without a
  source.
- The paper carries no revision comments.
- The paper follows the IEEE template (`paper/IEEEtran.cls`, V1.8b).
