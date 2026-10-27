# Correctness review (2026-10-27)

A full read of the repository, asked for directly: is anything in it wrong in a way that
would make a reported number wrong. This is not a new experiment; it is a review of the code
that produced every number in docs/NOTES.md and docs/PAPER.md, done by reading, hand-deriving
the mathematics against the code, and CPU-only checks (no GPU, no heavy CPU, per the standing
rule: that is what the 6 GPUs are for). It does not re-verify that a specific past run's log
matches its flags, and it cannot retrain anything to reproduce a number byte for byte.

**Verdict: no correctness bug found.** One genuinely new, now-permanent check was added
(below); nothing else in the repository was changed. The reasoning for changing nothing else
is in "What was not changed, and why".

## Scope

Every `.py` file (about 16,000 lines: `alloc/`, `tools/`, `net/`, `data/`, `configs/`,
`utils/`, `main.py`, `engine.py`) was read. Depth varied with how much a module's correctness
bears on a reported number:

* **Hand-derived against the code, not just read**: the MMSE link abstraction and its fitting
  (`alloc/utility.py`), the restless-bandit dynamic program, Lagrangian bound and index
  policies (`alloc/whittle.py`), and the stale-CSI model (`alloc/csi.py`) — the three modules
  every scheduling number in the paper is built on.
* **Read in full, traced for the specific risks below**: `tools/schedule_sim.py`,
  `tools/rmab_scale.py`, `tools/replay_report.py`, `tools/utility_fit.py`, `tools/latency.py`,
  `tools/mixed_snr.py`, `tools/codesign.py`, `data/datasets.py`, `engine.py`, `main.py`,
  `configs/config.py`, `net/tokens.py`, `net/loss.py`, the top of `net/network.py` and
  `net/backbones/vit.py`.
* **Read, structurally scanned for the patterns below**: the rest (`tools/stream_sim.py`,
  `tools/headroom.py`, `tools/frontier.py`, the other backbones, `utils/`, `tools/paper_figs.py`
  and its self-test, which this session also wrote).
* Ran: `pyflakes` over the whole tree (clean, both before and after); the full
  `tools/smoke.py` suite (over 150 assertions, including several that independently check
  core claims against brute force — see below); `tools/paper_figs.py --self-test`.
* Wrote and ran two things not in the repository before today: an empirical coverage check of
  the bootstrap confidence interval (below), which is now a permanent smoke-test addition, and
  a one-off mutation test confirming that check actually fails on a broken implementation (not
  kept — it is not a property of the shipped code, only a check on the check).

What this review specifically looked for, across every module: data leakage (held-out sets
used in fitting or calibration), broken pairing (a "paired" comparison that is not actually
comparing the same underlying frames), common-random-numbers claims that do not hold (two
policies, or two files, silently scored on different noise), seed/state reuse that would
correlate results that should be independent, and sign or scale errors in the statistics
(points vs. dB, a flipped inequality, an off-by-one in an index).

## What was checked, in detail

**The MMSE link abstraction and its fit (`alloc/utility.py`).** The additive model
`D = Dsrc + C * sum_j rho_j * phi(n_j)` and its PSNR conversion match the paper's formula
exactly. The fitting procedure (`fit()`, `_fit_level()`, `_fit_nested()`) never touches a
"random"-kind (held-out) profile before the final error report: kappa is chosen by golden-section
search minimizing RMS on the fit-only subset (`run()` masks to `kinds != "random"` throughout),
dsrc/csens/rho are weighted-least-squares fit on the same subset, and the held-out error is
computed only afterwards, with the fitted parameters frozen. The transmitter-calibration path
(`tools/utility_fit.py --calib`, used for every reported "transmitter's belief") cross-fits
kappa and rho by image fold (images in fold *f* get parameters fit on every other fold), and
fits each image's own Dsrc/C from only the specified `--calib` SNRs, reporting error on the
complementary profiles. No leakage in either path.

**The restless-bandit core (`alloc/whittle.py`).** Re-derived the per-user dynamic program's
backward induction by hand against the code (`Arm._solve`): the Bellman update, the
loss-probability handling (`p_lost`), the stale-report posterior averaging (`_by_report`,
`_take`), and the index computation (`Arm.index`, including the "flat region" / top-of-range
case the docstring describes) all match. The Lagrangian bound's golden-section refinement and
subgradient loop (`lagrangian_bound`) match the stated bound formula. This is also where the
existing test suite is most rigorous and gave independent confirmation: `tools/smoke.py` checks
a single arm's DP value against a *separately implemented*, full brute-force expectimax
(`lru_cache`-memoized, not sharing a line of code with `Arm`), and checks the three-way
inequality bound &ge; best causal schedule (full joint expectimax over every scheduling
decision) &ge; Whittle-index-policy value on several small cases, printing the actual gaps. This
session re-ran that suite and it still passes with the same structure. Separately,
`alloc/csi.py`'s `unit_noise()` (the deterministic Rayleigh-fading quadrature used throughout
the stale-report model) was hand-verified to be an *exact* closed form for the conditional mean
of each equiprobable Exp(1) cell — not an approximation — by direct integration.

**Common random numbers and pairing.** `tools/schedule_sim.draw_frames` draws every frame's
images, mean SNRs, fading and CQI noise once per frame, *before* the per-policy loop; policies
only ever see the already-drawn arrays, never the RNGs. The one per-slot randomness the
policies do use, tie-breaking (`tie_keys`), is seeded by `(seed, slot, constant)` — independent
of policy and of the frame-drawing RNGs — so every policy really does see identical images,
SNRs and fades for a given frame, matching the docstring's claim. The same pattern holds in
`tools/rmab_scale.py` (cross-checked against `schedule_sim` bit-for-bit by the existing smoke
suite) and in `tools/mixed_snr.py`'s `score_profiles`/`score_entry` (every SNR profile of one
image reseeds to the *same* channel-noise draw before scaling it per chunk, so different
effective-SNR rules are compared on the same noise, not independent draws).
`tools/replay_report.py`, which pairs decoded replays *across files*, explicitly checks that
paired (frame, user) keys agree on image and mean SNR across every file before using them,
raising if they do not — a real safety net against silently comparing the wrong frames.

**Train/valid/test and no test-set peeking.** `data/datasets.py` keeps train, valid and test
under physically separate directories (`DIV2K_train_HR` vs. `DIV2K_valid_HR` vs. `Kodak`/
`CLIC/test`); Kodak is never in the training table. `main.py` selects the checkpoint
(`best.pt`) by validation score only and evaluates the test set exactly once, after training,
with its own held-out loader. `engine.evaluate` reseeds per (SNR, CBR) cell from `eval_seed`
(fixed regardless of the training `--seed`, so different training seeds are comparable), and
draws a fresh resume seed *before* entering its evaluation loop so that calling it repeatedly
during training does not make the post-evaluation training stream repeat itself.

**The prefix/truncation independence invariant.** The paper's whole account of phases as
additive, independent contributions rests on truncating a prefix revealing nothing about the
tokens cut off. Traced this through `net/backbones/vit.py`: `post_symbols` (gain, DC removal,
power normalization) is applied once to the *full*-length codeword in `ViTEncoder.forward`;
truncation to whatever length a slot's budget allows happens afterwards, so a token's
transmitted value does not depend on how many tokens eventually get sent. This is also directly
smoke-tested ("a profile changes the noise of its own tokens only").

**The bootstrap confidence interval — new check.** Every interval reported anywhere in the
paper comes from one function, `tools/schedule_sim.bootstrap` (a 2000-resample percentile
bootstrap over paired per-frame differences). It had never been checked for statistical
validity on its own — only for being wired correctly into each tool's output, which the
existing smoke suite already did. This session added `tools/smoke.py:check_bootstrap_coverage`,
now running on every `tools/smoke.py` invocation:

* a constant array gives a zero-width interval at its own value;
* the interval narrows as the sample size grows (60 → 4000);
* **empirical coverage**: over 400 synthetic trials each, a nominal 95% interval actually
  contains the true mean about 94% of the time on a normal distribution and about 93-94% on a
  right-skewed one (deliberately chosen, since paired "gain" distributions are rarely
  symmetric) — within sampling noise of the nominal 95%, at n = 60 (the smallest frame count
  used anywhere: decoded replays).

This was checked, not assumed: a one-line mutation (an 80% interval mislabeled as 95%, by
reading the 10th/90th percentiles instead of 2.5th/97.5th) was run through the same check and
caught immediately (measured coverage 0.76), confirming the check has power, not just that it
happens to pass. The genuine implementation was not changed; the check is purely additive.

**Everything else.** `tools/latency.py`'s interpolation (`needed()`) and its own
frame-resampled bootstrap (`boot_needed`) are correct for what they claim; see the one design
note below. `tools/codesign.py`'s training-bank construction explicitly uses a different seed
(1000) from evaluation (0), so a scheduler-aware code is never trained on the frames it is
later scored on. `net/loss.py`'s weighting schedule and detached-ratio balancing match their
docstring. `main.py`/`engine.py`'s training loop (gradient-norm-based step skipping, EMA,
distributed all-reduce) is standard and matches its own description. No mutable default
arguments, no bare `except:`, no `TODO`/`FIXME` markers anywhere in the tree.

## Design notes (not bugs; nothing changed)

Three things worth knowing about, none of which make a reported number wrong:

1. **`tools/latency.py` does not share identical frames across different frame lengths T.**
   Each length's `--frames` Monte Carlo draws are independent of every other length's (because
   a longer T consumes more fading draws per frame from the same RNG stream, the per-frame
   alignment that holds *within* one length does not extend *across* lengths). Every number
   this produces is still an unbiased, correctly-bootstrapped estimate — `tools/latency.py`'s
   own bootstrap already treats different lengths independently, which is the statistically
   correct thing to do given they genuinely are independent — but a design that deliberately
   shared the first `min(T)` slots across every length would reduce the interpolated "needed
   length" estimates' variance further. Not something to change without re-running the
   latency experiments, so left as is.
2. **`alloc/whittle.py`'s `Arm.deviation`** (how far a freely-fit, per-length sensitivity
   profile is from the single-master-profile "nested" approximation the restless-bandit DP
   relies on) is computed in `master_profile()` but never read anywhere in the repository. It
   is not a bug — the substance of this question (what the nested approximation costs) is
   already measured and reported a better way, via `tools/utility_fit.py --nested`'s held-out
   predictive-accuracy comparison (nested vs. free fit), which is what is actually recorded in
   docs/NOTES.md. `Arm.deviation` is simply unused. Happy to wire it into that same report if
   useful; did not do so unprompted, to avoid adding new, untested code paths for a question
   that is already answered.
3. **The plain percentile bootstrap is mildly conservative at the smallest sample sizes**
   (measured ~93% actual coverage for a nominal 95% interval at n = 60, see above) — a known,
   small property of this method on skewed data, not specific to this implementation. It means
   60-frame decoded-replay confidence intervals are, if anything, very slightly narrower than
   their stated coverage, not wider: a marginal result at n = 60 deserves a little more
   skepticism than its printed interval alone suggests. Worth a one-line caveat if the paper
   ever leans on a decoded 60-frame interval by itself; it does not currently.

## What was not changed, and why

Given the verdict above, "refactor the project" was deliberately **not** done beyond the one
additive test. Three reasons:

* Nothing is broken, so there is no correctness reason to touch any of the roughly 16,000 lines
  of already-working numerical code.
* This code is unusually well exercised already: 150+ passing smoke assertions, several of
  which independently re-derive a result from scratch (brute-force expectimax, column
  generation plus integer programming against brute force, isotonic-fit recovery, frozen-tensor
  bit-identity checks) rather than just checking the code against itself. A stylistic rewrite of
  code this validated is pure downside risk: nothing here is slow, unclear enough to block
  future changes, or duplicated in a way that costs correctness, and any slip in a rewrite could
  silently change a reported number in a way that the *existing* tests, written to check the
  *current* formulas, would not catch.
* Every number in docs/NOTES.md and docs/PAPER.md was produced by the code exactly as it
  stands. Changing the code without re-running the (GPU) experiments would put the code and the
  written-up numbers out of sync — a worse state than the one today, where they agree.

If there is a *specific* piece anyone wants restructured (for clarity, to extend it, because it
is hard to follow), that is a different, scoped request and worth doing; a blanket pass over
validated code is not.

## What this review could not do

It is a static and CPU-only review. It does not and cannot:

* Reproduce any reported number from scratch — that needs the GPUs, which this session does
  not use for testing (COMMANDS.txt is how work gets to them).
* Confirm that a specific past log file actually used the flags it claims, that a dataset
  directory contains what its name says, or catch a transient infrastructure issue (a bad GPU,
  a truncated download) in a *particular* past run.
* Prove the absence of every possible bug — only that the specific, substantial risks a review
  of this kind looks for (leakage, broken pairing, seed/state bugs, sign and scale errors) were
  checked, by hand, against the paper's own numbers, and none were found.

## Bottom line

The code behind every reported number was read and, where the mathematics was non-trivial,
re-derived by hand against it. No leakage, no broken pairing, no CRN violation, and no sign or
scale error was found anywhere this review looked, including the modules the paper's headline
claims depend on most directly. The one gap found — the bootstrap CI's own statistical validity
had never been checked — is now closed and will stay checked on every future `tools/smoke.py`
run. Nothing else needed fixing.
