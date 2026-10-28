# Introduction: paragraph-by-paragraph outline (drafted 2026-10-28)

Maps the supervisor's structure (`/root/.claude/uploads/.../paper_instruction.txt`, section 1) onto
this paper's content. This is the OUTLINE only — what each paragraph argues and which already-measured
numbers it draws on (all from docs/PAPER.md and docs/NOTES.md; nothing here is a new claim). The
prose itself is next turn. Paragraph numbers match the supervisor's six; two open decisions are
flagged where the template asks for something this project has not settled yet.

A freshness check today (two targeted searches; not a full new survey pass) found nothing that
changes the gap statement below: no paper combines an effective-SNR-style link abstraction with
multi-user restless-bandit scheduling for a learned code. One adjacent, newer citation worth adding
to Related Work later: "Deep Index Policy for Multi-Resource Restless Matching Bandit" (arXiv
2408.07205) — a *learning* restless bandit (indices learned by RL, no known transition model),
which is the family docs/SURVEY.md's section F5 already contrasts this paper's *planning* bandit
against; it is a concrete example to cite there, not a threat to any claim.

## Two open decisions (my recommendation, your call)

1. **A name for the method.** The supervisor's template wants an acronym ("use an acronym... try to
   form it into a common word... can reflect the method's distinctive features"). Nothing in
   docs/PAPER.md or docs/PAPER_OUTLINE.md has coined one — everything is referred to descriptively
   ("the Lagrangian index," "the MMSE link abstraction," "the calibration-aware reward"). My
   recommendation: **don't force one.** This is IEEE TWC, a register where a CV-style catchy acronym
   ("octopus") would read as out of place against the paper's theoretical content, and nothing I
   tried (LASER, MARS, CALMS) earned its letters without straining. I'd instead fix two short,
   plain handles and use them consistently from the introduction on: **"the MMSE link abstraction"**
   for the model, and **"the calibration-aware Lagrangian scheduler"** (or just "our scheduler")
   for the policy. If you or your supervisor want an acronym anyway, say so and I'll propose a
   shortlist next turn before we commit to it everywhere.
2. **Four pillars, three contributions.** The template's sixth paragraph wants exactly three bullets.
   docs/PAPER.md's story has four (the abstraction; scheduling on it; the system verdict against
   digital; the scheduler-aware training "reverse interface"). I'd fold the fourth (scheduler-aware
   training, C8-C11 — real but the smallest and most hedged result: "small... a low-mobility effect")
   into the performance clause of bullet 3 rather than giving it its own bullet. It stays a full
   contribution in the body (Section VI of the method) and in the abstract; it just is not one of
   the introduction's headline three. Flagging this because it is a compression choice, not a cut.

## Paragraph 1 — why this matters

Role (template): establish the task's importance before naming any gap.

- Deep JSCC (image symbols mapped straight to channel symbols, no separate source/channel code)
  is a serious 6G candidate: matches or beats separation at low SNR, degrades gracefully instead
  of falling off a cliff.
- But it is designed and evaluated **as a single link at a given SNR** (one sender, one receiver,
  the channel condition handed to it). A cellular/WLAN downlink is not that: a base station splits
  a frame of slots among many users through a scheduler, each user's slot fades on its own, the
  base station knows only an estimate of the *current* slot (often late or wrong), and each
  delivery has a deadline (one frame, one request).
- So the gain the point-to-point papers report is not automatically the gain a cell gets: that
  depends on a scheduler, and no existing scheduler knows what a slot is worth to a learned code.
  That is the importance claim driving the whole paper.
- Close with the two things that make the problem hard and set up Paragraph 2: the scheduler needs
  a *value* for "give user k this slot now," and that value does not exist for a trained,
  nonlinear codec the way it exists for a known rate-distortion curve.

No numbers needed here — this paragraph is motivation, not evidence.

## Paragraph 2 — classification of existing approaches, with Figure 1

Role (template): split prior work into the categories that motivate ours by contrast; land on
Figure 1 (abstract, low-contrast for others, bright + bold for ours).

Two categories (not three — the related-work section's five subsections are for Section II; the
introduction only needs the minimal split that motivates the gap):

* **(a) Single-link learned JSCC.** DeepJSCC and its rate/SNR-adaptive successors (PADC,
  SwinJSCC, DeepJSCC-MIMO, JSCCformer-f, DeepJSCC-l, DRJSCC, NTSCC...) — all evaluated at a
  *given* SNR, sampled or fixed; no scheduler, no competing users, nothing says what happens if
  the symbols of one codeword cross slots at *different* SNRs.
* **(b) Scheduling theory for content with a known value.** Deadline scheduling, restless
  bandits and Lagrangian index policies with proven, vanishing optimality gaps (RaDiO; Hosseini
  and Panwar's layered-video bandit; Hao-Xu-Tong's Lagrangian priority policy; Brown-Smith;
  Zhang-Frazier; Gast-Gaujal-Yan) — mature, but built around sources whose slot-by-slot worth is
  already known: digital layers that decode or do not, or an effective-SNR abstraction (EESM,
  MIESM) built for conventional modulation and coding, never tested on a learned code.
* **(c) Ours**, the bridge: show a learned code can be TRAINED so it has a simple value per slot,
  then apply (b)'s machinery to it exactly.

**Figure 1 plan** (teaser/classification figure, abstract, to draw once the outline is settled —
left-right for TWC's two-column layout):
- (a) one encoder → one channel at one given SNR → one decoder. No scheduler, one user. Grey/low
  contrast.
- (b) K users with a *known* value curve (a step function for digital layers, or a textbook
  effective-SNR curve) feeding a scheduler/index-policy box. Grey/low contrast.
- (c) K users, each a **learned prefix code**; training gives it a link abstraction (highlighted,
  bright colour) that turns accumulated phases into a two-number state; the *same* scheduler box
  from (b) now sits on top of it, in bright colour, with a short highlighted arrow showing the new
  piece (the abstraction) is what makes (c) possible. One or two words in red on the single element
  that is new (the abstraction arrow), nothing else red.
- Caption, point by point: "(a) learned codes are evaluated on one link at a known SNR; (b) mature
  scheduling theory needs a known per-slot value, which no model gives a learned code; (c) we show
  training gives a learned code that value, so (b)'s theory applies to it for the first time."

## Paragraph 3 — shortcomings / the gap, as a question

Role (template): this is a new-category method, so the simplest form applies — motivate briefly,
end on a question — but still name 2-3 concrete sub-problems a reviewer can track, since they map
directly onto the three contributions in Paragraph 6.

Opening move (per the template's note on pilot-experiment papers: introduce with a question):
"JSCC papers assume the channel is simply given; scheduling papers assume the source's value is
already known. Neither asks what a slot is worth to a *learned* code — a trained network with no
declared rule for composing what it has already received with what a new slot adds. Naturally, we
ask: does a learned code even have a predictable, composable value per use of the channel, the way
a digital code does?"

Then the three sub-problems (maps 1:1 onto Paragraph 6's three bullets and onto docs/PAPER.md's
C12/C1/C2, C13, C5):

1. No link abstraction exists for learned codes, so none of (b)'s machinery — built around a known
   per-slot value — can reach them at all.
2. Even granted an estimated value, a scheduler aiming at a hard quality target has to trust that
   estimate exactly; an imperfect, calibrated curve makes a hard threshold brittle (a user judged
   just above the target by the model can land just below it when decoded).
3. It is not established whether a *scheduled* learned code still beats a well-engineered digital
   chain once the channel information the scheduler actually gets (CQI error, aged reports) is
   realistic, rather than the exact CSI single-link comparisons assume.

## Paragraph 4 — the pilot finding (template: optional, required here because we have one)

Role: state the empirical discovery that makes the rest of the paper possible, as findings, not
method. General → specific → general.

"To find out, we trained [the prefix code] under slot-varying SNR and decoded it on mixed-SNR
histories. We found:"

1. A prefix code trained with its phases at *different* slot SNRs has distortion that composes
   almost exactly additively across slots — a sensitivity-weighted sum of per-slot MMSE terms —
   held-out error 0.055 dB on Kodak (0.083 dB on CLIC), against 0.49 dB / 0.19 dB for the standard
   MIESM / EESM mappings tested on the same code (docs/PAPER.md C12).
2. A code trained the ordinary way, at one constant SNR per image, obeys **no** such rule (held-out
   error 0.22 dB) — the composable value is a property of *training*, not of the architecture.
3. The rule is cheap to calibrate (12 extra decodes per image) and predicts the *decoded* quality
   of a fully scheduled, multi-slot delivery to within 0.05-0.09 dB, including the schedules our
   own scheduler later produces (docs/PAPER.md C1/C2).

Close with one sentence turning the finding into the paper's thesis: training, not architecture,
is what gives a learned code a link abstraction — and once it has one, deadline scheduling of it
becomes an exactly solvable restless bandit.

## Paragraph 5 — method summary

Role: name it (see open decision 1), contrast with the closest alternative, state the two main
designs against the two hardest sub-problems from Paragraph 3, close on performance.

- "To address this, we propose [the MMSE link abstraction and a calibration-aware restless-bandit
  scheduler built on it]." One sentence naming what it is.
- Contrast: "Unlike digital system-level modelling, which reuses EESM/MIESM derived for
  conventional coding and has never been tested on a learned code, and unlike the one existing
  learned-code scheduler (McDowell et al. 2026), which sends a fresh, self-contained codeword every
  slot and has to learn slot values by reinforcement learning for lack of an abstraction, our
  per-user state accumulates across slots and is solved exactly."
- Specifically: "we show training under slot-varying SNR gives the code a two-number state (phases
  held, accumulated weighted noise) that a finite-horizon restless bandit can use directly — a
  per-user dynamic program, a Lagrangian bound on every causal scheduler, and a Lagrangian index
  policy, all exact and CPU-cheap — addressing [sub-problem 1]; and we reward the index's programs
  with the *probability* of reaching the quality target under the abstraction's own measured
  calibration error, rather than a hard threshold, addressing [sub-problem 2]."
- Performance clause (covers sub-problem 3 + the folded-in fourth pillar): "Decoded over real
  schedules, it gains 0.5-0.6 dB over proportional fair and brings 8-15 points more users to a
  30 dB target than per-image rate selection; its gap to the bound vanishes as the cell grows; and
  under CQI errors and aged reports it beats a digital chain with CQI-table link adaptation and
  incremental-redundancy HARQ by 1.1-1.5 dB, a lead that grows further when the code is trained on
  the very channel its own scheduler creates."

## Paragraph 6 — three contributions

Role: exactly three, one of the template's three formats. Recommended format: **Discovery-Method-
Performance** (the finding in Paragraph 4 genuinely is the thing that makes the rest possible, so
leading with it as a discovery is the honest and the most memorable framing).

1. **We show** that a learned prefix code trained under slot-varying SNR acquires a simple, additive
   MMSE link abstraction — held-out error 0.055-0.083 dB against 0.19-0.49 dB for the EESM/MIESM
   mappings digital systems use — and that a code trained the ordinary way has none; to our
   knowledge the first test of effective-SNR link abstraction on a learned code.
2. **We turn this into an exact scheduler**: each user becomes a two-number state for a
   finite-horizon restless bandit, giving a per-user dynamic program, a Lagrangian bound on every
   causal scheduler and a Lagrangian index policy, whose gap to the bound vanishes as the cell
   grows (0.21 to 0.005 dB per user, 8 to 128 users) once its programs reward the probability of
   success under the abstraction's own calibration error rather than a hard threshold.
3. **We show it delivers**: decoded, 0.5-0.6 dB over proportional fair and 8-15 points more users
   at a quality target than per-image rate selection; and, under the CQI errors and aged reports a
   real base station has, the scheduled code beats a digital chain with link adaptation and HARQ by
   1.1-1.5 dB — training the code on the channel its own scheduler creates adds a further 0.28 dB.

(Alternative, if you'd rather keep the fourth pillar visible as its own bullet and compress
differently: General-Specific-Performance — (1) propose the abstraction+scheduler to solve "what is
a slot worth to a learned code"; (2) two mechanisms: the calibration-aware reward, and
scheduler-aware training (the code learns the channel its scheduler makes); (3) performance. Happy
to switch to this if you'd rather both the calibration-aware reward and the reverse-interface
result get their own line.)

## What Paragraph 6 leaves for the rest of the paper

Everything compressed above stays a full result: C3/C4 (the closed-form certainty-equivalent
policy; admission+stopping for the quality target), C6 (encode once vs. a decoder per length), C7
(which policy when, under CSI errors), C8-C11 (the reverse interface) all keep their place in
Sections IV-V; the introduction's three bullets are a map of the paper, not the whole paper.

## Organisation paragraph (closing the introduction)

One short paragraph, standard form: Section II related work; III system model; IV the link
abstraction; V scheduling on it (the bandit, the bound, the index, the calibration-aware reward);
VI the scheduler-made channel; VII evaluation; VIII conclusion and limits. (Matches
docs/PAPER.md's existing outline — no change needed, just restate briefly in prose.)
