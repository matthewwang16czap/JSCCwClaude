# Manuscript draft (IEEE TWC)

The actual paper prose, section by section, as we write it — distinct from docs/PAPER_OUTLINE.md
(the planning outline) and docs/INTRO_OUTLINE.md (the paragraph-by-paragraph plan this section was
drafted from). Every number here is copied from docs/PAPER.md / docs/NOTES.md, never recomputed or
estimated; nothing is a new claim.

**Citation style in this draft.** Works are cited by author name in the prose (narrative style:
"Bourtsoulatze et al. showed...", not "[1]" or "(Author, Year)"), matching how docs/SURVEY.md and
docs/PAPER.md already refer to them. This sidesteps pinning an exact publication year for every
entry before we have a bibliography tool assembling one, and it is a mechanical, not a creative,
step to turn each name into a numbered `\cite{}` once we have a `.bib` file — the "Working
references" list below is that `.bib` file's content in prose form, with a confidence note on any
entry whose exact venue/year I have not independently re-verified today. Two formatting conventions
to restore at LaTeX time, dropped here only because Markdown has no small caps / proper em dashes:
IEEE renders "Fig." and section numbers as given; I use "--" for em dashes throughout.

---

## I. Introduction

**STALE (2026-10-29): paragraph 2's classification, paragraph 3's opening sentence, and
paragraph 5's contrast clause below do not match the revised outline in docs/INTRO_OUTLINE.md
and need rewriting (they still split prior work into "single-link JSCC" vs. "scheduling theory,"
which the outline revision replaced). Paragraphs 1, 4 and 6, and the organisation paragraph, are
unaffected and still good. Left as-is pending review of the revised outline, per the
paragraph-by-paragraph process -- do not treat paragraph 2 below as final.**

Deep joint source-channel coding (JSCC) is a strong candidate for image and video delivery in 6G
semantic communication: mapping source symbols directly to channel symbols lets a learned code
match or exceed the rate-distortion performance of a separate source and channel code at low
signal-to-noise ratio (SNR), and degrade gracefully as the channel worsens instead of falling off
the cliff a mismatched digital code does, as Bourtsoulatze, Kurka and Gündüz showed for the
foundational deep JSCC architecture. Every evaluation of this promise that we are aware of,
however, treats the channel as a single link at one SNR, known or sampled once per transmission. A
cellular or WLAN downlink is not that channel: a base station serves many users out of one frame of
slots through a scheduler, each user's channel fades slot by slot on its own, the base station
knows only an estimate of the current slot's quality -- often delayed or in error -- and each
delivery, one image or one video frame, must complete by a deadline. The gain a point-to-point
study reports is therefore not the gain a cell delivers: that depends on which user's data a
scheduler puts into which slot, a decision scheduling theory has studied for decades for other
kinds of content, but never for a learned code, because no existing model says what a slot is worth
to one.

Prior work splits along exactly this line. One body of work designs and evaluates learned codes
for a single, given channel: rate- and SNR-adaptive deep JSCC now spans convolutional backbones
with a predicted, per-image rate choice (PADC), Swin-transformer backbones with rate and SNR
modulation (SwinJSCC), and vision-transformer backbones that map patch tokens straight to channel
symbols (DeepJSCC-MIMO); successive-refinement, multi-decoder codes that approach the rate of
per-rate specialists (DeepJSCC-l); and codes that stop transmitting once a quality target is met,
watching channel-output feedback (JSCCformer-f) or a learned entropy model (NTSCC). All of them
assume the SNR a codeword will see is given, sampled once per image; none asks what happens when
one codeword's symbols cross several slots of a fading channel at different SNRs, because none
competes for slots with other users. A second, separate body of work has the opposite gap: mature
scheduling theory for deadline-constrained, quality-graded content, built around sources whose
per-slot value is already known -- a digital video layer that either decodes or does not, as in
Hosseini and Panwar's restless-bandit SVC scheduler, or a quality curve read off an effective-SNR
mapping such as EESM or MIESM, designed for conventional modulation and coding and, to our
knowledge, never tested on a learned one. The resulting Lagrangian index policies carry an
optimality gap that vanishes as the number of users grows, a line of results due to Hao, Xu and
Tong, Brown and Smith, Zhang and Frazier, and Gast, Gaujal and Yan, but every one of them assumes
the thing a learned code does not provide: a known value per use of the channel. Fig. 1 contrasts
the two bodies of work with ours: (a) single-link JSCC, where the channel is given; (b) scheduling
theory, where the source's value is given; (c) this paper, where training gives a learned code the
value scheduling theory needs, letting it reach a class of codes it has never been applied to.

JSCC papers assume the channel is simply given; scheduling papers assume the source's value is
already known. Neither line of work asks what a slot is worth to a learned code -- a trained
network with no declared rule for combining what a user has already received with what one more
slot would add. We therefore ask a basic question: does a learned code even have a predictable,
composable value per use of the channel, the way a digital code does? The answer matters for three
reasons. First, without one, none of the scheduling machinery in Fig. 1(b) -- built entirely around
a known per-slot value -- can reach a learned code at all. Second, even a calibrated estimate of
that value is imperfect, and a scheduler that holds it to a hard quality threshold is brittle: a
user the model judges just above a target can decode just below it. Third, it is untested whether a
scheduled learned code still outperforms a well-engineered digital chain once the channel
information a scheduler actually has -- erroneous or outdated channel quality indicator (CQI)
reports, not the perfect channel state information (CSI) single-link comparisons assume -- is
accounted for.

To find out, we trained a progressive (prefix) learned code, whose tokens are delivered and decoded
in order of importance, and decoded it on histories whose tokens arrive at different SNRs -- the
condition a scheduled, slot-by-slot delivery actually creates. Three findings make the rest of the
paper possible. (1) A code trained with its tokens at different SNRs has distortion that composes
almost exactly additively across the slots it is decoded from -- a sensitivity-weighted sum of
per-slot minimum mean-square-error (MMSE) terms -- with 0.055 dB held-out error on Kodak and
0.083 dB on CLIC, against 0.49 dB and 0.44 dB for the MIESM mapping on the same two test sets
(0.19 dB and 0.16 dB for EESM). (2) A code trained the ordinary way, at one constant SNR per image,
obeys no such rule at all (0.22 dB held-out error): the composable value is a property of how a
code is trained, not of its architecture. (3) The rule is cheap to calibrate -- twelve extra
decodes per image -- and predicts the decoded quality of a fully scheduled, multi-slot delivery to
within 0.05-0.09 dB, including the schedules our own scheduler later produces. Training, not
architecture, is what gives a learned code a link abstraction; once it has one, deadline scheduling
of it becomes an exactly solvable restless bandit.

To address this, we propose the MMSE link abstraction and a calibration-aware restless-bandit
scheduler built on it. Unlike digital system-level modelling, which reuses EESM- or MIESM-style
mappings designed for conventional coding and has, to our knowledge, never been applied to a
learned code, and unlike the one existing scheduler for a learned code, due to McDowell, Panebianco,
Chakraborty and Sun, which sends a fresh, self-contained codeword every slot and must learn each
slot's value by reinforcement learning for lack of an abstraction, our abstraction turns each user
into a two-number state that accumulates across slots and is solved exactly, with no learning or
exploration required. Specifically, we show that training under slot-varying SNR gives a prefix
code this two-number state -- phases held, and their accumulated weighted noise -- which a
finite-horizon restless bandit uses directly: a per-user dynamic program, a Lagrangian bound on
every causal scheduler, and a Lagrangian index policy, all exact and solvable on a CPU in seconds,
addressing the first problem above; and we reward these programs with the probability of reaching a
quality target under the abstraction's own measured calibration error, rather than a hard
threshold, addressing the second. Decoded over real schedules, the result gains 0.5-0.6 dB over
proportional fair and brings 8-15 points more users to a 30 dB quality target than per-image rate
selection, with a gap to the optimal-causal-scheduler bound that vanishes as the cell grows; and
under the CQI errors and aged reports a real base station has, it beats a digital chain with
standard link adaptation and incremental-redundancy HARQ by 1.1-1.5 dB, a lead that grows further
once the code is trained on the very channel its own scheduler creates.

Overall, our contributions are threefold.

1) We show that a learned prefix code trained under slot-varying SNR acquires a simple, additive
   MMSE link abstraction -- 0.055-0.083 dB held-out error across two test sets, against 0.44-0.49
   dB for the MIESM mapping used in digital system-level simulation (0.16-0.19 dB for EESM) -- and
   that a code trained the ordinary way has none; to our knowledge this is the first test of
   effective-SNR link abstraction on a learned code.
2) We turn this abstraction into an exact scheduler: each user becomes a two-number state for a
   finite-horizon restless bandit, yielding a per-user dynamic program, a Lagrangian bound on every
   causal scheduler, and a Lagrangian index policy whose gap to the bound vanishes as the cell
   grows -- from 0.21 to 0.005 dB per user between 8 and 128 users -- once its programs are
   rewarded with the probability of success under the abstraction's own calibration error rather
   than a hard threshold.
3) We show the result delivers: decoded, it gains 0.5-0.6 dB over proportional fair and brings
   8-15 points more users to a quality target than per-image rate selection; under realistic CQI
   errors and aged reports it beats a digital chain with link adaptation and HARQ by 1.1-1.5 dB;
   and training the code on the very channel its own scheduler creates adds a further 0.28 dB.

The rest of the paper is organised as follows. Section II surveys related work. Section III states
the system model. Section IV develops the MMSE link abstraction. Section V builds the
restless-bandit scheduler on it: the per-user program, the Lagrangian bound, the index policy, and
the calibration-aware reward. Section VI shows how a scheduler, once built, changes the channel a
code should be trained for. Section VII evaluates both against proportional fair, per-image rate
selection, and a digital baseline with link adaptation and HARQ. Section VIII concludes and states
the paper's limitations.

---

## Notes for revision

**Two defaults I used, per my recommendation two turns ago (you did not explicitly confirm either,
so flagging again now that they are load-bearing in actual prose):**
- No acronym for the method; it is "the MMSE link abstraction" and "the calibration-aware Lagrangian
  scheduler" (or "our scheduler") throughout. Say the word if you want one instead.
- Three contributions, not four: scheduler-aware training (the fourth pillar, C8-C11) is folded into
  contribution 3's last clause rather than given its own bullet. It still gets full treatment in
  Section VI and the abstract.

**Open items before this is submission-ready:**
- **Fig. 1 is referenced but not drawn.** The plan (three abstract panels, (a)/(b) low-contrast,
  (c) bright) is in docs/INTRO_OUTLINE.md. Say when you want it built; it is a drawing task, not a
  writing one.
- **EESM/MIESM's own citation is still open.** I looked for a canonical original reference today and
  could not verify one to this project's standard (a search surfaced a 2011 paper that *revisits*
  EESM, not an originating source, and an unverified "attributed to Brueninghaus et al., PIMRC 2005"
  recollection I would not put in the paper without reading it first). The prose above names EESM and
  MIESM without a citation marker rather than guess. If you or your supervisor already have the
  standard reference for this (a 3GPP TR or an IEEE 802.16 evaluation-methodology document is the
  usual source), tell me and I will slot it in; otherwise I can go read candidates properly before we
  finalise this section.
- **Length.** This draft runs to roughly 1,250 words, a little over the "~1.5 pages" estimate in
  docs/PAPER_OUTLINE.md once Fig. 1 is laid out beside it. Normal at this stage; a trim pass belongs
  after the figure exists and we can see the real column layout, not before.

## Working references cited in Section I

Full detail in docs/SURVEY.md; repeated here as this section's reference list, confidence-flagged
where I have not independently re-verified the exact year beyond what SURVEY.md already recorded.

| name used in text | full reference | confidence |
|---|---|---|
| Bourtsoulatze, Kurka and Gündüz | "Deep Joint Source-Channel Coding for Wireless Image Transmission," arXiv:1809.01733, IEEE Trans. Cogn. Commun. Netw., 2019 | venue + year verified (docs/SURVEY.md) |
| PADC (Zhang, Zhang, Ma, Shao, Wang, Leung) | "Predictive and Adaptive Deep Coding for Wireless Image Transmission in Semantic Communication," IEEE Trans. Wireless Commun., vol. 22, no. 8, pp. 5486-5501, 2023, DOI 10.1109/TWC.2023.3234408 | read in full with its code (docs/SURVEY.md) |
| SwinJSCC (Yang et al.) | arXiv:2308.09361 | arXiv ID verified; full author list and exact venue not independently re-checked today |
| DeepJSCC-MIMO (Wu, Shao, Bian, Mikolajczyk, Gündüz) | arXiv:2309.00470, IEEE Trans. Wireless Commun., vol. 23, no. 10, 2024, DOI 10.1109/TWC.2024.3422794 | read in full (docs/SURVEY.md) |
| JSCCformer-f (Wu, Shao, Ozfatura, Mikolajczyk, Gündüz) | arXiv:2306.09101, IEEE Trans. Wireless Commun., vol. 23, no. 9, pp. 11904-11919, 2024 | read in full (docs/SURVEY.md) |
| DeepJSCC-l (Kurka and Gündüz) | arXiv:2009.12480, IEEE Trans. Wireless Commun., 2021, DOI 10.1109/TWC.2021.3090048 | read in full (docs/SURVEY.md) |
| NTSCC (Dai et al.) | arXiv:2112.10961, IEEE J. Sel. Areas Commun. | venue verified; exact year not independently re-checked today |
| Hosseini and Panwar | "Restless Video Bandits: Optimal SVC Streaming in a Multi-user Wireless Network," arXiv:1612.03985 | read in full (docs/SURVEY.md) |
| Hao, Xu and Tong | "Asymptotically Optimal Lagrangian Priority Policy for Deadline Scheduling with Processing Rate Limits," IEEE Trans. Autom. Control, DOI 10.1109/TAC.2021.3049340 | read in full (docs/SURVEY.md); exact year not independently re-checked today |
| Brown and Smith | "Index Policies and Performance Bounds for Dynamic Selection Problems," Manage. Sci., vol. 66, no. 7, 2020 | venue + year from docs/SURVEY.md |
| Zhang and Frazier | "Restless Bandits with Many Arms: Beating the Central Limit Theorem," arXiv:2107.11911 | read in full (docs/SURVEY.md) |
| Gast, Gaujal and Yan | "LP-based Policies for Restless Bandits," Math. Oper. Res., vol. 49, no. 4, 2024, arXiv:2106.10067 | read in full (docs/SURVEY.md) |
| McDowell, Panebianco, Chakraborty and Sun | "Significance-Driven Semantic Communication," arXiv:2608.28441 | read in full (docs/SURVEY.md) |

EESM / MIESM: no reference listed, per the open item above.
