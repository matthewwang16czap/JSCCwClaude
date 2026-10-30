# Section I (Introduction): paragraph-by-paragraph outline

Seven paragraphs, in the order of the supervisor's template. Each numbered item is one sentence
(or two short ones) of the prose. [Keys] point to the reference list at the end. Every number,
with the round and the result file it comes from, is in `docs/INTRO_NUMBERS.md`.

**Terms used throughout.** *Learned code*: a deep JSCC encoder-decoder pair. *Progressive code*: a
learned code whose symbols are sent in phases and that decodes from whatever prefix of phases has
arrived. *Slot*, *frame*, *deadline*: the base station has a frame of T slots and serves K users;
every image must be delivered by the end of the frame. *Link abstraction*: a rule that maps the
SNRs a codeword's symbols met to its decoded quality. *Physical link design* and *the JSCC way*:
the two existing categories (paragraph 2). *CLASP*: our method (paragraph 5).

## Paragraph 1. Why the problem matters

1. Deep JSCC maps an image straight to channel symbols with one learned encoder and decoder, with
   no separate source and channel codes [Bourtsoulatze19]. It matches or beats separation-based
   digital transmission at low SNR and degrades gracefully as the channel worsens, instead of
   failing once the SNR drops below what a digital code was designed for [Bourtsoulatze19,
   Gunduz23]. That makes it a strong candidate for image delivery in 6G [Gunduz23].
2. But a learned code is designed and evaluated as a single link at a given SNR: one transmitter,
   one receiver, the channel condition handed to it [Bourtsoulatze19, Xu22, Yang25].
3. A cellular downlink is different:
   - a base station splits a frame of slots among many users through a scheduler;
   - each user's slots fade independently;
   - the base station knows only an estimate of the current slot, a channel-quality report
     [TS36213] that is often late or wrong;
   - each delivery, one image, has a deadline.
4. So the gain a point-to-point study reports is not automatically the gain a cell delivers. That
   depends on the scheduler, and no existing scheduler knows what a slot is worth to a learned
   code.
5. To schedule, the base station needs the value of "give user k this slot now". For a digital
   code that value follows from the rate the slot carries and the codec's rate-distortion curve.
   For a trained, nonlinear code no such value exists.

## Paragraph 2. Existing approaches: physical link design and the JSCC way (Fig. 1)

Order: physical link design and its limitations first; then "In contrast," the JSCC way, the
category ours belongs to.

1. Existing approaches to serving images to many users over a fading downlink fall into two
   categories (Fig. 1):
   - physical link design, which keeps the image codec separate and engineers the link that
     carries its bits;
   - the JSCC way, which lets a learned code take on the channel itself.
2. Physical link design is how cellular systems deliver images today:
   - a standard codec compresses the image into bits once (HEVC intra, as in HEIC
     [Sullivan12]);
   - for each slot, the physical layer picks the modulation and coding rate from the user's
     channel-quality report [Goldsmith97, TS36213];
   - failed blocks are retransmitted with incremental-redundancy HARQ [Caire01];
   - a channel-aware scheduler, such as proportional fair, picks the user each slot serves
     [Viswanath02].
3. It has mature tools for many users and for codewords that span many resources:
   - effective-SNR link abstractions (EESM, MIESM) predict a coded block's error rate from the
     SNRs of all the resources it occupies, so link adaptation and system-level evaluation can
     value a set of resources without decoding [Ericsson03, EMD09, Lagen20];
   - media-aware schedulers weigh each user's next packet by the distortion it removes before
     its deadline [Chou06, Pahalawatta07, Fu10, Hosseini16].
4. Limitation 1: every one of these tools rests on a binary outcome per block, decoded or lost.
   - A user's quality is therefore a step function of its channel: a slot below the block's
     design SNR delivers nothing, and SNR above it is wasted.
   - This is the cliff effect [Bourtsoulatze19, Gunduz23].
5. Limitation 2: the rate is committed before the slot is seen, from a report that may be late or
   wrong.
   - An optimistic report costs a retransmission; a pessimistic one leaves capacity unused.
   - Link adaptation and HARQ protect the bits only by spending extra slots.
   - In our simulations, going from fresh reports to aged ones (correlation 0.8) costs the digital
     chain 1.9 dB, against 0.5 dB for the learned code (Section VII).
6. In contrast, the JSCC way replaces the two codes with one learned code that maps the image
   straight to channel symbols [Bourtsoulatze19].
   - Its quality degrades smoothly with the SNR its symbols meet, so no block is lost outright
     and no coding rate has to be matched to an uncertain report.
   - The same symbols serve receivers of different SNRs, each at a quality matched to its
     channel, as linear analog codes showed for multicast video [Jakubczak11].
7. One line of work makes a single link adapt to the channel and the budget:
   - one model for all SNRs [Xu22];
   - one model for many bandwidths and rates [Kurka21, Dai22, Zhang23, Yang25];
   - feedback that stops the transmission once a target quality is reached [Kurka20, Wu24f];
   - MIMO channels [Wu24m];
   - a predictor of each image's quality that picks its rate [Zhang23].
8. To serve many users, the JSCC way splits the shared channel by each user's predicted quality
   at its SNR:
   - channel assignment and symbol counts from a semantic spectral efficiency [Yan22];
   - user association and bandwidth [Xia24];
   - compression ratios and resources by Bayesian optimisation over a learned quality oracle
     [BO26].

   Other work builds the sharing into the code, letting users transmit non-orthogonally
   [Yilmaz23]. Where the allocation must change over time, a scheduler is learned by deep
   reinforcement learning [Wang22, Zhang23t]. The most recent such work is a restless bandit
   over fading slots whose slot values are learned [McDowell26].
9. Our method belongs to the JSCC way: it schedules a learned code to many users across fading
   slots.

### Figure 1

The figure is one column wide, with three panels stacked top to bottom: (a), (b) and (c).

- **Style.** All text is bold. Panels (a) and (b) use grey, low-contrast tones; panel (c) uses
  saturated colour. Red appears only on the words "link abstraction" in (c). There are no
  equations and no axes beyond the insets.
- **(a) Physical link design.**
  - The chain: image → "codec" → bits → "AMC + HARQ" → a row of slots.
  - Each slot holds one user's block, marked ✓ (decoded) or ✗ (lost).
  - A "scheduler (CQI)" box above the row picks the user for each slot.
  - Inset: quality against SNR as a step (the cliff).
- **(b) The JSCC way.**
  - The chain: image → "learned JSCC" → one codeword per user.
  - Each codeword has a single arrow to its own channel, labelled with one SNR.
  - An "allocator q(SNR)" box sets every user's share once.
  - A dashed "RL scheduler" box carries a "?" for the value of a slot.
  - Inset: quality against SNR as a smooth curve.
- **(c) CLASP.**
  - The chain: image → "progressive JSCC" → phases spread over slots of different SNRs (slot
    colour = SNR) → "link abstraction" → each user's two-number state → "restless-bandit
    scheduler".
  - The scheduler serves users slot by slot up to a "deadline" marker.
  - Inset: predicted against decoded quality, on the diagonal.

**Caption.** "Fig. 1. Approaches to serving images to many users over a fading downlink.
(a) Physical link design: a separate codec over adaptive modulation and coding, HARQ and a
channel-aware scheduler; each block is decoded or lost. (b) The JSCC way: a learned code degrades
gracefully, but each user is valued at a single SNR and allocated once, or scheduled by a policy
learned by trial and error. (c) CLASP: a progressive learned code trained under slot-varying SNR
obeys a link abstraction that values every slot, and a restless-bandit scheduler built on it
serves every user slot by slot up to the deadline."

## Paragraph 3. Challenges of the JSCC way, and the question they raise

1. Although existing JSCC methods are effective, they still face the following challenges when a
   base station schedules a learned code to many users over fading slots.
2. **(1) No link abstraction.**
   - They value a user's transmission at a single SNR, with a fitted quality-SNR curve or a
     learned quality predictor [Yan22, Zhang23, BO26].
   - A scheduled image, however, is spread over slots of different SNRs. For a nonlinear learned
     code, no rule is known for how these SNRs combine into decoded quality.
   - The EESM and MIESM mappings of physical link design predict a digital block's error rate,
     not a learned code's distortion. To our knowledge, they have not been validated on a learned
     code.
3. **(2) No tractable, certifiable scheduler.**
   - Deciding which user each slot serves, before a deadline and without knowing future slots,
     is a stochastic dynamic program. Its state grows exponentially with the number of users.
   - Existing methods take one of two routes, and neither plans over the frame with a guarantee:
     - some decide each transmission on its own, at its current SNR, without planning over the
       slots left before the deadline [Yan22, Xia24, BO26];
     - others learn slot values by reinforcement learning, which needs many interactions to
       train and gives no certificate of how far the learned policy is from optimal [Wang22,
       Zhang23t, McDowell26].
4. **(3) Imperfect information.**
   - The scheduler's quality model and its channel reports are both estimates, yet existing
     schedulers take them at face value.
     - PADC picks a rate from a predicted quality as if the prediction were exact [Zhang23].
     - The learned schedulers assume perfect channel knowledge [McDowell26].
   - So a user predicted just above a quality target can decode just below it.
   - The JSCC way's advantage over physical link design was measured on single links with an
     idealised digital side: exact channel knowledge and capacity-achieving codes [Kurka20].
     Whether that advantage survives the reports a real base station has is open.
5. Behind all three challenges lies one question: does a learned code have a link abstraction at
   all? That is, is there a rule that, like EESM for a digital block, turns the SNRs its symbols
   met into its decoded quality? And if there is, what decides whether a code obeys it?

## Paragraph 4. Pilot findings

1. To answer it, we trained a progressive code under slot-varying SNR, with the chunks of each
   codeword at independent SNRs. We then decoded it on held-out histories of mixed slot SNRs and
   tested candidate rules (Section IV).
   - The code has six phases, any prefix of which decodes.
   - It is a vision transformer whose encoder and decoder never see the SNR.

   We found the following.
2. **(1) Its distortion adds up over slots.** The source distortion plus a sensitivity-weighted
   sum of per-slot MMSE terms, D = Dsrc + C Σ_j ρ_j φ(n_j), predicts held-out decodes far better
   than the digital mappings. Mean absolute error, every rule given the same per-phase
   sensitivity weights (in brackets, the rule's standard equal-weight form):

   | test set | MMSE rule | EESM | MIESM |
   |---|---|---|---|
   | Kodak | 0.055 dB (0.37) | 0.19 dB (0.37) | 0.34 dB (0.49) |
   | CLIC | 0.083 dB (0.41) | 0.16 dB (0.37) | 0.29 dB (0.44) |

   Both parts are needed: the MMSE form, and weights per phase, because the code is ordered and
   which phase a slot carries matters.

   The form is the one linear analog codes obey by construction [Jakubczak11]. Here a nonlinear
   learned code acquires it.
3. **(2) Training decides it.**
   - The same architecture trained at one constant SNR per image obeys no such rule: the MMSE
     rule errs by 0.22 dB, and no rule does better than 0.15 dB.
   - The reason is that such a code codes its later phases against the base reconstruction it
     expects at one SNR. DeepJSCC-l observed the same mismatch for layers [Kurka21].
4. **(3) It is cheap and accurate, and its small error still matters.**
   - Twelve decodes per image calibrate it.
   - It predicts the decoded quality of complete scheduled deliveries with a mean absolute error
     of 0.05-0.09 dB with fresh reports (up to 0.12 dB with aged ones), and a bias within
     ±0.05 dB.
   - But its calibration error (0.03-0.05 dB) is first-order at a hard quality target. A
     scheduler that trusts it loses 1.7-2.0 points of the share of users reaching 30 dB, at
     every cell size from 16 to 128 users.
5. In short, a learned code can be given a link abstraction through training. With it, each
   user's state reduces to two numbers: the phases received, and their accumulated weighted noise.
   Scheduling then becomes a planning problem, provided the abstraction's residual error is part
   of the objective.

## Paragraph 5. The method: CLASP

1. To address the aforementioned challenges, we propose CLASP (Calibration-aware,
   Link-Abstraction-based Scheduling of Progressive codes), a scheduler for learned progressive
   codes in a multi-user fading downlink.
2. Existing JSCC methods value each user at one SNR and allocate once, or learn slot values by
   reinforcement learning. Unlike them, CLASP values every slot for every user through the code's
   link abstraction, and plans each user's delivery to the deadline exactly.
3. Specifically, CLASP has one design for each challenge:
   - **Challenge (1), the link abstraction.** We train the progressive code under slot-varying
     SNR and calibrate its MMSE link abstraction per image from 12 decodes. Each user then becomes
     a two-number state.
   - **Challenge (2), the scheduler.** On this state the frame becomes a finite-horizon restless
     bandit [Whittle88]:
     - a Lagrangian relaxation splits it into one small dynamic program per user under per-slot
       prices, solved exactly on a CPU;
     - the relaxation's value is an upper bound on every causal scheduler;
     - users are served in the order of the Lagrangian index those programs define [Hao22,
       Brown20].
   - **Challenge (3), imperfect information.**
     - Each program rewards the probability of reaching the quality target under the
       abstraction's measured calibration error, rather than a hard threshold.
     - Each program decides on the posterior of the channel given an aged report, not on the
       report itself.
4. Results:
   - Decoded by the codec, CLASP gains 0.5-0.6 dB of mean PSNR over proportional fair, and brings
     8-15 points more users to 30 dB than per-image rate selection [Zhang23].
   - Simulated against the bound, its gap shrinks to 0.005 dB per user at 128 users for mean
     PSNR, and stays within ±0.3 points from 16 users on for the target.
   - Simulated under CQI errors and aged reports, the learned code scheduled on its abstraction
     beats a digital chain with CQI-table link adaptation and incremental-redundancy HARQ by
     1.1-1.5 dB of mean PSNR. With error-free reports the two are on par.
   - Decoded on the same slots, training the code on the channel its scheduler creates adds
     0.28 dB under the closed-form version of the index. Of that, 0.08-0.12 dB is beyond training
     for the one prefix length that scheduler gives.

## Paragraph 6. Contributions (Discovery-Method-Performance)

Overall, our main contributions are as follows.

1. **Discovery.**
   - We find that a learned progressive code trained under slot-varying SNR obeys an additive
     MMSE link abstraction. It is accurate to 0.055-0.083 dB on two test sets. Given the same
     per-phase weights, EESM errs by 0.16-0.19 dB and MIESM by 0.29-0.34 dB; in their standard
     form, by 0.37 dB and 0.44-0.49 dB.
   - A code trained at constant SNR obeys none.
   - A transmitter can calibrate the abstraction from 12 decodes per image.
   - To our knowledge, effective-SNR mappings had not been tested on a learned code before.
2. **Method.** We propose CLASP.
   - It turns multi-user deadline scheduling of learned codes into a finite-horizon restless
     bandit with a two-number user state.
   - It solves that bandit with exact per-user dynamic programs, a Lagrangian bound on every
     causal scheduler, and an index policy.
   - A calibration-aware reward and report-aware programs make it robust to the abstraction's
     error and to aged channel reports.
3. **Performance.**
   - Decoded, CLASP gains 0.5-0.6 dB over proportional fair, and 8-15 points of users at 30 dB
     over per-image rate selection.
   - Simulated, its gap to the bound vanishes as the cell grows:
     - for mean PSNR, from 0.21 to 0.005 dB per user between 8 and 128 users;
     - for the target, within ±0.3 points from 16 users on.
   - Simulated under CQI errors and aged reports, the learned code scheduled on the abstraction
     outperforms a digital chain with link adaptation and HARQ by 1.1-1.5 dB and 10.7-13.5
     points. This uses the best JSCC policy at each point: the closed form for the mean, a
     planner for the target.
   - Decoded, training the code on its scheduler's channel adds 0.28 dB.

**How paragraphs 3, 5 and 6 line up:**

| challenge (paragraph 3) | design in CLASP (paragraph 5) | evidence (paragraph 6) |
|---|---|---|
| (1) no link abstraction | slot-varying-SNR training; MMSE link abstraction calibrated per image from 12 decodes | mean absolute error 0.055-0.083 dB, against EESM 0.16-0.19 and MIESM 0.29-0.34 with the same weights; decoded schedules 0.05-0.09 dB with fresh reports |
| (2) no tractable, certifiable scheduler | two-number state → finite-horizon restless bandit: exact per-user programs, Lagrangian bound, Lagrangian index | per-user gap to the bound 0.21 → 0.005 dB, 8 → 128 users (simulated); +0.5-0.6 dB over PF (decoded) |
| (3) imperfect information | calibration-aware reward; programs on the channel posterior given aged reports | +1.7 points over the hard target, users missed 0.0% against 1.2% (decoded); +1.1-1.5 dB over digital under CQI errors and aged reports (simulated, closed-form policy) |

## Paragraph 7. Organisation

The rest of the paper is organised as follows:

- Section II reviews related work in three subsections: physical link design for multi-user media
  delivery, the JSCC way (adaptive and multi-user learned codes), and restless bandits with index
  policies.
- Section III states the system model.
- Section IV reports the pilot experiments and derives the MMSE link abstraction.
- Section V develops CLASP.
- Section VI studies the channel the scheduler creates, and trains the code on it.
- Section VII evaluates CLASP against proportional fair, per-image rate selection, the Lagrangian
  bound and the digital chain.
- Section VIII concludes and states the limitations.

## References cited in Section I

| key | reference |
|---|---|
| Bourtsoulatze19 | E. Bourtsoulatze, D. B. Kurka, D. Gündüz, "Deep joint source-channel coding for wireless image transmission," IEEE Trans. Cogn. Commun. Netw., vol. 5, no. 3, pp. 567-579, 2019 |
| Gunduz23 | D. Gündüz, Z. Qin, I. E. Aguerri, H. S. Dhillon, Z. Yang, A. Yener, K. K. Wong, C.-B. Chae, "Beyond transmitting bits: Context, semantics, and task-oriented communications," IEEE J. Sel. Areas Commun., vol. 41, no. 1, pp. 5-41, 2023 |
| Xu22 | J. Xu, B. Ai, W. Chen, A. Yang, P. Sun, M. Rodrigues, "Wireless image transmission using deep source channel coding with attention modules," IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 4, pp. 2315-2328, 2022 |
| Yang25 | K. Yang, S. Wang, J. Dai, X. Qin, K. Niu, P. Zhang, "SwinJSCC: Taming Swin transformer for deep joint source-channel coding," IEEE Trans. Cogn. Commun. Netw., vol. 11, no. 1, pp. 90-104, 2025 |
| TS36213 | 3GPP TS 36.213, "Evolved Universal Terrestrial Radio Access (E-UTRA); Physical layer procedures" (CQI table 7.2.3-1) |
| Sullivan12 | G. J. Sullivan, J.-R. Ohm, W.-J. Han, T. Wiegand, "Overview of the High Efficiency Video Coding (HEVC) standard," IEEE Trans. Circuits Syst. Video Technol., vol. 22, no. 12, pp. 1649-1668, 2012 |
| Goldsmith97 | A. J. Goldsmith, S.-G. Chua, "Variable-rate variable-power MQAM for fading channels," IEEE Trans. Commun., vol. 45, no. 10, pp. 1218-1230, 1997 |
| Caire01 | G. Caire, D. Tuninetti, "The throughput of hybrid-ARQ protocols for the Gaussian collision channel," IEEE Trans. Inf. Theory, vol. 47, no. 5, pp. 1971-1988, 2001 |
| Viswanath02 | P. Viswanath, D. N. C. Tse, R. Laroia, "Opportunistic beamforming using dumb antennas," IEEE Trans. Inf. Theory, vol. 48, no. 6, pp. 1277-1294, 2002 |
| Ericsson03 | Ericsson, "System-level evaluation of OFDM: further considerations," 3GPP TSG-RAN WG1 #35, R1-031303, Nov. 2003 |
| EMD09 | IEEE 802.16 Broadband Wireless Access Working Group, "IEEE 802.16m evaluation methodology document (EMD)," IEEE 802.16m-08/004r5, Jan. 2009 |
| Lagen20 | S. Lagen et al., "New radio physical layer abstraction for system-level simulations of 5G networks," Proc. IEEE ICC, 2020 |
| Chou06 | P. A. Chou, Z. Miao, "Rate-distortion optimized streaming of packetized media," IEEE Trans. Multimedia, vol. 8, no. 2, pp. 390-404, 2006 |
| Pahalawatta07 | P. Pahalawatta, R. Berry, T. Pappas, A. Katsaggelos, "Content-aware resource allocation and packet scheduling for video transmission over wireless networks," IEEE J. Sel. Areas Commun., vol. 25, no. 4, pp. 749-759, 2007 |
| Fu10 | F. Fu, M. van der Schaar, "A systematic framework for dynamically optimizing multi-user wireless video transmission," IEEE J. Sel. Areas Commun., vol. 28, no. 3, pp. 308-320, 2010 |
| Hosseini16 | S. A. Hosseini, S. S. Panwar, "Restless video bandits: Optimal SVC streaming in a multi-user wireless network," arXiv:1612.03985, 2016 |
| Jakubczak11 | S. Jakubczak, D. Katabi, "A cross-layer design for scalable mobile video," Proc. ACM MobiCom, 2011 |
| Kurka20 | D. B. Kurka, D. Gündüz, "DeepJSCC-f: Deep joint source-channel coding of images with feedback," IEEE J. Sel. Areas Inf. Theory, vol. 1, no. 1, pp. 178-193, 2020 |
| Kurka21 | D. B. Kurka, D. Gündüz, "Bandwidth-agile image transmission with deep joint source-channel coding," IEEE Trans. Wireless Commun., 2021, DOI 10.1109/TWC.2021.3090048 |
| Dai22 | J. Dai, S. Wang, K. Tan, Z. Si, X. Qin, K. Niu, P. Zhang, "Nonlinear transform source-channel coding for semantic communications," IEEE J. Sel. Areas Commun., 2022 (arXiv:2112.10961) |
| Zhang23 | W. Zhang et al., "Predictive and adaptive deep coding for wireless image transmission in semantic communication" (PADC), IEEE Trans. Wireless Commun., vol. 22, no. 8, pp. 5486-5501, 2023 |
| Wu24f | H. Wu, Y. Shao, E. Ozfatura, K. Mikolajczyk, D. Gündüz, "Transformer-aided wireless image transmission with channel feedback" (JSCCformer-f), IEEE Trans. Wireless Commun., vol. 23, no. 9, pp. 11904-11919, 2024 |
| Wu24m | H. Wu, Y. Shao, C. Bian, K. Mikolajczyk, D. Gündüz, "Deep joint source-channel coding for adaptive image transmission over MIMO channels," IEEE Trans. Wireless Commun., vol. 23, no. 10, 2024, DOI 10.1109/TWC.2024.3422794 |
| Yan22 | L. Yan, Z. Qin, R. Zhang, Y. Li, G. Y. Li, "Resource allocation for text semantic communications," IEEE Wireless Commun. Lett., vol. 11, no. 7, pp. 1394-1398, 2022 |
| Xia24 | L. Xia, Y. Sun, D. Niyato, X. Li, M. A. Imran, "Joint user association and bandwidth allocation in semantic communication networks," IEEE Trans. Veh. Technol., vol. 73, no. 2, pp. 2699-2711, 2024 |
| BO26 | "Reliable online resource allocation for multi-user semantic communications: A constraint Bayesian optimization approach," arXiv:2604.10931, 2026 |
| Yilmaz23 | S. F. Yilmaz, C. Karamanlı, D. Gündüz, "Distributed deep joint source-channel coding over a multiple access channel," Proc. IEEE ICC, 2023 |
| Wang22 | Y. Wang, M. Chen, T. Luo, W. Saad, D. Niyato, H. V. Poor, S. Cui, "Performance optimization for semantic communications: An attention-based reinforcement learning approach," IEEE J. Sel. Areas Commun., vol. 40, no. 9, pp. 2598-2613, 2022 |
| Zhang23t | H. Zhang, H. Wang, Y. Li, K. Long, A. Nallanathan, "DRL-driven dynamic resource allocation for task-oriented semantic communication," IEEE Trans. Commun., vol. 71, pp. 3992-4004, 2023 |
| McDowell26 | C. McDowell, A. Panebianco, S. Chakraborty, Y. Sun, "Significance-driven semantic communication," arXiv:2608.28441, 2026 |
| Whittle88 | P. Whittle, "Restless bandits: Activity allocation in a changing world," J. Appl. Probab., vol. 25A, pp. 287-298, 1988 |
| Hao22 | L. Hao, Y. Xu, L. Tong, "Asymptotically optimal Lagrangian priority policy for deadline scheduling with processing rate limits," IEEE Trans. Autom. Control, vol. 67, no. 1, pp. 236-250, 2022 |
| Brown20 | D. B. Brown, J. E. Smith, "Index policies and performance bounds for dynamic selection problems," Manage. Sci., vol. 66, no. 7, 2020 |

## Numbers used

Each number above, with the result file, the COMMANDS round that wrote it, the line to look
for and whether it is simulated or decoded: `docs/INTRO_NUMBERS.md`.
