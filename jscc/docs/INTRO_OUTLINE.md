# Section I (Introduction): paragraph-by-paragraph outline

Seven paragraphs, in the order of the supervisor's template. Each numbered item is one sentence
(or two short ones) of the prose. [Keys] point to the reference list at the end. Every number,
in the order it appears here, with the printed line, the result file and the round it comes
from, is in `docs/INTRO_NUMBERS.md`.

**Terms used throughout.** *Learned code*: a deep JSCC encoder-decoder pair. *Progressive code*: a
learned code whose symbols are sent in phases and that decodes from whatever prefix of phases has
arrived. *Slot*, *frame*, *deadline*: the base station has a frame of T slots and serves K users;
every image must be delivered by the end of the frame. *Link abstraction*: a rule that maps the
SNRs a codeword's symbols met to its decoded quality. *Physical link design* and *the JSCC way*:
the two existing categories (paragraph 2). *Calibration-aware*: deciding on a calibrated
predictive distribution of a user's decoded quality, whose spread is the measured error of the
prediction, from the quality model and from the channel reports. *CLASP* (Calibration-aware,
Link-Abstraction-based Scheduling of Progressive codes): our method (paragraph 5). *Share of
users at the target*: the percentage of users whose decoded PSNR reaches the quality target by
the deadline. A difference between two shares is the difference of the two percentages, given in
% of all users: "an extra X% of the users" means X% more of all users reach the target.

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
2. Physical link design is how cellular systems deliver images today: a standard codec
   compresses the image into bits once, and the physical layer carries them with link adaptation
   from CQI reports, HARQ and a channel-aware scheduler such as proportional fair [Goldsmith97,
   Caire01, Viswanath02].
3. Its tools for many users are mature (Section II-A: link adaptation and HARQ, effective-SNR
   mappings, media-aware schedulers).
4. Limitation 1: every one of these tools rests on a binary outcome per block, decoded or lost.
   - A user's quality is therefore a step function of its channel: a slot below the block's
     design SNR delivers nothing, and SNR above it is wasted.
   - This is the cliff effect [Bourtsoulatze19, Gunduz23].
5. Limitation 2: the rate is committed before the slot is seen, from a report that may be late or
   wrong.
   - An optimistic report costs a retransmission; a pessimistic one leaves capacity unused.
   - Link adaptation and HARQ protect the bits only by spending extra slots.
   - In our simulations, going from fresh reports to aged ones (correlation 0.8) costs the digital
     chain 1.9 dB of mean PSNR, against 0.6 dB for a learned code under our scheduler
     (Section V-D).
6. In contrast, the JSCC way replaces the two codes with one learned code that maps the image
   straight to channel symbols [Bourtsoulatze19].
   - Its quality degrades smoothly with the SNR its symbols meet, so no block is lost outright
     and no coding rate has to be matched to an uncertain report.
   - The same symbols serve receivers of different SNRs, each at a quality matched to its
     channel, as linear analog codes showed for multicast video [Jakubczak11].
7. Most of this work adapts a single link to the channel and the budget (Section II-B).
8. To serve many users, the JSCC way splits the shared channel by each user's predicted quality
   at its SNR [Yan22, Xia24, Hou26], or learns the allocation by deep reinforcement learning
   [Wang22, Zhang23t], most recently inside a restless bandit over fading slots [McDowell26].
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
     learned quality predictor [Yan22, Zhang23, Hou26].
   - A scheduled image, however, is spread over slots of different SNRs. For a nonlinear learned
     code, no rule is known for how these SNRs combine into decoded quality.
   - The exponential and the mutual-information effective SNR mappings (EESM, MIESM) of physical
     link design predict a digital block's error rate from the SNRs of the resources it occupies
     [Ericsson03, EMD09], not a learned code's distortion. To our knowledge, they have not been
     validated on a learned code.
3. **(2) No tractable, certifiable scheduler.**
   - Deciding which user each slot serves, before a deadline and without knowing future slots,
     is a stochastic dynamic program. Its state grows exponentially with the number of users.
   - Existing methods take one of two routes, and neither plans over the frame with a guarantee:
     - some decide each transmission on its own, at its current SNR, without planning over the
       slots left before the deadline [Yan22, Xia24, Hou26];
     - others learn the allocation by reinforcement learning, which needs many interactions to
       train and gives no certificate of how far the learned policy is from optimal [Wang22,
       Zhang23t, McDowell26].
4. **(3) Imperfect information.**
   - The scheduler's quality model and its channel reports are both estimates, yet existing
     schedulers take them at face value.
     - PADC picks a rate from a predicted quality as if the prediction were exact [Zhang23].
     - The closest learned scheduler assumes exact channel knowledge [McDowell26].
   - So a user predicted just above a quality target can decode just below it.
   - The JSCC way's advantage over physical link design was measured on single links with an
     idealised digital side: exact channel knowledge and capacity-achieving codes
     [Bourtsoulatze19, Kurka20].
     Whether that advantage survives the reports a real base station has is open.
5. Behind all three challenges lies one question: does a learned code have a link abstraction at
   all? That is, is there a rule that, like EESM for a digital block, turns the SNRs its symbols
   met into its decoded quality? And if there is, what decides whether a code obeys it?

## Paragraph 4. Pilot findings

1. To answer it, we trained a progressive code under slot-varying SNR, with the chunks of each
   codeword at independent SNRs. We then decoded it on held-out histories of mixed slot SNRs and
   tested candidate rules (Sections IV-B and V-A).
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
   - Twelve decodes per image calibrate it. The calibrated curves predict held-out mixed-SNR
     decodes with a mean absolute error of 0.03-0.06 dB per prefix length (0.05-0.09 dB on CLIC).
   - Used by a scheduler, the model predicts the decoded quality of complete scheduled deliveries
     with a mean absolute error of 0.06-0.09 dB and a bias within ±0.03 dB, whether the reports
     are exact, carry 2 dB CQI errors, or are aged (correlation 0.8).
   - But even this small error is first-order at a hard quality target. A scheduler that takes
     the calibrated curves as exact leaves an extra 1.7-2.0% of the users below 30 dB, at every
     cell size from 16 to 128 users.
5. In short, a learned code can be given a link abstraction through training. With it, each
   user's state reduces to two numbers: the phases received, and their accumulated weighted noise.
   Scheduling then becomes a planning problem, provided the abstraction's residual error is part
   of the objective.

## Paragraph 5. The method: CLASP

1. To address the aforementioned challenges, we propose CLASP (Calibration-aware,
   Link-Abstraction-based Scheduling of Progressive codes), a scheduler for learned progressive
   codes in a multi-user fading downlink.
2. Existing JSCC methods value each user at one SNR and allocate once, or learn the allocation by
   reinforcement learning. Unlike them, CLASP values every slot for every user through the code's
   link abstraction, and plans each user's delivery to the deadline exactly.
3. Specifically, CLASP has one design for each challenge:
   - **Challenge (1), the link abstraction.** We train the progressive code under slot-varying
     SNR and calibrate its MMSE link abstraction per image from 12 decodes. Each user then becomes
     a two-number state.
   - **Challenge (2), the scheduler.** On this state the frame becomes a finite-horizon restless
     bandit [Whittle88], a model already used to schedule layered video [Hosseini19] and learned
     semantic codes [McDowell26]:
     - a Lagrangian relaxation splits it into one small dynamic program per user under per-slot
       prices, solved exactly on a CPU;
     - the relaxation's value is an upper bound on every causal scheduler;
     - users are served in the order of the Lagrangian index those programs define [Hao22,
       Brown20].
   - **Challenge (3), imperfect information.**
     - Each program rewards the probability of reaching the quality target under a calibrated
       predictive distribution of the user's decoded quality, rather than a hard threshold on
       the predicted quality.
     - The distribution's spread is the measured error of the prediction, measured offline. With
       exact reports it is the calibrated curves' error alone, 0.03-0.05 dB per prefix length.
       With noisy or aged reports it includes the reports' error, 0.18-0.97 dB.
     - Each program decides on the posterior of the channel given an aged report, not on the
       report itself.
4. Results:
   - Decoded by the codec, in all four decoded cases (Kodak with fresh reports, 2 dB CQI errors
     or aged reports; CLIC with fresh reports), CLASP gains 0.5-0.6 dB of mean PSNR over
     proportional fair and brings an extra 8-15% of the users to 30 dB over per-image rate
     selection [Zhang23].
   - Simulated against the bound, its gap shrinks to 0.005 dB per user at 128 users for mean
     PSNR; for the target, its share of users at 30 dB is within ±0.3% of the bound from 16
     users on.
   - Simulated on the same frames as a digital chain with CQI-table link adaptation and
     incremental-redundancy HARQ, CLASP beats it at every CSI condition tested: by 1.2-1.6 dB of
     mean PSNR and an extra 13-16% of the users at 30 dB with 2 dB CQI errors or aged reports
     (correlation 0.8), and by 0.2 dB and 3.6% with exact reports.
   - An idealised digital code 3 dB from capacity, with the same HARQ, brings an extra 2.4% of
     the users to 30 dB over CLASP with exact reports; from 1 dB CQI errors or correlation 0.95
     on, CLASP brings an extra 5-14% of the users there.
   - Decoded on the same slots, training the code on the channel CLASP creates adds 0.21 dB (two
     seeds) over training it as long on the usual channel. Training on CLASP's prefix lengths
     with blindly taken slots gains nothing (-0.09 dB), so the gain comes from the slots CLASP
     selects.

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
   - A reward that knows the measured error of its own prediction, from the abstraction and
     from the channel reports, makes it robust to both; report-aware programs decide on aged
     reports.
3. **Performance.**
   - Decoded, CLASP gains 0.5-0.6 dB over proportional fair, and an extra 8-15% of the users at
     30 dB over per-image rate selection, with fresh, noisy or aged reports.
   - Simulated, its gap to the bound vanishes as the cell grows:
     - for mean PSNR, from 0.21 to 0.005 dB per user between 8 and 128 users;
     - for the target, within ±0.3% of the bound from 16 users on.
   - Simulated, CLASP outperforms a digital chain with link adaptation and HARQ at every CSI
     condition tested: by 1.2-1.6 dB and an extra 13-16% of the users at 30 dB with 2 dB CQI
     errors or reports aged to correlation 0.8. Aged reports cost the digital chain 1.9 dB and CLASP 0.6 dB.
   - Once reports are imperfect, from 1 dB CQI errors or correlation 0.95 on, CLASP also leads an
     idealised digital code 3 dB from capacity, with an extra 5-14% of the users at 30 dB.
   - Decoded, training the code on CLASP's channel adds 0.21 dB.

**How paragraphs 3, 5 and 6 line up:**

| challenge (paragraph 3) | design in CLASP (paragraph 5) | evidence (paragraph 6) |
|---|---|---|
| (1) no link abstraction | slot-varying-SNR training; MMSE link abstraction calibrated per image from 12 decodes | mean absolute error 0.055-0.083 dB, against EESM 0.16-0.19 and MIESM 0.29-0.34 with the same weights; decoded schedules 0.06-0.09 dB, bias within ±0.03 dB |
| (2) no tractable, certifiable scheduler | two-number state → finite-horizon restless bandit: exact per-user programs, Lagrangian bound, Lagrangian index | per-user gap to the bound 0.21 → 0.005 dB, 8 → 128 users (simulated); +0.5-0.6 dB over PF (decoded) |
| (3) imperfect information | a calibration-aware reward: the probability of reaching the target under the prediction's measured error (the curves' and the reports'); programs on the channel posterior given aged reports | an extra 1.7% of the users at 30 dB over the hard target, users missed 0.0% against 1.2% (decoded); an extra 13.3% / 12.5% over PADC at 2 dB CQI errors / r = 0.8 (decoded); +1.2-1.6 dB and an extra 13-16% over digital (simulated); aged reports cost CLASP 0.6 dB, digital 1.9 dB |

## Paragraph 7. Organisation

The rest of the paper is organised as follows:

- Section II reviews related work in three subsections: physical link design for multi-user media
  delivery, the JSCC way (adaptive and multi-user learned codes), and restless bandits with index
  policies.
- Section III states the system model and the problem.
- Section IV presents the method: training the progressive code under slot-varying SNR, its
  MMSE link abstraction, CLASP, and training the code on the channel CLASP creates.
- Section V evaluates it: the abstraction's accuracy, and CLASP against proportional fair,
  per-image rate selection, the Lagrangian bound and the digital chain.
- Section VI concludes and states the limitations.

## References cited in Section I

In the order Section I first cites them, numbered as in the PDF. The BibTeX entries are in
`paper/references.bib`, each checked against Crossref, the 3GPP archive or the IEEE 802.16
document list; why each one qualifies, and the text that supports what the paper says about
it, are in `docs/RELATED_SOURCES.md`.

| key | reference |
|---|---|
| [1] Bourtsoulatze19 | E. Bourtsoulatze, D. B. Kurka, and D. Gündüz, "Deep joint source-channel coding for wireless image transmission," IEEE Trans. Cogn. Commun. Netw., vol. 5, no. 3, pp. 567–579, Sep. 2019, doi: 10.1109/TCCN.2019.2919300. |
| [2] Gunduz23 | D. Gündüz, Z. Qin, I. E. Aguerri, H. S. Dhillon, Z. Yang, A. Yener, K. K. Wong, and C.-B. Chae, "Beyond transmitting bits: Context, semantics, and task-oriented communications," IEEE J. Sel. Areas Commun., vol. 41, no. 1, pp. 5–41, Jan. 2023, doi: 10.1109/JSAC.2022.3223408. |
| [3] Xu22 | J. Xu, B. Ai, W. Chen, A. Yang, P. Sun, and M. Rodrigues, "Wireless image transmission using deep source channel coding with attention modules," IEEE Trans. Circuits Syst. Video Technol., vol. 32, no. 4, pp. 2315–2328, Apr. 2022, doi: 10.1109/TCSVT.2021.3082521. |
| [4] Yang25 | K. Yang, S. Wang, J. Dai, X. Qin, K. Niu, and P. Zhang, "SwinJSCC: Taming Swin transformer for deep joint source-channel coding," IEEE Trans. Cogn. Commun. Netw., vol. 11, no. 1, pp. 90–104, Feb. 2025, doi: 10.1109/TCCN.2024.3424842. |
| [5] TS36213 | 3GPP, "Evolved Universal Terrestrial Radio Access (E-UTRA); Physical layer procedures (Release 8)," 3rd Generation Partnership Project, TS 36.213 V8.8.0, Sep. 2009. |
| [6] Goldsmith97 | A. J. Goldsmith and S.-G. Chua, "Variable-rate variable-power MQAM for fading channels," IEEE Trans. Commun., vol. 45, no. 10, pp. 1218–1230, Oct. 1997, doi: 10.1109/26.634685. |
| [7] Caire01 | G. Caire and D. Tuninetti, "The throughput of hybrid-ARQ protocols for the Gaussian collision channel," IEEE Trans. Inf. Theory, vol. 47, no. 5, pp. 1971–1988, Jul. 2001, doi: 10.1109/18.930931. |
| [8] Viswanath02 | P. Viswanath, D. N. C. Tse, and R. Laroia, "Opportunistic beamforming using dumb antennas," IEEE Trans. Inf. Theory, vol. 48, no. 6, pp. 1277–1294, Jun. 2002, doi: 10.1109/TIT.2002.1003822. |
| [9] Jakubczak11 | S. Jakubczak and D. Katabi, "A cross-layer design for scalable mobile video," in Proc. 17th Annu. Int. Conf. Mobile Comput. Netw. (MobiCom), Las Vegas, NV, USA, Sep. 2011, pp. 289–300, doi: 10.1145/2030613.2030646. |
| [10] Yan22 | L. Yan, Z. Qin, R. Zhang, Y. Li, and G. Y. Li, "Resource allocation for text semantic communications," IEEE Wireless Commun. Lett., vol. 11, no. 7, pp. 1394–1398, Jul. 2022, doi: 10.1109/LWC.2022.3170849. |
| [11] Xia24 | L. Xia, Y. Sun, D. Niyato, X. Li, and M. A. Imran, "Joint user association and bandwidth allocation in semantic communication networks," IEEE Trans. Veh. Technol., vol. 73, no. 2, pp. 2699–2711, Feb. 2024, doi: 10.1109/TVT.2023.3318510. |
| [12] Hou26 | H. Hou, S. Bi, X. Li, H. Zhang, and Z. Quan, "Reliable online resource allocation for multi-user semantic communications: A constraint Bayesian optimization approach," IEEE Trans. Wireless Commun., vol. 25, pp. 22323–22338, 2026, doi: 10.1109/TWC.2026.3724562. |
| [13] Wang22 | Y. Wang, M. Chen, T. Luo, W. Saad, D. Niyato, H. V. Poor, and S. Cui, "Performance optimization for semantic communications: An attention-based reinforcement learning approach," IEEE J. Sel. Areas Commun., vol. 40, no. 9, pp. 2598–2613, Sep. 2022, doi: 10.1109/JSAC.2022.3191112. |
| [14] Zhang23t | H. Zhang, H. Wang, Y. Li, K. Long, and A. Nallanathan, "DRL-driven dynamic resource allocation for task-oriented semantic communication," IEEE Trans. Commun., vol. 71, no. 7, pp. 3992–4004, Jul. 2023, doi: 10.1109/TCOMM.2023.3274145. |
| [15] McDowell26 | C. McDowell, A. Panebianco, S. Chakraborty, and Y. Sun, "Significance-driven semantic communication," arXiv:2608.28441, Aug. 2026. |
| [16] Zhang23 | W. Zhang, H. Zhang, H. Ma, H. Shao, N. Wang, and V. C. M. Leung, "Predictive and adaptive deep coding for wireless image transmission in semantic communication," IEEE Trans. Wireless Commun., vol. 22, no. 8, pp. 5486–5501, Aug. 2023, doi: 10.1109/TWC.2023.3234408. |
| [17] Ericsson03 | Ericsson, "System-level evaluation of OFDM – further considerations," 3GPP TSG-RAN WG1 #35, Lisbon, Portugal, Tdoc R1-031303, Nov. 2003. |
| [18] EMD09 | IEEE 802.16 Broadband Wireless Access Working Group, "IEEE 802.16m evaluation methodology document (EMD)," IEEE 802.16m-08/004r5, Jan. 2009. |
| [19] Kurka20 | D. B. Kurka and D. Gündüz, "DeepJSCC-f: Deep joint source-channel coding of images with feedback," IEEE J. Sel. Areas Inf. Theory, vol. 1, no. 1, pp. 178–193, May 2020, doi: 10.1109/JSAIT.2020.2987203. |
| [20] Kurka21 | D. B. Kurka and D. Gündüz, "Bandwidth-agile image transmission with deep joint source-channel coding," IEEE Trans. Wireless Commun., vol. 20, no. 12, pp. 8081–8095, Dec. 2021, doi: 10.1109/TWC.2021.3090048. |
| [21] Whittle88 | P. Whittle, "Restless bandits: Activity allocation in a changing world," J. Appl. Probab., vol. 25, no. A, pp. 287–298, 1988, doi: 10.2307/3214163. |
| [22] Hosseini19 | S. A. Hosseini and S. S. Panwar, "Restless video bandits: Optimal SVC streaming in a multi-user wireless network," IEEE Access, vol. 7, pp. 150805–150822, 2019, doi: 10.1109/ACCESS.2019.2947902. |
| [23] Hao22 | L. Hao, Y. Xu, and L. Tong, "Asymptotically optimal Lagrangian priority policy for deadline scheduling with processing rate limits," IEEE Trans. Autom. Control, vol. 67, no. 1, pp. 236–250, Jan. 2022, doi: 10.1109/TAC.2021.3049340. |
| [24] Brown20 | D. B. Brown and J. E. Smith, "Index policies and performance bounds for dynamic selection problems," Manage. Sci., vol. 66, no. 7, pp. 3029–3050, Jul. 2020, doi: 10.1287/mnsc.2019.3342. |

## Numbers used

Each number above, in order of appearance, with the line the run printed, the result file, the
COMMANDS round that wrote it and whether it is simulated or decoded: `docs/INTRO_NUMBERS.md`.
