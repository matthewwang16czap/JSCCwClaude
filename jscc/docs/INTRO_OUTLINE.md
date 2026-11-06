# Section I (Introduction): paragraph-by-paragraph outline

Seven paragraphs, in the order of the supervisor's template. Each numbered item is one sentence
(or two short ones) of the prose. [Keys] point to the reference list at the end. Every number,
in the order it appears here, with the printed line, the result file and the round it comes
from, is in `docs/INTRO_NUMBERS.md`.

**Terms used throughout.** *Learned code*: a deep JSCC encoder-decoder pair. *Progressive code*: a
learned code whose symbols are sent in phases and that decodes from whatever prefix of phases has
arrived. *Slot*, *frame*, *deadline*: the base station has a frame of T slots and serves K users;
every image must be delivered by the end of the frame. *Link abstraction*: a rule that maps the
SNRs a codeword's symbols met to its decoded quality. *Physical link design* and *JSCC design*:
the two existing categories (paragraph 2). *Calibration-aware*: deciding on a calibrated
predictive distribution of a user's decoded quality, whose spread is the measured error of the
prediction, from the quality model and from the channel reports. *CLASP* (Calibration-aware,
Link-Abstraction-based Scheduling of Progressive codes): our method (paragraph 5). *Share of
users at the target*: the percentage of users whose decoded PSNR reaches the quality target by
the deadline. A difference between two shares is the difference of the two percentages, given in
% of all users: "an extra X% of the users" means X% more of all users reach the target.

## Paragraph 1. Why the problem matters

1. Deep JSCC can map an image directly to channel symbols with a single learned encoder and
   decoder, eliminating the separate source and channel codes that digital coding needs (widely
   known in the field, so not cited).
2. Trained end to end through the channel, deep JSCC beats digital transmission at low SNR and
   degrades gracefully as the channel worsens, instead of failing once the SNR drops below what a
   digital code was designed for [Bourtsoulatze19, Gunduz23].
3. That makes it a strong candidate for image delivery in 6G, yet a learned code is designed and
   evaluated on a single link at a given SNR.
4. A cellular downlink is a multi-user, multi-slot system with fading, imperfect channel knowledge
   and deadlines, so the gain a learned code reports on a single link [Bourtsoulatze19, Xu22,
   Yang25] is reduced when a base station delivers to many users over fading slots.
5. A base station splits a frame of slots among many users through a scheduler, and each user's
   slots fade independently.
6. Moreover, the base station knows only an estimate of the current slot's channel, making a
   channel-quality report [TS36213] often late or wrong. Since each delivery has a deadline, a
   low-latency scheduling method is necessary as well.

## Paragraph 2. Existing approaches: physical link design and JSCC design (Fig. 1)

Order: physical link design and its limitations first; then "In contrast," JSCC design, the
category ours belongs to. Both are described by their schedulers, since from paragraph 1 on the
transmission is a learned code.

1. Existing approaches to serving images to many users over a fading downlink fall into two
   categories, namely physical link design and JSCC design. The scheduler of the former serves
   the bits of a separate image codec, whereas that of the latter serves a learned code.
2. Physical link design is how cellular systems deliver images today. As illustrated in
   Fig. 1(a), its scheduler selects both the served users and the rate of their blocks in every
   slot.
3. Link adaptation sets each rate from the CQI, HARQ retransmits failed blocks, and PF
   scheduling serves each user near the peak of its channel quality [Goldsmith97, Caire01,
   Viswanath02].
4. **Limitation 1.** These designs assume that each block is either decoded or lost and produce
   the cliff effect, in which a slot below the design SNR of its block delivers nothing and any
   SNR above it adds no quality [Bourtsoulatze19, Gunduz23].
5. **Limitation 2.** Link adaptation and HARQ protect the bits from imperfect channel knowledge
   only by spending extra slots.
6. In contrast, JSCC design constructs the downlink on a learned code, whose distortion varies
   continuously with the SNR of its symbols. Hence no transmission is lost outright, and no coding
   rate has to be matched to an uncertain report.
7. On a single link, per-image rate selection sends each image at the least rate at which its
   predicted quality meets a target [Zhang23].
8. As illustrated in Fig. 1(b), multi-user schedulers in this category value each user by such a
   predicted quality rather than by bits.
9. They allocate the shared channel by the quality predicted at each user's SNR [Yan22, Xia24,
   Hou26], train an allocation policy by deep reinforcement learning [Wang22, Zhang23t], or embed
   such learning in a restless bandit over fading slots [McDowell26] (Section II-B).

### Figure 1

The figure is one column wide, with three panels stacked top to bottom: (a), (b) and (c). It is
drawn in TikZ in `paper/figs/overview.tex`, which the figure environment of
`paper/sections/introduction.tex` includes.

- **Style.** All text is bold. Panels (a) and (b) use grey, low-contrast tones; panel (c) uses
  saturated colour. Red appears only on the words "link abstraction" in (c). There are no
  equations and no axes beyond the insets.
- **(a) Physical link design.**
  - The chain: image → "codec" → bits ("0110" on the arrow) → "link adaptation + HARQ" → a row
    of slots. The text says link adaptation and never defines AMC, so the box does too.
  - Each slot holds one user's block, marked inside the slot ✓ (decoded) or ✗ (lost). Exactly
    one block is lost. As in a real HARQ process, its NACK returns to the HARQ box and the HARQ
    box resends the same block in a later slot, where it decodes. The two arrows form one loop
    under the slot row, labelled "NACK" and "resend" side by side.
  - A "scheduler (CQI)" box above the row picks the user and the rate for each slot; it values a
    slot by the bits it delivers.
  - Inset: quality against SNR as a step (the cliff).
- **(b) JSCC design.**
  - The chain: image → "learned JSCC" → one shared channel, a bar split into unequal shares for
    users 1-3, each share labelled with the single SNR its user is valued at. As in (a) and (c),
    one image icon stands for the images of all users.
  - One "scheduler" box, labelled "q(SNR) or RL policy", sets the split with a single arrow into
    the channel. Existing designs use one of the two, never both together.
  - Inset: quality against SNR as a smooth curve.
- **(c) CLASP.**
  - The chain: image → "progressive JSCC" → phases spread over slots of different SNRs (slot
    colour = SNR) → "link abstraction" → each user's two-number state → "restless-bandit
    scheduler".
  - The scheduler serves users slot by slot up to a "deadline" marker. A shade bar labelled SNR
    gives the slot colour scale.
  - Inset: predicted against decoded quality, drawn as a schematic band along the diagonal with no
    data points, so that it cannot be read as a measurement.

**Caption.** "Fig. 1. Approaches to serving images to many users over a fading downlink.
(a) Physical link design transmits the bits of a separate codec at a rate set from a channel
report. Every block either decodes or fails, and HARQ resends a failed block in a later slot.
(b) JSCC design transmits a learned code, and its scheduler splits the channel among users
according to a quality predicted at one SNR or a policy learned by reinforcement learning.
(c) CLASP schedules a progressive code trained under slot-varying SNR, whose link abstraction
values every slot. This restless-bandit scheduler serves every user slot by slot up to the
deadline."

## Paragraph 3. Challenges of JSCC design

1. Although JSCC design avoids the cliff effect of physical link design, it leaves three
   challenges unresolved (a list, each item an italic label followed by a colon).
2. **(1) Absence of a link abstraction.**
   - The quality models of JSCC design, whether fitted curves or learned predictors [Yan22,
     Zhang23, Hou26], take a single SNR as input. Each model maps this SNR to the predicted
     quality of a whole image, as if every symbol of the image met the same channel.
   - Physical link design provides rules for the error rate of a digital block, namely the
     exponential and mutual-information effective SNR mappings (EESM and MIESM) [Ericsson03,
     EMD09], but no rule is known for how the SNRs of the slots that carry one image combine into
     the quality of a nonlinear code.
3. **(2) Absence of tractable and certifiable scheduling.**
   - Choosing the users of every slot up to the deadline, without knowledge of future slots, is a
     stochastic dynamic program whose state grows exponentially with the number of users.
   - The quality-based allocations are myopic, since they ignore the slots that remain in the
     frame [Yan22, Xia24, Hou26].
   - Learned policies [Wang22, Zhang23t, McDowell26] require many training interactions and offer
     no certificate of their distance from the optimum.
4. **(3) Imperfect information.**
   - The scheduler makes decisions based on predictions of both the quality and the channel, both
     of which are imperfect.
   - Per-image rate selection [Zhang23] treats its predicted quality as exact, so a user predicted
     just above a quality target may decode just below it.
   - The learned restless-bandit scheduler likewise assumes exact channel knowledge
     [McDowell26].
   - The advantage that motivates these designs was measured against an idealized digital link
     with exact channel knowledge and capacity-achieving codes [Bourtsoulatze19, Kurka20]. Whether
     this advantage persists under the reports of a real base station remains open.

## Paragraph 4. Why the first challenge comes first, and the pilot findings

Only the findings are stated, without their numbers. The numbers behind each finding are listed
under it as evidence, in the order of `docs/INTRO_NUMBERS.md`, and are quoted in Sections IV-V.

1. The first challenge underlies the other two, since a scheduler needs a link abstraction both to
   plan over the remaining slots with a certificate and to account for prediction errors. We
   therefore ask whether the decoded quality of a learned code can be predicted from its slot SNRs
   at all, and how the code should be trained to make this possible.
   - To answer these questions, we trained the same progressive architecture in two ways, under
     slot-varying SNR and at one constant SNR per image. We then compared candidate rules on both
     codes and evaluated the most accurate rule within a scheduler (Sections IV-B, V-A and V-F).
     Three findings emerged.
   - Background, not in the prose: the code has six phases, and in training the chunks of each
     codeword meet independent SNRs. It is a vision transformer whose encoder and decoder never
     see the SNR.
2. **(1) An additive MMSE abstraction.** First, the code trained under slot-varying SNR has an
   additive link abstraction, a property that linear analog codes have by construction
   [Jakubczak11]. A source term plus a weighted sum of per-slot minimum mean-square-error (MMSE)
   terms predicts its decoded quality more accurately than EESM and MIESM.
   - Evidence, not in the prose: D = Dsrc + C Σ_j ρ_j φ(n_j). Mean absolute error on held-out
     decodes, every rule given the same per-phase sensitivity weights (in brackets, the rule's
     standard equal-weight form):

   | test set | MMSE rule | EESM | MIESM |
   |---|---|---|---|
   | Kodak | 0.055 dB (0.37) | 0.19 dB (0.37) | 0.34 dB (0.49) |
   | CLIC | 0.083 dB (0.41) | 0.16 dB (0.37) | 0.29 dB (0.44) |

   Both parts are needed, the MMSE form and a weight per phase, because the code is ordered and
   the phase a slot carries matters.
3. **(2) The role of training.** Second, the training condition determines whether the abstraction
   holds. For the code trained at one constant SNR per image, none of the candidate rules predicts
   the decoded quality with comparable accuracy.
   - Evidence, not in the prose: on that code the MMSE rule errs by 0.22 dB, and no candidate rule
     does better than 0.15 dB.
   - Kept for Sections IV-V: a constant-SNR code encodes its later phases against the base
     reconstruction it expects at its training SNR, but the actual base reconstruction changes
     when the slot SNRs vary. A similar mismatch reduces the gain of the refinement layers in
     DeepJSCC-l below the training SNR (Kurka21, cited in Section II-B).
4. **(3) Calibration, accuracy and the remaining error.** Third, this abstraction can be
   calibrated from a small number of decodes per image. Within a scheduler, it then predicts the
   quality of complete deliveries accurately, even with noisy or aged channel reports. The remaining prediction error
   is small but not negligible, since a scheduler that ignores it leaves more users below a
   quality target.
   - Evidence, not in the prose: twelve decodes per image calibrate the curves, which predict
     held-out mixed-SNR decodes with a mean absolute error of 0.03-0.06 dB per prefix length
     (0.05-0.09 dB on CLIC).
   - Evidence, not in the prose: the predicted quality of complete scheduled deliveries has a
     mean absolute error of 0.06-0.09 dB and a bias within ±0.03 dB, whether the reports are
     exact, carry 2 dB CQI errors, or are aged (correlation 0.8).
   - Evidence, not in the prose: a scheduler that takes the calibrated curves as exact leaves an
     extra 1.7-2.0% of the users below 30 dB, against one that models their error, at every cell
     size from 16 to 128 users.
5. These findings show that a learned code can be scheduled through its link abstraction, provided
   that the scheduler models this error.
   - Background, not in the prose: under the abstraction, each user's state reduces to two
     numbers, the phases received and their accumulated weighted noise (paragraph 5).

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
| (1) link abstraction | slot-varying-SNR training; MMSE link abstraction calibrated per image from 12 decodes | mean absolute error 0.055-0.083 dB, against EESM 0.16-0.19 and MIESM 0.29-0.34 with the same weights; decoded schedules 0.06-0.09 dB, bias within ±0.03 dB |
| (2) tractable and certifiable scheduling | two-number state → finite-horizon restless bandit: exact per-user programs, Lagrangian bound, Lagrangian index | per-user gap to the bound 0.21 → 0.005 dB, 8 → 128 users (simulated); +0.5-0.6 dB over PF (decoded) |
| (3) imperfect information | a calibration-aware reward: the probability of reaching the target under the prediction's measured error (the curves' and the reports'); programs on the channel posterior given aged reports | an extra 1.7% of the users at 30 dB over the hard target, users missed 0.0% against 1.2% (decoded); an extra 13.3% / 12.5% over PADC at 2 dB CQI errors / r = 0.8 (decoded); +1.2-1.6 dB and an extra 13-16% over digital (simulated); aged reports cost CLASP 0.6 dB, digital 1.9 dB |

## Paragraph 7. Organisation

The rest of the paper is organised as follows:

- Section II reviews related work in three subsections: physical link design for multi-user media
  delivery, JSCC design (adaptive and multi-user learned codes), and restless bandits with index
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
| [9] Zhang23 | W. Zhang, H. Zhang, H. Ma, H. Shao, N. Wang, and V. C. M. Leung, "Predictive and adaptive deep coding for wireless image transmission in semantic communication," IEEE Trans. Wireless Commun., vol. 22, no. 8, pp. 5486–5501, Aug. 2023, doi: 10.1109/TWC.2023.3234408. |
| [10] Yan22 | L. Yan, Z. Qin, R. Zhang, Y. Li, and G. Y. Li, "Resource allocation for text semantic communications," IEEE Wireless Commun. Lett., vol. 11, no. 7, pp. 1394–1398, Jul. 2022, doi: 10.1109/LWC.2022.3170849. |
| [11] Xia24 | L. Xia, Y. Sun, D. Niyato, X. Li, and M. A. Imran, "Joint user association and bandwidth allocation in semantic communication networks," IEEE Trans. Veh. Technol., vol. 73, no. 2, pp. 2699–2711, Feb. 2024, doi: 10.1109/TVT.2023.3318510. |
| [12] Hou26 | H. Hou, S. Bi, X. Li, H. Zhang, and Z. Quan, "Reliable online resource allocation for multi-user semantic communications: A constraint Bayesian optimization approach," IEEE Trans. Wireless Commun., vol. 25, pp. 22323–22338, 2026, doi: 10.1109/TWC.2026.3724562. |
| [13] Wang22 | Y. Wang, M. Chen, T. Luo, W. Saad, D. Niyato, H. V. Poor, and S. Cui, "Performance optimization for semantic communications: An attention-based reinforcement learning approach," IEEE J. Sel. Areas Commun., vol. 40, no. 9, pp. 2598–2613, Sep. 2022, doi: 10.1109/JSAC.2022.3191112. |
| [14] Zhang23t | H. Zhang, H. Wang, Y. Li, K. Long, and A. Nallanathan, "DRL-driven dynamic resource allocation for task-oriented semantic communication," IEEE Trans. Commun., vol. 71, no. 7, pp. 3992–4004, Jul. 2023, doi: 10.1109/TCOMM.2023.3274145. |
| [15] McDowell26 | C. McDowell, A. Panebianco, S. Chakraborty, and Y. Sun, "Significance-driven semantic communication," arXiv:2608.28441, Aug. 2026. |
| [16] Ericsson03 | Ericsson, "System-level evaluation of OFDM – further considerations," 3GPP TSG-RAN WG1 #35, Lisbon, Portugal, Tdoc R1-031303, Nov. 2003. |
| [17] EMD09 | IEEE 802.16 Broadband Wireless Access Working Group, "IEEE 802.16m evaluation methodology document (EMD)," IEEE 802.16m-08/004r5, Jan. 2009. |
| [18] Kurka20 | D. B. Kurka and D. Gündüz, "DeepJSCC-f: Deep joint source-channel coding of images with feedback," IEEE J. Sel. Areas Inf. Theory, vol. 1, no. 1, pp. 178–193, May 2020, doi: 10.1109/JSAIT.2020.2987203. |
| [19] Jakubczak11 | S. Jakubczak and D. Katabi, "A cross-layer design for scalable mobile video," in Proc. 17th Annu. Int. Conf. Mobile Comput. Netw. (MobiCom), Las Vegas, NV, USA, Sep. 2011, pp. 289–300, doi: 10.1145/2030613.2030646. |
| [20] Whittle88 | P. Whittle, "Restless bandits: Activity allocation in a changing world," J. Appl. Probab., vol. 25, no. A, pp. 287–298, 1988, doi: 10.2307/3214163. |
| [21] Hosseini19 | S. A. Hosseini and S. S. Panwar, "Restless video bandits: Optimal SVC streaming in a multi-user wireless network," IEEE Access, vol. 7, pp. 150805–150822, 2019, doi: 10.1109/ACCESS.2019.2947902. |
| [22] Hao22 | L. Hao, Y. Xu, and L. Tong, "Asymptotically optimal Lagrangian priority policy for deadline scheduling with processing rate limits," IEEE Trans. Autom. Control, vol. 67, no. 1, pp. 236–250, Jan. 2022, doi: 10.1109/TAC.2021.3049340. |
| [23] Brown20 | D. B. Brown and J. E. Smith, "Index policies and performance bounds for dynamic selection problems," Manage. Sci., vol. 66, no. 7, pp. 3029–3050, Jul. 2020, doi: 10.1287/mnsc.2019.3342. |

## Numbers used

Each number above, in order of appearance, with the line the run printed, the result file, the
COMMANDS round that wrote it and whether it is simulated or decoded: `docs/INTRO_NUMBERS.md`.
