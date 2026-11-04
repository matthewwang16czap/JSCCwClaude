# Sources of the paper's references: why each qualifies, and what supports each claim

**The rule.** Cite top venues from the last three years, or highly cited work. Applied as:

- the work is peer reviewed, in an IEEE or ACM transactions, journal or letters, a flagship
  conference, or an operations-research or applied-probability journal;
- and it was published in 2023-2026, or it is cited at least 100 times (Crossref
  `is-referenced-by-count`, read on 2026-11-03).

Two kinds are cited whatever their count: works whose method, definition or baseline the paper
uses (marked *used*), and standards documents. Two works that fail the rule are kept at the
authors' request as the closest prior work (marked *exception*): Hosseini and Panwar (IEEE Access)
and McDowell et al. (an arXiv preprint, the only one cited).

**How each claim was checked.** Every statement Sections I-III make about a work was compared
with the work's abstract (Crossref, OpenAlex or Semantic Scholar record), with its full text where
the abstract does not settle it (the readings in
`research_notes/Related work for JSCC scheduling/fulltext_readings.md`, or the documents
downloaded on 2026-11-03), or both. The support column quotes the text. Numbers are as in the PDF.

## References

**[1] Bourtsoulatze19.** IEEE TCCN 2019; 1534 citations; highly cited.
- Says (I, II-A, II-B): deep JSCC maps the image straight to channel symbols with a CNN encoder
  and decoder trained through the channel; beats JPEG/JPEG2000 with a capacity-achieving code at
  low SNR and bandwidth; no cliff effect; degrades gracefully when the SNR departs from training;
  evaluated as a single link; idealized digital side.
- Support (abstract, re-read on arXiv 1809.01733 on 2026-11-04): "directly maps the image pixel
  values to the complex-valued channel input symbols"; "two convolutional neural networks (CNNs),
  which are trained jointly, and can be considered as an autoencoder with a non-trainable layer in
  the middle that represents the noisy communication channel" (Section I: trained end to end
  through the channel); "outperforms digital transmission concatenating JPEG or JPEG2000
  compression with a capacity achieving channel code at low signal-to-noise ratio (SNR) and
  channel bandwidth values"; "does not suffer from the 'cliff effect'"; "graceful performance
  degradation as the channel SNR varies with respect to the SNR value assumed during training".

**[2] Gunduz23.** IEEE JSAC 2023; 824; recent and highly cited.
- Says (I, II-B): learned codes degrade gracefully instead of failing below a design SNR; a
  tutorial on semantic and task-oriented communications. Section I's 6G sentence is not cited;
  the full-text line "future wireless communications systems, such as 6G" would support it.
- Support (abstract): "This tutorial summarizes the efforts to date ... semantic-aware and
  task-oriented communications". Full text (arXiv 2207.09353v2): "future wireless communications
  systems, such as 6G" (Sec. I); "their performance does not fall apart when the channel quality
  falls below a certain threshold"; "these JSCC schemes also achieve graceful degradation with
  channel quality" (Sec. VI).

**[3] Xu22.** IEEE TCSVT 2022; 530; highly cited.
- Says (I, II-B): a single-link code whose gains are reported per SNR (I); one model serves a
  range of SNRs by scaling features with the SNR (II-B).
- Support (abstract): "can successfully operate with different SNR levels during transmission";
  "uses the channel-wise soft attention to scaling features according to SNR conditions".

**[4] Yang25.** IEEE TCCN 2025; 246; recent and highly cited.
- Says (I, II-B): a single-link code whose gains are reported per channel state (I); Swin
  Transformer backbone that adapts to the channel state and the target rate within one model;
  compared with BPG and 5G LDPC codes (II-B).
- Support (abstract): "neural JSCC codec built on the emerging Swin Transformer backbone";
  "modulation modules that scale latent representations based on the channel state information
  and target transmission rate"; "better or comparable performance versus the state-of-the-art
  engineered BPG + 5G LDPC coded transmission system".

**[5] TS36213.** 3GPP TS 36.213 V8.8.0 (Release 8), Sep. 2009; standard; *used* (CQI table of the
digital baseline).
- Says (I, II-A, III-D): the base station knows the channel through a report; in LTE, a 4-bit
  CQI; the efficiencies of the CQI table; a CQI index is one whose transport block could be
  received with an error probability of at most 0.1.
- Support (V8.8.0 text): "a single 4-bit wideband CQI is reported according to Table 7.2.3-1";
  "could be received with a transport block error probability not exceeding 0.1"; Table 7.2.3-1
  holds the efficiencies 0.1523 to 5.5547 used by `alloc/digital.py`. The SNR thresholds
  theta_c are not from the specification: Section III-D calls them the AWGN SNRs we take.

**[6] Goldsmith97.** IEEE TCOM 1997; 1851; highly cited.
- Says (I, II-A): adaptive modulation from the channel state; its bit error rate depends on the
  error and the delay of the channel estimate.
- Support (abstract): "variable-rate and variable-power MQAM modulation scheme ... over fading
  channels"; "We also determine the effect of channel estimation error and delay on the BER
  performance of our adaptive scheme."

**[7] Caire01.** IEEE TIT 2001; 559; highly cited; *used* (IR-HARQ).
- Says (I, II-A, III-D): HARQ; the throughput of such protocols over a fading Gaussian collision
  channel has closed forms; incremental redundancy accumulates (III-D: a block is decoded once
  the information of its transmissions adds up to its rate, eq. (14)).
- Support (abstract): "protocols for reliable packet communication based on 'hybrid-ARQ,' over a
  slotted multiple-access Gaussian channel with fading"; "we obtain closed-form throughput
  formulas". The accumulated-information rule for IR is attributed to Caire and Tuninetti by
  Steiner and Shamai (IEEE TWC 2008, read in full, `fulltext_readings.md` batch 4).

**[8] Viswanath02.** IEEE TIT 2002; 2115; highly cited; *used* (PF baseline).
- Says (I, II-A): a channel-aware scheduler such as proportional fair serves each user when its
  channel quality is near its peak.
- Support (abstract): "scheduling transmissions to users when their instantaneous channel quality
  is near the peak". The paper itself could not be fetched on 2026-11-03 (TLS failure, then HTTP
  503). Web-search summaries of two later sources, an RWTH Aachen paper and Tse's IMA 2001
  slides, tie it to proportional fair scheduling; neither source could be fetched either.

**[9] Jakubczak11.** ACM MobiCom 2011; 193; highly cited.
- Says (I, II-B): a linear analog code (SoftCast) serves receivers of different SNRs with one
  stream, each at a quality matched to its channel; noise reduces sharpness instead of causing
  glitches.
- Support (abstract): "SoftCast changes the network stack to act like a linear transform"; "a
  video source to multicast a single stream that each receiver decodes to a video quality
  commensurate with its channel quality"; errors "now reduce the sharpness of the received pixels
  but do not cause the video to glitch or stall".

**[10] Yan22.** IEEE WCL 2022; 337; highly cited.
- Says (I, II-B): channel assignment and the number of semantic symbols from a semantic spectral
  efficiency; values a transmission at one SNR.
- Support (abstract): "the semantic spectral efficiency (S-SE) is defined ... and is used to
  optimize resource allocation in terms of channel assignment and the number of transmitted
  semantic symbols".

**[11] Xia24.** IEEE TVT 2024; 43; recent.
- Says (I, II-B): user association and bandwidth from a message-level throughput.
- Support (abstract): "two fundamental problems of user association (UA) and bandwidth allocation
  (BA)"; "a new semantics-level metric, namely system throughput in message (STM)".

**[12] Hou26.** IEEE TWC 2026; 0; recent.
- Says (I, II-B): compression ratios and transmission rates by Bayesian optimization over
  Gaussian-process models of quality against SNR and compression ratio.
- Support (abstract): "a Bayesian optimization (BO)-based online algorithm"; "The edge server
  jointly determines each user's CR and transmission rate by exploiting Gaussian-process (GP)
  models that capture the relationship between reconstruction performance, signal-to-noise ratio
  (SNR), and CR".

**[13] Wang22.** IEEE JSAC 2022; 227; highly cited.
- Says (I, II-B): proximal policy optimization with an attention network assigns resource blocks
  and the part of the semantic content to send.
- Support (abstract): "the BS must select an appropriate resource block for each user as well as
  determine and transmit part of the semantic information"; "a proximal-policy-optimization-based
  reinforcement learning (RL) algorithm integrated with an attention network".

**[14] Zhang23t.** IEEE TCOM 2023; 191; recent and highly cited.
- Says (I, II-B): a DDPG agent sets each user's compression ratio, power and bandwidth.
- Support (abstract): "a deep deterministic policy gradient (DDPG) agent at the micro base
  station"; "a joint optimization problem of the semantic compression ratio, transmit power, and
  bandwidth of each user".

**[15] McDowell26.** arXiv 2608.28441 (v2, 31 Aug 2026); preprint; *exception*, kept at the
authors' request as the closest prior learned scheduler.
- Says (I, II-C): learns the allocation by reinforcement learning inside a restless bandit over
  independent Rayleigh-fading slots, for a learned semantic code; a Lagrangian relaxation with one
  price splits it into per-sensor programs whose values are learned by reinforcement learning; the
  reward is a task value computed from the decoder's posterior; a fluid upper bound is stated but
  not plotted, so the learned policy has no certificate; each slot carries a fresh codeword of the
  newest observation, so nothing accumulates across slots toward a deadline; the channel state is
  known exactly.
- Support: read in full (arXiv v2). The notes of that reading (`fulltext_readings.md` item 6)
  record: N sensors over an orthogonal multiple-access channel with i.i.d. Rayleigh block fading
  per slot; a meta-learned code; the per-slot budget relaxed to a discounted one with one
  multiplier found by bisection; each sensor's Q-function learned by PPO; the reward computed from
  the decoder's own posterior; the fluid LP upper bound stated in Appendix C and plotted in no
  figure; a fresh, self-contained codeword of the newest observation in every slot, no deadline,
  and CSI at both ends.

**[16] Zhang23.** IEEE TWC 2023; 277; recent and highly cited; *used* (per-image rate selection).
- Says (I, II-B): PADC predicts each image's PSNR from its content, the SNR and the compression
  ratio, and sends it at the least compression ratio meeting a quality constraint, treating the
  prediction as exact.
- Support (abstract): "an Oracle Network (OraNet) model for predicting peak-signal-to-noise (PSNR)
  value ... according to its contents, channel signal to noise ratio (SNR) and the compression
  ratio (CR) value, and a CR optimizer aims at finding the minimal data-level or instance-level CR
  with a PSNR quality constraint". Read in full with its code (`docs/SURVEY.md` A).

**[17] Ericsson03.** 3GPP TSG-RAN WG1 #35, Tdoc R1-031303, Nov. 2003; standards contribution, the
origin of EESM; *used* (EESM baseline).
- Says (I, II-A): an effective SINR mapping turns the SNRs of a block's resources into one
  effective SNR, from which the block error rate is read off the AWGN curve; EESM is derived from
  the union-Chernoff bound with an MCS-dependent parameter, for OFDM.
- Support (the Tdoc): "maps the instantaneous channel state into a single scalar value, an
  effective SIR, which then is used to find an estimate of the Block-Error probability";
  "The effective SNR is then used to find an estimate of the block-error probability from basic
  AWGN link-level-performance"; "The exponential ESM is derived based on the Union-Chernoff
  bound"; "generalized to include an MCS-dependent parameter"; "system-level evaluation of e.g.
  OFDM performance".

**[18] EMD09.** IEEE 802.16m-08/004r5, Jan. 2009; standard; *used* (MIESM baseline).
- Says (I, II-A): MIESM maps the SINRs through the mutual information per received symbol or per
  bit.
- Support (Sec. 4.3): "Within the class of MIESM there are two variants, one is based on the
  mutual information per received symbol normalized to yield the bit mutual information and the
  other directly computes the bit mutual information. Each of these PHY abstractions uses a
  different function to map the vector of SINR values to a single number."

**[19] Kurka20.** IEEE JSAIT 2020; 492; highly cited.
- Says (I, II-B): channel output feedback, with shorter variable-length transmission; the digital
  side of the comparison has exact channel knowledge and capacity-achieving codes.
- Support (abstract): "exploits the channel output feedback, and provides considerable
  improvements in terms of the end-to-end reconstruction quality for fixed-length transmission, or
  in terms of the average delay for variable-length transmission". Full text (arXiv 1911.11174,
  `deepjscc_link_level.md`): the separation scheme "inherently assumes perfect CSI at the
  receiver"; its baselines are BPG with capacity-achieving codes.

**[20] Kurka21.** IEEE TWC 2021; 195; highly cited; *used* (the layered-code mechanism).
- Says (I, II-B): layers as successive refinements or multiple descriptions with negligible loss;
  refinements lose value below the training SNR.
- Support (abstract): "the well known problems of successive refinement and multiple
  descriptions"; "can learn to transmit the source progressively with negligible losses in the
  end-to-end performance compared with a single transmission". Full text (`fulltext_readings.md`
  item 8): "since the reconstructed base layer becomes significantly different from what the
  encoder expects based on the training data".

**[21] Whittle88.** J. Appl. Probab. 1988; 814; highly cited.
- Says (I, II-C, III-E): the constraint on the number of projects operated is relaxed to its
  expected value; the multiplier defines an index; the conjecture of near-optimality (III-E: the
  scheduling problem becomes a restless bandit).
- Support (abstract): "under a constraint upon the expected number of projects in operation. The
  Lagrange multiplier associated with this constraint defines an index"; "for m and n large in
  constant ratio, the policy of operating the m projects of largest current index is nearly
  optimal" (a conjecture).

**[22] Hosseini19.** IEEE Access 2019; 0 citations; *exception*, kept at the authors' request as
the closest restless-bandit work on video. The published version of arXiv 1612.03985 (same
authors and title).
- Says (I, II-A, II-C): scalable video streaming to many users over Markov channels, modeled as a
  restless bandit and relaxed to a linear program whose value is plotted beside the policies; up to
  twice the network utilization of PF; layered video already scheduled by a restless bandit.
- Support (abstract): "We model the QA-adaptive scheduling ... as a Restless Bandit"; "our
  QA-Aware strategy can achieve up to two times improvement in network utilization compared to
  popular baseline algorithms such as Proportional Fairness". Read in full (arXiv v2); the notes
  of that reading (`fulltext_readings.md` item 1) record a finite Markov channel per user that sets
  how many SVC layers a subchannel carries, the relaxed problem solved as a linear program, its
  value plotted as "the optimum" beside the policies at 20 users, and layers that decode or fail.

**[23] Hao22.** IEEE TAC 2022; 5; *used* (the Lagrangian priority policy is CLASP's policy).
- Says (I, II-C): the Lagrangian priority value is the greatest tax under which processing is
  optimal; serving in that order is asymptotically optimal; one multiplier per stage.
- Support (abstract): "we define the Lagrangian priority value as the greatest tax under which it
  is optimal to activate the arm"; "establish its asymptotic optimality as the system scales".
  Read in full (`fulltext_readings.md` item 4): "Lagrangian relaxation with one multiplier per
  stage".

**[24] Brown20.** Manage. Sci. 2020; 72; *used* (the Lagrangian index policy and bound).
- Says (I, II-C): relaxing the limit on the number of items selected gives an upper bound and the
  Lagrangian index policy, both asymptotically optimal with many items; tie-breaking matters.
- Support (abstract): "a Lagrangian relaxation of the DP that relaxes the constraint limiting the
  number of items that may be selected"; "upper bounds on the performance of an optimal policy";
  "under mild conditions, these policies and bounds are asymptotically optimal for problems with
  many items; mixed policies and tiebreaking play an essential role".

**[25] Sullivan12.** IEEE TCSVT 2012; 7753; highly cited; *used* (HEVC in the digital baseline).
- Says (II-A, III-D): HEVC as the standard codec; the HEIC files of the digital baseline use HEVC
  intra coding ([25] is cited for HEVC; HEIC is the HEIF image format with HEVC-coded images).
- Support (abstract): "This paper provides an overview of the technical features and
  characteristics of the HEVC standard."

**[26] Chase85.** IEEE TCOM 1985; 743; highly cited.
- Says (II-A, III-D): the receiver combines the noisy copies of a retransmitted block (III-D
  models it as the copies' SNRs adding, eq. (15); the equation is ours, [26] is cited for the
  scheme).
- Support (title): "Code combining - A maximum-likelihood decoding approach for combining an
  arbitrary number of noisy packets".

**[27] Lagen20.** IEEE ICC 2020; 101; highly cited.
- Says (II-A): the 5G NR abstraction of the ns-3 simulator calibrates EESM and obtains SINR-BLER
  tables with a link-level simulator.
- Support (abstract): "its integration into an open-source ns-3 based NR system-level simulator";
  "we used an NR-compliant link-level simulator to calibrate the EESM method as well as to obtain
  SINR-block error rate (BLER) lookup tables".

**[28] Chou06.** IEEE TMM 2006; 405; highly cited.
- Says (II-A): RaDiO reduces streaming to the error-cost optimized transmission of each data unit,
  an MDP with a Lagrange multiplier pricing the rate, essentially at the operational D(R) of the
  source at the erasure channel's capacity.
- Support (abstract): "can be reduced to the problem of error-cost optimized transmission of an
  isolated data unit"; "the framework of Markov decision processes"; "essentially achieve the best
  possible performance: the operational distortion-rate function of the source at the capacity of
  the packet erasure channel". Read in full (`fulltext_readings.md` item 5): "priced by a Lagrange
  multiplier".

**[29] Pahalawatta07.** IEEE JSAC 2007; 118; highly cited.
- Says (II-A): gradient-based schedulers set each user's rate from its channel quality and the
  gradient of a utility built from its received-video distortion.
- Support (abstract): "user data rates are dynamically adjusted based on channel quality as well
  as the gradients of a utility function. The user utilities are designed as a function of the
  distortion of the received video."

**[30] HouBorkar09.** IEEE INFOCOM 2009; 220; highly cited.
- Says (II-A): packets expire at the end of each frame; feasible per-client delivery ratios are
  characterized; largest-debt-first fulfills every feasible set.
- Support: read in full; the notes of that reading (`fulltext_readings.md` batch 4, `docs/SURVEY.md`
  F2) record frames of tau slots with one packet per client per frame, expired at the frame's end,
  per-client long-run delivery ratios, and that largest-debt-first is feasibility optimal.

**[31] Dai22.** IEEE JSAC 2022; 378; highly cited.
- Says (II-B): learns an entropy model of its latent representation and adapts the rate to it.
- Support (abstract): "learns both the source latent representation and an entropy model as the
  prior on the latent representation. Accordingly, novel adaptive rate transmission ...
  mechanisms are developed".

**[32] YangBianKim22.** IEEE TCCN 2022; 201; highly cited.
- Says (II-B): deep JSCC with OFDM over multipath fading; 2.5-4 dB of SNR gain over BPG with LDPC.
- Support (abstract): "we combine an autoencoder with orthogonal frequency division multiplexing
  (OFDM) to cope with multipath fading"; "2.5 - 4 dB SNR gain for the equivalent image quality
  compared to ... BPG and ... LDPC".

**[33] Wu24m.** IEEE TWC 2024; 121; recent and highly cited.
- Says (II-B): MIMO channels, with robustness to channel estimation errors.
- Support (abstract): "over multiple-input multiple-output (MIMO) channels"; "exhibiting
  robustness to channel estimation errors".

**[34] Wu24f.** IEEE TWC 2024; 60; recent.
- Says (II-B): feedback refines the receiver's estimate block by block.
- Support (abstract): "employs a single encoder to facilitate transmission over multiple blocks,
  refining the receiver's estimation at each block".

**[35] Mu23.** IEEE JSAC 2023; 88; recent.
- Says (II-B): in a non-orthogonal uplink shared with a bit-based user, the semantic user's
  communication mode is chosen in each fading state.
- Support (abstract): "a secondary far user (F-user) employs the semantic communication (SemCom)
  while a primary near user (N-user) employs the conventional bit-based communication (BitCom)";
  "via the most suitable communication method at each fading state".

**[36] Xie22.** IEEE JSAC 2022; 469; highly cited.
- Says (II-B): multi-user semantic transceivers trained end to end for image retrieval, machine
  translation and visual question answering.
- Support (abstract): "deep learning (DL) based multi-user semantic communication systems"; "image
  retrieval, machine translation, and visual question answering (VQA)"; "DeepSC-IR is trained to
  ...", "DeepSC-MT is trained to ...".

**[37] WeberWeiss90.** J. Appl. Probab. 1990; 337; highly cited.
- Says (II-C): Whittle's conjecture holds when the fluid approximation has a globally stable
  equilibrium; a counterexample otherwise.
- Support (abstract): "the conjecture is true if the differential equation describing the fluid
  approximation to the index policy has a globally stable equilibrium point"; "we present an
  example for which the index policy is not asymptotically optimal".

**[38] Adelman08.** Oper. Res. 2008; 169; highly cited.
- Says (II-C): subproblems coupled only through action constraints admit Lagrangian and LP
  relaxations whose value functions separate by subproblem.
- Support (abstract): "multiple subproblems that are independent of each other except for a
  collection of coupling constraints on the action space"; "an additively separable value
  function approximation using two techniques, namely, Lagrangian relaxation and the linear
  programming (LP) approach".

**[39] Gast24.** Math. Oper. Res. 2024; 19; recent.
- Says (II-C): an LP-based index policy is asymptotically optimal over finite and infinite
  horizons, at a square-root rate in general and an exponential one under nondegeneracy.
- Support (abstract): "under both finite and infinite time horizons"; "the LP-index policy that is
  asymptotically optimal with square root convergence rate on all models and with exponential rate
  if the model is nondegenerate in finite horizon".

**[40] LiuZhao10.** IEEE TIT 2010; 284; highly cited.
- Says (II-C): closed-form Whittle index for dynamic multichannel access; a Lagrangian relaxation
  bounds performance for non-identical channels.
- Support (abstract): "establish the indexability and obtain Whittle index in closed form"; "For
  nonidentical arms, we develop efficient algorithms for computing a performance upper bound given
  by Lagrangian relaxation."

**[41] Kadota18.** IEEE/ACM ToN 2018; 575; highly cited.
- Says (II-C): for the age of information over unreliable channels, Whittle's index policy is
  near optimal.
- Support (abstract): "a Whittle's Index policy"; "both the Max-Weight and Whittle's Index policies
  ... achieve near optimal performance".

**[42] Shisher23.** IEEE JSAIT 2023; 29; recent.
- Says (II-C): feature-length selection and transmission scheduling form a restless multi-armed,
  multi-action bandit.
- Support (abstract): "the co-design problem becomes a restless multi-arm multi-action bandit
  problem".

**[43] Jakes94.** W. C. Jakes (ed.), Microwave Mobile Communications, IEEE Press, 1994 (the
reissue of the 1974 book); 2145 citations; highly cited.
- Says (III-B): the magnitude of the correlation of the fading process at lag tau under Jakes'
  isotropic-scattering model is |J0(2 pi f_D tau)|.
- Support: Crossref record (doi: 10.1109/9780470545287). The correlation is the inverse Fourier
  transform of the Doppler spectrum of that model, 1 / (pi sqrt(f_D^2 - nu^2)) for |nu| < f_D,
  and that transform equals J0(2 pi f_D tau), checked numerically on 2026-11-03 (five lags, to
  six decimals). Wikipedia's "Rayleigh fading" attributes the spectrum to Clarke (1968) and its
  popularization to Jakes' book, and the IEEE 802.16m EMD cites the book for the Jakes Doppler
  spectrum.

## Considered and left out

These works came up in the survey (`docs/SURVEY.md`, `reports/`) and fail the rule (Crossref
citations, 2026-11-03):

| work | venue, year | citations | reason |
|---|---|---|---|
| Fu and van der Schaar, multi-user wireless video as per-user MDPs with prices | IEEE JSAC 2010 | 74 | before 2023, under 100 |
| Yilmaz, Karamanlı and Gündüz, distributed DeepJSCC over a MAC | IEEE ICC 2023 | 38 | conference, under 100 |
| Yu, Xu and Tong, deadline scheduling as restless bandits | IEEE TAC 2018 | 40 | before 2023, under 100 |
| Ouyang, Eryilmaz and Shroff, downlink scheduling over Markovian fading channels | IEEE/ACM ToN 2016 | 17 | before 2023, under 100 |
| Ouyang, Murugesan, Eryilmaz and Shroff, channel memory and Whittle indexability | IEEE TIT 2015 | 18 | before 2023, under 100 |
| Verloop, priority policies for restless bandits | Ann. Appl. Probab. 2016 | 86 | before 2023, under 100 |
| Zayas-Cabán, Jasin and Wang, finite-horizon restless bandits | Adv. Appl. Probab. 2019 | 37 | before 2023, under 100 |
| Ayesta, Erausquin and Jacko, potential-improvement index | Perform. Eval. 2010 | 51 | before 2023, under 100 |
| Steiner and Shamai, multi-layer broadcasting HARQ | IEEE TWC 2008 | 26 | before 2023, under 100 |
| Le Ny, Feron and Dahleh, scheduling Kalman filters | IEEE TAC 2011 | 31 | before 2023, under 100 |
| He et al., progressive pseudo-analog video | IEEE TMM 2017 | 28 | before 2023, under 100 |
| Borkar et al., opportunistic scheduling as restless bandits | IEEE TCNS 2018 | 34 | before 2023, under 100 |
| Brown and Zhang, dynamic fluid policies | Oper. Res. 2022 | 23 | before 2023, under 100 |
| Hawkins, Lagrangian decomposition (thesis); Hu and Frazier; Zhang and Frazier | thesis, arXiv | - | not peer reviewed |
