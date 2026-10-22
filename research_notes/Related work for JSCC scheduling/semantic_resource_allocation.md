# Semantic, task-oriented and DeepJSCC resource allocation and multi-user scheduling (2019-2026): prior work for deadline scheduling of learned image codes

*How these notes were made (2026-10-20). Please read this before citing anything below.* The only tool that worked in this sandbox was web search. arxiv.org, export.arxiv.org, ar5iv, alphaxiv, Semantic Scholar, Crossref, IEEE Xplore, ResearchGate and the university repositories were all blocked, either by a proxy 403 or by a DNS failure. Then the shared web-search budget ran out after 33 successful queries. So every entry comes from abstracts, search-engine summaries and repository metadata, **not from full texts**. A venue, volume or DOI appears only when a search result showed it. Anything recalled from memory is marked "(unverified)". The "Lacks vs ours" lines are my inferences from those abstract-level descriptions. I did not re-read the entries that `jscc/docs/SURVEY.md` read in full (PADC, JSCCformer-f, DeepJSCC-MIMO, 2604.10931, 2509.19754), and those readings stand.

The labels below refer to our components:
- **(a)** slot-by-slot multi-user scheduling of a prefix code over a block-fading downlink with a T-slot deadline.
- **(b)** the slot-SNR link abstraction, a sensitivity-weighted additive MMSE rule.
- **(c)** the objectives: mean PSNR at the deadline, and the share of users reaching a target.
- **(d)** the finite-horizon restless bandit: a Lagrangian relaxation into per-user dynamic programs, an upper bound, and the Whittle and Lagrangian indices.
- **(e)** the closed-form advantage index with quotas, and admission plus online stopping for a target.
- **(f)** stale CSI.
- **(g)** the digital baseline (HEVC + CQI link adaptation + HARQ).
- **(h)** delay.
- **(i)** scheduler-aware training.

## 1. Multi-user DeepJSCC and semantic bandwidth, power and scheduling papers: which quality model, static or slot-by-slot, and any deadlines or targets?

### Takeaway
Almost every multi-user semantic or DeepJSCC allocator plugs in a quality-versus-SNR curve measured at constant SNR. The forms are:
- the lookup tables behind Yan et al.'s semantic spectral efficiency (S-SE);
- logistic or generalized-logistic fits of DeepSC similarity;
- logarithmic PSNR fits;
- the "Alpha-Beta-Gamma" MS-SSIM fit;
- learned predictors.

These papers allocate once per transmission, or per slot (by DRL) with transmissions that are complete within that slot. None found decodes one image from symbols sent in slots of different SNR, so none needs an abstraction like (b). Quality thresholds, PSNR requirements and task success under a delay limit are common. A multi-slot deadline for a refinable code is not.

### Cited Findings
**1a. The semantic spectral efficiency (S-SE) family (Yan, Qin, G. Y. Li et al.)**
- **L. Yan, Z. Qin, R. Zhang, Y. Li, G. Y. Li, "Resource Allocation for Text Semantic Communications," arXiv 2201.06023 (Jan 2022, revised Apr 2022)**. The venue, IEEE WCL 2022, is unverified.
  - *Setting / method:* it defines a semantic spectral efficiency and uses it to optimise channel assignment and the number of semantic symbols each user transmits. A transform converts bit-based spectral efficiency into S-SE so the two systems can be compared fairly — [arXiv](https://arxiv.org/abs/2201.06023v2)
  - *Quality model:* the journal follow-up says the S-SE formulation treats the semantic-information parameters as constants for a single task. It can therefore only allocate among users that run the same task — [arXiv 2305.06543](https://arxiv.org/abs/2305.06543)
  - *Overlap:* quality-aware multi-user allocation, related to (c). *Lacks vs ours:* any time dimension, fading within a transmission, a deadline or a bound. Each user–channel pair has one SNR (inference).
- **L. Yan, Z. Qin, R. Zhang, Y. Li, G. Y. Li, "QoE-Aware Resource Allocation for Semantic Communication Networks," arXiv 2205.14530, IEEE GLOBECOM 2022**, extended as "QoE-based Semantic-Aware Resource Allocation for Multi-Task Networks," arXiv 2305.06543 (journal venue unverified).
  - *Quality model:* semantic entropy measures the task-relevant information, and a QoE model is built on it.
  - *Method:* it chooses the number of semantic symbols, channel assignment and power. The problem is split into an exhaustive search over the symbol count and a many-to-one matching game for channel and power — [arXiv 2205.14530](https://arxiv.org/pdf/2205.14530); [arXiv 2305.06543](https://arxiv.org/abs/2305.06543)
  - *Lacks:* this is a static, one-shot allocation, with no horizon or deadline (inference).
- **X. Pu et al., arXiv 2401.01143** (title not captured): maximises S-SE for joint processing across several base stations, with deep RL — [arXiv](https://www.arxiv.org/pdf/2401.01143)

**1b. The logistic-fit family (DeepSC similarity as a function of SNR)**
- **X. Mu, Y. Liu, "Exploiting Semantic Communication for Non-Orthogonal Multiple Access," arXiv 2209.06006**. The venue, IEEE JSAC 2023, is unverified.
  - *Setting:* two-user uplink NOMA. The secondary far user uses semantic communication and the primary near user uses bit communication.
  - *Quality model:* for each K (symbols per word), semantic similarity is fitted by a logistic-type curve with lower and upper asymptotes, a growth rate and a midpoint. An indicator sets the "effective semantic rate" to zero when similarity falls below a minimum threshold.
  - *Results:* semantic communication helps the far user when its transmit power is strictly capped, and can do worse when its power is high. An "opportunistic" scheme chooses semantic or bit communication at each fading state — [arXiv](https://arxiv.org/pdf/2209.06006)
  - *Overlap:* a quality threshold (c) and a decision per fading state. *Lacks:* multi-slot scheduling of several users and a deadline (inference).
- **X. Mu, Y. Liu et al., "Heterogeneous Semantic and Bit Communications: A Semi-NOMA Scheme," arXiv 2205.02620** (venue unverified): semantic similarity is fitted by a generalized logistic function, using data regression — [ar5iv](https://ar5iv.labs.arxiv.org/html/2205.02620)
- **L. Xia, Y. Sun, D. Niyato, X. Li, M. A. Imran, "Joint User Association and Bandwidth Allocation in Semantic Communication Networks," IEEE TVT 73(2):2699-2711, 2024; arXiv 2212.14142; a preliminary version appeared at the INFOCOM 2022 workshops.**
  - It defines a semantic metric, "system throughput in message" (STM). It covers networks with perfect and with imperfect knowledge matching, solves joint user association and bandwidth allocation optimally, and beats two benchmarks — [arXiv](https://arxiv.org/pdf/2212.14142); [Glasgow listing](https://eprints.gla.ac.uk/306689)
  - Whether STM rests on a (generalized) logistic fit could **not** be confirmed (no full text) — [search summary of arXiv 2212.14142](https://arxiv.org/html/2212.14142v5)
- Related: X. He, Y. Sun et al., "Joint User Association and Resource Allocation for Adaptive Semantic Communication in 5G and Beyond Networks" (another version is titled "…for Multi-Cell Networks with Adaptive Semantic Communication"), arXiv 2312.01049 — [arXiv](https://arxiv.org/pdf/2312.01049). Details not verified.

**1c. PSNR and MS-SSIM fits at constant SNR for image DeepJSCC**
- **S. Ma, B. Shen et al., "Modeling and Performance Analysis for Semantic Communications Based on Empirical Results," arXiv 2504.21055 (Apr 2025, preprint).**
  - *Quality model:* an "Alpha-Beta-Gamma" (ABG) formula links an end-to-end metric (MS-SSIM, or inference accuracy) to the SNR. Its parameters are fitted by non-linear least squares. The MS-SSIM upper bound is tied to the number of quantised bits the encoder outputs.
  - *Method:* adaptive power control over fading that guarantees a target QoS; energy-efficiency maximisation by a Dinkelbach-type algorithm; and, for a multi-user OFDMA downlink, max-min MS-SSIM by bisection — [arXiv](https://arxiv.org/abs/2504.21055); [Moonlight review](https://www.themoonlight.io/en/review/modeling-and-performance-analysis-for-semantic-communications-based-on-empirical-results)
  - *Overlap:* a quality target under fading (c) on a multi-user downlink. *Lacks:* the metric depends on one SNR per transmission, which is the abstraction our MMSE rule replaces. There is no slot-by-slot scheduling, horizon or deadline (inference).
- **"Resource Allocation for the Training of Image Semantic Communication Networks," arXiv 2501.04408 (2025)** (authors not captured): PSNR is fitted as a logarithmic function of compression rate and SNR, on CIFAR-10 — [arXiv](https://arxiv.org/pdf/2501.04408)
- **"Resource Allocation for Capacity Optimization in Joint Source-Channel Coding Systems," arXiv 2211.11412** (already in SURVEY F, re-checked): each user has a PSNR requirement, and the paper derives the minimum transmit power for each user–subchannel pair. At a fixed PSNR, the required SNR is roughly inversely proportional to the compression ratio — [arXiv](https://arxiv.org/html/2211.11412)
- **Distributed DeepJSCC over a multiple access channel, arXiv 2211.09920** (SURVEY F, re-checked): maximises PSNR under a power constraint at CR 1/3 and 1/6. NOMA beats TDMA at every SNR tested — [arXiv](https://arxiv.org/pdf/2211.09920)
- **C. Xu, M. B. Mashhadi, Y. Ma, R. Tafazolli, J. Wang, "Generative Semantic Communications with Foundation Models: Perception-Error Analysis and Semantic-Aware Power Allocation," arXiv 2411.04575 (v2, Jan 2025).**
  - It proves that the relation between regenerated quality and transmission reliability is non-decreasing. "Semantic values" measure similarity. Two power-allocation methods follow, tested on Kodak, with up to 90% power saved in the coded case — [arXiv](https://arxiv.org/html/2411.04575v2)
  - An earlier, narrower version covers the uncoded case, solved by constraint decoupling and bisection: arXiv 2407.03050 — [arXiv](https://arxiv.org/html/2407.03050v1)
- **"Semantic Importance-Aware Based for Multi-User Communication Over MIMO Fading Channels," arXiv 2312.16057** (authors not captured).
  - It defines a semantic information distortion (SID) and a semantic outage probability (SOP): the probability that the SID exceeds a constraint. Both are derived for single- and multi-user SISO and MIMO over Rayleigh fading. The single-user SISO SOP is a tail of a generalized noncentral chi-squared variable, approximated by the central limit theorem. It also adds semantic interference cancellation — [arXiv](https://arxiv.org/pdf/2312.16057)
  - *Overlap:* an importance-weighted distortion under fading, with an outage defined by a distortion target, related to (b) and (c). *Lacks:* scheduling, time slots, and weights calibrated per image (inference).

**1d. "Dynamic" allocators: decisions per slot (DRL or Lyapunov), each transmission complete within its slot**
- **H. Zhang, H. Wang, Y. Li, K. Long, A. Nallanathan, "DRL-Driven Dynamic Resource Allocation for Task-Oriented Semantic Communication," IEEE TCOM 71:3992-4004, 2023, DOI 10.1109/TCOMM.2023.3274145.** A DDPG agent at a micro base station sets each user's semantic compression ratio, transmit power and bandwidth jointly. The example task is image classification, and data with more semantic value gets priority — [QMRO](https://qmro.qmul.ac.uk/xmlui/handle/123456789/98084)
- **L. Wang, W. Wu, F. Zhou, Z. Yang, Z. Qin, "Adaptive Resource Allocation for Semantic Communication Networks," arXiv 2312.01081 (Dec 2023)** (journal version unverified).
  - It quantises semantic features into bits ("semantic-bit quantisation") and adds an offset compensator for semantic noise. Its metric is "SC-QoS".
  - It optimises base-station beamforming, the bits per semantic representation, bandwidth and subchannel assignment jointly. SAC and D-SAC make real-time decisions from the sensed tasks and channels — [arXiv](https://arxiv.org/pdf/2312.01081)
- **Y. Wang, M. Chen, T. Luo, W. Saad, D. Niyato, H. V. Poor, S. Cui, "Performance Optimization for Semantic Communications: An Attention-based Reinforcement Learning Approach," IEEE JSAC 40(9), Sept 2022, DOI 10.1109/JSAC.2022.3191112; arXiv 2208.08239.**
  - Text is sent as knowledge-graph triples. The full graph cannot always be sent within the delay limit, so the base station chooses a resource block for each user and how much semantic content (which triples) to send. The goal is total semantic similarity (accuracy plus completeness).
  - The solver is PPO with an attention network that weights triples, and it converges to a local optimum — [arXiv](https://arxiv.org/abs/2208.08239v1); [NSF PAR](https://par.nsf.gov/servlets/purl/10380112)
  - *Overlap:* partial, prefix-like content under a delay limit, with a resource choice per user, related to (a), (c) and the spirit of (e). *Lacks:* refinement over several fading slots, a quality model for mixed-SNR decoding, and a bound (inference).
- **J. Su et al. (with D. Niyato), "Semantic Communication-Based Dynamic Resource Allocation in D2D Vehicular Networks," IEEE TVT 72(8):10784-10796, 2023, DOI 10.1109/TVT.2023.3257770.** A long-term robust allocation weighs user satisfaction, queue stability and delay. A Bernstein approximation handles channel fluctuations. Lyapunov optimisation splits the problem into semantic access control (application layer) and power control (physical layer) — [OpenAlex](https://api.openalex.org/works/doi:10.1109%2FTVT.2023.3257770); [HKU](https://hub.hku.hk/handle/10722/353093)

**1e. Delay constraints and quality targets inside allocation**
- **C. Liu, C. Guo, Y. Yang, N. Jiang, "Adaptable Semantic Compression and Resource Allocation for Task-Oriented Communications," IEEE TCCN (online 28 Dec 2023), DOI 10.1109/TCCN.2023.3346481; arXiv 2204.08910.**
  - Adaptable semantic compression (ASC) compresses semantics according to their importance to the task. More compression needs fewer channel resources but distorts the semantics. Less compression may make the transmission fail under the delay constraint.
  - Its algorithms, CRRA and CRRAUS (which adds user selection), gain at least 15% and 10% in task success — [arXiv](https://www.arxiv.org/abs/2204.08910); [OpenAlex](https://api.openalex.org/works/doi:10.1109%2FTCCN.2023.3346481)
- **L. X. Nguyen, Y. Park, A. D. Raha, H. Q. Le, Z. Han, E.-N. Huh, C. S. Hong, "Energy-Aware Compression-Computation Co-Adaptation for Latency Minimization in Multi-User Semantic Communication" (CoCo), arXiv 2608.13632 (Aug 2026, preprint).**
  - For each user, the base station either sends a longer signal or lets the device denoise a shorter one by diffusion. The choice depends on the user's quality target and energy budget. The code is an adaptive-rate DeepJSCC.
  - The mixed-integer program is split into offline training and an online greedy subcarrier allocation, which is proved globally optimal when denoising time is negligible.
  - Tests are on AWGN: latency falls 8-18% against rate-only adaptation. It trains on DIV2K and reports MS-SSIM at 1-5 dB and CR 1/48-1/12 — [arXiv](https://arxiv.org/abs/2608.13632); [Pith](https://pith.science/paper/2608.13632)
- "Bandwidth and Power Allocation for Task-Oriented Semantic Communication," arXiv 2201.10795: the paper exists ([arXiv](https://arxiv.org/pdf/2201.10795)), but its details were not verified.

### Inferences
- **Quality-model taxonomy.** Only our rule scores one image whose phases saw different SNRs:

  | family | examples above | what it maps | can it score one image whose phases saw different SNRs? |
  |---|---|---|---|
  | lookup table / S-SE | Yan et al. 2022 | (symbols, SNR) → similarity | no: one SNR per user–channel pair |
  | logistic / generalized logistic | Mu & Liu; semi-NOMA | SNR → similarity, per K | no |
  | logarithmic / ABG fits | 2501.04408; Ma et al. 2504.21055 | (CR, SNR) → PSNR / MS-SSIM | no |
  | learned predictor | PADC OraNet, 2604.10931 (SURVEY A, F) | (image, SNR, rate) → PSNR | no: one SNR per image |
  | outage of an importance-weighted distortion | 2312.16057 | fading realisation → P(SID > target) | per realisation, with fixed importance; closest in form (not verified in depth) |
  | ours (b) | — | per-phase slot noise → distortion, with weights per image | yes |

- A reviewer can therefore map all of these to our "mean-dB" baseline: the curve evaluated at one SNR. Our evidence that this fails for codes split across slots is what sets (b) apart (SURVEY F4 says the same).
- The DRL allocators (Zhang TCOM 2023; Wang 2312.01081; Pu 2401.01143) decide every time step. From their abstracts, though, each user's transmission is complete within the step. No state carries over, such as "phases delivered so far and the noise they met". They are not restless bandits over a refinement process.
- Quality targets are standard: a threshold (Mu & Liu), a PSNR requirement (2211.11412), a QoS target (Ma), per-user targets (CoCo), task success under a delay limit (Liu TCCN). So our objective (c) is not new in form, only in combination.
- Delay limits appear (Wang JSAC 2022; Liu TCCN 2023), but as latency constraints on each transmission, not as a T-slot horizon over which a code is refined.

### Gaps
- No full text could be read, so these remain unconfirmed: whether Xia et al.'s STM uses a logistic fit; whether Wang et al. (JSAC 2022) decide per slot or per frame; Zhang et al.'s classification-accuracy model; and the SC-QoS formula in Wang 2312.01081.
- No multi-user allocator built on NTSCC, SwinJSCC rate modulation or ADJSCC was found. The search budget ran out before targeted queries.
- These venues remain unverified: Mu & Liu, semi-NOMA, Yan 2201.06023, Yan 2305.06543, Wang 2312.01081. Authors were not captured for 2501.04408 and 2312.16057.

## 2. Scheduling semantic users over time-varying channels: DRL, Lyapunov, age of semantic information, semantic HARQ, and importance-aware scheduling in edge learning and inference

### Takeaway
The closest analog of our advantage index comes from Kaibin Huang's group:
- a scheduler whose metric couples each device's SNR with its data's uncertainty;
- an "importance ARQ" with a threshold on SNR against uncertainty;
- progressive feature transmission with a threshold stopping rule.

Elsewhere, semantic scheduling uses one of three things: DQN/DDPG/SAC or LLM agents on semantic rewards, Lyapunov drift-plus-penalty for long-term constraints, or freshness metrics of the age-of-information kind (AoII, AoSI). Simple channel-first or meaning-first rules also appear. None found schedules a refinable image code across fading slots to a deadline.

### Cited Findings
**2a. Importance-aware scheduling and retransmission (Huang group): the analog of "content value × channel" in our index**
- **D. Liu, G. Zhu, J. Zhang, K. Huang, "Data-Importance Aware User Scheduling for Communication-Efficient Edge Machine Learning," IEEE TCCN 7(1):265-278, 2021, DOI 10.1109/TCCN.2020.2999606; arXiv 1910.02214; an earlier version appeared at an IEEE ICC 2020 workshop.**
  - The scheduler weighs how informative each device's data is, not just its channel. Its "data importance indicator" combines the channel SNR with the data's uncertainty, so it exploits both multi-user channel diversity and data diversity.
  - Uncertainty is the distance to the decision boundary for SVMs, and the entropy for CNNs. Models converge faster than under schedulers that exploit only one kind of diversity — [arXiv](https://arxiv.org/pdf/1910.02214); [Glasgow](https://eprints.gla.ac.uk/320738)
- **D. Liu, G. Zhu, Q. Zeng, J. Zhang, K. Huang, "Wireless Data Acquisition for Edge Learning: Data-Importance Aware Retransmission," IEEE TWC 20(1):406-420, 2021; arXiv 1812.02030.**
  - "Importance ARQ" retransmits a sample selectively, according to its uncertainty under the model being trained. A relation between SNR and uncertainty gives a simple threshold policy. It converges faster than channel-aware ARQ, with larger gains on imbalanced data.
  - The authors suggest applying the principle to scheduling and power control — [arXiv](https://arxiv.org/pdf/1812.02030); [Glasgow](https://eprints.gla.ac.uk/320649)
  - The DOI 10.1109/TWC.2020.3024980 is from memory (unverified).
- **"An Overview of Data-Importance Aware Radio Resource Management for Edge Machine Learning," arXiv 1911.03878** (same group): covers extensions such as scheduling on both channel and uncertainty — [arXiv](https://arxiv.org/pdf/1911.03878)
- **Q. Lan, Q. Zeng, P. Popovski, D. Gündüz, K. Huang, "Progressive Feature Transmission for Split Classification at the Wireless Edge," IEEE TWC (search metadata gives 2022; issue and DOI unverified); arXiv 2112.07244, titled "…for Split Inference…"; a SPAWC 2022 version exists.**
  - The protocol, ProgressFTX, sends features progressively until a target inference confidence is reached. The server selects features by importance and controls when transmission stops.
  - Stopping follows a threshold rule: stop once another feature's extra uncertainty reduction no longer exceeds its communication cost. The stopping analysis is for Gaussian channels (the same gain every slot), and fading channels are also considered.
  - Latency drops substantially against feature pruning and random transmission — [arXiv](https://arxiv.org/pdf/2112.07244); [Imperial](https://spiral.imperial.ac.uk/entities/publication/ed038593-31d8-4942-b8cf-ee15080d60e0); [SPAWC'22](https://t4awstest.cc.ic.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/LZPGH_SPAWC22.pdf)
  - The sources conflict on optimality. The arXiv and journal versions call the feature selection optimal, the journal abstract calls it sub-optimal, and the SPAWC version calls the CNN policy sub-optimal — [search summary](https://arxiv.org/pdf/2112.07244)

**2b. DRL and LLM schedulers of semantic users**
- **"Context-Aware Semantic Communication for the Wireless Networks," arXiv 2505.23249 (May 2025, preprint; authors not captured).** Users have Rayleigh-fading SNRs. An LLM-assisted scheduler with a DRL fallback is compared with DQN, greedy and random selection. The reward combines ImageReward and SSIM, and the LLM variant converges fastest — [arXiv](https://arxiv.org/pdf/2505.23249)
- **L. Chen, J. Gong, "Multi-source Scheduling and Resource Allocation for Age-of-Semantic-Importance Optimization in Status Update Systems," arXiv 2403.07386.** It defines the "age of semantic importance" (AoSI), which combines freshness with semantic importance. Average AoSI is minimised over multi-source scheduling and resource allocation by a DQN-based algorithm — [arXiv](https://arxiv.org/pdf/2403.07386)
- **"Semantic-aware Transmission Scheduling: a Monotonicity-driven Deep Reinforcement Learning Approach," arXiv 2305.13706** (remote estimation; authors and venue not verified). MSE falls about 30% against conventional DRL. Time-correlated fading is left as future work — [arXiv](https://arxiv.org/pdf/2305.13706)
- Also: "Semantic-Aware Spectrum Sharing in Internet of Vehicles Based on Deep Reinforcement Learning," arXiv 2406.07213 — [arXiv](https://arxiv.org/html/2406.07213v1) (details not verified). The DRL scheduling papers for IRS-assisted semantic communication (2504.07498, 2603.21097) are in SURVEY F4 and were not re-checked.

**2c. Lyapunov and drift-plus-penalty semantic control**
- Su et al., IEEE TVT 2023 (section 1d): Lyapunov optimisation splits the problem into semantic access control and power control, under user satisfaction, queue stability and delay — [OpenAlex](https://api.openalex.org/works/doi:10.1109%2FTVT.2023.3257770)
- **"Semantic-Aware Remote Estimation of Multiple Markov Sources Under Constraints," arXiv 2403.16855:** an online, low-complexity drift-plus-penalty algorithm replaces the high-dimensional MDP and meets a transmission-frequency constraint — [arXiv](https://arxiv.org/html/2403.16855v2)
- **"Dynamic Relative Representations for Goal-Oriented Semantic Communications," arXiv 2403.16986:** stochastic Lyapunov optimisation, with virtual queues enforcing average constraints — [arXiv](https://arxiv.org/pdf/2403.16986)

**2d. Timeliness metrics: AoII, AoSI and the age of semantic information**
- The age of incorrect information (AoII) was introduced by Maatouk et al. (IEEE/ACM ToN 2020), and a 2023 IEEE TWC follow-up frames it as an enabler of semantics-empowered communication. Both are cited in the reference list of arXiv 2609.07563 — [arXiv 2609.07563](https://arxiv.org/pdf/2609.07563)
- A 2026 survey describes AoSI as extending AoI with information significance and task effectiveness, in reconstruction-oriented and task-oriented variants — [arXiv 2603.24328](https://arxiv.org/pdf/2603.24328)
- **"Rate-Splitting Multiple Access for Semantic-Aware Networks: an Age of Incorrect Information Perspective," arXiv 2312.07159:** a multi-user downlink with joint user scheduling, precoding and power allocation. Rate splitting gives lower AoII than SDMA — [arXiv](https://arxiv.org/pdf/2312.07159)
- "Age of Semantic Information-Aware Wireless Transmission for Remote Monitoring Systems," arXiv 2508.12248 — [arXiv](https://arxiv.org/html/2508.12248v2) (details not verified).

**2e. Heuristic schedulers that weigh channel or meaning**
- **"Channel-Aware Preemptive Scheduling for Semantic Communication with Truncated Diffusion and Path Compensation," arXiv 2604.11849 (Apr 2026, preprint).** Each user keeps a countdown inversely proportional to its instantaneous channel gain. The user with the shortest countdown transmits at once, even if its diffusion process has not finished, and the receiver compensates for the semantic loss — [arXiv](https://arxiv.org/html/2604.11849)
- **"Semantic security-aware multi-user resource allocation: A novel meaning-first scheduling," ScienceDirect (PII S1874490725003313, 2025; journal not verified).** In image-classification experiments, a semantic-importance-first policy beats channel-state-first scheduling at comparable delay — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1874490725003313)

**2f. Semantic HARQ (retransmission control on one link)**
- **SemHARQ, arXiv 2404.08490:** a feature-importance ranking protects the important features, and a distortion-evaluation network detects corrupted ones. Those are retransmitted, and leftover resources carry incremental data. Rank-1 vehicle re-identification improves by more than 20% at low SNR — [arXiv](https://arxiv.org/pdf/2404.08490)
- **"Semantic HARQ for Intelligent Transportation Systems: Joint Source-Channel Coding-Powered Reliable Retransmissions," arXiv 2504.14615:** three schemes (Type-I, weighted combining, synonymous combining). Synonymous combining adds incremental semantic redundancy and performs best, at higher complexity — [arXiv](https://arxiv.org/pdf/2504.14615)
- **IK-HARQ, "Adaptive Bit Rate Control in Semantic Communication with Incremental Knowledge-based HARQ," arXiv 2203.06634:** one JSCC encoder, and a decoder that uses incremental knowledge from failed transmissions. It adds adaptive rate control and a self-attention denoiser that needs no SNR — [ar5iv](https://ar5iv.labs.arxiv.org/html/2203.06634)
- **SCHARQ**, as cited in a 2024 survey: adapts incremental-redundancy HARQ to a DeepJSCC architecture, with lower word error rate than a Reed-Solomon-coded HARQ baseline (SC-RS-HARQ) at low SNR — [arXiv 2402.14297](https://arxiv.org/pdf/2402.14297)

### Inferences
- **The data-importance indicator (Liu et al., TCCN 2021) is the closest analog to (e).** Both rank users by content value times channel state.
  - Its value is a training sample's learning gain, sent in one shot per slot.
  - Ours is the PSNR gain of the next phase of a refinable code, *relative to that user's gain in a typical slot at its mean SNR*: an opportunistic advantage in quality units, with deadline quotas.
  - That relative form is closer in spirit to proportional-fair scheduling (rate now over average rate) than to anything found in the semantic literature. The proportional-fair link is background knowledge, not verified here.
- **ProgressFTX's stopping rule is the single-link analog of our online stopping for a target.** The rule is "stop when the next feature's uncertainty reduction no longer pays for its cost". Ours stops users who have reached the target, across many users sharing a slot budget with a deadline. Importance ARQ's threshold on SNR against uncertainty is the retransmission analog of spending a good slot on a valuable phase.
- The meaning-first result (importance-first beats channel-first) and the channel-gain countdown (2604.11849) each use one side of what our index combines (content gain and channel advantage).
- **The DRL and LLM schedulers are learned policies.** Our causal upper bound limits what any such policy can add on our problem (SURVEY F4). None found reports a bound.
- **AoII and AoSI value the freshness of status updates in an infinite-horizon, average-cost world.** Our arms refine one fixed image before a deadline (finite horizon), so the state and reward differ even where the machinery (Whittle, Lyapunov) is shared.

### Gaps
- The search budget ran out before a planned query on the Huang group's 2022-2026 follow-ups: multi-device progressive feature transmission, and importance-aware scheduling for edge inference. These may contain a multi-user version of ProgressFTX and should be searched before submission.
- Whether the rewards of the DRL schedulers (2505.23249; Chen & Gong) are measured decoded quality or proxies is not verified.
- No semantic scheduler using stale or outdated CSI was found. The closest are Su et al.'s robust Bernstein constraint and the belief-state AoII Whittle work (section 4). Only abstracts were seen.

## 3. Rate-adaptive DeepJSCC with per-image allocation, used for multi-user allocation; online stopping; per-user quality targets

### Takeaway
Rate adaptation per image, driven by predicted quality, exists: PADC (IEEE TWC 2023) and 2604.10931 (both in SURVEY, read in full). Allocation over time for a learned code exists for a single user: DeepWiVe's RL bandwidth allocation across video frames. Multi-user allocation of an adaptive-rate DeepJSCC with per-user quality targets appears in CoCo (Aug 2026), but over AWGN and as a one-shot subcarrier allocation. Online stopping at a target exists on single links: JSCCformer-f with feedback, and ProgressFTX for features. No paper found combines a prefix code's online stopping with multi-user slot scheduling under a deadline.

### Cited Findings
- **T.-Y. Tung, D. Gündüz, "DeepWiVe: Deep-Learning-Aided Wireless Video Transmission," IEEE JSAC 40(9):2570-2583, 2022, DOI 10.1109/JSAC.2022.3191354; arXiv 2111.13034.**
  - One neural transform does compression, channel coding and modulation. Separate bandwidth-allocation networks per frame allow variable bandwidth, and an RL network divides the limited bandwidth among the frames of a group of pictures (GOP).
  - The learned allocation beats uniform and heuristic allocation. The system avoids the cliff effect, and gains up to 0.0485 MS-SSIM over H.264 + LDPC — [UNIMORE](https://iris.unimore.it/handle/11380/1286021); [arXiv](https://arxiv.org/abs/2111.13034v1)
  - *Overlap:* allocation over time for a learned code; RL against uniform allocation. *Lacks:* multiple users, a deadline across users, and a fading-dependent quality model (inference).
- **CoCo (arXiv 2608.13632; section 1e):** an adaptive-rate DeepJSCC per user, per-user quality targets, and an online greedy subcarrier allocation that is optimal when denoising time is negligible. AWGN only — [arXiv](https://arxiv.org/abs/2608.13632)
- **PADC; JSCCformer-f; "reliable online resource allocation" (2604.10931); timeliness-aware JSCC (2509.19754)**: see SURVEY A and F, read in full there and not re-read here. PADC gives each image the least rate meeting a PSNR floor, using its OraNet predictor. JSCCformer-f stops at a target PSNR using feedback, on one link. 2604.10931 predicts PSNR with an oracle network and chooses compression ratios by constrained Bayesian optimisation. 2509.19754 chooses the code length with a PPO policy.
- **DeepJSCC-f (Kurka and Gündüz)**: the paper exists — [Imperial PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/KurkaGunduz_deepJSCC-f.pdf). From memory (unverified): IEEE JSAIT 2020, with channel-output feedback and variable-length transmission that stops when the target quality is reached.
- **ProgressFTX** (section 2a): progressive transmission until a target confidence, with a threshold stopping rule — [arXiv](https://arxiv.org/pdf/2112.07244)
- **DRJSCC, arXiv 2311.15309:** splits the channel symbols into blocks sent progressively, and re-encodes the remaining blocks when the channel changes. It matches existing methods on stable channels and is more robust on time-varying ones — [arXiv](https://arxiv.org/pdf/2311.15309)
- **"Scalable Deep Joint Source-Channel Coding for Multi-User Wireless Image Transmission With Diverse Bandwidth Conditions" (2025).** Layered DeepJSCC for heterogeneous users, per SURVEY A. Multicasting street-scene images to 10 users cuts per-user latency at a fixed power budget, against non-generative multicasting — [ResearchGate](https://www.researchgate.net/publication/397693533_Scalable_Deep_Joint_Source-Channel_Coding_for_Multi-User_Wireless_Image_Transmission_with_Diverse_Bandwidth_Conditions)
- **"When Semantic Communication Meets Queueing: Cross-Layer Latency and Task Fidelity Optimization," arXiv 2605.05514 (May 2026, preprint).** The latent dimension of a learned encoder-decoder becomes a cross-layer control that trades semantic fidelity against channel use. It assumes block Rayleigh fading plus AWGN, with queueing analysis — [arXiv](https://arxiv.org/abs/2605.05514)
- **"Deadline-Aware Bandwidth Allocation for Semantic Generative Communication with Diffusion Models," arXiv 2508.12701 (Aug 2025, preprint).**
  - It defines a "semantic deadline": the latest time conditioning data can be injected and still meet a performance threshold. Bandwidth is allocated so each semantic element arrives in time, the elements travelling over independent channels and arriving asynchronously.
  - The search summary describes it as framed on one link — [arXiv](https://arxiv.org/abs/2508.12701)
- **Shisher, Ji, Hou, Sun (JSAIT 2023; section 4):** chooses feature length jointly with transmission scheduling. This is the remote-inference analog of choosing how many phases to send — [arXiv](https://arxiv.org/pdf/2308.10094)
- **Information-theory anchor:** "Joint Source-Channel Coding with Time-Varying Channel and Side-Information" treats sending a Gaussian source over a delay-limited block-fading channel, with side information about the fading — [Imperial PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/EG_IT15.pdf). The authors (I. Estella Aguerri and D. Gündüz) and the venue (IEEE Trans. IT, 2016) are from memory (unverified).
- Also found, details not verified: "RL-Driven Semantic Compression Model Selection and Resource Allocation in Semantic Communication Systems," arXiv 2506.18660 — [arXiv](https://arxiv.org/html/2506.18660)

### Inferences
- Our "admit the smallest remaining needs, then stop online" applies PADC's per-image rule (the least rate meeting a floor) online and across users under a shared, time-varying budget. Classical shortest-remaining-work scheduling is the obvious cousin. That is background knowledge, not verified here.
- **CoCo is the closest multi-user DeepJSCC allocator with per-user targets**, but its allocation is static per request on AWGN. It needs neither a slot-SNR abstraction nor a deadline horizon. Its "greedy is optimal" result is the static counterpart of our upper bound.
- DeepWiVe and Wang et al. (JSAC 2022) show reviewers already know RL allocation of a learned code's length or partial content. A model-based index with a bound is the contrast to draw.

### Gaps
- No paper was found using NTSCC, SwinJSCC rate modulation or ADJSCC as the code inside a multi-user scheduler. The planned queries were not run because the budget ran out.
- The DeepJSCC-f details come from memory.
- Whether 2508.12701 is multi-user is unclear.

## 4. Restless bandits, Whittle or Lagrangian indices, per-user decomposition and bounds in semantic or task-oriented multi-user scheduling (2021-2026)

### Takeaway
Yes, such work exists, but for remote inference and age-of-information scheduling, not image codes. Yin Sun's group casts task-oriented multi-source scheduling as (multi-action) restless bandits. They solve it with Lagrangian or dual decomposition, Whittle-index source selection and a "net gain" rule that needs no indexability, with asymptotic-optimality proofs:
- Shisher, Ji, Hou and Sun (JSAIT 2023);
- Shisher, Sun and Hou (IEEE/ACM ToN 2024);
- McDowell, Panebianco, Chakraborty and Sun (arXiv, Aug 2026).

AoII scheduling has closed-form and learned Whittle indices. Hosseini and Panwar remain the only restless-bandit precedent found that uses a layered (digital) video code with deadlines. No work found uses a finite-horizon deadline with a decoded-image-quality reward, or reports a Lagrangian upper bound as an optimality certificate for a DeepJSCC scheduler.

### Cited Findings
- **M. K. C. Shisher, B. Ji, I.-H. Hou, Y. Sun, "Learning and Communications Co-Design for Remote Inference Systems: Feature Length Selection and Transmission Scheduling," IEEE JSAIT vol. 4, pp. 524-538, 2023; arXiv 2308.10094.**
  - For one sensor–predictor pair, it gives optimal scheduling policies for fixed and for time-varying feature length. A third-party summary says the optimal policy is a threshold that is non-monotonic and depends on both the age and the feature length.
  - For many pairs, a **Lagrangian decomposition of a relaxed formulation** yields a new algorithm. A secondary summary calls it "net gain maximization" and says it avoids the indexability conditions Whittle policies need. It beats maximum-age-first scheduling — [arXiv](https://arxiv.org/pdf/2308.10094); [Lacuna summary](https://lacuna.tiptreesystems.com/work/learning-and-communications-co-design-for-remote-inference)
- **M. K. C. Shisher, Y. Sun, I.-H. Hou, "Timely Communications for Remote Inference," IEEE/ACM ToN 32(5):3824-3839, 2024, DOI 10.1109/TNET.2024.3408673; arXiv 2404.16281.**
  - Multi-source, multi-channel scheduling is posed as a **multi-action restless bandit**. The policy combines **Whittle-index source selection** with **duality-based** selection of which buffered feature to send, and is proved **asymptotically optimal**.
  - It holds for general age functions, monotonic or not. The authors note that indexability is hard to establish because of the complicated transitions, unreliable channels and general loss functions — [ACM DL](https://dl.acm.org/doi/10.1109/TNET.2024.3408673); [arXiv](https://arxiv.org/pdf/2404.16281)
- The same group's lineage, found but not read: "Goal-Oriented Communications for Remote Inference under Two-Way Delay with Memory," arXiv 2311.11143 ([arXiv](https://arxiv.org/pdf/2311.11143)); "Computation and Communication Co-scheduling for Multi-Task Remote Inference," arXiv 2501.04231 ([arXiv](https://arxiv.org/html/2501.04231v2)); "AoI-based Scheduling of Correlated Sources for Timely Inference," arXiv 2509.01926 ([arXiv](https://arxiv.org/html/2509.01926v1)); "Remote Safety Monitoring: Significance-Aware Status Updating for Situational Awareness," arXiv 2507.09833 ([arXiv](https://arxiv.org/pdf/2507.09833)).
- **C. McDowell, A. Panebianco, S. Chakraborty, Y. Sun, "Significance-Driven Semantic Communication," arXiv 2608.28441 (28 Aug 2026, preprint; SURVEY F2 lists it, re-checked here with more detail).**
  - A per-sample significance metric comes from statistical decision theory, inside a cross-layer problem that maximises "semantic spectrum efficiency".
  - The MAC layer is a **multi-action restless bandit** solved by a "Q-Maximization" algorithm. Whittle's index is not applied directly, because multiple actions break the binary-action and indexability assumptions. Instead they use **Lagrangian relaxation with dual decomposition** and an actor-critic conditioned on λ, trained offline, so the policy adapts to the number of sensors and the bandwidth.
  - The physical layer is "Meta-VIB", a meta-learned hypernetwork transceiver. The data is from traffic cameras — [arXiv](https://arxiv.org/html/2608.28441)
- **Ayik et al., "Optimization of AoII and QAoII in Multi-User Links," IEEE INFOCOM Workshops 2023; arXiv 2305.00191** (full author list not captured). It derives closed-form Whittle indices for a push-based multi-user network over unreliable channels with costs that depend on AoII, and a query-based variant (QAoII) for pull-based systems. The Whittle policies beat the benchmarks — [arXiv](https://arxiv.org/abs/2305.00191v1); [DeepAI](https://deepai.org/publication/optimization-of-aoii-and-qaoii-in-multi-user-links)
- **Kriouile and Assaad, "Minimizing the Age of Incorrect Information for Real-time Tracking of Markov Remote Sources," ISIT 2021; arXiv 2102.03245.** Tracking under AoII is a POMDP. A scheduler based on Whittle's index keeps a belief about each source's state — [ar5iv](https://ar5iv.arxiv.org/html/2102.03245). The search summary contradicts itself on whether the abstract treats one source or several, so check before citing it as multi-source.
- **Jonah et al., "Adaptive Scheduling: A Reinforcement Learning Whittle Index Approach for Wireless Sensor Networks," arXiv 2601.01179 (2026, preprint).** RL learns the Whittle index for polling sensors when only a subset can be polled per step, using AoII as the priority — [arXiv](https://arxiv.org/html/2601.01179v2)
- A 2024 paper on learning Whittle indices frames restless-bandit models in semantic communication as maximising relevant semantic information over channels rather than raw throughput, with users or channels as arms — [arXiv 2409.04605](https://arxiv.org/pdf/2409.04605)
- **S. A. Hosseini, S. S. Panwar, "Restless Video Bandits: Optimal SVC Streaming in a Multi-user Wireless Network," arXiv 1612.03985 (Dec 2016; v2 Aug 2017; SURVEY F2, re-checked).**
  - Scheduling scalable-video users is a restless bandit solved as a linear program. It admits up to twice as many users at a given quality and scales without exceeding single-user complexity.
  - The best policy favours users with better channels and lower buffer occupancy. A "QA-Blind" heuristic needs no higher-layer knowledge and was tested on a testbed — [arXiv](https://www.arxiv.org/abs/1612.03985v2)
  - A precursor is "SVC-based Multi-user Streamloading for Wireless Networks," arXiv 1512.00137 — [arXiv](https://arxiv.org/pdf/1512.00137)
- **General wireless Whittle precedents with closed-form indices or indexability proofs:**
  - age scheduling with stochastic arrivals, arXiv 1801.03422 — [arXiv](https://arxiv.org/pdf/1801.03422)
  - opportunistic scheduling over Markovian channels, IEEE TWC 2019 (a PDF on a personal page at lri.fr; title and authors not captured) — [PDF](https://www.lri.fr/~chen/index_fichiers/publications/twc2019.pdf)
  - remote estimation of Gauss-Markov processes over multiple channels, arXiv 2305.04809 — [arXiv](https://arxiv.org/pdf/2305.04809)
  - multichannel scheduling of queues, arXiv 1902.02277 — [arXiv](https://arxiv.org/pdf/1902.02277)
  - the uncertainty-of-information restless bandit, arXiv 2102.06384 — [arXiv](https://arxiv.org/pdf/2102.06384)
  - restless bandits with individual penalty constraints (possibly relevant to quotas), arXiv 2604.04101 — [arXiv](https://arxiv.org/pdf/2604.04101)

### Inferences
- **Yin Sun's group is the main threat to our method contribution (d).** Reviewers from goal-oriented communication will see "a Lagrangian relaxation into per-user problems, then an index policy with an optimality argument" as their recipe. In particular, the "net gain" rule of Shisher et al. (JSAIT 2023) looks conceptually like our Lagrangian index (the gain at the bound's prices); the description comes from a secondary summary.
- What still separates ours:
  1. a **finite horizon with deadlines**, where their problems are infinite-horizon, average-cost age problems;
  2. **arm states** made of the phases delivered and the per-phase slot noise, with a **decoded-PSNR reward** from a calibrated abstraction per image, where theirs are the age and the inference error;
  3. an **explicit Lagrangian upper bound used as a numerical certificate** (our 0.26 dB gap), where theirs are asymptotic-optimality proofs;
  4. a **closed-form advantage index**;
  5. **stale CSI** through a kernel from report to channel.
- **McDowell et al. (2026) is the learned-policy counterpart:** multiple actions, dual decomposition and a λ-conditioned actor-critic. Ours is model-based with closed-form indices, so the contrast is "index and bound against learned dual policy".
- **Hosseini and Panwar is still the closest *setting* precedent:** a layered code, deadlines, multiple users and a restless bandit solved by LP. Their "users admitted at a quality" is our target objective. Their layers are digital and do not degrade smoothly with SNR.

### Gaps
- I could not confirm from full texts:
  - whether Shisher et al. compute and report the Lagrangian bound numerically;
  - how their unreliable channels are modelled (Bernoulli or SNR-graded);
  - whether McDowell et al.'s channel fades from slot to slot, and whether their decomposition gives an upper bound;
  - whether Hosseini and Panwar ever appeared in a journal (none was found).
- No query was run for classical Whittle-index scheduling of video QoE; the budget was exhausted.

## 5. Cross-layer or MAC-aware semantic communication: designing the learned code and the scheduler together, or training the code on the channel statistics a scheduler produces

### Takeaway
Only McDowell et al. (arXiv, Aug 2026) claim a cross-layer design that jointly optimises physical-layer semantic compression and MAC-layer allocation (a meta-learned transceiver plus a λ-conditioned bandit policy). Shisher et al. (JSAIT 2023) co-design feature length and scheduling, and 2605.05514 treats the latent dimension as a cross-layer control. **No paper found trains a learned code on the SNR distribution its scheduler induces**, which is consistent with SURVEY F3.

### Cited Findings
- **McDowell et al., arXiv 2608.28441:** say they believe theirs is the first cross-layer semantic communication design that jointly optimises physical-layer semantic compression and MAC allocation (the authors' claim). The transceiver adapts to changing channels without online retraining. The MAC is a multi-action restless bandit — [arXiv](https://arxiv.org/html/2608.28441)
- **Shisher, Ji, Hou, Sun, JSAIT 2023:** a "learning and communications co-design framework" that chooses feature length and transmission scheduling jointly — [arXiv](https://arxiv.org/pdf/2308.10094)
- **arXiv 2605.05514:** the latent dimension of a learned encoder-decoder is a cross-layer control variable, with block Rayleigh fading and queueing analysis — [arXiv](https://arxiv.org/abs/2605.05514)
- **Su et al., IEEE TVT 2023:** an application-layer semantic access control and a physical-layer power control are coupled by Lyapunov optimisation — [OpenAlex](https://api.openalex.org/works/doi:10.1109%2FTVT.2023.3257770)
- **SCSC (arXiv 2501.02738):** notes that jointly optimising separate source and channel codes is recognised as useful for cross-layer design, but does not extend this to MAC scheduling — [arXiv](https://arxiv.org/html/2501.02738v1)
- One targeted search for a learned codec trained jointly with a MAC scheduler, end to end on channel statistics, found no match. The search summary called the combination "an open niche" — [search results incl. arXiv 2608.28441](https://arxiv.org/html/2608.28441)

### Inferences
- Our scheduler-aware training is a different kind of co-design. The scheduler shapes the SNR law that each phase meets, and the code is then fine-tuned to that law (specialisation).
- McDowell et al.'s meta-learned transceiver goes the other way: it is built to work across channel conditions without retraining (robustness).
- Shisher's co-design picks a feature length from a fixed menu of predictors, and does not retrain the predictor on the age distribution the scheduler induces (inferred from the abstract).
- So claim (i) can stand, provided McDowell et al. and Shisher et al. are cited and the distinction is made explicit.

### Gaps
- Whether McDowell et al. train Meta-VIB on channel conditions produced by their MAC policy could not be checked without the full text. That is the one detail that could weaken claim (i).
- Classical cross-layer joint source-channel coding and scheduling for video was not searched; the budget was exhausted.

## 6. Synthesis: the five closest works, and an overlap verdict for each of our contributions

### Takeaway
The closest works split by what they share with us:
- **Method (d, e):** Yin Sun's restless-bandit work on remote inference and significance.
- **Setting (a, c):** Hosseini and Panwar's layered-video restless bandit.
- **Index and stopping (e):** Kaibin Huang's importance-aware scheduling and progressive transmission.
- **Quality model and targets (b, c):** the constant-SNR metric models used for multi-user DeepJSCC allocation, exemplified by Ma et al.'s ABG fit.

No single work combines a learned code refined across fading slots, a deadline, a slot-SNR distortion abstraction and a restless-bandit index with an upper bound.

### Cited Findings
The five closest works:
1. **Shisher, Sun, Hou, "Timely Communications for Remote Inference," IEEE/ACM ToN 32(5):3824-3839, 2024, DOI 10.1109/TNET.2024.3408673**, with its companion **Shisher, Ji, Hou, Sun, JSAIT vol. 4, pp. 524-538, 2023 (arXiv 2308.10094)** — [ACM DL](https://dl.acm.org/doi/10.1109/TNET.2024.3408673); [arXiv](https://arxiv.org/pdf/2308.10094)
   - *Overlap:* (d) a multi-action restless bandit, Whittle plus duality, a Lagrangian decomposition of a relaxed problem, a "net gain" index without indexability, asymptotic optimality; (i) a co-design of feature length and scheduling.
   - *Lacks:* image codes, PSNR rewards, finite-horizon deadlines, slot-SNR-dependent rewards and a numerical bound, as far as the abstracts show.
2. **Hosseini, Panwar, "Restless Video Bandits," arXiv 1612.03985 (2016/2017)** — [arXiv](https://www.arxiv.org/abs/1612.03985v2)
   - *Overlap:* (a) layered video, multiple users, deadlines through buffers, and a restless bandit solved by LP; (c) users admitted at a quality; a policy favouring good channels and low buffers.
   - *Lacks:* a learned, smoothly degrading code, a quality abstraction per image, stale CSI and scheduler-aware training.
3. **McDowell, Panebianco, Chakraborty, Sun, "Significance-Driven Semantic Communication," arXiv 2608.28441 (Aug 2026, preprint)** — [arXiv](https://arxiv.org/html/2608.28441)
   - *Overlap:* (d) a multi-action restless bandit with Lagrangian dual decomposition; (i) a cross-layer co-design of the PHY and the MAC.
   - *Lacks:* image reconstruction quality, deadlines, a slot-SNR abstraction and closed-form indices; its policy is a learned actor-critic.
4. **Liu, Zhu, Zhang, Huang, "Data-Importance Aware User Scheduling…," IEEE TCCN 7(1):265-278, 2021**, with **Liu et al., importance ARQ, IEEE TWC 20(1):406-420, 2021** and **Lan et al., ProgressFTX, IEEE TWC** — [arXiv 1910.02214](https://arxiv.org/pdf/1910.02214); [arXiv 1812.02030](https://arxiv.org/pdf/1812.02030); [arXiv 2112.07244](https://arxiv.org/pdf/2112.07244)
   - *Overlap:* (e) indices combining content value and channel, thresholds on SNR against uncertainty, and threshold stopping at a target confidence over fading.
   - *Lacks:* a deadline across users, PSNR as the value, a restless-bandit bound and a prefix code split across slots.
5. **Ma, Shen et al., "Modeling and Performance Analysis for Semantic Communications Based on Empirical Results," arXiv 2504.21055 (Apr 2025, preprint)** — [arXiv](https://arxiv.org/abs/2504.21055)
   - *Overlap:* (b, c) a calibrated end-to-end metric model, power control with a QoS target over fading, and multi-user max-min MS-SSIM on an OFDMA downlink.
   - *Lacks:* scoring a code spread over slots of different SNR, slot-by-slot scheduling and a deadline.

Runners-up, from the sections above:
- **CoCo**, arXiv 2608.13632: per-user targets, adaptive-rate DeepJSCC, greedy allocation optimal on AWGN.
- **Wang et al.**, IEEE JSAC 2022: partial content under a delay limit, resource blocks per user, PPO.
- **Liu, Guo et al.**, IEEE TCCN 2023: task success under delay, with user selection.
- **Context-aware semantic communication**, arXiv 2505.23249: a DRL/LLM scheduler over Rayleigh fading with an image-quality reward.
- **PADC and JSCCformer-f**: SURVEY A, read in full there.

### Inferences
Overlap verdict per contribution (my judgment from the abstract-level evidence above):

| our contribution | overlap | closest prior work | what remains ours |
|---|---|---|---|
| (a) slot-by-slot multi-user scheduling of a learned prefix code over block fading, with a T-slot deadline | LOW-MED | Hosseini & Panwar (digital SVC); Wang et al. JSAC 2022 (delay-limited partial content); DRL semantic schedulers over fading (2505.23249; Zhang TCOM 2023) | a learned code refined across fading slots to a deadline, for many users; none found |
| (b) slot-SNR link abstraction (sensitivity-weighted additive MMSE; better than MIESM/EESM) | LOW | constant-SNR curves (S-SE, logistic, logarithmic, ABG, OraNet); importance-weighted distortion outage under fading (2312.16057); linear analog SoftCast/ParCast (SURVEY F4) | calibrated, per image, for a *nonlinear learned* code spanning slots of different SNR; no semantic paper found evaluates mixed-SNR decoding |
| (c) mean PSNR at the deadline; share of users reaching a target | HIGH (in form) | thresholds and targets in Mu & Liu, 2211.11412, Ma et al., CoCo, Liu TCCN, ProgressFTX; "users admitted at a quality" (Hosseini & Panwar) | only the combination with (a) and (d) |
| (d) finite-horizon restless bandit, Lagrangian per-user DPs, upper bound, Whittle and Lagrangian indices | MED-HIGH (method) | Shisher et al. ToN 2024 and JSAIT 2023; McDowell et al. 2026; AoII Whittle (Ayik et al.; Kriouile & Assaad); Hosseini & Panwar (LP) | finite horizon with deadlines; arm = phases × slot noise with a decoded-PSNR reward; the bound used as a numerical certificate for DeepJSCC scheduling; position against Yin Sun's line explicitly |
| (e) advantage index (gain now minus gain at a typical slot at mean SNR) with quotas; admission plus online stopping | MED | data-importance indicator (SNR × uncertainty); importance-ARQ threshold; ProgressFTX stopping; Shisher's "net gain"; meaning-first scheduling; the channel-gain countdown (2604.11849) | the relative, PSNR-unit opportunistic form, quotas against a deadline, and multi-user admission by smallest remaining need |
| (f) stale CSI | LOW | robust Bernstein constraint (Su et al. TVT 2023); belief-state AoII Whittle (Kriouile & Assaad) | a kernel from report to channel inside the per-user DP and the bound |
| (g) digital comparison with CQI link adaptation and HARQ | LOW | semantic-vs-bit comparisons (Mu & Liu; Yan's S-SE transform; DeepWiVe against H.264/H.265 + LDPC) | a closed link-adaptation and HARQ loop under multi-user scheduling; none found |
| (h) delay | MED | latency, queueing and age metrics (CoCo; 2604.11849; 2605.05514; Su et al.; Chen & Gong AoSI; 2508.12701) | delay of deadline-scheduled DeepJSCC images against digital |
| (i) scheduler-aware training | LOW-MED | McDowell et al. (cross-layer, meta-learned, built for robustness); Shisher et al. (feature length co-design); 2605.05514 | training the code on the SNR law the scheduler induces; none found |

- Net assessment: the **setting (a) and abstraction (b) look new**, and so does the scheduler-aware training (i), modulo McDowell et al.
- The **method (d) is an application of a recipe** that Yin Sun's group already uses in goal-oriented communication; it must be cited and contrasted on finite horizon, decoded quality and the numerical bound.
- The **objective forms (c) are standard.**

### Gaps
- None of the five closest works was read in full; the blocked access is explained at the top. Before submission, read these in full:
  - Shisher, Sun, Hou (ToN 2024), and Shisher, Ji, Hou, Sun (JSAIT 2023): the bound and the "net gain" rule;
  - McDowell et al. (2026): the training data of its transceiver, and whether its decomposition yields a bound;
  - Lan et al. (ProgressFTX): whether a multi-device version exists;
  - Ma et al. (2504.21055): how it models fading.
- Not searched because the shared budget ran out:
  - the Huang group's multi-device importance-aware scheduling for inference, 2022-2026;
  - surveys of semantic resource allocation, for citation trails;
  - NTSCC, SwinJSCC and ADJSCC inside multi-user schedulers;
  - proportional-fair or opportunistic semantic scheduling;
  - Whittle scheduling of video QoE.

  Found but not opened, for citation trails: arXiv 2212.08487 (tutorial-survey), 2506.11779 (6G coexistence and multiple access), 2411.18199 (semantic edge computing survey).
- These SURVEY entries were not re-checked here: the IRS DRL papers (2504.07498, 2603.21097), the IoV Lyapunov paper (PMC13165843), PADC, JSCCformer-f, DeepJSCC-MIMO, 2604.10931 and 2509.19754.
