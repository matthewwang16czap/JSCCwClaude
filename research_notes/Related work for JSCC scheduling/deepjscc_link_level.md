# DeepJSCC link-level properties for system design: link abstraction, mixed-SNR training, progressive/HARQ transmission, analog vs digital under imperfect CSI, channel-distribution training

Scope: 2018-2026 DeepJSCC / semantic link-level work that overlaps or bears on our contributions (a prefix-decodable ViT JSCC sent as up to 6 successive-refinement phases; per-phase-SNR "piecewise" training; an additive sensitivity-weighted MMSE link abstraction compared with MIESM/EESM; scheduling over a fading downlink; JSCC vs HEVC + LTE CQI link adaptation + IR-HARQ under CQI errors and outdated CSI; training on the scheduler-induced per-phase SNR distribution). It builds on `jscc/docs/SURVEY.md` (sections A, F3, F4). Items already listed there are tagged [K] and only re-described where they matter here.

**Verification legend. Read this before citing anything below.** This session could not open full texts. The egress proxy refused arxiv.org and the Crossref, OpenAlex and Semantic Scholar APIs (HTTP 403). WebFetch failed DNS on every host tried (arxiv.org, imperial.ac.uk, link.springer.com). The shared per-turn web-search budget also ran out. So everything below comes from search-engine extracts of the primary pages (arXiv abstract, HTML or PDF; IEEE Xplore; Springer; ResearchGate listings), plus SURVEY.md's earlier full readings. Tags:
- **[A]**: content from the primary page's abstract or text, as the search surfaced it.
- **[M]**: bibliographic metadata seen on a listing page.
- **[K]**: already in SURVEY.md.
- **[?]**: uncertain, or recalled from memory. Verify before citing.

Numbers are as the sources report them. None were checked against full texts.

## Q1. Link abstraction for learned codes: has anyone mapped multi-SNR transmissions of a learned code to its quality, tested EESM/MIESM on learned codes, or defined a semantic link-to-system interface?

### Takeaway
No work found maps a learned code sent over several SNRs (per symbol, subcarrier, block or phase) to its decoded quality with a validated composition rule. No work found tests how accurate EESM or MIESM is against a learned code's distortion. What does exist:
1. Single-SNR "semantic performance functions" fitted to SNR sweeps, then used for power control or allocation.
2. Two 2026 preprints that simply adopt digital EESM as the link-to-system (L2S) model for semantic OFDM links.
3. Neural predictors of per-image quality from SNR or CSI.
4. The linear-analog precedent (SoftCast/ParCast), where additive per-component MMSE holds by construction.

Semantic system-level simulators found so far leave out PHY modelling entirely.

### Cited Findings
- **[A] Ma, Shen, Zhang, Wu, Li, Li, Shi, Al-Dhahir, "Modeling and Performance Analysis for Semantic Communications Based on Empirical Results," arXiv:2504.21055 (Apr 2025; venue not verified).**
  - Setting: proposes an "Alpha-Beta-Gamma" (ABG) formula linking an end-to-end metric to SNR, for image reconstruction (MS-SSIM) and for inference tasks.
  - Finding: the formula fits common encoders, including SCUNet and a Vision Transformer. The MS-SSIM ceiling depends on the number of quantized encoder output bits, with a closed-form fit for that dependence. The model is then used to study adaptive power control over random fading channels.
  - Form: a third-party review gives φ(ρ, n_b) = α(n_b) − γ/(1+(βρ)^τ). This is unverified against the paper.
  - Overlap: a fitted link-performance model for a learned (ViT) codec, used for system-level design over fading.
  - Lacks: one SNR per transmission (no rule for a codeword spanning several SNRs), no ESM comparison, quantized digital encoders, no scheduling.
  - Sources: [arXiv abs](https://arxiv.org/abs/2504.21055); [third-party review, formula](https://www.themoonlight.io/en/review/modeling-and-performance-analysis-for-semantic-communications-based-on-empirical-results)
- **[A] P. Jiang et al., "Intention-Aware Semantic Agent Communications for AI Glasses," arXiv:2604.23691 (Apr 2026).**
  - Setting: Section III introduces "an OFDM transmission model and an effective SNR abstraction that links physical-layer channel conditions to semantic transmission reliability."
  - Mechanism: the semantic encoder's N_sym complex symbols are mapped onto K active subcarriers. EESM then maps the per-subcarrier SNRs to the AWGN SNR that "yields the same block error rate (BLER) as the actual fading channel," with a parameter β that depends on modulation and coding.
  - Other reported results: more than 50% bandwidth reduction at maintained task performance, and graceful degradation at low SNR.
  - Overlap: the only work found that uses an effective-SNR mapping as the L2S for a semantic link.
  - Lacks, as far as the extract shows: any test of whether EESM predicts the learned code's distortion. The abstraction is borrowed from BLER-based digital practice [?]. No composition across time slots, no training dependence.
  - Sources: [arXiv HTML](https://arxiv.org/html/2604.23691v1); [arXiv PDF](https://arxiv.org/pdf/2604.23691)
- **[A] "AgentComm: Semantic Communication for Embodied Agents," arXiv:2604.13558 (2026; authors not verified).** Also defines the effective SNR (ESNR) over K subcarriers with non-uniform SNRs, as the equivalent AWGN SNR that achieves the same BLER as the fading channel. This is a second 2026 instance of digital EESM adopted for semantic links. — [arXiv HTML](https://arxiv.org/html/2604.13558v1)
- **[A] "Adaptive CSI Feedback for Deep Learning-Enabled Image Transmission," arXiv:2302.13477 (2023; authors and venue not verified [?]).**
  - Setting: a lightweight CNN "performance evaluator" predicts each image's D-JSCC PSNR, trained by regression on actual PSNR labels. The prediction decides how much CSI feedback to send.
  - Finding: adaptive feedback beats a fixed feedback budget for all images, and needs less feedback for the same share of transmissions above a PSNR threshold. The results come via a third-party summary.
  - Overlap: a learned quality predictor conditioned on channel state.
  - Lacks: a black-box predictor for one transmission, not a composition rule across slots; no ESM comparison; single link.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2302.13477); [Lacuna summary](https://lacuna.tiptreesystems.com/work/adaptive-csi-feedback-for-deep-learning-enabled-image-transmission/wrk_242650a3a943d368f9a799cd1ff2b4fb)
- **[A][M?] G. Zhang, Q. Hu, Y. Cai, G. Yu, "SCAN: Semantic Communication with Adaptive Channel Feedback," arXiv:2306.15534; cited as IEEE Trans. Cogn. Commun. Netw. (year not verified [?]).** A lightweight evaluator at the transmitter estimates reconstruction quality, distilled from DeepSC-MIMO. It is used to adapt feedback. Overlap and gaps are as for the entry above. — [arXiv abs](https://arxiv.org/abs/2306.15534)
- **[K] Other quality predictors, all one SNR per transmission:**
  - PADC's OraNet (IEEE TWC 2023, DOI 10.1109/TWC.2023.3234408): an MLP on code statistics, SNR and rate predicts PSNR, with 0.07-0.42 dB MAE. — [DOI](https://doi.org/10.1109/TWC.2023.3234408)
  - Oracle PSNR network + constrained Bayesian optimisation for multi-user compression ratios, arXiv:2604.10931. — [arXiv](https://arxiv.org/abs/2604.10931)
  - Yan, Qin et al., text semantic resource allocation from a fitted similarity-vs-SNR curve, arXiv:2201.06023. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2201.06023)
- **[K] SoftCast / ParCast (linear analog video).** Distortion is a sum of per-component LLSE errors by construction, and ParCast matches high-energy components to high-gain subchannels. This is the precedent for the form of our rule. — [MSR ParCast page](https://www.microsoft.com/en-us/research/?p=168037)
- **[A] Classical L2S, for contrast (digital BLER, not learned codes):**
  - The 5G NR PHY abstraction for system-level simulation (arXiv:2001.10309) uses EESM, covering multiple MCS tables up to 256-QAM. — [arXiv abs](https://arxiv.org/abs/2001.10309)
  - A DNN-based L2S for 5G/IoT (Sensors 2019, PubMed 30857237) trains a regressor on LTE data around EESM, for better BLER prediction and less processing time. — [PubMed](https://pubmed.ncbi.nlm.nih.gov/30857237/)
  - The standard structure is a link-quality model (per-subcarrier SINR) plus a link-performance model (SINR-to-PER lookup tables). — [MathWorks](https://www.mathworks.com/help/wlan/ug/physical-layer-abstraction-for-system-level-simulation.html)
  - BLER-vs-SNR curves are "the important L2S interfaces" in C-V2X simulators. — [arXiv:1907.04693](https://arxiv.org/pdf/1907.04693)
- **[A] Semantic system-level simulation:**
  - An ns-3 + Python "knowledge plane" simulator for semantic-aware 6G management (arXiv:2603.12695) measures semantic fidelity and routing but deliberately skips detailed PHY modelling. — [arXiv HTML](https://arxiv.org/html/2603.12695)
  - A 6G standardisation roadmap for semantic communication (arXiv:2509.12758) calls for "scalable testbeds and simulation platforms" that emulate noise, mobility and interference while keeping semantic fidelity end to end. It proposes no interface. — [arXiv PDF](https://arxiv.org/pdf/2509.12758)
  - Seq2Seq-SC (arXiv:2210.15237) argues that a link-level simulation is required to verify realistic end-to-end performance of semantic systems. — [arXiv PDF](https://arxiv.org/pdf/2210.15237)
- **[A] SGD-JSCC, "Semantics-Guided Diffusion for Deep JSCC in Wireless Image Transmission," arXiv:2501.01138.** Treats a slow-fading block as AWGN with effective SNR |h|²/σ², estimated from the received signal rather than from pilots. This is the trivial single-SNR "abstraction" used by most DeepJSCC fading papers. — [arXiv HTML](https://arxiv.org/html/2501.01138v1)

### Inferences
- The default practice now emerging for semantic links is to borrow digital EESM (2604.23691, 2604.13558) or to fit a single-SNR curve (ABG; Yan et al.; OraNet). Our finding answers exactly this practice: MIESM is about 9x less accurate, EESM is worse, and a learned code obeys an additive sensitivity-weighted MMSE rule only when trained for mixed SNR. These 2026 preprints should be cited as the motivating counter-practice, not as prior art for the rule.
- A single-SNR curve is the "mean-dB" abstraction. By construction it cannot tell which phase got the bad slot, so the composition rule across slots, and its per-image sensitivity weights, is the novel part. The additive-MMSE form itself is SoftCast/ParCast's for linear codes, so claim only the learned, nonlinear, training-dependent version.
- Neural per-image predictors (OraNet, the 2302.13477 evaluator, SCAN) are the natural competitor a reviewer will propose: "just train a net on per-slot SNRs." Our 12-decode per-image fit is the cheap, interpretable alternative. A comparison against such a predictor would pre-empt this objection.

### Gaps
- Not checked: whether 2604.23691 validated EESM against its semantic metric, or treated its semantic link as digital with BLER semantics. Section III-B was not visible in the extracts.
- Not checked: whether the ABG paper's power control assumes one SNR per image or per block.
- Leads not verified at all (titles only; they may define a semantic-to-PHY "interface"): "Multi-level Reliability Interface for Semantic …" ([arXiv:2407.05487](https://arxiv.org/pdf/2407.05487)) and "Learning-Based Interface for Semantic Communication with Bit Importance Awareness" ([arXiv:2507.12850](https://arxiv.org/pdf/2507.12850)).
- No 3GPP or ITU semantic-communication study item defining a link abstraction was found. The only standards-flavoured source was the roadmap preprint above.

## Q2. DeepJSCC when the SNR varies within a codeword (OFDM, block or fast fading): what is known about per-symbol or per-chunk sensitivity and about training with varying SNR?

### Takeaway
Training and adaptation for SNR that varies within one image is established. DRJSCC changes the SNR on every block in training. CFA-JSCC adapts to both average and instantaneous SNR. euJSCC fine-tunes on time-varying channels. CA-JSCC learns subcarrier power. All of these adapt the encoder to the instantaneous CSI (re-encoding, attention, or power maps), and the channel is always exogenous. None measures per-chunk sensitivity weights, proposes an additive distortion model, or reports that constant-SNR training breaks such a model.

### Cited Findings
- **[A] J. Pan, H. Li, G. Zhang, Y. Cai, G. Yu (Zhejiang Univ.), "Deep Refinement-Based Joint Source Channel Coding over Time-Varying Channels" (DRJSCC), arXiv:2311.15309 (Nov 2023). A later paper cites it as IEEE WCNC 2024 [?].**
  - Setting: coded symbols are split into blocks and sent progressively. When the channel changes, the remaining blocks are re-encoded with the new CSI [K].
  - Training: SNR ~ U[0, 20] dB, changed on every block. The ADJSCC baseline changes SNR only every 8 blocks. Tests include SNR changing once, twice or on every block (Fig. 6).
  - Finding: the gap over ADJSCC widens as more blocks are sent over a changing channel.
  - Overlap: per-block varying-SNR training (our piecewise training is of this kind) and progressive block delivery.
  - Lacks: an encode-once prefix (it re-encodes using fed-back CSI), any distortion model or abstraction, multiple users or scheduling, and digital baselines.
  - Sources: [arXiv abs](https://arxiv.org/abs/2311.15309); [arXiv HTML](https://arxiv.org/html/2311.15309)
- **[A][M] H. Li, G. Zhang, K. Zhou, Y. Cai, G. Yu, "Coarse-to-Fine: A Dual-Phase Channel-Adaptive Method for Wireless Image Transmission" (CFA-JSCC), arXiv:2412.08211; IEEE Trans. Wireless Commun. (Xplore document 11130661; one aggregator says 2026, early access is probably 2025 [?]).**
  - Setting: block fading in which "a single image transmission can span multiple fading events."
  - Mechanism: a coarse phase sets the encoding from average SNR. A fine phase refines it from instantaneous, per-symbol channel quality, "re-encoding the remaining channel symbols whenever the channel conditions change." An RL method quantizes the continuous SNR into a small CQI set to cut feedback bits.
  - Finding: beats ADJSCC, DeepJSCC-V, CA-JSCC and DRJSCC in PSNR and MS-SSIM. Instantaneous-SNR adaptation beats average-SNR or channel-gain inputs.
  - Overlap: SNR varying within an image, CQI-quantized feedback.
  - Lacks: an encode-once prefix, an abstraction, a scheduler, digital or HARQ baselines.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2412.08211); [IEEE Xplore PDF](https://ieeexplore.ieee.org/iel8/7693/4656680/11130661.pdf); [Lacuna listing](https://lacuna.tiptreesystems.com/work/coarse-to-fine-a-dual-phase-channel-adaptive)
- **[A] "Extended Universal Joint Source-Channel Coding for Digital Semantic Communications: Improving Channel-Adaptability" (euJSCC), arXiv:2602.14018 (Feb 2026; authors and venue not verified).**
  - Setting: digital, with hypernetwork-based normalization and a dynamic codebook-generation network that refines modulation-specific codebooks from CQI feedback.
  - Mechanism: outer modules track long-term channel statistics; inner modules refine features block by block. Training is in two phases: pre-train on AWGN, then fine-tune on time-varying channels.
  - Finding: beats channel-adaptive digital JSCC under block fading and AWGN.
  - Related: JSCC for channel-adaptive digital semantic communications ([IEEE 10584091](https://ieeexplore.ieee.org/document/10584091/), arXiv:2311.08146) and "Blind Training for Channel-Adaptive Digital Semantic Communications" ([arXiv:2501.02273](https://arxiv.org/pdf/2501.02273)).
  - Overlap: CQI-driven block-wise adaptation, training on time-varying channels.
  - Lacks: an analog code, an abstraction, a scheduler.
  - Source: [arXiv HTML](https://arxiv.org/html/2602.14018)
- **[A] H. Wu, Y. Shao, K. Mikolajczyk, D. Gündüz, "Channel-Adaptive Wireless Image Transmission with OFDM" (CA-JSCC), arXiv:2205.02417 (IEEE WCL 2022 [?]).**
  - Mechanism: dual attention on estimated CSI learns the feature-to-subcarrier mapping and the power allocation.
  - Finding: it "generally allocates more power to subcarriers with better channel conditions." The non-adaptive baseline needs matched train and test SNR.
  - Bears on: learned per-subcarrier sensitivity, exposed only implicitly through the power map.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2205.02417); [Imperial PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/Channel-Adaptive-Wireless-Image-Transmission-with-OFDM.pdf)
- **[A] M. Yang, C. Bian, H.-S. Kim [authors and venues from memory, ?]: "Deep Joint Source Channel Coding for Wireless Image Transmission with OFDM," arXiv:2101.03909 (ICC 2021 [?]), and "OFDM-guided Deep Joint Source Channel Coding for Wireless Multipath Fading Channels," arXiv:2109.05194 (IEEE TCCN 2022 [?]).**
  - In the OFDM-guided paper, a model trained at a low SNR favours low SNR, and one trained at 15 dB favours high SNR.
  - The same paper reports robustness to moderate mismatch in the number of multipath taps.
  - For the digital comparisons, see Q4.
  - Sources: [2109.05194](https://arxiv.org/pdf/2109.05194); [2101.03909](https://arxiv.org/pdf/2101.03909)
- **[A] SGD-JSCC (arXiv:2501.01138).** Under fast fading, "symbols experience different instantaneous channel conditions," and small-gain symbols give highly noisy elements. A training-free denoising step absorbs per-symbol gain variation. This is direct evidence that per-symbol SNR heterogeneity matters to the decoder, though no sensitivity model is given. — [arXiv HTML](https://arxiv.org/html/2501.01138v1)
- **[K] DeepJSCC-MIMO (IEEE TWC 2024, arXiv:2309.00470).** A "channel heatmap" (each symbol's post-equalisation noise variance) is fed to the decoder, and the encoder learns power over eigenmodes. Per-symbol SNR is side information, not a quality model. — [arXiv](https://arxiv.org/abs/2309.00470)
- **[K] Importance-to-subchannel mapping, single link:** importance-aware MIMO-OFDM ViT ([2508.07696](https://arxiv.org/pdf/2508.07696)), FAST ([2401.14614](https://www.arxiv.org/pdf/2401.14614)), DeepStream ([2509.05971](https://arxiv.org/html/2509.05971v1)). Two more of the same kind:
  - [A] "Image value-based dynamic spatial stream allocation in DeepJSCC-coded MIMO" (EURASIP JWCN 2026) assigns high-gain streams to important image regions, with gains at low SNR. — [Springer](https://link.springer.com/article/10.1186/s13638-026-02633-6)
  - [A] "Importance-Aware Resource Allocations for MIMO Semantic Communication" (2025). — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12191451/)
- **[A] MERL TR2025-069, rateless deep JSCC for 3D point clouds.** Argues that conventional autoencoders "treat all coded symbols equally," and uses weighted dropout to give symbols different importance in training. — [MERL PDF](https://www.merl.com/publications/docs/TR2025-069.pdf)
- **[A] Training-SNR evidence:**
  - Over-the-air study (arXiv:2405.13403): training at random SNRs beats fixed-SNR training at low SNR. Under Rayleigh fading, fixed- and random-SNR training perform about the same, and only fine-tuning at a specific SNR gives a slight gain. — [arXiv PDF](https://arxiv.org/pdf/2405.13403)
  - Learnable pilot design for OFDM DeepJSCC (arXiv:2610.25979, Oct 2026) trains on SNR drawn from {0, 5, 10, 15, 20} dB. — [arXiv HTML](https://arxiv.org/html/2610.25979)
  - A 2026 receiver-side enhancer paper names intra-frame SNR variation in fast fading as future work. — [Sensors 26(16):5134](https://doi.org/10.3390/s26165134)

### Inferences
- "Training with varying SNR" is not novel on its own. Piecewise per-phase training should be presented as the DRJSCC/CFA-JSCC mechanism applied to an encode-once prefix code. The new findings are what it enables, an additive and calibratable distortion model, and that constant-SNR training breaks that model.
- No paper found reports per-chunk or per-phase sensitivity weights (our rho_L[j]) measured per image. Learned power maps (CA-JSCC, DeepJSCC-MIMO) and importance-to-subcarrier rules imply such weights exist, but nobody exposes them as a link model.

### Gaps
- Not checked: CFA-JSCC's and DRJSCC's numerical gains, block lengths and fading models. The full texts were inaccessible.
- Not checked: whether any OFDM DeepJSCC paper reports PSNR as a function of per-subcarrier SNR profiles, which would be a de facto abstraction test.
- The ADJSCC (Xu et al., TCSVT) and SNR-adaptive DeepJSCC ([Ding et al., arXiv:2102.00202](https://arxiv.org/pdf/2102.00202)) details were not re-verified this session.

## Q3. Progressive, layered and variable-length DeepJSCC, feedback, and semantic HARQ / prefix decoding

### Takeaway
Prefix-style progressive DeepJSCC is established: Kurka and Gündüz (SPAWC 2019, then TWC 2021), DeepJSCC-f, JSCCformer-f, DeepJSCC-l++, and others in SURVEY A. Semantic HARQ is an active 2022-2026 line, but mostly for text or multi-task features. It works by error detection plus retransmission or "incremental knowledge" combining. No work found pairs an encode-once, any-prefix-decodable analog code with an HARQ-like phase-by-phase delivery at different SNRs, a quality abstraction across phases, and a scheduler.

### Cited Findings
- **[A] D. B. Kurka, D. Gündüz, "Successive Refinement of Images with Deep Joint Source-Channel Coding," IEEE SPAWC 2019, arXiv:1903.06333.** Three CNN strategies for progressive transmission, with different complexity-performance trade-offs. The learned layered representation comes close to a single-layer scheme, with graceful degradation in SNR. — [arXiv abs](https://arxiv.org/abs/1903.06333)
- **[A][M] D. B. Kurka, D. Gündüz, "Bandwidth-Agile Image Transmission With Deep Joint Source-Channel Coding" (DeepJSCC-l), IEEE TWC 2021 (Xplore 9464731; DOI 10.1109/TWC.2021.3090048 per SURVEY), arXiv:2009.12480.**
  - Setting: layers are sent either sequentially (successive refinement) or so that any subset can be decoded (multiple descriptions).
  - Finding: small loss against single transmission, and comparable to digital progressive schemes at low SNR and small bandwidth.
  - Per a search extract of the journal PDF [?]: refinement-layer quality drops when the test SNR falls below the training SNR, because the reconstructed base layer no longer matches what the encoder expects.
  - Overlap: the prefix or successive-refinement code.
  - Lacks: per-layer SNR composition, scheduling, HARQ baselines.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2009.12480); [IEEE Xplore](https://ieeexplore.ieee.org/document/9464731/); [Imperial PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/layered_JSCC_journal.pdf)
- **[A] D. B. Kurka, D. Gündüz, "DeepJSCC-f: Deep Joint Source-Channel Coding of Images with Feedback," arXiv:1911.11174 (IEEE JSAIT 2020 [venue from memory, ?]; ICASSP 2020 version).**
  - Setting: a noisy copy of the channel output reaches the transmitter with unit delay. The variable-length mode "greatly reduc[es] the average bandwidth required for a target quality."
  - Training needs: it learns without pilots or explicit channel estimation.
  - Baselines: separation baselines are given perfect CSI and BPG with capacity-achieving codes, i.e. deliberately generous. The authors note that feedback "can only improve channel coding" in the separate architecture.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/1911.11174); [Imperial PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/KurkaGunduz_deepJSCC-f.pdf)
- **[K] JSCCformer-f (IEEE TWC 2024, arXiv:2306.09101).** ViT with channel-output feedback, re-encoding every block, and a variable-rate stop at a target PSNR. The nesting penalty is 0.58-1.73 dB at the last block. **[K]** Also in this family: DeepJSCC-l++ ([2305.13161](https://arxiv.org/abs/2305.13161)) and Yang and Kim's prefix mask ([2110.04456](https://arxiv.org/abs/2110.04456)); timeliness-aware JSCC, TS-JSCC and rateless NTRSCC are in SURVEY A. — [arXiv](https://arxiv.org/abs/2306.09101)
- **[A][M] J. Hu, F. Wang, W. Xu, H. Gao, P. Zhang, "SemHARQ: Semantic-Aware Hybrid Automatic Repeat Request for Multi-Task Semantic Communications," IEEE TWC, DOI 10.1109/TWC.2025.3602146 (per ResearchGate; the listing's "January 2025" date conflicts with the DOI's numbering [?]), arXiv:2404.08490.**
  - Mechanism: a multi-task semantic encoder, feature-importance ranking, and a feature-distortion evaluation network that flags corrupted features for retransmission.
  - Finding: 16.27%-166.48% multi-task gain over semantic-aware and traditional HARQ at low SNR.
  - Overlap: semantic HARQ against a traditional-HARQ baseline.
  - Lacks: image-reconstruction PSNR with an any-prefix code, a digital CQI/MCS pipeline, stale CSI.
  - Sources: [arXiv PDF](https://arxiv.org/pdf/2404.08490); [ResearchGate](https://www.researchgate.net/publication/395273948_SemHARQ_Semantic-Aware_Hybrid_Automatic_Repeat_Request_for_Multi-Task_Semantic_Communications)
- **[A][M] P. Jiang, C.-K. Wen, S. Jin, G. Y. Li, "Deep Source-Channel Coding for Sentence Semantic Transmission With HARQ," IEEE TCOM 70(8):5225-5240, 2022, DOI 10.1109/TCOMM.2022.3180997, arXiv:2106.03009.**
  - Mechanism: SC-RS-HARQ and an end-to-end SCHARQ with chase-combining and incremental-redundancy variants in latent space. A similarity network (Sim32) replaces CRC.
  - Finding: fewer bits and a lower sentence error rate.
  - Scope: text only.
  - Sources: [ResearchGate](https://www.researchgate.net/publication/361177941_Deep_Source-Channel_Coding_for_Sentence_Semantic_Transmission_with_HARQ); [arXiv](https://arxiv.org/pdf/2106.03009)
- **[A] Zhou et al., "Adaptive Bit Rate Control in Semantic Communication with Incremental Knowledge-based HARQ," IEEE OJ-COMS vol. 3, 2022, arXiv:2203.06634.** IK-HARQ with rate control for text. — [arXiv PDF](https://arxiv.org/pdf/2203.06634)
- **[A] "Topology Data Analysis-based Error Detection for Semantic Image Transmission with Incremental Knowledge-based HARQ," arXiv:2403.11542 (2024; authors and venue not verified).** Image IK-HARQ that uses TDA-based semantic error detection to trigger retransmission. Per a related paper, convincing error detection for images was under-investigated. — [arXiv PDF](https://arxiv.org/pdf/2403.11542)
- **[A][M] Y. Li, X. Wang, Z. Shi, Y. Fu, "Semantic HARQ for Intelligent Transportation Systems: Joint Source-Channel Coding-Powered Reliable Retransmissions," arXiv:2504.14615. The journal version, "Semantic HARQ: Joint Source-Channel Coding-Powered Reliable Retransmissions for IoT Networks," is on IEEE Xplore (document 11372719; the Xplore path suggests IEEE IoT Journal [?]).**
  - Mechanism: three text schemes: sem-HARQ-I, weighted combining, and synonymous combining. A generative reconstructor uses a discriminator for error detection.
  - Finding: beats no-HARQ JSCC, most at low SNR.
  - Caveat: a third-party review calls the gains under-supported.
  - Sources: [arXiv](https://arxiv.org/html/2504.14615v1); [IEEE Xplore](https://ieeexplore.ieee.org/iel8/6488907/6702522/11372719.pdf); [review](https://pith.science/paper/2504.14615)
- **[A] B. Li, S. Wang, S. Bi, L. Qian, Y. Cai, "Joint Source-Channel-Check Coding with HARQ for Reliable Semantic Communications" (S3CHARQ), arXiv:2603.23869 (Mar 2026).** Folds the check codeword into JSCC, so one codeword serves both fidelity checking and reconstruction. An RL module makes per-sample retransmission decisions. Gains over HARQ-based semantic methods are reported; the numbers were not visible. — [arXiv HTML](https://arxiv.org/html/2603.23869v1)
- **[A] B. Li, M. Gong, S. Wang, J. Yan, S. Bi, Y.-J. A. Zhang, "Toward Reliable Semantic Communication: Beyond Average Performance," arXiv:2606.01284 (31 May 2026).** A perspective that reviews three reliability approaches:
  - channel-aware adaptation, which channel-state mismatch can hurt;
  - robust codec design, which may trade average quality for stability;
  - HARQ, which adds overhead and latency.

  It proposes robust adaptive semantic communication under imperfect CSI, and JSCC-check coding with adaptive retransmission. — [arXiv abs](https://arxiv.org/abs/2606.01284)
- **[A] Others, existence and gist only:**
  - Generative Semantic HARQ (text). A stochastic Transformer-VAE produces a different latent on each retransmission, the authors' analogue of IR with one model. — [arXiv:2603.15068](https://arxiv.org/html/2603.15068)
  - HARQ for cooperative-perception semantic communication. — [arXiv:2409.09042](https://arxiv.org/html/2409.09042)
  - Predictive semantic-aware HARQ (Electronics 15(19):4482, 2026). — [DOI](https://doi.org/10.3390/electronics15194482)

### Inferences
- Our phases behave like analog incremental redundancy: each phase refines an already decodable prefix, and nothing is re-encoded. That is a different design point from:
  - semantic HARQ, which detects errors and then retransmits or combines;
  - DeepJSCC-f and JSCCformer-f, which use channel-output feedback and re-encoding;
  - DRJSCC and CFA-JSCC, which re-encode the remaining symbols.
- None of the semantic-HARQ papers models what a retransmission at a different SNR does to final quality, which is what our abstraction provides.
- The DeepJSCC-l observation (refinement layers degrade when SNR departs from training) is consistent with our finding that a constant-SNR-trained code obeys no composition rule. It is worth citing once the full text is checked.

### Gaps
- Not verified: whether DeepJSCC-l (TWC 2021) evaluates layers at different SNRs. The extracts say nothing about how layers are assigned to SNRs.
- Not verified: authors or venue of 2403.11542, or whether its image IK-HARQ uses prefix-decodable retransmissions.
- No image DeepJSCC paper was found that combines learned IR or chase combining with NACK-driven retransmission and compares against digital IR-HARQ.

## Q4. DeepJSCC vs separation under imperfect or outdated CSI, mobility or realistic channels: what margins, against which baselines, and is HARQ ever in the digital baseline?

### Takeaway
Image DeepJSCC-vs-digital margins are usually reported against BPG, JPEG2000 or VTM with LDPC or capacity-achieving codes at a fixed or best MCS, often with perfect CSI given to the digital side. Typical margins are 2.5-4 dB, or more than 3 dB once the digital side has imperfect CSI. None of the image comparisons found includes HARQ or CQI-table link adaptation in the digital baseline, and none models outdated (aged) CQI reports. The one close study, Mohajer et al. (Wireless Networks 2026), contrasts analog JSCC with separation driven by OLLA link adaptation under imperfect CSIT. It is task-oriented and analytic/RL-based, and its HARQ treatment is unknown.

### Cited Findings
- **[A][M] Mohajer, Bavaghar, Mirzaei, Fernando, "Link adaptation and agentic semantic communication under imperfect CSIT for task-oriented wireless links," Wireless Networks 32:2149-2171 (published 22 Jun 2026), DOI 10.1007/s11276-026-04165-9.**
  - Setting: defines semantic rate and distortion, and a "semantic-capacity" metric accounting for finite blocklength and CSIT uncertainty.
  - Finding: the analysis "reveals when analog JSCC yields better task performance than separation with conventional outer-loop link adaptation (OLLA)-driven MCS/rate selection." OLLA targets bit-level reliability, not task success.
  - Controller: constrained RL allocates the pilot and feedback budget and adapts power, coding rate and JSCC parameters. A multi-agent variant separates channel probing from payload adaptation.
  - Numbers: on time-varying fading and interference traces, up to ≈76% lower mean distortion and ≈17 points higher task success at equal latency. Simulation only.
  - Overlap: (c) analog vs digital link adaptation under CSIT error, plus co-adapting JSCC parameters.
  - Lacks, as far as the abstract shows: learned image DeepJSCC with PSNR, an IR-HARQ digital baseline, a Jakes-type aging model, multi-user scheduling, a link abstraction.
  - Companion paper: "Robust Semantic Signal Processing and Distribution-Shift-Aware Link Adaptation for Task-Oriented Wireless Communications" (Wireless Pers. Commun. 2026, DOI 10.1007/s11277-026-12232-9), covering analog, digital and hybrid modes with a two-agent controller.
  - Sources: [Springer](https://link.springer.com/article/10.1007/s11276-026-04165-9); [companion](https://link.springer.com/article/10.1007/s11277-026-12232-9)
- **[A] Yang, Bian, Kim, DeepJSCC-OFDM, arXiv:2101.03909.** Separation baselines get perfect CSI, while DeepJSCC never gets perfect CSI at the decoder. "With imperfect CSI, the performance of the baseline scheme is significantly worse (e.g., >3 dB PSNR) at the same low rate target." — [arXiv PDF](https://arxiv.org/pdf/2101.03909)
- **[A] OFDM-guided DeepJSCC, arXiv:2109.05194.** Reports a "2.5-4 dB SNR gain for the equivalent image quality" over separate source and channel coding (attribution to this paper is via a search extract [?]). — [arXiv PDF](https://arxiv.org/pdf/2109.05194)
- **[A][M] E. Bourtsoulatze, D. B. Kurka, D. Gündüz, "Deep Joint Source-Channel Coding for Wireless Image Transmission," IEEE TCCN 2019 (Xplore 8723589), arXiv:1809.01733.** Over slow Rayleigh fading, DeepJSCC "learns noise resilient coded representations and significantly outperforms separation-based digital communication at all SNR and channel bandwidth values," without pilots. — [arXiv abs](https://arxiv.org/abs/1809.01733); [IEEE](https://ieeexplore.ieee.org/abstract/document/8723589)
- **[A] DeepJSCC-f (arXiv:1911.11174).** The separation scheme "inherently assumes perfect CSI at the receiver." DeepJSCC-f still outperforms "despite all the impractical assumptions to the advantage of the latter." — [Imperial PDF](https://www.imperial.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/KurkaGunduz_deepJSCC-f.pdf)
- **[A] DeepJSCC-Q (arXiv:2206.08100).** Constellation-constrained DeepJSCC outperforms BPG + LDPC and avoids the cliff effect. — [arXiv PDF](https://arxiv.org/pdf/2206.08100)
- **[A] "Lightweight Low-SNR-Robust Semantic Communication System for Autonomous Driving," arXiv:2604.20278 (2026).** Below 14.5 dB, DeepJSCC PSNR is "significantly superior" to BPG-LDPC, whose PSNR "drops sharply to nearly 0 dB" when LDPC decoding fails. This is a cliff with no link adaptation or HARQ. — [arXiv PDF](https://arxiv.org/pdf/2604.20278)
- **[A] "Image Semantic Communication with Quadtree Partition-based Coding" (Quad-DeepSC), arXiv:2512.05395 (Dec 2025).** Evaluated in Sionna. It sweeps the MCS index per channel condition, keeps the configuration with the lowest bandwidth ratio, and compares against VTM + 3GPP (5G LDPC) channel coding. No retransmissions, and it is unclear whether CDL or TDL models were used. — [arXiv HTML](https://arxiv.org/html/2512.05395v1); [arXiv PDF](https://arxiv.org/pdf/2512.05395)
- **[A] Robustness to channel-estimation error, against learned baselines only:**
  - HANA-JSCC, "Robust MIMO Semantic Communication with Imperfect CSI via Knowledge Distillation" (arXiv:2509.04005): 0.40-0.54 dB above state-of-the-art learned benchmarks. — [arXiv PDF](https://arxiv.org/pdf/2509.04005)
  - "Robust Image Semantic Coding With Learnable CSI Fusion Masking Over MIMO Fading Channels" (IEEE TWC 2024, DOI 10.1109/TWC.2024.3409735; arXiv:2406.07389). — [ACM DL](https://dl.acm.org/doi/abs/10.1109/TWC.2024.3409735)
  - SNR-aware image transmission with DL channel estimation (arXiv:2504.20557; Physical Communication 2026) observes that train/deploy discrepancies "can significantly impact performance." — [arXiv](https://arxiv.org/html/2504.20557); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1874490726001254)
- **[A] CLEAR (arXiv:2412.08978; Springer Discover AI 2026).** Conditions denoising on full CSI including Doppler and phase noise, for +2.31 dB over DeepJSCC-V at about 5.6x the runtime. A mobility-flavoured setting, but with no staleness model. — [arXiv PDF](https://arxiv.org/pdf/2412.08978); [Springer](https://link.springer.com/article/10.1007/s44163-026-02499-z)
- **[A] "SNR-Independent Joint Source-Channel Coding for wireless image transmission," arXiv:2306.15183.** Argues that channel-adaptive JSCC conditioned on SNR degrades through "channel mismatch effects," so SNR-conditioned codes themselves suffer when the SNR report is wrong. — [arXiv PDF](https://arxiv.org/pdf/2306.15183)
- **[K]** SURVEY F4 had already found the DeepJSCC-vs-separation OFDM studies (2101.03909, 2109.05194) and the digital-compatible JSCC surveys. None has a CQI-driven link-adaptation loop with report errors. — [2412.18876](https://arxiv.org/pdf/2412.18876)

### Inferences
- The widely quoted DeepJSCC margins (more than 3 dB, 2.5-4 dB, "cliff" curves) come from digital baselines that have no HARQ and, under imperfect CSI, no link-adaptation loop. HARQ with combining is exactly what a real digital system uses to absorb CQI error and staleness. Our margins (+1.3 dB at 2 dB CQI error, +1.7 dB at report correlation 0.8, with LTE CQI-table link adaptation and IR-HARQ) are therefore the more conservative and realistic estimate. Present this as a contribution in its own right, citing Mohajer et al. as the only analytic precedent and distinguishing it (task metric, OLLA, no stated HARQ).
- No paper found ties DeepJSCC's advantage to report age (Jakes or Gauss-Markov correlation). "The lead grows with report age" appears untested elsewhere.

### Gaps
- Not checked in the Mohajer et al. full text: whether HARQ is in its separation baseline, how CSIT error is modelled (estimation error or delay), and whether its "analog JSCC" is learned or analytic.
- No paper was found that compares DeepJSCC with an HEVC/VTM + 5G/LTE MCS + HARQ chain in a 3GPP system-level setting. Searches for 3GPP CDL/TDL + DeepJSCC + Sionna found only Quad-DeepSC (single link, no HARQ).

## Q5. Training DeepJSCC on realistic channel traces or on MAC- or scheduler-induced channel distributions; co-design with scheduling or link adaptation

### Takeaway
Papers on distribution mismatch show that DeepJSCC degrades when the deployment channel differs from training, and they fix it with robust regularisers, domain adaptation, or training on time-varying channels. In every case the channel distribution is exogenous: i.i.d. models, block fading, or recorded traces. No work found trains a learned code on the per-symbol or per-phase SNR distribution that its own scheduler or MAC induces. The closest co-design is Mohajer et al.'s RL controller, which jointly sets JSCC parameters and link adaptation but does not retrain the code. This confirms SURVEY F3.

### Cited Findings
- **[A] E. Hong, T. Park, Y. Kim, "A Survey on Robust Deep Joint Source-Channel Coding for Semantic Communications," arXiv:2604.04413 (6 Apr 2026).** DeepJSCC performance "often degrades when the deployment channels differ from the training channel conditions," so robustness is "a critical requirement." — [arXiv abs](https://arxiv.org/abs/2604.04413)
- **[A] T. Park et al., "Robust Deep JSCC for Task-Oriented Semantic Communications," arXiv:2503.12907.** A KL regulariser between noisy and noise-free posteriors. Trained on AWGN and tested on Rayleigh, it stays stable where vanilla DeepJSCC does not. — [arXiv PDF](https://arxiv.org/pdf/2503.12907)
- **[A] Lee et al., "Deep JSCC with Iterative Source Error Correction," AISTATS 2023 (PMLR v206), arXiv:2302.09174.** Trained on Gaussian noise and tested on Laplace noise of the same variance, the decoding improvement holds. DeepJSCC decoders can generalise better than separate schemes. — [PMLR PDF](https://proceedings.mlr.press/v206/lee23c/lee23c.pdf)
- **[A] Domain and distribution shift (task-oriented):**
  - Domain-adaptive DeepJSCC (arXiv:2607.28907): vanilla DeepJSCC has the lowest target-domain accuracy; 98.15% on SVHN→MNIST at 10 dB. — [arXiv HTML](https://arxiv.org/html/2607.28907)
  - Information-bottleneck robustness to spurious-feature shift (arXiv:2405.09514). — [arXiv PDF](https://arxiv.org/pdf/2405.09514)
- **[A] Over-the-air testing (arXiv:2405.13403).** Random-SNR training helps at low SNR. Under Rayleigh, fixed- and random-SNR training are about equal, and only SNR-specific fine-tuning gives a slight gain. — [arXiv PDF](https://arxiv.org/pdf/2405.13403)
- **[A] Exogenous time-varying training:**
  - DRJSCC: SNR changed per block. — [arXiv](https://arxiv.org/abs/2311.15309)
  - CFA-JSCC: block fading with average and instantaneous SNR. — [arXiv](https://arxiv.org/pdf/2412.08211)
  - euJSCC: fine-tuned on time-varying channels after AWGN pre-training. — [arXiv](https://arxiv.org/html/2602.14018)
- **[A] Mohajer et al. (Wireless Networks 2026).** A constrained RL controller adapts power, coding rate and JSCC parameters together with the pilot and feedback budget, evaluated on "time-varying fading and interference traces." The companion paper adds distribution-shift-aware link adaptation. This is co-design of link adaptation with JSCC parameters, not retraining the code on the distribution the controller induces. — [Springer](https://link.springer.com/article/10.1007/s11276-026-04165-9); [companion](https://link.springer.com/article/10.1007/s11277-026-12232-9)
- **[A] "Semantic Communication in Dynamic Channel Scenarios: Collaborative Optimization of Dual-Pipeline JSCC and Personalized Federated Learning," arXiv:2503.14084.** Notes that a CSI-estimation module inside the JSCC decoder "requires extensive training and large parameter spaces for unpredictable channels" (satellite links). — [arXiv HTML](https://arxiv.org/html/2503.14084)
- **[A] A 6G simulator survey (Electronics 14(16):3313, 2025).** Mentions a semantic-communication study that built channel datasets from a cellular network simulator for an RL-based semantic method. Which study this is was not verified [?]. — [MDPI](https://www.mdpi.com/2079-9292/14/16/3313)
- **[A] DeepMA (arXiv:2303.11543).** End-to-end deep multiple access for image semantic communication. It learns the access scheme with the code, but has no scheduler-induced SNR statistics. — [arXiv PDF](https://arxiv.org/pdf/2303.11543)
- **[K]** SURVEY F3 (searches of 2026-10-13): distributed DeepJSCC over a MAC, NOMA JSCC, JSCC resource allocation, DD-JSCC and IoV semantic scheduling all design the code and the allocation separately. None trains the code on the channel the scheduler induces. — [SURVEY source list, e.g. 2211.09920](https://arxiv.org/pdf/2211.09920)

### Inferences
- Training on the scheduler-induced distribution is new in kind: the training distribution is endogenous to the system design. Its mechanism is ordinary matched-distribution training, as in DRJSCC, CFA-JSCC and euJSCC.
- Our small gain (+0.1-0.3 dB) fits the over-the-air finding that the training-SNR distribution matters little under Rayleigh. Frame it as a robustness and consistency result ("the code-scheduler fixed point is close to the piecewise-trained code"), not as a large win. Reviewers will call it matched training otherwise.
- Mohajer et al. is the closest "co-design" citation. The difference is that they tune JSCC parameters online, while we retrain the code's weights on the per-phase SNR distribution the scheduler produces.

### Gaps
- Not found: a DeepJSCC trained on measured or 3GPP system-level channel traces with a quantified sim-to-real gap. Only OTA tests and synthetic Doppler or phase-noise channels turned up.
- Not checked: "Lossy Joint Source-Channel Coding over Unknown Channels" ([arXiv:2606.07933](https://arxiv.org/html/2606.07933)) and "Foundation Model-Based Adaptive Semantic Image Transmission for Dynamic Wireless Environments" ([arXiv:2509.23590](https://arxiv.org/pdf/2509.23590)). Both may train under channel uncertainty. Only their titles were seen.

## Q6 (coordinator's synthesis request). The five closest works, and an overlap verdict for (a) the link abstraction, (b) scheduler-aware training, (c) analog vs digital with HARQ under stale CSI

### Takeaway
Our individual mechanisms each have precedents:
- prefix codes: DeepJSCC-l, JSCCformer-f;
- mixed-SNR training: DRJSCC, CFA-JSCC, euJSCC;
- analog vs digital under CSIT error: Mohajer et al.;
- semantic performance functions: ABG.

Our three claims are not covered by any work found:
- (a) a validated multi-SNR composition rule for a learned code, beating MIESM/EESM;
- (b) training on the scheduler-induced distribution;
- (c) a comparison against a digital chain with CQI-table link adaptation and IR-HARQ under aged CQI reports.

### Cited Findings
The five closest works, ranked by overall overlap:
1. **CFA-JSCC.** Li, Zhang, Zhou, Cai, Yu, IEEE TWC (Xplore 11130661, 2025/26 [?]), arXiv:2412.08211.
   - Overlaps: SNR varying within one image under block fading; adaptation to both average and instantaneous SNR; CQI-quantised SNR feedback chosen by RL.
   - Lacks: an encode-once prefix (it re-encodes the remaining symbols), a distortion model or abstraction, multiple users or scheduling, digital or HARQ baselines.
   - Sources: [arXiv](https://arxiv.org/pdf/2412.08211); [IEEE](https://ieeexplore.ieee.org/iel8/7693/4656680/11130661.pdf)
2. **DRJSCC.** Pan, Li, Zhang, Cai, Yu, arXiv:2311.15309 (WCNC 2024 [?]).
   - Overlaps: per-block varying-SNR training, the closest analogue of our piecewise training; progressive block delivery over a time-varying channel.
   - Lacks: an encode-once prefix (CSI-driven re-encoding), an abstraction, a scheduler, digital baselines.
   - Source: [arXiv](https://arxiv.org/abs/2311.15309)
3. **The DeepJSCC-l family (Kurka and Gündüz).** TWC 2021 (arXiv:2009.12480), with SPAWC 2019 (arXiv:1903.06333) and DeepJSCC-f (arXiv:1911.11174).
   - Overlaps: a successive-refinement, prefix-decodable learned code; variable length with feedback; refinement layers that are sensitive to SNR mismatch [?].
   - Lacks: a per-layer SNR composition rule, scheduling, digital HARQ baselines (separation is given perfect CSI and capacity-achieving codes).
   - Sources: [TWC](https://ieeexplore.ieee.org/document/9464731/); [SPAWC](https://arxiv.org/abs/1903.06333); [DeepJSCC-f](https://arxiv.org/pdf/1911.11174)
4. **Mohajer, Bavaghar, Mirzaei, Fernando.** Wireless Networks 32:2149-2171 (2026), DOI 10.1007/s11276-026-04165-9.
   - Overlaps: when analog JSCC beats separation with OLLA-driven MCS under imperfect CSIT; joint adaptation of JSCC parameters and link adaptation on fading traces.
   - Lacks: learned image DeepJSCC, an IR-HARQ baseline (not mentioned), report aging, multi-user scheduling, an abstraction.
   - Source: [Springer](https://link.springer.com/article/10.1007/s11276-026-04165-9)
5. **ABG semantic performance model.** Ma et al., arXiv:2504.21055 (2025).
   - Overlaps: a fitted SNR-to-quality link model for learned (ViT, SCUNet) codecs, used for system-level power control over fading.
   - Lacks: multi-SNR composition, an ESM comparison, training dependence; the encoders are quantized digital.
   - Source: [arXiv](https://arxiv.org/abs/2504.21055)

Must-cite runners-up:
- For (a), the two 2026 preprints that adopt digital EESM for semantic OFDM links without a visible validation: P. Jiang et al., arXiv:2604.23691 ([arXiv](https://arxiv.org/html/2604.23691v1)), and AgentComm, arXiv:2604.13558 ([arXiv](https://arxiv.org/html/2604.13558v1)).
- [K] SoftCast/ParCast, for the additive-MMSE form. — [MSR](https://www.microsoft.com/en-us/research/?p=168037)
- [K] JSCCformer-f and PADC, for the prefix and per-image-quality side. — [JSCCformer-f](https://arxiv.org/abs/2306.09101); [PADC DOI](https://doi.org/10.1109/TWC.2023.3234408)
- SemHARQ, as the HARQ-side precedent with a traditional-HARQ baseline. — [arXiv](https://arxiv.org/pdf/2404.08490)

### Inferences
These verdicts are my assessment from the cited abstracts and extracts.
- **(a) Link abstraction. Overlap: LOW (new in kind), with two precedents that must be acknowledged.**
  - Nothing found composes the quality of a learned code from several SNRs, and nothing tests EESM or MIESM on learned codes.
  - The additive-MMSE form is SoftCast/ParCast's for linear codes.
  - Effective-SNR mapping is already being adopted, unvalidated, for semantic links (2604.23691, 2604.13558).
  - Single-SNR fits (ABG, OraNet, Yan et al.) are the "mean-dB" abstraction.
  - Strongest framing: "learned codes need their own L2S; the borrowed digital ESMs are wrong by about 9x (MIESM) and lose to a sensitivity-weighted MMSE rule, which holds only under mixed-SNR training."
  - Pre-empt the reviewer's "train a neural predictor instead" by comparing against an OraNet- or SCAN-style per-slot-SNR predictor.
- **(b) Scheduler-aware training. Overlap: LOW for the endogenous-distribution idea; HIGH for the underlying mechanism.**
  - Varying-SNR training over exogenous channels is DRJSCC's, CFA-JSCC's and euJSCC's.
  - No paper trains on scheduler-induced statistics (confirmed again here, in line with SURVEY F3).
  - Given the +0.1-0.3 dB size, present it as a co-design consistency result, citing Mohajer et al. as parameter-level co-design and the over-the-air study (2405.13403) for why training-distribution effects are small under Rayleigh.
- **(c) Analog vs digital with HARQ under stale CSI. Overlap: LOW-MEDIUM.**
  - Image DeepJSCC-vs-digital studies report more than 3 dB (imperfect CSI, 2101.03909) or 2.5-4 dB (2109.05194). Their baselines have no HARQ or link-adaptation loop, often with perfect digital CSI.
  - Quad-DeepSC sweeps MCS in Sionna, with no retransmissions.
  - Semantic-HARQ papers compare semantic HARQ against traditional HARQ in text or feature settings, not an analog image code against a CQI-table digital chain.
  - Mohajer et al. is the only analog-vs-OLLA study, and it is task-level.
  - Our smaller, HARQ-inclusive margins (1.3 dB and 1.7 dB) and their growth with report age appear to be the first such measurement for image DeepJSCC. This is a credible "realistic baseline" contribution.

### Gaps
- Every verdict above rests on abstracts and search extracts. The full texts of CFA-JSCC, DRJSCC, DeepJSCC-l, Mohajer et al., 2604.23691 and ABG could not be opened (arXiv and metadata APIs blocked by egress policy; WebFetch DNS failures). They should be read before the paper's related-work section is finalised.
- Highest priority:
  - whether Mohajer et al. include HARQ, and how they model CSIT error;
  - whether 2604.23691 validates EESM against its semantic metric;
  - whether DeepJSCC-l tests layers at different SNRs.
- The web-search budget ran out before the following could be checked: 3GPP and ITU semantic-communication study material; the authors and venues of 2302.13477 and 2403.11542; the leads 2407.05487 and 2507.12850 (possible semantic-to-PHY "interfaces"); and a dedicated search for EESM or MIESM applied to DeepJSCC over OFDM beyond the two preprints found.
