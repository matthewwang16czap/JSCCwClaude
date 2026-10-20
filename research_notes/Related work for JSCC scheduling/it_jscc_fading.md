# Information-theoretic JSCC over block-fading channels: successive refinement, deadlines, retransmissions, multiple users

Scope: prior theory for deadline scheduling of a learned, prefix-decodable deep-JSCC image code. The code has up to 6 phases, one phase per slot, K users and a T-slot deadline. CSIT is causal (or stale) per slot. The link abstraction is D = Dsrc(L) + C(L)·Σ_j ρ_L[j]·φ(n_j) with φ(n) = n/(1+n/κ). Each user runs its own DP, the users are coupled by a Lagrange price, and the code is compared with HEVC + rate adaptation + IR-HARQ. Component tags used below: **[LA]** link abstraction, **[DP]** per-user dynamic program, **[MAP]** phase-to-slot mapping, **[MU]** multi-user scheduling, **[CMP]** HARQ-vs-analog comparison.

**How far the entries were checked.** WebFetch could not resolve any host in this sandbox (arxiv.org, ieeexplore, technion, repec and others all failed with DNS errors). curl to scholarly APIs was refused by the proxy. The shared web-search budget ran out after about 30 queries. So every entry below rests on search-engine summaries of abstracts, bibliographic records and citing papers. **No full text was read.** Volume, issue, page and DOI details marked "(unconfirmed)" come from memory. Works already in `jscc/docs/SURVEY.md` are not repeated unless a new detail is added: Equitz–Cover, Rimoldi and Shoham–Gersho (G); the restless-bandit lineage (F2/F5), including Derman–Lieberman–Ross; and SoftCast/ParCast (F4).

## 1. Expected-distortion minimisation of a Gaussian source over block fading with layered / successive-refinement codes

### Takeaway
Information theory answers "successive refinement over fading" in two ways:
- **Power splits across superposed digital layers when the transmitter has no CSI.** Tian–Steiner–Shamai–Diggavi and Ng–Gündüz–Goldsmith–Erkip solve this with O(M), recursive and variational methods.
- **High-SNR distortion exponents for layered, hybrid digital-analog (HDA) or broadcast codes spread over L blocks.** See Gündüz–Erkip, Bhattad–Narayanan–Caire, Caire–Narayanan and Laneman et al.

None of these works lets the transmitter see a block's SNR before choosing what goes into it. None has a deadline shared by several users. All of them treat a layer as decoded or lost (a mutual-information threshold), never as a smooth per-block MMSE term.

### Cited Findings

**1.1 Shamai and Steiner: the broadcast approach for slowly fading MIMO channels (2003).**
- **Citation:** S. Shamai (Shitz) and A. Steiner, "A broadcast approach for a single-user slowly fading MIMO channel," IEEE Trans. Inf. Theory, vol. 49, no. 10, pp. 2617–2635, Oct. 2003 — [reference list of a citing paper](https://arxiv.org/pdf/1208.4790); [Technion draft](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_661-1.pdf). The DOI of the journal paper is unconfirmed.
  - Conference version: "Single user broadcasting in a MIMO channel," IEEE ITW 2003 — [DOI 10.1109/ITW.2003.1216689 via Unpaywall](https://unpaywall.org/10.1109%2FITW.2003.1216689).
  - Shamai's original SISO "broadcast strategy" paper (ISIT 1997) was not verified in this session.
- **Setting:**
  - Slowly fading Gaussian MIMO channel; perfect CSI at the receiver only; no feedback link.
  - Superposition (layered) coding lets the receiver decode as many layers as its realised state allows.
  - The layering is motivated by majorization theory — [Technion draft](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_661-1.pdf).
- **Result:** A single-layer outage approach performs reasonably well for MIMO in average reliably decoded rate, and a "multiple-access broadcast" variant performs relatively well — [Technion draft](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_661-1.pdf).
- **Relates to:**
  - [CMP]: the digital hedge against unknown CSI, which degrades by layers. This is the digital counterpart of an analog code's soft degradation.
  - [LA]: a layer counts only if the state exceeds that layer's threshold.
- **Lacks vs ours:**
  - It maximises expected rate, not distortion.
  - It has one block and no CSIT at all; our transmitter knows the slot SNR when the slot starts.
  - It has no deadline and no multiple users.

**1.2 Tajer, Steiner and Shamai: survey of the broadcast approach (2021).**
- **Citation:** A. Tajer, A. Steiner, S. Shamai (Shitz), "The Broadcast Approach in Communication Networks," *Entropy* 23(1):120, Jan. 2021, DOI 10.3390/e23010120, arXiv 2101.07173 — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7831149/); [arXiv](https://arxiv.org/pdf/2101.07173).
- **Content:** A 137-page open-access review of the broadcast approach for channels whose states the transmitter knows only statistically, applied to various channel and network models and operational constraints. Its keywords include slow fading, superposition coding, MAC, interference and relay. It belongs to the special issue "Multiuser Information Theory III" — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7831149/); [Zenodo](https://zenodo.org/records/4540439).
- **Relates to:** The single best entry point for a "related theory" paragraph on [CMP] and [LA].
- **Lacks / unknown:** I could not check which of its chapters cover source coding (expected distortion) or broadcast HARQ.

**1.3 Tian, Steiner, Shamai and Diggavi: successive refinement via broadcast (2008).**
- **Citation:** C. Tian, A. Steiner, S. Shamai (Shitz), S. N. Diggavi, "Successive refinement via broadcast: Optimizing expected distortion of a Gaussian source over a Gaussian fading channel," IEEE Trans. Inf. Theory, vol. 54, pp. 2903–2918, 2008 — [EPFL Infoscience](https://infoscience.epfl.ch/record/160848?ln=fr); [Technion PDF](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_665-1.pdf). The issue (no. 7, July 2008) and the DOI are unconfirmed.
- **Setting:**
  - Gaussian source, MSE distortion, slowly fading Gaussian channel.
  - CSI is known at the receiver but not at the transmitter.
  - The source is successive-refinement (SR) coded and sent with the broadcast strategy, so each SR layer is one superposition layer — [EPFL](https://infoscience.epfl.ch/record/160848?ln=fr).
- **Main results:** An algorithm that finds the optimal power allocation across layers in linear time O(M) for M discrete fading states, and the optimal power allocation for a continuum of fading states by the classical variational method — [EPFL](https://infoscience.epfl.ch/record/160848?ln=fr).
- **Relates to:**
  - [LA]: expected distortion is a probability-weighted sum, over fading states, of the distortion at the total rate of the decoded layers.
  - [DP]: "what a phase is worth" given the channel law.
- **Lacks vs ours:**
  - There is no CSIT; the whole construction exists because the transmitter does not know the state.
  - It has one block, no slot-by-slot scheduling, no deadline and no multiple users.
  - Layers are decoded or lost, not degraded smoothly.

**1.4 Ng, Gündüz, Goldsmith and Erkip: layered broadcast coding with successive refinement (2007–2009).**
- **Citations:**
  - (a) C. T. K. Ng, D. Gündüz, A. J. Goldsmith, E. Erkip, "Optimal Power Distribution and Minimum Expected Distortion in Gaussian Layered Broadcast Coding with Successive Refinement," arXiv 0705.3099 (2007, revised 2009) — [arXiv](https://arxiv.org/abs/0705.3099v1).
  - (b) An earlier preprint, arXiv 0704.2659 — [arXiv](https://www.arxiv.org/pdf/0704.2659).
  - (c) "Recursive power allocation in Gaussian layered broadcast coding with successive refinement," IEEE ICC 2007 — [Princeton record](https://collaborate.princeton.edu/en/publications/recursive-power-allocation-in-gaussian-layered-broadcast-coding-w/).
  - (d) "Minimum expected distortion in Gaussian layered broadcast coding with successive refinement" (ISIT 2007 per my memory) — [Princeton record](https://collaborate.princeton.edu/en/publications/minimum-expected-distortion-in-gaussian-layered-broadcast-coding-/).
  - The journal version is, from memory, "Distortion minimization in Gaussian layered broadcast coding with successive refinement," IEEE Trans. Inf. Theory 55(11):5074–5086, Nov. 2009 (unconfirmed).
- **Setting:**
  - A transmitter without CSI sends a delay-limited Gaussian source over a slowly fading channel.
  - The source is coded in superimposed layers, each refining the previous one.
  - Expected distortion is minimised by allocating transmit power among the layers — [arXiv 0705.3099](https://arxiv.org/abs/0705.3099v1).
- **Main results:**
  - **Two layers:** it is optimal to give power first to the higher layer, up to a "power ceiling" that depends only on the fading distribution, and the rest to the lower layer — [arXiv 0705.3099](https://arxiv.org/abs/0705.3099v1).
  - **Finite discrete states:** the expected distortion is written as recurrence relations and minimised by applying the two-layer procedure at each step — [Princeton ICC record](https://collaborate.princeton.edu/en/publications/recursive-power-allocation-in-gaussian-layered-broadcast-coding-w/).
  - **Continuum of layers:** the optimal power distribution solves linear differential equations in the fading density (calculus of variations) — [arXiv 0705.3099](https://arxiv.org/abs/0705.3099v1); [arXiv 0704.2659](https://www.arxiv.org/pdf/0704.2659).
  - A search summary of this line says that at high SNR the diversity gains of layering can outweigh the benefit of transmitter CSI — [arXiv 0705.3099](https://arxiv.org/abs/0705.3099v1). This is summary-level and needs checking in the full text.
- **Relates to:**
  - [LA]: the value of each layer.
  - [CMP]: CSIT vs no CSIT.
  - [DP]: the backward recursion over layers resembles our recursion over phases.
- **Lacks vs ours:** The same as 1.3: one block, no CSIT, digital threshold layers, no time scheduling, no deadline across slots, one user. "Delay-limited" here means the source must go in one fading block.

**1.5 Etemadi and Jafarkhani: layered transmission over fading and erasure channels (2007–2008).** Titles and venues come from DBLP; contents were not read.
- **Citations:** all from [DBLP author page](https://dblp.dagstuhl.de/pid/00/5715.html).
  - F. Etemadi, H. Jafarkhani, "A Unified Framework for Layered Transmission Over Fading and Packet Erasure Channels," IEEE Trans. Commun. 56(4):565–573, 2008.
  - "Rate and Power Allocation for Layered Transmission With Superposition Coding," IEEE Signal Process. Lett. 14(11):773–776, 2007. Its conference version is "Optimal Rate and Power Allocation for Layered Transmission with Superposition Coding," DCC 2007.
  - "Joint Source-Channel Coding for Quasi-Static Fading Channels with Quantized Feedback," ISIT 2007, pp. 2241–2245.
- **Not found:** the "Optimal layered transmission over quasi-static fading channels" (ISIT 2006) named in the brief did not appear in the search results. Treat it as uncertain.
- **Relates to:**
  - [LA] and [MAP]: one framework for fading and packet erasure, i.e. layer ↔ protection-level allocation.
  - [CMP]: quantized feedback means partial CSIT, which is our stale-report regime.
- **Lacks (judging by the titles):** slot-by-slot causal scheduling, deadlines across users, analog layers.

**1.6 Gündüz and Erkip: JSCC for MIMO block-fading channels (2008).**
- **Citation:** D. Gündüz, E. Erkip, "Joint Source–Channel Codes for MIMO Block-Fading Channels," IEEE Trans. Inf. Theory, vol. 54, no. 1, pp. 116–134, 2008; arXiv 0711.1161 — [arXiv](https://arxiv.org/pdf/0711.1161); [Imperial Spiral](https://spiral.imperial.ac.uk/entities/publication/b55458c0-5256-493e-8670-60698c553ac0).
- **Setting:**
  - A continuous-amplitude (complex Gaussian) source over an L-block Rayleigh MIMO channel, with CSI only at the receiver.
  - The figure of merit is the distortion exponent (the decay rate of expected distortion with SNR) as a function of the bandwidth ratio — [arXiv](https://arxiv.org/pdf/0711.1161).
- **Main results:**
  - Separation is not optimal because the channel is not ergodic.
  - It gives upper and lower bounds on the exponent. The lower bounds come from layered source coding combined with progressive (time-layered), superposition, or hybrid digital/analog transmission.
  - The upper bound is achieved with one degree of freedom (L = 1 and min{Mt, Mr} = 1). Otherwise the achievable exponents meet the bound only for some ranges of the bandwidth ratio — [arXiv](https://arxiv.org/pdf/0711.1161).
- **Relates to:**
  - [MAP]: "progressive transmission" (layers in separate time portions or blocks) is the closest IT analogue of "one phase per slot".
  - [LA].
  - [CMP]: HDA (with an analog component) vs pure digital layering.
- **Lacks vs ours:**
  - Layers are placed in blocks without knowing the blocks' SNRs.
  - The results are high-SNR exponents, not finite-SNR policies.
  - It has no deadline coupling users.

**1.7 Bhattad, Narayanan and Caire: distortion exponents of layered schemes (2008).**
- **Citation:** K. Bhattad, K. R. Narayanan, G. Caire, "On the Distortion SNR Exponent of Some Layered Transmission Schemes," IEEE Trans. Inf. Theory (2008), DOI 10.1109/TIT.2008.924703; arXiv cs/0703035 — [arXiv](https://arxiv.org/abs/cs/0703035). From memory, vol. 54, no. 7, pp. 2943–2958 (unconfirmed).
- **Setting:** A complex Gaussian source over block-fading MIMO (M transmit, N receive antennas), coding across L blocks — [arXiv](https://arxiv.org/abs/cs/0703035).
- **Main results:**
  - A broadcast strategy with a new power and rate allocation achieves the optimal distortion-SNR exponent over a range of bandwidth efficiencies, the first characterisation in part of that range. For large bandwidth efficiency it achieves the optimal exponent MNL.
  - A digital layered scheme combining time layering and superposition is at least as good as the best previously known schemes everywhere, and strictly better for some (M, N, b) — [arXiv](https://arxiv.org/abs/cs/0703035).
- **Relates to:**
  - [MAP]: time layering assigns layers to blocks.
  - [CMP].
- **Lacks vs ours:** CSIT, causal finite-SNR policies, deadlines and multiple users.

**1.8 Caire and Narayanan: hybrid digital-analog space-time coding (2007).**
- **Citation:** G. Caire, K. R. Narayanan, "On the distortion SNR exponent of hybrid digital–analog space–time coding," IEEE Trans. Inf. Theory, 2007 (IEEE document 4276911) — [IEEE record mirrored by Infona](https://www.infona.pl/resource/bwmeta1.element.ieee-art-000004276911).
  - Bhattad et al. cite this work as submitted to IEEE Trans. IT in 2006 — [arXiv cs/0703035](https://arxiv.org/abs/cs/0703035).
  - The record does not name the authors. The attribution is mine, from the title and the citation trail.
  - From memory, vol. 53, no. 8, pp. 2867–2878, Aug. 2007 (unconfirmed).
- **Setting:** A real i.i.d. "analog" source over a quasi-static M×N MIMO block-fading channel; the distortion SNR exponent as a function of spectral efficiency — [Infona](https://www.infona.pl/resource/bwmeta1.element.ieee-art-000004276911).
- **Main results:**
  - It derives an upper bound, a lower bound from separated coding, and a tighter lower bound from new hybrid digital-analog space-time codes — [Infona](https://www.infona.pl/resource/bwmeta1.element.ieee-art-000004276911).
  - A later paper attributes to Caire–Narayanan the optimum exponent with perfect CSIT, which serves as the upper bound — [arXiv 0902.3595](https://arxiv.org/pdf/0902.3595).
- **Relates to:** [CMP]. It is the theoretical reason an analog component helps when CSIT is missing.
- **Lacks vs ours:** causal CSIT over many blocks, scheduling, deadlines, users and learned codes.

**1.9 Laneman, Martinian, Wornell and Apostolopoulos: source-channel diversity (2005).**
- **Citation:** J. N. Laneman, E. Martinian, G. W. Wornell, J. G. Apostolopoulos, "Source-Channel Diversity for Parallel Channels," IEEE Trans. Inf. Theory, vol. 51, no. 10, pp. 3518–3539, Oct. 2005; arXiv cs/0412113 — [Notre Dame page](https://wireless.nd.edu/publications/source-channel-diversity-for-parallel-channels/); [arXiv](https://arxiv.org/abs/cs/0412113).
- **Setting:** It compares three ways to exploit parallel channels: channel coding across the channels at the physical layer, multiple-description (MD) source coding, and MD with joint source-channel decoding — [arXiv](https://arxiv.org/abs/cs/0412113).
- **Main results:**
  - It introduces the distortion exponent.
  - On on-off channels, MD source diversity wins. On continuous-state channels (AWGN with Rayleigh fading), physical-layer coding across channels wins.
  - MD with joint source-channel decoding matches the physical-layer exponent on continuous channels and keeps the on-off advantage — [arXiv](https://arxiv.org/abs/cs/0412113).
- **Relates to:** [CMP]. Which architecture wins depends on whether a block's badness is binary (an erasure or outage, like a HARQ-protected digital packet) or graded (like an analog code).
- **Lacks vs ours:** CSIT, scheduling, deadlines and linear/analog coding.

**1.10 Estella Aguerri and Gündüz: time-varying channel and side information (peripheral).**
- arXiv 1312.0932 adds receiver side information that varies over time. It finds that separate source and channel coding is suboptimal in general when both the channel state and the side-information state vary — [EmergentMind summary of 1312.0932](https://api.emergentmind.com/papers/1312.0932).
- From memory, its title is "Joint source-channel coding with time-varying channel and side-information" by I. Estella Aguerri and D. Gündüz (unconfirmed).
- A companion paper is "Distortion Exponent in MIMO Fading Channels with Time-Varying Source Side Information," arXiv 1409.0494 — [arXiv](https://arxiv.org/pdf/1409.0494).
- **Relates to:** [LA], peripherally.
- **Lacks:** CSIT scheduling and deadlines.

### Inferences
- **The broadcast approach is the answer to the wrong CSI regime.** It is the information-theoretic solution when the transmitter knows only the fading law. Our transmitter knows the slot SNR when the slot starts. So the theory closest to our per-user problem is adaptation with causal CSIT (section 4). The broadcast approach becomes relevant only for stale reports (Steiner–Shamai partial CSIT, section 2; Etemadi–Jafarkhani quantized feedback, 1.5).
- **The layer abstraction in this literature is multiplicative, ours is additive.** A layer counts only if it is decoded, and decoding is a mutual-information threshold event. Distortion is then a function of the total decoded rate; for a successively refinable Gaussian source this is σ²·2^(−2ΣR_i/b), the standard Equitz–Cover/Rimoldi fact. So each layer multiplies the distortion (it adds in log-distortion). Our phases add sensitivity-weighted MMSE terms. This difference in form is what separates our [LA] from everything in this section.
- **Gündüz–Erkip's "progressive" scheme is the nearest structure to ours,** and the time layering of Bhattad et al. is similar: one layer per time portion or block. Their analysis is high-SNR and has no CSIT. Our problem is the finite-SNR, causal-CSIT version, with the layer sizes fixed by a learned code. Only the assignment of phases to slots, and the number of phases, are free; these papers instead optimise the power split.
- **The O(M), recursive and variational algorithms (Tian et al.; Ng et al.) optimise power inside one block.** They have no direct counterpart in our problem, because the phase sizes and power are fixed. Their value to us is the precedent that "expected distortion of a successive-refinement code = Σ_states P(state)·D(decoded prefix)".

### Gaps
- None of the full texts was read. The issue numbers and DOIs of Tian et al., Ng et al. (TIT version), Caire–Narayanan and Shamai–Steiner are unconfirmed.
- Etemadi–Jafarkhani's ISIT 2006 paper was not found.
- I found no broadcast-approach or distortion-exponent paper that assigns layers to several blocks with causal CSIT (each block's state known before transmitting in it). This rests on the searches above, which were not exhaustive.
- Other candidates I know of but could not verify (all unconfirmed):
  - Sesia, Caire and Vivier, a lossy slow-fading comparison of progressive, superposition and hybrid schemes (ISIT 2005). A EURECOM PDF named "cm-sesist-060904" appeared in a result list, but its title was not shown.
  - Zachariadis, Honig and Katsaggelos, "Source fidelity over fading channels: erasure vs scalable codes" (IEEE TCOM 2008).
  - Mittal and Phamdo, HDA codes (IEEE Trans. IT 2002).

## 2. JSCC with ARQ/HARQ or feedback under delay constraints; any theory comparing analog/uncoded with digital-plus-HARQ under outdated CSI

### Takeaway
Theory exists for four nearby problems:
- Digital JSCC with ARQ under delay constraints. Holliday–Goldsmith–Poor work on the diversity–multiplexing–delay tradeoff and show that a DP is needed once a source delay constraint is imposed.
- Layered digital codes with per-layer HARQ (Steiner–Shamai broadcast HARQ).
- HARQ throughput with ideal codes (Caire–Tuninetti).
- Partial CSIT in the broadcast approach (Steiner–Shamai).

I found no theoretical or simulated comparison of an analog or linear code against digital IR-HARQ under outdated CSI. The theory supporting our 0.5 dB vs 1.9 dB result is indirect: without reliable CSIT, graded codes (layered, HDA, analog) have better distortion exponents than separated fixed-rate codes.

### Cited Findings

**2.1 Holliday, Goldsmith and Poor: "robust or quick?" (2008).**
- **Citation:** T. Holliday, A. J. Goldsmith, H. V. Poor, "Joint Source and Channel Coding for MIMO Systems: Is it Better to be Robust or Quick?" IEEE Trans. Inf. Theory, vol. 54, no. 4, pp. 1393–1405, Apr. 2008; arXiv 0801.3702 — [arXiv](https://arxiv.org/pdf/0801.3702); [Princeton record](https://collaborate.princeton.edu/en/publications/joint-source-and-channel-coding-for-mimo-systems-is-it-better-to-/).
- **Setting:**
  - A vector quantizer concatenated with a MIMO space-time channel code.
  - The framework trades off diversity, multiplexing and delay (ARQ) to minimise end-to-end distortion — [arXiv](https://arxiv.org/pdf/0801.3702).
- **Main results:**
  - It gives closed-form high-SNR results for the distortion-optimal operating point, and convex optimisation at finite SNR.
  - Without a source delay constraint, distortion is minimised by maximising the ARQ window.
  - With a source delay constraint, the problem needs a DP that captures the diversity-multiplexing tradeoff as well as random arrivals and retransmissions. A delay-sensitive system gains by adapting its operating point on the diversity-multiplexing-delay region to the system dynamics — [Princeton record](https://collaborate.princeton.edu/en/publications/joint-source-and-channel-coding-for-mimo-systems-is-it-better-to-/); [arXiv](https://arxiv.org/pdf/0801.3702).
  - A search summary adds that the paper gives the optimal high-SNR distortion exponent for the ARQ-enhanced tradeoff, a regime where retransmissions become rare — [arXiv](https://arxiv.org/pdf/0801.3702) (summary-level).
- **Relates to:**
  - [DP]: the IT precedent for "retransmissions plus a deadline require a DP over the distortion objective".
  - [CMP]: the digital side.
- **Lacks vs ours:**
  - The code is digital (a quantizer plus channel code at a DMT operating point); there is no analog or smooth-degradation code.
  - Its DP state is a queue, arrival and ARQ state, not accumulated distortion.
  - It covers a single link, not several users sharing slots.
  - It is high-SNR and asymptotic, not finite-SNR with stale CSI.

**2.2 Steiner and Shamai: multi-layer broadcasting HARQ.**
- **Citation:** A. Steiner, S. Shamai (Shitz), "Multi-Layer Broadcasting Hybrid-ARQ Strategies for Block Fading Channels," Technion CCIT Report #660, Nov. 2007 (draft dated 31 Aug. 2007) — [Technion PDF](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_660-1.pdf); [abstract](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_abstract_660-1.pdf).
  - From memory, the journal version is IEEE Trans. Wireless Commun., vol. 7, no. 7, pp. 2640–2650, July 2008 (**unconfirmed**; the search could find only the report).
- **Setting:**
  - Block fading, no CSIT.
  - Each code layer of a broadcast (superposition) code runs its own HARQ process, so each block retransmits only the layers not yet decoded.
  - It contrasts this with conventional HARQ, which targets throughput at the cost of latency — [Technion PDF](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_660-1.pdf).
- **Main results:** It analyses broadcast chase combining (BCC) and broadcast incremental redundancy (BIR), with infinitely many layers and with finite-level coding. With continuous broadcasting, BCC closely approximates BIR, using a sub-optimal power distribution — [Technion PDF](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_660-1.pdf).
- **Relates to:**
  - [CMP]: the strongest digital competitor in theory. Layered digital with per-layer IR is the digital analogue of our prefix code with per-slot phases.
  - [LA]: per-layer accumulation.
- **Lacks vs ours:**
  - The objective is rate or throughput, not image distortion.
  - There is no causal or stale CSIT, no deadline across users, and no analog layers.

**2.3 Park, Simeone, Sahin and Shamai: multi-layer HARQ for an out-of-band relay (peripheral).**
- **Citation:** S.-H. Park, O. Simeone, O. Sahin, S. Shamai, "Multi-Layer Hybrid-ARQ for an Out-of-Band Relay Channel," arXiv 1303.4782 (2013; a PIMRC 2013 version is on Simeone's page) — [arXiv](https://arxiv.org/pdf/1303.4782); [KCL PDF](https://www.nms.kcl.ac.uk/osvaldo.simeone/ml%20pimrc%2013.pdf).
- **Content:** It combines multi-layer broadcast HARQ with a relay connected by a limited out-of-band link. It cites A. Steiner, S. Shamai, "Single-user broadcasting protocols over a two-hop relay fading channel," IEEE Trans. Inf. Theory 52(11):4821–4838, Nov. 2006 — [arXiv 1303.4782](https://arxiv.org/pdf/1303.4782).
- **Relates to:** [CMP], peripherally. It shows broadcast HARQ is an active line of work.

**2.4 Caire and Tuninetti: the information-theoretic HARQ model (2001).**
- **Citation:** G. Caire, D. Tuninetti, "The throughput of hybrid-ARQ protocols for the Gaussian collision channel," IEEE Trans. Inf. Theory 47(5):1971–1988, July 2001, DOI 10.1109/18.930931 — [EURECOM](https://www.eurecom.fr/en/node/4906931); [ResearchGate PDF](https://researchgate.net/profile/Daniela-Tuninetti/publication/3080461_The_throughput_of_Hybrid-ARQ_protocols_for_the_Gaussian_collision_channel/links/00b49517819029e399000000/The-throughput-of-Hybrid-ARQ-protocols-for-the-Gaussian-collision-channel.pdf).
- **Setting and results:**
  - It analyses HARQ over a slotted multiple-access Gaussian channel with fading, and studies throughput and average delay.
  - It gives closed-form throughput formulas from the renewal–reward theorem.
  - ARQ systems are not interference-limited even without multiuser detection, unlike conventional CDMA — [EURECOM](https://www.eurecom.fr/en/node/4906931).
  - Its use of Gaussian random coding and typical-set decoding makes the results a limiting performance independent of the code — [ar5iv of a citing paper](https://ar5iv.labs.arxiv.org/html/0712.4135).
- **From memory, not confirmed in this session:** IR decoding succeeds once the accumulated mutual information over rounds exceeds the rate.
- **Relates to:**
  - [CMP]: the information-theoretic model of our digital baseline's IR-HARQ.
  - [LA] and [DP]: digital HARQ's sufficient statistic is accumulated MI.
- **Lacks:** source distortion, analog codes and deadlines per image.

**2.5 Tuninetti and Caire: delay constraint and causal feedback in multi-access block fading (2001).**
- **Citation:** D. Tuninetti, G. Caire, "The effect of delay constraint and causal feedback on the wideband performance of multi-access block fading channels," Asilomar 2001 — [EURECOM](https://www.eurecom.fr/fr/publication/784).
- **Content:** It extends the analysis of delay-constrained systems with causal feedback to the multiuser case — [EURECOM](https://www.eurecom.fr/fr/publication/784).
- **Relates to:** [MU] and [DP], with causal CSI and a deadline.
- **Lacks:** a distortion objective.

**2.6 Steiner and Shamai: broadcast approach with imperfect CSIT.**
- **Citation:** A. Steiner, S. Shamai (Shitz), "Achievable rates with imperfect transmitter side information using a broadcast transmission strategy," Technion CCIT Report #663, Nov. 2007; DBLP lists it as a journal article — [Technion PDF](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_663-1.pdf); [DBLP](https://dblp.org/pid/26/538). From memory, IEEE TWC 2008 (unconfirmed).
- **Setting:** The broadcast approach under several models of partial CSIT:
  - quantized fading feedback;
  - a Gaussian mean/variance model of channel-estimation error, which in SISO is equivalent to Ricean fading.
- **Main results:**
  - **Quantized feedback, Rayleigh fading:** with more than one feedback bit the broadcasting gain becomes negligible, because the channel uncertainty shrinks.
  - **Mean/variance model:** it derives maximal broadcasting rates.
  - **MISO:** it derives an optimal power allocation, and uniform power beats beamforming where the broadcasting gain is non-negligible — [Technion PDF](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_663-1.pdf).
- **Relates to:** [CMP] and [DP] under stale reports. It is the only theory found on how much "graceful degradation" is worth as CSIT quality varies.
- **Lacks:** a source-distortion objective, analog codes and scheduling.

**2.7 Taherzadeh and Poor: limits on the robustness of MIMO JSCC.**
- **Citation:** M. Taherzadeh, H. V. Poor, "Limits on the Robustness of MIMO Joint Source-Channel Codes," arXiv 0910.5950 (venue unconfirmed) — [arXiv](https://arxiv.org/pdf/0910.5950); [Princeton record](https://collaborate.princeton.edu/en/publications/limits-on-the-robustness-of-mimo-joint-source-channel-codes/).
- **Main result:** There is a considerable gap between the diversity–fidelity tradeoff of robust JSCC codes and the optimum tradeoff without the robustness constraint — [arXiv](https://arxiv.org/pdf/0910.5950).
- **Relates to:** [CMP]. Robustness (an encoder that does not depend on the SNR) has a provable cost, which is the price our encode-once code pays when CSI is good.
- **Lacks:** scheduling and HARQ.

**2.8 Joda, Lahouti and Erkip: delay–distortion–power tradeoffs with buffering.**
- **Citation:** R. Joda, F. Lahouti, E. Erkip, "Delay-Distortion-Power Trade Offs in Quasi-Stationary Source Transmission over Block Fading Channels," arXiv 1505.00651 (venue unconfirmed) — [ar5iv](https://ar5iv.labs.arxiv.org/html/1505.00651).
- **Content:** Buffering techniques for quasi-stationary sources over block fading, with four coding schemes designed to minimise reconstruction distortion under buffer and power constraints — [ar5iv](https://ar5iv.labs.arxiv.org/html/1505.00651).
- **Related titles found (authors not confirmed):**
  - "Delay-limited Source and Channel Coding of Quasi-Stationary Sources over Block Fading Channels: Design and Scaling Laws," arXiv 1202.6175 — [arXiv](https://arxiv.org/pdf/1202.6175).
  - "On the End-to-End Distortion for a Buffered Transmission over Fading Channel," arXiv 0706.0225 — [arXiv](https://arxiv.org/pdf/0706.0225).
- **Relates to:** [DP], as distortion–delay tradeoffs over block fading.
- **Lacks:** per-slot scheduling across users, analog codes and HARQ.

### Inferences
- **Theory predicts the direction of our result, not its size.** Our measurement: at report correlation 0.8, JSCC loses 0.5 dB and digital with rate adaptation and IR-HARQ loses 1.9 dB. The exponent results above (1.6–1.9) say that when the transmitter does not know the state, codes whose quality degrades in grades beat a separated fixed-rate code, which suffers outage and needs extra rounds. Steiner–Shamai (2.6) add that the value of graded layering shrinks quickly as CSIT improves (one feedback bit already suffices in Rayleigh fading). A stale report at correlation 0.8 is partial CSIT, the regime where gracefulness matters.
- **Reviewers may ask for a stronger digital competitor.** From theory, the fair digital competitor under stale CSI is layered superposition with per-layer HARQ (Steiner–Shamai broadcast HARQ) or an HDA code, not single-layer IR-HARQ. Cite 2.2 and 1.8, and consider a two-layer superposition digital baseline before submission.
- **Both baselines carry an additive sufficient statistic.** Digital IR-HARQ carries accumulated mutual information, Σ_k ½log(1+s_k) (2.4, from memory). Chase combining carries accumulated SNR. Our DP carries accumulated sensitivity-weighted MMSE, Σ_j ρ_j φ(n_j). This is a clean way to present our DP state: it is the analog code's counterpart of HARQ's accumulated MI.
- **Why stale CSI hurts the digital code more (my reasoning, not from a source).**
  - With a stale report, our transmitter's expected cost of a slot is ρ·E[φ(n) | report]. The cost is linear in φ, so the report enters only through one conditional mean.
  - For digital HARQ, success is a threshold event on accumulated MI. Its expected value is a nonlinear function of the posterior, and a mispredicted outage costs a whole round in a deadline-limited window.
  - This is consistent with our larger digital loss, but no paper found proves it.
- **Holliday et al. sit closest on the digital side:** distortion objective, ARQ, delay constraint, DP. They differ on every axis that defines our problem: code type, CSIT model, users, finite SNR.

### Gaps
- No theoretical comparison of analog or uncoded codes against digital-with-HARQ under outdated CSI was found. The supporting evidence above is indirect and at exponent level.
- Not verified this session, so treat as candidates (from memory, unconfirmed):
  - El Gamal, Caire and Damen, "The MIMO ARQ channel: diversity–multiplexing–delay tradeoff" (IEEE Trans. IT 2006), which Holliday et al. build on.
  - Schalkwijk and Kailath (1966); Gallager and Nakiboğlu, "Variations on a theme by Schalkwijk and Kailath" (IEEE Trans. IT 2010). These cover linear (analog) retransmission with feedback. That retransmission refines the estimation error, giving multiplicative distortion, unlike our additive phases.
  - Kostina, Polyanskiy and Verdú, "Joint source-channel coding with feedback" (IEEE Trans. IT 2017).
  - Kittipiyakul, Elia and Javidi, "High-SNR analysis of outage-limited communications with bursty and delay-limited information" (IEEE Trans. IT 2009).
  - Tuninetti, partial CSI for repetition protocols (IEEE Trans. IT 2011).
  - Makki and Eriksson on HARQ with outdated or quantized CSI.
  - Goblick (1965) and Gastpar, Rimoldi and Vetterli, "To code or not to code" (IEEE Trans. IT 2003). These establish that uncoded transmission is optimal for a bandwidth-matched Gaussian source over AWGN.
- Also unconfirmed: a search summary said Kumar and Caire (IEEE Trans. IT 2011) apply the distortion-exponent framework to the MIMO-MAC. arXiv 1005.1594, "Channel State Feedback over the MIMO-MAC", may be that paper; I could not confirm.
- The search budget ran out before these could be checked.

## 3. Multi-user: sharing a fading channel among several sources to minimise total expected distortion before a deadline; optimal-policy structure

### Takeaway
Two targeted searches found no information-theory paper that schedules several sources slot by slot on a shared fading channel to minimise total expected distortion before a deadline. None derives an index or threshold structure for such a problem. The closest IT-flavoured models are:
- energy minimisation for N users with a common deadline, solved by DP (Tarello–Sun–Zafer–Modiano);
- a DP for power control in a two-user broadcast channel with causal CSI and stringent delay (ICC 2002);
- multi-access block fading with a delay constraint and causal feedback (Tuninetti–Caire).

The machinery our scheduler uses (Lagrangian relaxation, Whittle and LP indices) comes from operations research, which `SURVEY.md` F2/F5 already covers, not from JSCC theory.

### Cited Findings

**3.1 Tarello, Sun, Zafer and Modiano: minimum-energy scheduling with a common deadline (2005).**
- **Citation:** A. Tarello, J. Sun, M. Zafer, E. Modiano, "Minimum energy transmission scheduling subject to deadline constraints," WiOpt 2005, DOI 10.1109/WIOPT.2005.28 — [EUDL](https://eudl.eu/doi/10.1109/WIOPT.2005.28).
  - A later journal version appears to be in Springer *Wireless Networks* (2008), DOI 10.1007/s11276-007-0093-y — [DOI](https://www.doi.org/10.1007/S11276-007-0093-Y). The link between the two is from a catalogue entry and is unconfirmed.
- **Setting:**
  - N users, each with a fixed amount of data due by a common deadline, over a fading channel.
  - The fade state sets each user's throughput per unit energy, and the goal is minimum expected total energy.
  - It treats linear and strictly convex rate–power curves — [EUDL](https://eudl.eu/doi/10.1109/WIOPT.2005.28).
- **Main results:** Optimal solutions by DP plus tractable heuristics. For a non-fading channel with a convex rate–power curve, the optimum is a shortest-path problem — [EUDL](https://eudl.eu/doi/10.1109/WIOPT.2005.28).
- **Relates to:** [MU] and [DP]. It is multi-user, deadline-constrained scheduling over fading.
- **Lacks vs ours:** The objective is energy for fixed bits, not distortion. It has no learned code, no phase structure and no stale CSI, and the abstract mentions no Lagrangian decomposition or index policy.

**3.2 "Optimal power allocation over fading channels with stringent delay constraints" (IEEE ICC 2002; authors not confirmed).**
- **Source:** [Princeton record](https://collaborate.princeton.edu/en/publications/optimal-power-allocation-over-fading-channels-with-stringent-dela/).
- **Content:** With only causal CSI the power-allocation problem becomes harder and is solved by DP. Constant power is optimal in the high-SNR limit. The power control problem is also solved by DP for a two-user broadcast channel — [Princeton record](https://collaborate.princeton.edu/en/publications/optimal-power-allocation-over-fading-channels-with-stringent-dela/) (summary-level).
- **Relates to:** [DP] and [MU], with causal CSI and a deadline.
- **Lacks:** a distortion objective, a learned code and many users.

**3.3 Tuninetti and Caire (Asilomar 2001), item 2.5.** It extends delay-constrained causal-feedback analysis to multi-access block fading — [EURECOM](https://www.eurecom.fr/fr/publication/784).

**3.4 Tian, Chen and Diggavi: multiuser successive refinement (wireline).**
- **Citation:** C. Tian, J. Chen, S. N. Diggavi, "Multiuser Successive Refinement and Multiple Description Coding," arXiv 0707.4133 (venue unconfirmed) — [arXiv](https://arxiv.org/pdf/0707.4133).
- **Setting:** Users linked to a server by links of different capacities, each reconstructing the same source scalably.
- **Main results:**
  - An achievable region for two users and two layers, which gives the complete rate-distortion region for a Gaussian source under MSE.
  - With multiple users, the loss of multistage (successive-refinement) coding can be unbounded for the Gaussian source — [arXiv](https://arxiv.org/pdf/0707.4133).
- **Relates to:** [MU], loosely. It concerns one layered source serving many users; we have many users with separate images.
- **Lacks:** fading, scheduling and deadlines.

**3.5 Alay and Argyriou (peripheral; details uncertain).** A repository record describes optimum transmission of correlated Gaussian sources that allows opportunistic interference to minimise expected distortion, with successive interference cancellation at the destination — [University of Thessaly repository](https://ir.lib.uth.gr/xmlui/handle/11615/70400). Which of the two repository handles shown holds this paper, and its venue, are unconfirmed.

**3.6 Negative results of the searches.**
- A search for "opportunistic scheduling to minimise expected distortion, multiple users, Gaussian sources, fading" returned only single-user layered-broadcast and interference-channel work — [search pointer: Ng et al.](https://arxiv.org/pdf/0705.3099v1).
- A search for "multiuser diversity scheduling, Gaussian source distortion, successive refinement, fading downlink" returned only Tian–Chen–Diggavi (wireline) and Ng et al. — [arXiv 0707.4133](https://arxiv.org/pdf/0707.4133).

### Inferences
- **Our [MU] component has no precedent in IT JSCC theory.** The IT JSCC literature in sections 1–2 is single-link, or sends one source to many receivers. "Multiuser diversity for source transmission" does not appear to be a body of theory, judging from two searches.
- **The nearest optimal-policy structure is not in IT.** Index and threshold results for deadline-constrained multi-user scheduling come from operations research: Derman–Lieberman–Ross intervals (section 4); Whittle and Lagrangian indices and Brown–Smith (SURVEY F2/F5).
- **Tarello et al. are the IT-side multi-user deadline DP.** Positioned against them, we change the reward from energy-for-bits to calibrated image distortion, and replace a joint DP with a per-user DP plus a price.
- **The users' coupling is a slot-budget constraint, so the relaxation is classical.** A unit-per-slot constraint relaxed with a price makes each user's problem the single-user DP of section 4 with a per-slot charge. That is the weakly coupled DP decomposition (Hawkins; Adelman–Mersereau; SURVEY F2) applied to a JSCC reward. IT supplies the reward (the MMSE abstraction), not the decomposition.

### Gaps
- Classical multiuser-diversity and utility-scheduling works could not be verified this session (from memory, unconfirmed): Knopp and Humblet (ICC 1995); Viswanath, Tse and Laroia (IEEE Trans. IT 2002); Liu, Chong and Shroff's opportunistic-scheduling framework (2003); Stolyar's gradient scheduling (Oper. Res. 2005); Agrawal and Subramanian (Allerton 2002).
- Pahalawatta, Berry, Pappas and Katsaggelos, content-aware video scheduling (IEEE JSAC 2007), is applied work and also unverified.
- Whittle-index work on remote estimation and MSE-driven multi-source scheduling was not searched, because the budget ran out. It may hold the only index-structure results with an MSE reward.

## 4. Optimal transmission of a source over several fading blocks with causal CSI (single-user DP) and sequential stochastic assignment (layers to blocks)

### Takeaway
The single-user template exists in the deadline-scheduling literature:
- finite-horizon DPs over block fading in which the current state is known before transmitting (Fu–Modiano–Tsitsiklis; Negi–Cioffi, as cited by Lee–Jindal; Lee–Jindal);
- a distortion-minimising, deadline-constrained version with separation-based coding and energy limits (Orhan–Gündüz–Erkip).

Derman–Lieberman–Ross's sequential stochastic assignment gives the exact structure of phase-to-slot mapping when the phase order is free. The optimal rule uses intervals (thresholds) that do not depend on the phase weights.

### Cited Findings

**4.1 Fu, Modiano and Tsitsiklis: scheduling over fading with energy and deadline constraints (2006).**
- **Citation:** A. Fu, E. Modiano, J. N. Tsitsiklis, "Optimal Transmission Scheduling over a Fading Channel with Energy and Deadline Constraints," IEEE Trans. Wireless Commun., vol. 5, no. 3, Mar. 2006; conference version at CISS 2002 — [MIT PDF of the CISS version](https://stuff.mit.edu/people/jnt/Papers/C-02-fu-wrlss-CISS.pdf). The journal details come from the search summary.
- **Setting:** One transmitter with limited energy and time sends data over a fading channel, and maximises expected throughput.
- **Main results:**
  - The problem is solved by DP, first when the current channel state is known before transmission, then when it is learned only afterwards and follows a Markov process.
  - It extends to minimising the expected energy needed to send a fixed amount of data by a deadline — [MIT PDF](https://stuff.mit.edu/people/jnt/Papers/C-02-fu-wrlss-CISS.pdf).
  - Lee and Jindal describe it as a finite-horizon DP for one packet over block fading with a hard deadline — [Lee–Jindal, arXiv 0807.3332](https://arxiv.org/pdf/0807.3332).
  - A third-party summary claims a closed-form optimal schedule. The abstract says only "optimal transmission schedule", so this is unconfirmed.
- **Relates to:** [DP]. It is the structural template: finite horizon, deadline, block fading, state known at the start of the slot (our perfect-CQI case) or later (our stale case).
- **Lacks vs ours:**
  - The reward is bits per energy, not distortion.
  - Its state is (time, remaining energy or bits); ours is (time, phases held, accumulated weighted noise).
  - It has one user and no learned code.

**4.2 Negi and Cioffi (via Lee–Jindal) and Lee–Jindal (2008).**
- **Negi–Cioffi as described by Lee–Jindal:** they "studied the dual problem of maximizing the expected number of transmitted bits in a finite number of slots subject to a finite energy constraint". For more than two slots the optimal policy must generally be found numerically — [J. Lee, N. Jindal, "Energy-efficient Scheduling of Delay Constrained Traffic over Fading Channels," arXiv 0807.3332](https://arxiv.org/pdf/0807.3332).
- **Negi–Cioffi bibliographic details (from memory, unconfirmed):** R. Negi, J. M. Cioffi, "Delay-constrained capacity with causal feedback," IEEE Trans. Inf. Theory 48(9), Sept. 2002. Two searches did not surface the paper itself.
- **Lee–Jindal venue:** IEEE TWC 2009 (from memory, unconfirmed).
- **Relates to:** [DP]. Causal CSI, a finite number of slots and DP, with no closed form beyond two slots, which matches our need for a numerical DP.
- **Lacks:** distortion, phases and users.

**4.3 Causal-CSI delay-constrained DPs in other settings.**
- **ICC 2002 stringent-delay paper (item 3.2):** DP under causal CSI; constant power optimal at high SNR — [Princeton record](https://collaborate.princeton.edu/en/publications/optimal-power-allocation-over-fading-channels-with-stringent-dela/).
- **Li and Dey (2008, 2010):** DP for a K-block delay-constrained problem with causal channel feedback in a cooperative relay. The 2008 paper maximises expected rate; the 2010 paper (European Trans. Telecommun.) minimises outage — [Maynooth repository](https://mural.maynoothuniversity.ie/id/eprint/14467/). The titles were not shown.
- **An unidentified IEEE letter:** power and rate allocation for expected capacity under causal CSIT. At high and low SNR the optimal policy resembles water-filling over the current channel and estimates of future channels, and causal CSIT matters when capacity is delay-constrained — [Infona record 4020529](https://www.infona.pl/resource/bwmeta1.element.ieee-art-000004020529). Authors and year not identified.

**4.4 Orhan, Gündüz and Erkip: source-channel coding under energy, delay and buffer constraints (2015).**
- **Citation:** O. Orhan, D. Gündüz, E. Erkip, "Source-Channel Coding under Energy, Delay and Buffer Constraints," IEEE Trans. Wireless Commun. 14(7):3836–3849, 2015; arXiv 1408.2302 — [arXiv](https://arxiv.org/pdf/1408.2302).
- **Setting:**
  - An energy-limited sensor observes independent Gaussian samples whose variances change over time slots, and sends them over a flat-fading channel.
  - Compressed samples sit in a finite buffer and must be delivered within d slots.
  - The goal is minimum average distortion. It covers a finite battery, energy harvesting, and the energy cost of processing and sampling — [arXiv](https://arxiv.org/pdf/1408.2302).
- **Main result:** For the strict-delay case (d = 1), a two-dimensional water-filling interpretation — [arXiv](https://arxiv.org/pdf/1408.2302).
- **Relates to:** [DP]. It is the closest IT single-user problem with **distortion + deadline + fading**.
- **Lacks vs ours:** It is separation-based (compression into a buffer, then channel coding); it has no analog code, no phases and no users; the constraint is energy rather than slots. Whether it treats causal (online) CSI explicitly was not confirmed.

**4.5 Derman, Lieberman and Ross: sequential stochastic assignment (1972; already in SURVEY F2, details added here).**
- **Citation:** C. Derman, G. J. Lieberman, S. M. Ross, "A Sequential Stochastic Assignment Problem," *Management Science* 18(7):349–355, 1972 — [IDEAS/RePEc](https://ideas.repec.org/a/inm/ormnsc/v18y1972i7p349-355.html). DOI 10.1287/mnsc.18.7.349 is unconfirmed.
- **Model:**
  - n jobs arrive in sequence, each with a random value x.
  - n men have known "success probabilities" p, and a man assigned to a job is unavailable afterwards.
  - The reward of assigning job x to man i is p_i·x, and the goal is maximum expected total reward.
- **Main theorem:** The optimal policy is defined by non-overlapping intervals of the real line. The intervals depend on n and on the distribution of job values but **not on the p's** — [IDEAS/RePEc](https://ideas.repec.org/a/inm/ormnsc/v18y1972i7p349-355.html).
  - My recollection of the rule (unconfirmed): if the arriving value falls in the i-th interval, the job goes to the man with the i-th smallest p. The thresholds follow from an expectation recursion over the number of remaining jobs.
- **Related papers:**
  - The same authors' "A stochastic sequential allocation model" (title from memory), *Operations Research* 23(6):1120–1130, 1975. A fixed budget is split across randomly arriving investment opportunities — [IDEAS/RePEc](https://ideas.repec.org/a/inm/oropre/v23y1975i6p1120-1130.html).
  - "A Unified Approach for Solving Sequential Selection Problems," arXiv 1901.04183 (authors unconfirmed). It lists applications including machine scheduling, packet switching and kidney allocation — [arXiv](https://arxiv.org/pdf/1901.04183).
  - "The Stochastic Sequential Assignment Problem With Random Deadlines" (title only; authors and venue unconfirmed) — [academia.edu](https://www.academia.edu/126220778/The_Stochastic_Sequential_Assignment_Problem_With_Random_Deadlines).
  - A search for a wireless application of the DLR model found none — [search pointer](https://ideas.repec.org/a/inm/ormnsc/v18y1972i7p349-355.html).
- **Relates to:** [MAP] directly, and to [DP] (see Inferences).
- **Lacks:** It has no prefix (ordering) constraint, no choice of how many phases to send, no price on slots and no multiple users.

### Inferences
- **Our per-user problem without a price and with free phase order is exactly DLR (my mapping, not a source's).** Take a fixed number of phases L, price λ = 0, and phases that may go into any slot (the decoder knows which phase came in which slot). Then the problem is DLR with:
  - jobs = the T slots, valued x = −φ(n) (or −E[φ(n) | report] under stale CQI, since the cost is linear in φ);
  - men = the L phases, weighted p_j = ρ_L[j];
  - T − L dummy men of weight 0, which stand for skipped slots.

  DLR's theorem then gives:
  - The optimal online mapping is an interval rule on the slot's φ(n) that does not depend on the sensitivities ρ. The ρ's only fix the ranking: the most sensitive phase gets the best interval.
  - Skipping is the bottom interval (the dummies).
  - Ex post, the best mapping sorts phases by ρ and slots by φ (the rearrangement inequality). This is the ParCast/"sensitivity-ordered mapping" rule from SURVEY F4; DLR is its causal counterpart.
- **What breaks the DLR structure, and why our DP state is needed.** Three features of our problem do:
  - the prefix constraint (the j-th taken slot carries phase j);
  - the choice of L, whose Dsrc(L), C(L) and ρ_L change non-additively with L;
  - a Lagrange price per slot.

  Under the prefix constraint the question becomes "take this slot for the next phase or wait". When ρ decreases with j, this can conflict with DLR's preference to save good slots for sensitive phases. Because the objective is additive given L, (phases held, accumulated Σρφ, time) is a sufficient state.
- **Stale reports shrink the value of mapping.** With stale reports the job-value distribution is that of E[φ(n) | report]. Its spread shrinks as report correlation falls, so the DLR intervals collapse and both mapping and opportunism gain less. This matches the "information limits opportunism" point in SURVEY F5 (Grunewalder and Khaleghi).
- **Fu–Modiano–Tsitsiklis is the template to cite for [DP],** with Negi–Cioffi for causal CSI and Orhan–Gündüz–Erkip for a distortion objective. Ours differs in the objective (calibrated learned-code distortion), the state (accumulated weighted MMSE) and the coupling (price across users).

### Gaps
- None of these papers' full texts was read, so the exact form of the Fu et al. policy (threshold or closed form) is unconfirmed.
- The DLR threshold recursion is from memory.
- Not verified (from memory): Albright (1974) on random arrival times, and Albright and Derman (1972) on asymptotics of the SSAP; Zafer and Modiano on delay-constrained rate control (IEEE Trans. IT 2008 / ToN 2009); Chen, Neely and Mitra on energy-efficient transmission with individual packet deadlines (IEEE Trans. IT 2008); Castiglione, Simeone, Erkip and Zemen on energy-neutral source-channel coding (IEEE TCOM 2012).
- No IT paper was found that solves the single-user "Gaussian source over T blocks with causal CSIT and a deadline, choosing which blocks carry which refinement layer" problem at finite SNR. The theory comes in pieces: no-CSIT layering (section 1), causal-CSI rate scheduling (4.1–4.3) and assignment (4.5).

## 5. Linear analog codes: MSE as a sum of per-component MMSE terms, and an information-theoretic case for an MMSE (not mutual-information) link abstraction

### Takeaway
That the MSE of a linear code over parallel components is a sum of per-component LMMSE terms is classical linear-coding theory (Lee–Petersen 1976). Our φ(n) = n/(1+n/κ) = κn/(κ+n) is exactly the MMSE of estimating a variance-κ Gaussian from an observation in Gaussian noise of variance n.

The I-MMSE identity (Guo–Shamai–Verdú 2005) is the information-theoretic bridge. Digital and HARQ codes are governed by accumulated mutual information; an analog code's distortion is the MMSE itself. That justifies an MMSE, not mutual-information, abstraction for analog codes. I found no paper that states this as a link-abstraction principle or uses it for scheduling, apart from the linear-video systems already in SURVEY F4.

### Cited Findings

**5.1 Lee and Petersen: optimal linear coding for vector channels (1976).**
- **Citation:** K.-H. Lee, D. P. Petersen, "Optimal linear coding for vector channels," IEEE Trans. Commun., vol. 24, no. 12, pp. 1283–1290, 1976 — [reference list of arXiv 1302.3660](https://arxiv.org/pdf/1302.3660).
- **Precursor thesis:** Kyong-Hwa Lee, "Optimal Linear Coding For a Multichannel System," Univ. of New Mexico dissertation, 1975, chaired by Daniel Paul Petersen — [UNM repository](https://digitalrepository.unm.edu/ece_etds/436).
  - The thesis finds the optimal linear coding that matches an r-dimensional vector signal to a k-dimensional channel under a power constraint.
  - The encoder decorrelates the r correlated variables into independent ones, and keeps at most k of them, those with the k largest eigenvalues of the signal covariance — [UNM repository](https://digitalrepository.unm.edu/ece_etds/436).
  - That the thesis and the 1976 paper match exactly is unconfirmed.
- **Relates to:** [LA] and [MAP]. This is the canonical result on linear source-to-channel matching:
  - The distortion of a linear code with a linear MMSE decoder over independent subchannels is the sum of per-component LMMSE errors.
  - The best mapping pairs strong source components with good subchannels.
- **Lacks:** nonlinear (learned) codes, time-varying slots, causal CSI and scheduling.

**5.2 Other linear and zero-delay JSCC work (titles verified; contents not read).**
- "On Zero Delay Source-Channel Coding," arXiv 1302.3660 — [arXiv](https://arxiv.org/pdf/1302.3660). From memory, Akyol, Viswanatha, Rose and Ramstad, IEEE Trans. IT 2014 (unconfirmed).
- "Optimal Linear Joint Source-Channel Coding with Delay Constraint," arXiv 1203.6318 — [arXiv](https://arxiv.org/pdf/1203.6318). Authors unconfirmed.
- A University of Minnesota record whose title is truncated to "Linear joint source channel coding for Gaussian sources through f[ading channels]" — [UMN Experts](https://experts.umn.edu/en/publications/linear-joint-source-channel-coding-for-gaussian-sources-through-f/). Authors and venue unconfirmed; it is likely the most directly relevant (linear JSCC over fading) and should be checked.

**5.3 Guo, Shamai and Verdú: I-MMSE (2005).**
- **Citation:** D. Guo, S. Shamai (Shitz), S. Verdú, "Mutual Information and Minimum Mean-Square Error in Gaussian Channels," IEEE Trans. Inf. Theory 51(4):1261–1282, Apr. 2005; arXiv cs/0412108 — [Princeton record](https://collaborate.princeton.edu/en/publications/mutual-information-and-minimum-mean-square-error-in-gaussian-chan/); [arXiv](https://arxiv.org/abs/cs/0412108).
- **Main results:**
  - The derivative of mutual information (in nats) with respect to SNR equals half the MMSE, whatever the input statistics. This holds for scalar and vector signals and for discrete- and continuous-time non-causal estimation.
  - The causal filtering MMSE at a given SNR equals the non-causal smoothing MMSE averaged over SNRs uniform on [0, SNR] — [arXiv](https://arxiv.org/abs/cs/0412108).
- **Follow-up:** Guo, Wu, Shamai and Verdú study properties of the MMSE, including concavity in the input distribution at any SNR — [arXiv 1004.3332](https://arxiv.org/pdf/1004.3332). From memory, the title is "Estimation in Gaussian noise: properties of the minimum mean-square error" (unconfirmed).
- **Relates to:** [LA] and [CMP]. It is the formal link between an MI-based abstraction (digital codes, HARQ) and an MMSE-based one (analog codes).
- **Lacks:** any application to link abstraction or scheduling.

**5.4 HARQ's mutual-information state.** Caire–Tuninetti (item 2.4) is the information-theoretic HARQ model with ideal codes — [EURECOM](https://www.eurecom.fr/en/node/4906931). The accumulated-MI decoding rule is from memory.

### Inferences (my derivations; no source states them in this form)
- **φ is a Gaussian MMSE.** φ(n) = κn/(κ+n) = κ·mmse_G(κ/n), with mmse_G(s) = 1/(1+s), the normalised MMSE of a Gaussian input at SNR s (the setting of 5.3). So our abstraction is the distortion of a linear code in which phase j's content, weighted by ρ_L[j], rides on the slot that carries it and is decoded by per-slot LMMSE.
  - Additivity across slots follows from independent per-slot noise and a decoder that is block-diagonal across slots: the trace of a block-diagonal error covariance is a sum. This is Lee–Petersen's setting (5.1) with "subchannel" read as "slot".
  - What is not standard is that a learned nonlinear code obeys the same form after calibration, as SURVEY F4 already concluded.
- **I-MMSE explains why the abstraction must be MMSE, not MI, for analog codes.**
  - I(s) = ½∫₀^s mmse(γ)dγ, so MI and MMSE are integral and derivative of one another and rank mixtures of slot SNRs differently.
  - A digital or HARQ code decodes when accumulated MI, Σ½log(1+s_k), exceeds the rate. An analog code's distortion is the (weighted) sum of MMSEs.
  - So MI-type effective-SNR mappings (MIESM or EESM, used for digital link-to-system abstraction; not verified this session) are the wrong functional for an analog code. The natural "effective SNR" for an analog code is the one whose MMSE equals the weighted mean MMSE.
- **Separation beats a linear code over unequal slots when CSI is perfect (the AM–GM gap).**
  - Take K Gaussian samples sent in K real channel uses with SNRs s_k, one deadline window and perfect CSIT.
  - Ideal separation gives D_sep = σ²·Π_k(1+s_k)^(−1/K): the geometric mean of 1/(1+s_k).
  - Uncoded LMMSE gives D_lin = σ²·(1/K)Σ_k 1/(1+s_k): the arithmetic mean.
  - By AM ≥ GM, D_lin ≥ D_sep, with equality only if all s_k are equal. This is the classical reason uncoded transmission is optimal on AWGN but not over varying slots.
  - Consequences for us:
    - (i) The analog code pays a Jensen-type penalty for SNR variation inside its window. This is why phase-to-slot mapping and opportunistic scheduling matter for it.
    - (ii) Digital can collect the geometric-mean advantage only with accurate rate choice. With stale CSI it backs off or suffers outage and HARQ rounds, which gives the robustness asymmetry we measure (0.5 vs 1.9 dB at correlation 0.8).
  - This gives a theory-backed way to state the [CMP] result: digital's ideal is better, but it needs CSI to reach it.
- **HDA refinement multiplies distortion; our phases add to it.** In HDA schemes (1.6, 1.8) the analog part usually sends the quantisation error, so distortion is roughly D_q/(1+s), multiplicative. Linear feedback schemes (Schalkwijk–Kailath type) refine the estimation error round by round, also multiplicative. Our additive form is that of parallel independent components (Lee–Petersen; SoftCast in F4). This distinction is worth one sentence in the paper, because "successive refinement" in IT usually means a multiplicative refinement.

### Gaps
- I found no paper that proposes an MMSE-based link abstraction (an effective-SNR mapping) for analog or uncoded codes and uses it for scheduling, apart from the SoftCast/ParCast systems in SURVEY F4. The searches for this were cut short by the budget.
- The MIESM/EESM link-abstraction references (e.g., Brueninghaus et al., PIMRC 2005) are from memory and unconfirmed.
- From memory and unconfirmed: the Gaussian input is the worst case for MMSE among inputs of the same variance, which would make φ an upper bound for non-Gaussian phase content.

## 6. Synthesis: the closest works and an overlap verdict per contribution

### Takeaway
The five closest works are:
1. Holliday–Goldsmith–Poor 2008;
2. Gündüz–Erkip 2008, with Bhattad–Narayanan–Caire 2008;
3. Ng et al. and Tian et al., 2008–09;
4. Steiner–Shamai broadcast HARQ;
5. Fu–Modiano–Tsitsiklis 2006, with Derman–Lieberman–Ross 1972 as the structural template.

None overlaps the combination. Overlap per contribution is highest for phase-to-slot mapping (DLR) and the per-user DP template (Fu et al.). It is lowest for multi-user distortion scheduling and for the analog-vs-HARQ comparison under stale CSI.

### Cited Findings
1. **Holliday, Goldsmith, Poor, IEEE Trans. IT 54(4):1393–1405, 2008** — [arXiv 0801.3702](https://arxiv.org/pdf/0801.3702).
   - Closest in spirit: a distortion objective, retransmissions and a delay constraint lead to a DP over the operating point.
   - It differs in code (digital VQ plus space-time code), state (queue and ARQ), regime (high SNR) and users (one link).
2. **Gündüz, Erkip, IEEE Trans. IT 54(1):116–134, 2008** — [arXiv 0711.1161](https://arxiv.org/pdf/0711.1161); and **Bhattad, Narayanan, Caire, IEEE Trans. IT 2008, DOI 10.1109/TIT.2008.924703** — [arXiv cs/0703035](https://arxiv.org/abs/cs/0703035).
   - Layered source codes spread over L fading blocks, with a progressive (time-layered) variant and an HDA (analog) variant: the IT version of "one phase per slot".
   - They have no CSIT, use high-SNR exponents and have no users.
3. **Ng, Gündüz, Goldsmith, Erkip** — [arXiv 0705.3099](https://arxiv.org/abs/0705.3099v1); and **Tian, Steiner, Shamai, Diggavi, IEEE Trans. IT 54, 2008** — [EPFL](https://infoscience.epfl.ch/record/160848?ln=fr).
   - The canonical "expected distortion of a successively refined Gaussian source over fading" results.
   - They have no CSIT, one block, power split only, and digital threshold layers.
4. **Steiner, Shamai, broadcast HARQ** — [Technion #660](https://ece.technion.ac.il/wp-content/uploads/2021/01/publication_660-1.pdf).
   - Layered digital code with per-layer HARQ over block fading: the strongest theoretical digital competitor for our [CMP] comparison.
   - Its objective is rate, and it has no CSIT and no distortion.
5. **Fu, Modiano, Tsitsiklis, IEEE TWC 5(3), 2006** — [MIT PDF](https://stuff.mit.edu/people/jnt/Papers/C-02-fu-wrlss-CISS.pdf); with **Derman, Lieberman, Ross, Manag. Sci. 18(7):349–355, 1972** — [IDEAS](https://ideas.repec.org/a/inm/ormnsc/v18y1972i7p349-355.html).
   - The finite-horizon deadline DP over fading with the current state known before transmitting, plus the interval-rule theorem for assigning weighted items to sequential random slots.
   - Their objectives are bits or energy, or abstract rewards.
- **Runner-up:** Orhan, Gündüz, Erkip, IEEE TWC 14(7), 2015 — [arXiv 1408.2302](https://arxiv.org/pdf/1408.2302). A distortion objective with a deadline over fading, but separation-based, energy-limited and single-user.

### Inferences (overlap verdicts per contribution)
- **[LA] additive, sensitivity-weighted MMSE abstraction.**
  - **Form:** HIGH overlap with classical theory. For linear codes it is Lee–Petersen; φ is the Gaussian MMSE (I-MMSE setting); for systems it is SoftCast/ParCast (SURVEY F4).
  - **Fading-JSCC literature:** LOW overlap. Sections 1–2 use mutual-information threshold (outage) abstractions for digital layers and multiplicative refinement for HDA, never an additive per-block MMSE rule.
  - **Verdict:** cite the form as the linear-code form. The claim that stands is that a learned nonlinear prefix code obeys it after per-image calibration, and that it makes the scheduling DP additive. The I-MMSE argument plus the AM–GM comparison give a theory paragraph explaining why an analog code needs an MMSE abstraction and a digital code an MI one.
- **[DP] per-user finite-horizon DP over (phases held, accumulated weighted noise).**
  - MEDIUM overlap: the template (finite horizon, deadline, causal CSIT, numerical DP) is Fu–Modiano–Tsitsiklis, Negi–Cioffi and Lee–Jindal.
  - The distortion version with retransmissions is Holliday et al.; the distortion version with a deadline and energy is Orhan et al.
  - New: the objective (calibrated learned-code distortion), the additive MMSE state, and the variable number of phases with L-dependent Dsrc and C. Present it as "the analog-code counterpart of HARQ's accumulated-MI state".
- **[MAP] phase-to-slot mapping.**
  - If phases can go into any slot: MEDIUM-HIGH overlap with Derman–Lieberman–Ross (1972). The optimal online rule is an interval rule independent of the phase sensitivities. Cite DLR and present our rule as an instance with dummy phases for skipped slots.
  - If phases are prefix-ordered and L is chosen under a price: LOW overlap. DLR's structure breaks, and our DP is the solution.
  - Ex post, the mapping is the rearrangement rule ParCast uses (SURVEY F4). Gündüz–Erkip's progressive transmission assigns layers to blocks without CSIT.
- **[MU] multi-user scheduling coupled by a Lagrange price.**
  - LOW overlap within IT JSCC theory: no paper found schedules several sources to minimise total distortion before a deadline over fading. The closest IT-side work is Tarello et al. (an energy-minimising multi-user deadline DP).
  - The decomposition and index machinery is known OR (SURVEY F2/F5). Novelty is the reward model, not the relaxation.
- **[CMP] analog JSCC vs digital (HEVC + rate adaptation + IR-HARQ) under CQI errors and outdated CSI.**
  - LOW overlap: no theoretical or simulated comparison of this kind was found.
  - Theory predicts the direction. No-CSIT distortion exponents favour graded, HDA and broadcast codes over separated fixed-rate codes (Laneman et al.; Gündüz–Erkip; Caire–Narayanan; Bhattad et al.). The value of graceful degradation falls as CSIT improves (Steiner–Shamai, partial CSIT). With perfect CSIT, ideal digital beats a linear code by the AM–GM gap.
  - Risk: a reviewer versed in this literature may ask for a layered-digital (superposition plus per-layer HARQ, Steiner–Shamai) or HDA baseline, not only single-layer IR-HARQ.

### Gaps
- **How these notes were produced:**
  - WebFetch failed on every host (DNS), and the proxy refused curl to arXiv, IEEE, Crossref, DBLP, OpenAlex and Semantic Scholar.
  - The shared web-search budget ran out after about 30 queries.
  - Every entry therefore rests on search-engine summaries of abstracts, bibliographic records and citing papers, not on full texts.
  - Before the paper cites any item as a precedent, re-check its volume, issue, pages and DOI, and read its main theorem. Do this first for Tian et al., the TIT version of Ng et al., Caire–Narayanan, Negi–Cioffi, the TWC version of Steiner–Shamai broadcast HARQ, and DLR.
- **Not searched for lack of budget:**
  - Whittle-index and remote-estimation scheduling with MSE rewards.
  - MMSE-based link abstraction for analog codes.
  - Analog retransmission with feedback (Schalkwijk–Kailath line).
  - HARQ under outdated CSI (Makki–Eriksson, Tuninetti).
  - The classical multiuser-diversity and gradient-scheduling papers.

  Each of these could add a precedent; each is listed under the Gaps of the relevant section, marked unconfirmed.
