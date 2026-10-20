# Analog and hybrid digital-analog (HDA) wireless image/video transmission: allocation, multi-user scheduling, CSI and digital comparisons (prior work for deadline scheduling of learned image codes)

> **How these notes were made, and their limits.** Sources are web-search result summaries and Microsoft Research (MSR) publication records. MSR was the only publisher host this sandbox could open: arXiv, IEEE Xplore, ACM DL, MERL, MIT DSpace, Stanford and university PDF hosts all failed DNS or were refused. The web-search budget shared by all agents ran out after about 45 queries, so several planned checks were not run (see each Gaps list). No full text was read. Every entry rests on an abstract, a record page or a search summary. Anything marked **(unverified)** comes from background knowledge or a summary I could not tie to a page, and must be checked before it is cited.
>
> **Tags for the parts of our work.** These tags are used in every entry.
> - [A] The link abstraction: distortion = Dsrc(L) + C(L)·Σ_j rho_L[j]·phi(n_j), an additive per-phase MMSE rule weighted by sensitivity, together with the finding that EESM/MIESM fail for the learned code.
> - [B] The important phases land in the good slots.
> - [C] Multi-user scheduling slot by slot to a deadline of T slots: a finite-horizon restless bandit, its Lagrangian bound, Whittle and Lagrangian indices, and the closed-form rule (this slot's gain against the gain from a typical slot).
> - [D] Stale or outdated CSI reports.
> - [E] The comparison with digital HEVC intra + LTE CQI link adaptation + HARQ.
> - [F] Delay measurement.
> - [G] Training the code on the SNR distribution the scheduler induces.
> - [H] Prefix-decodable successive-refinement phases.
> - [I] Quality-target objectives: mean PSNR at the deadline, and the share of users reaching a PSNR target.

## 1. How do SoftCast-lineage and HDA schemes allocate power/bandwidth across chunks and subchannels, and does any do slot-by-slot multi-user scheduling over time-varying channels with deadlines?

### Takeaway
Every SoftCast-lineage and HDA scheme found solves allocation **inside one transmission**. It minimizes a sum of per-chunk (or per-component) squared errors over power, bandwidth or channel uses, for a known, worst-case or statistically known channel. The variants are SoftCast-style chunk scaling, energy-model refinement (Xiong et al.), sorted matching of important components to strong MIMO-OFDM subchannels (ParCast/ParCast+), precoders over unequal subchannels (Zheng/Cagnazzo/Kieffer) and HDA digital/analog splits solved as Lagrangian or MINLP problems (Lan et al.; LayerCast). Time variation enters in only three ways:
- statistical CSI with diversity-aware allocation (Cui et al., INFOCOM 2014);
- channel *prediction* (A-HDAVT);
- progressive bandwidth-share scheduling of one stream (He et al., TMM 2017).

**No analog/HDA paper was found that schedules several unicast users slot by slot over a fading downlink against a deadline.**

### Cited Findings

#### Pure analog (pseudo-analog) schemes
- **SoftCast.** S. Jakubczak, D. Katabi, "A Cross-Layer Design for Scalable Mobile Video," ACM MobiCom 2011 (Sept. 2011); DOI 10.1145/2030613.2030646 **(unverified: ACM DL unreachable)**. An overview version, "SoftCast: Clean-Slate Scalable Wireless Video," appeared at Allerton 2010 — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/72994). An earlier HotNets 2009 paper is at [softcast-hotnets2009.pdf](https://people.csail.mit.edu/rahul/papers/softcast-hotnets2009.pdf) (title and authors unverified), and a tech report at [tr2.pdf](https://people.csail.mit.edu/szym/softcast/tr2.pdf).
  - Method: the network stack behaves as a linear transform, so the transmitted signal is linearly related to pixel luminance, and channel noise becomes a gradual loss of sharpness. One stream is multicast and each receiver decodes to a quality matching its own channel — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/72994).
  - Results: a 20-node GNU Radio/USRP testbed. SoftCast gives up to 5.5 dB higher average PSNR across broadcast receivers than conventional single- or multi-layer video. For a single receiver it removes glitches caused by mobility and is an order of magnitude more robust to packet loss — [MIT thesis record](https://dspace.mit.edu/handle/1721.1/66006?show=full), [MIT DSpace](https://dspace.mit.edu/handle/1721.1/72994).
  - Overlap with ours: the form of [A] (linear code with an LLSE decoder; see section 2) and [I] (per-receiver quality).
  - Lacks: [B]-[H]. It does no scheduling at the transmitter (broadcast), has no time axis or deadline, uses no CSI reports and its code is fixed and linear.
- **ParCast+ (and ParCast).** X. L. Liu, W. Hu, C. Luo, Q. Pu, F. Wu, Y. Zhang, "ParCast+: Parallel Video Unicast in MIMO-OFDM WLANs," IEEE Trans. Multimedia 16(7):2038–2051, Nov. 2014, DOI 10.1109/TMM.2014.2331616 — [MSR record](https://www.microsoft.com/en-us/research/?p=168037). Its conference precursor is ParCast (ACM MobiCom 2012), already in SURVEY F4. ParCast's bibliographic details were not re-verified here.
  - Method: source and channel are each separated into independent components. The most important source components are paired with the highest-gain channel components, and power is allocated from both. Transmission uses pseudo-analog modulation, giving fine-grained unequal error protection.
  - Results: implemented in Matlab and on Sora; up to 6.4 dB PSNR over competing schemes in challenging scenarios — [MSR record](https://www.microsoft.com/en-us/research/?p=168037).
  - Overlap with ours: [A] and [B]. It is the canonical precedent for "the important components go to the good channels", but it works across frequency and space within one transmission, for one user, with the channel known at the transmitter.
  - Lacks: [C] (no other users, no slots, no deadline), [D], [G] and a learned code.
- **FlexCast.** S. T. Aditya, S. Katti, "FlexCast: Graceful Wireless Video Streaming," ACM MobiCom 2011, pp. 277–288 — [PDF](https://web.stanford.edu/~skatti/pubs/mobicom11-flexcast.pdf), [IITH record](https://publications.iith.ac.in/publication/flexcast-graceful-wireless-video-streaming).
  - Method and results: a joint source-channel code whose reconstruction quality tracks the bits received and the instantaneous channel quality. It is near-optimal and about 6 dB PSNR above prior graceful video systems.
  - Overlap with ours: graceful single-link delivery (compare [H]).
  - Lacks: analog symbols (it works on bits, as the abstract's "bits received" indicates), multi-user scheduling and deadlines.
- **DCast.** X. Fan, F. Wu, D. Zhao, O. C. Au, W. Gao, "DCAST" (distributed soft video broadcast with explicit motion). It sends the *coset code* of the video signal over raw OFDM instead of its linear transform, and distributed source coding replaces closed-loop prediction. It reports significant gains over SoftCast — [PKU-hosted PDF](https://idm.pku.edu.cn/__local/1/0E/10/7CB4C9BB8826C77C41E19CA4319_D5F0D321_43CAB.pdf?e=.pdf). A search summary identified this PDF; I did not open it. Venue and year are **unverified** (possibly DCC 2012, with a TCSVT 2013 journal version).
  - Lacks: scheduling, time and CSI.
- **WaveCast.** X. Fan, R. Xiong, F. Wu, D. Zhao, "WaveCast: Wavelet Based Wireless Video Broadcast Using Lossy Transmission," IEEE VCIP 2012, pp. 1–6, DOI 10.1109/VCIP.2012.6410743 — [pkuscholar](https://pkuscholar.onrender.com/conf/vcip/FanXWZ12).
  - Method: motion-compensated temporal filtering (MCTF). Motion vectors go over a conventional lossless (digital) channel and coefficients in lossy (analog) mode.
  - Results: about 2 dB over SoftCast at low SNR, and up to 8 dB over an H.264-based framework in broadcast.
  - Overlap with ours: an HDA split only.
  - Lacks: [B]-[I].
- **Cactus.** H. Cui, Z. Song, Z. Yang, C. Luo, R. Xiong, F. Wu, "Cactus: A Hybrid Digital-Analog Wireless Video Communication System," ACM MSWiM 2013, pp. 273–278, DOI 10.1145/2507924.2507987 — [MSR record](https://www.microsoft.com/en-us/research/?p=166044), [DOI](https://www.doi.org/10.1145/2507924.2507987).
  - Method: removes inter-frame redundancy but keeps intra-frame redundancy. Transform-domain scaled values are sent by amplitude modulation, and the receiver denoises the raw channel output. It runs in real time on Sora with a GPU, evaluated in 802.11a/g.
  - Results: on average 4.7 dB above SoftCast, and robust to packet loss.
  - Overlap with ours: none with scheduling (decoder-side priors only).
  - Lacks: [B]-[I].
- **LayerCast.** X. Fan, R. Xiong, D. Zhao, F. Wu, "Layered Soft Video Broadcast for Heterogeneous Receivers," IEEE TCSVT 25(11):1801–1814, Nov. 2015, DOI 10.1109/TCSVT.2015.2402831 — [HIT record](https://scholar.hit.edu.cn/en/publications/layered-soft-video-broadcast-for-heterogeneous-receivers/), [pkuscholar](https://pkuscholar.onrender.com/journals/tcsv/FanXZW15/).
  - Method: serves users with diverse SNRs **and** bandwidths through layered coset coding. Power allocation between layers, and between DCT chunks within each layer, is globally optimal. Narrowband users decode only the base layer.
  - Results: up to 4 dB over SoftCast, and up to 8 dB over an H.264-based framework in broadcast.
  - Overlap with ours: [H] (each receiver decodes a prefix of the layers) and per-chunk power allocation as in [A].
  - Lacks: [C] (all layers are broadcast at once, with no per-slot user choice), [D] and [G].
- **KMV-Cast and its follow-ups** (Xin-Lin Huang's group, Tongji).
  - The scheme is described as knowledge-enhanced: a hierarchical Bayesian model folds correlated, cloud-side information into the reconstruction — [as cited in arXiv 2005.09302](https://arxiv.org/pdf/2005.09302).
  - A GLOBECOM 2016 analysis sends the prior knowledge itself over a Gaussian channel. With imperfect knowledge, PSNR is worst of three schemes, so the prior matters — [infona record](https://www.infona.pl/resource/bwmeta1.element.ieee-art-000007842128).
  - "Maximum a Posteriori Decoding for KMV-Cast Pseudo-Analog Video Transmission," Mobile Networks and Applications, 2018 — [Springer Professional](https://www.springerprofessional.de/en/maximum-a-posteriori-decoding-for-kmv-cast-pseudo-analog-video-t/15136070).
  - "Improved KMV-Cast with BM3D Denoising" (X.-L. Huang, X. Tang, X. Huan, P. Wang, J. Wu) — [dblp](https://dblp1.uni-trier.de/pers/hd/h/Huan:Xiaoning).
  - A Wiener-filter receiver gains about 2 dB PSNR at −10 dB SNR and about 1.5 dB at 10 dB — [EUDL](https://eudl.eu/doi/10.1007/978-3-030-98002-3_17).
  - The venue of the original KMV-Cast paper is **unverified**. It is possibly IEEE TCSVT 2017, "Knowledge-enhanced mobile video broadcasting framework with cloud support."
  - Overlap with ours: none (receiver priors).
  - Lacks: [B]-[I].
- **Power-distortion optimization (PDO) of SoftCast.**
  - R. Xiong, F. Wu, X. Fan, C. Luo, S. Ma, W. Gao, "Power-Distortion Optimization for Wireless Image/Video SoftCast by Transform Coefficients Energy Modeling with Adaptive Chunk Division," IEEE VCIP 2013, DOI 10.1109/VCIP.2013.6706344 (author order unverified) — [MSR record](https://www.microsoft.com/en-us/research/?p=168062). SoftCast allocates power unequally by coefficient energy, so sender and receiver must share energy metadata. The paper formulates the PDO problem and chooses chunk boundaries adaptively, gaining 4–8 dB over the original SoftCast with fewer chunks.
  - Journal version: R. Xiong, J. Zhang, F. Wu, J. Xu, W. Gao, "Power Distortion Optimization for Uncoded Linear Transformed Transmission of Images and Videos," IEEE TIP 26(1):222–236, 2017, DOI 10.1109/TIP.2016.2621478 — [pkuscholar](https://pkuscholar.onrender.com/journals/tip/XiongZWX017/), [SciSpace](https://stagecf.scispace.com/papers/power-distortion-optimization-for-uncoded-linear-transformed-45tfsan6af). Exploiting the energy diversity of the transformed signal is the key to the transform gain. It reports 3–5 dB with metadata cut to a few values. A companion analysis, "Analysis of Decorrelation Transform Gain for Uncoded Wireless Image and Video Communication" (IEEE TIP 2016), is known only from a search summary.
  - Overlap with ours: the per-component weights of [A]. Their energy-based weights correspond to our learned rho_L[j].
  - Lacks: [B]-[I].
- **Robust uncoded video over fast fading.** H. Cui, C. Luo, C. W. Chen, F. Wu, "Robust Uncoded Video Transmission over Wireless Fast Fading Channel," IEEE INFOCOM 2014 — [MSR record](https://www.microsoft.com/en-us/research/?p=166043), [Buffalo record](https://researchconnect.buffalo.edu/en/publications/robust-uncoded-video-transmission-over-wireless-fast-fading-chann/).
  - Setting: the sender has **only statistical CSI**.
  - Finding: high-priority data needs more channel diversity.
  - Method: power and channel uses are reallocated between high- and low-priority data under total power and channel-use limits, minimizing squared error. The paper gives power allocation for a fixed channel allocation, and a channel allocation that trades high-priority diversity against low-priority information loss. For multicast it gives an optimal resource allocation when the distribution of noise power is known.
  - Results against SoftCast: unicast gains about 2 dB in average PSNR and 5 dB in outage PSNR; multicast gains about 1.4 dB and 4 dB.
  - Overlap with ours: [A] (expected distortion over the noise law), close to [D] (no instantaneous CSIT) and close to [I] (outage PSNR is like a quality-target share).
  - Lacks: [C] (no user selection, slots or deadline), [G] and a learned code.
- **Denoising and resource allocation.** H. Cui, R. Xiong, C. Luo, Z. Song, F. Wu, "Denoising and Resource Allocation in Uncoded Video Transmission," IEEE JSTSP 9(1):102–112, 2015, DOI 10.1109/JSTSP.2014.2338279 — [MSR record](https://www.microsoft.com/en-us/research/?p=168039).
  - Method: MCTF at the sender and denoising at the receiver, using spatial-domain transmission with cascaded denoising. Resource allocation uses variable-size L-shaped chunks, which are more energy-efficient than fixed rectangles.
  - Results on 720p: up to 3.3 dB over SoftCast, and up to 5.3 dB over a digital scheme with robust rate adaptation and H.264 SVC.
  - Lacks: [B]-[I].

#### HDA schemes and HDA resource allocation
- **A practical HDA scheme.** C. Lan, C. Luo, W. Zeng, F. Wu, "A Practical Hybrid Digital-Analog Scheme for Wireless Video Transmission," IEEE TCSVT vol. 28, pp. 1634–1647, July 2018 (IEEE Xplore 7858688) — [MSR record](https://microsoft.com/en-us/research/?p=381074).
  - Method: frames are modelled as a parallel Gaussian source and split by scalar quantization. The digital part gets entropy and channel coding; the analog part gets power scaling. Power and bandwidth are allocated across components using a closed-form expression for total distortion at the **worst-case** channel SNR. The resulting mixed-integer nonlinear program is NP-hard, so the authors simplify it to a near-optimal practical scheme.
  - Results: up to 5.0 dB over SoftCast.
  - Overlap with ours: [A] (a closed-form distortion drives the allocation).
  - Lacks: [C], [D] (it designs for the worst case) and [G].
- **Progressive pseudo-analog transmission.** D. He, C. Lan, C. Luo, E. Chen, F. Wu, W. Zeng, "Progressive Pseudo-Analog Transmission for Mobile Video Streaming," IEEE Trans. Multimedia vol. 19, Aug. 2017 (IEEE Xplore 7885562; issue and pages unverified) — [MSR record](https://microsoft.com/en-us/research/?p=381044). The conference version is "Progressive Pseudo-Analog Transmission for Mobile Video Live Streaming," IEEE VCIP 2015, DOI 10.1109/VCIP.2015.7457915 — [MSR record](https://microsoft.com/en-us/research/?p=238252).
  - Setting: SNR and bandwidth vary together.
  - Method: bandwidth adaptivity comes from a scheduling algorithm with optimal power allocation, solved by water-filling. A near-optimal approximation applies when only statistical channel information is available. The conference version adds bandwidth shares greedily, each improving quality, and derives optimal scheduling and power allocation under MSE.
  - Results: approaches the AWGN upper bound of pseudo-analog transmission and beats existing pseudo-analog schemes in fast Rayleigh fading. In trace-driven emulation and on a testbed it beats both digital and pseudo-analog baselines.
  - Overlap with ours: **strong for [H]**: every extra bandwidth share raises quality, the analog counterpart of our prefix phases. It also overlaps [A] (MSE with water-filling) and is close to [D] (statistical channel information).
  - Lacks: [C]. As far as the abstracts say, the "scheduling" decides what one stream sends in each share; it does not choose among competing users. It also lacks a deadline/RMAB formulation, [G] and a learned code.
- **A-HDAVT.** "Adaptive Hybrid Digital–Analog Video Transmission in Wireless Fading Channel," an IEEE journal article (IEEE Xplore 7123183; **authors, venue and year unverified**; the record is hosted by SUNY) — [IEEE Xplore](https://ieeexplore.ieee.org/document/7123183/authors), [SUNY record](https://researchconnect.suny.edu/en/publications/adaptive-hybrid-digital-analog-video-transmission-in-wireless-fad/).
  - Method: each group of pictures (GOP) becomes one low-pass frame, sent as a digital base layer, and high-pass frames, sent as analog enhancement layers. Power-distortion optimization based on **channel prediction** allocates power by content and by predicted channel state. Single-user and multi-user cases are studied, exploiting content and channel diversity.
  - Results: beats Robust Uncoded Video Transmission, ParCast and SoftCast.
  - Overlap with ours: [D] (it predicts the channel instead of trusting the last state), near [C] (multiple users) and [A].
  - Lacks: a deadline formulation, any bound or index policy, and a learned code (details unverified).
- **Analog Coded SoftCast.** B. Tan, J. Wu, Y. Li, H. Cui, W. Yu, C. W. Chen, "Analog Coded SoftCast: A Network Slice Design for Multimedia Broadcast/Multicast," IEEE Trans. Multimedia, 2017, DOI 10.1109/TMM.2017.2733303 — [Buffalo record](https://researchconnect.buffalo.edu/en/publications/analog-coded-softcast-a-network-slice-design-for-multimedia-broad/), [OA.mg](https://oa.mg/work/10.1109/tmm.2017.2733303).
  - Method: the content server compresses with a rateless Spinal code. The base station then applies a chaotic-function analog code designed for generalized-Gaussian sources, decoded by MAP with source priors.
  - Results: the 1/2-rate code gains about 2 dB over SoftCast with two repetitions, and the 1/3-rate code about 3 dB over SoftCast with three.
  - Lacks: scheduling.
- **Superposition-coding HDA.** B. Tan, H. Cui, J. Wu, C. W. Chen, "An Optimal Resource Allocation for Superposition Coding-Based Hybrid Digital-Analog System," IEEE Internet of Things J. 4(4):945–956, 2017, DOI 10.1109/JIOT.2017.2680407 — [researchr](https://researchr.org/publication/TanCWC17/bibtex). The content was not retrieved.
- **HDA over fading channels.** X. Jiang, H. Lu, "Joint Rate and Resource Allocation in Hybrid Digital-Analog Transmission over Fading Channels," arXiv:1812.07513 (2018) — [arXiv](https://arxiv.org/pdf/1812.07513), [DeepAI](https://deepai.org/publication/joint-rate-and-resource-allocation-in-hybrid-digital-analog-transmission-over-fading-channels).
  - Method: earlier HDA allocation assumed the digital part was error-free; here the digital rate enters through outage. The objective is expected distortion with analog and digital terms (bandwidth expansion, a Gaussian source, an entropy-constrained scalar quantizer), solved by block coordinate descent in the low-outage regime.
  - Results: up to 2.3 dB signal-to-distortion ratio (single Gaussian source) and up to 3.5 dB (multivariate).
  - Overlap with ours: [A] (expected distortion over fading).
  - Lacks: multiple users and scheduling.
- **Classic HDA power split.** Wang, Alajaji and Linder (IEEE Trans. Commun., 2009) split channel input power between the analog and digital signals over matched and mismatched Gaussian (not fading) channels — [PDF](https://mast.queensu.ca/~fady/Journal/tcom09-hda.pdf) (exact title not captured).
- **Precoding for linear video coding over unequal subchannels.** S. Zheng, M. Cagnazzo, M. Kieffer — [EURASIP thesis page](https://theses.eurasip.org/document/accounting-for-channel-constraints-in-joint-source-channel-video-coding-schemes/), [SPIC record](https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/248/paper-detail/40576), [SigPort](https://sigport.org/node/2387), [Cagnazzo's page](https://cagnazzo.wp.imt.fr/?p=1463).
  - ICIP 2016: SoftCast over parallel channels with per-carrier power limits and different noise (DSL, power line). The MSE-optimal precoder comes from an inverse eigenvalue problem; there are suboptimal alternatives, and the gain grows with the encoded quality.
  - ICASSP 2018: "Precoding matrix design in linear video coding."
  - Signal Processing: Image Communication 2019, "Optimal and suboptimal channel precoding and decoding matrices for linear video coding": per-subchannel power limits, low-complexity designs with little loss, and the **impact of mismatched channel conditions**.
  - Zheng's thesis: multi-level water-filling combined with a structured Hermitian inverse eigenvalue problem.
  - Overlap with ours: [A] (MSE over subchannels with unequal noise); the mismatch analysis is close to [D].
  - Lacks: multiple users and time.
- **The theoretical root.** K.-H. Lee, D. P. Petersen, "Optimal Linear Coding for Vector Channels," IEEE Trans. Commun. 24(12):1283–1290, 1976. These details come from a citing paper's reference list in a search summary; the paper was not opened. It grew out of K.-H. Lee's UNM dissertation, "Optimal Linear Coding For a Multichannel System" (1975) — [UNM repository](https://digitalrepository.unm.edu/ece_etds/436).
  - Method: the optimal linear map from an r-dimensional signal to a k-dimensional channel under a power limit. The encoder decorrelates the signal, keeps at most k components with the largest eigenvalues, and pairs them with the channel noise eigenvalues in a specific order.
  - Overlap with ours: the root of [A] and [B] (sorted matching plus per-component MMSE).
- **The book.** F. Wu, C. Luo, H. Lu, *Uncoded Multimedia Transmission*, CRC Press, 2021 (paperback 2023). Its parts are coded vs uncoded transmission, correlation processing, **resource allocation**, MIMO support and HDA — [tenlong listing](https://tenlong.com.tw/products/9780367632953), [Bookshare](https://bookshare.org/browse/book/5403377). The table of contents comes from a retailer listing.
- **The survey.** T. Fujihashi, T. Koike-Akino, T. Watanabe, "Soft Delivery: Survey on a New Paradigm for Wireless and Mobile Multimedia Streaming," ACM Computing Surveys 56(2):1–37, 2023, DOI 10.1145/3607139 (arXiv:2111.08189) — [MERL](https://merl.com/publications/TR2023-102), [Crossref](https://api.crossref.org/works/10.1145%2F3607139). It covers HDA designs. **It was not read** because its hosts were unreachable.
- **An analog multicast allocation with a capacity-type objective.** "Soft Video Multicasting Using Adaptive Compressed Sensing," arXiv:2003.03092 (authors not captured). It allocates subchannels first, then the optimal power for each packet on its subchannels, maximizing **a sum of log-rates** under total power — [arXiv](https://arxiv.org/pdf/2003.03092).

### Inferences
- In this literature, the allocation problem is always "one source block, one transmission, a known, worst-case or statistically known channel". The decision variables are power, chunk division, the subchannel/component pairing and the digital/analog split. Our problem moves the pairing decision onto the **time** axis and across **K competing users**, with causal and possibly stale reports. None of the works above has that structure.
- ParCast+'s pairing and Lee–Petersen's ordered pairing are the static, full-information versions of [B]. Our scheduler realizes [B] *ex post* and online: a phase sent in a good slot gets a small n_j.
- He et al. (TMM 2017) is the closest analog precedent for [H] combined with scheduling. Its unit is the "bandwidth share" of one user's stream over time, so it is the single-user cousin of our per-user dynamic program.

### Gaps
- No analog/HDA paper doing multi-user, slot-by-slot opportunistic scheduling with deadlines was found. The queries were: "soft video multicast OFDMA subcarrier power allocation user scheduling analog", "soft video delivery multi-user MIMO user scheduling analog", "pseudo-analog video multi-user MIMO precoding user selection soft multicast", "uncoded video unicast multiple users OFDMA resource allocation pseudo-analog", "analog joint source-channel coding opportunistic scheduling multiuser diversity", "hybrid digital-analog video multiuser scheduling fading channel fairness unicast", "soft video delivery scheduling time-varying channel deadline pseudo-analog" and "SoftCast proportional fair scheduling LTE". This is a search-engine negative, not proof of absence. The Fujihashi survey (ACM CSUR 2023) and the CRC book's resource-allocation part are the most likely places to find a counterexample, and neither was read.
- SoftCast's exact power rule is **unverified this session**. From memory: per-chunk gain g_i ∝ λ_i^(−1/4) under a total power limit, an LLSE decoder, and dropping of low-energy chunks when bandwidth is short. Check it in the MobiCom 2011 text before stating it.
- Not verified: ParCast (MobiCom 2012) DOI and pages; A-HDAVT's authors, venue and year; DCast's venue; the original KMV-Cast venue; and whether ParCast/ParCast+ need current CSI at the transmitter (almost certainly they need per-subcarrier CSI, but unconfirmed).
- Known from background but not searched or verified, for a follow-up: Mittal and Phamdo (IEEE TIT 2002, HDA codes for broadcasting); Skoglund, Phamdo and Alajaji (IEEE TIT 2006); Yu, Li and Li, HDA scalable video (IEEE TCSVT 2014); Song et al., layered image/video SoftCast with HDA (ICME 2014); Liu et al., compressive image broadcasting in MIMO with antenna heterogeneity (SPIC 2014); Supcast (arXiv 1812.06713, already in SURVEY; not re-checked).

## 2. Are analog/HDA distortion models over subchannels with different SNRs (sums of per-component MMSE errors) used as a link abstraction or an effective-SNR mapping in multi-user or system-level allocation? Does anyone say EESM/MIESM do not apply to analog?

### Takeaway
Within single transmissions the per-component MSE sum is used everywhere as the *optimization objective*: power scaling, sorted matching, water-filling and inverse-eigenvalue precoders, HDA splits, and expected distortion under fading. **No work was found that packages it as a link-to-system abstraction for a multi-user scheduler.** No analog paper was found that discusses EESM/MIESM at all, let alone states that they fail. The analog literature never needed an effective SNR because it optimizes distortion directly. One analog multicast paper even uses a log-rate (capacity-type) power objective.

### Cited Findings
- Linear (uncoded) transmission is MSE-optimal for a Gaussian source over AWGN at matched bandwidth, a rare case where joint source-channel coding (JSCC) is both simple and optimal — [JSCC fundamentals survey, arXiv 2409.17557](https://arxiv.org/pdf/2409.17557).
- Sorted pairing of source eigen-components with channel noise eigenvalues is the optimal linear code for vector channels — [Lee 1975 dissertation (UNM)](https://digitalrepository.unm.edu/ece_etds/436).
- ParCast+ pairs the most important source components with the highest-gain MIMO-OFDM channel components and allocates power from both — [MSR](https://www.microsoft.com/en-us/research/?p=168037).
- Over parallel subchannels with different noise and per-carrier power limits, the MSE-minimizing linear precoder comes from an inverse eigenvalue problem, or from multi-level water-filling plus a structured Hermitian inverse eigenvalue problem. Its loss under mismatched channel conditions has been analysed — [EURASIP thesis page](https://theses.eurasip.org/document/accounting-for-channel-constraints-in-joint-source-channel-video-coding-schemes/), [SPIC record](https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/248/paper-detail/40576).
- Exploiting the energy diversity of the coefficients is the key to the gain of uncoded linear transmission. The accuracy of the per-chunk energy model matters (3–5 dB) — [Xiong et al., TIP 2017](https://pkuscholar.onrender.com/journals/tip/XiongZWX017/).
- A closed-form total distortion for an HDA split, evaluated at the worst-case channel SNR, drives the power and bandwidth allocation in Lan et al. — [MSR](https://microsoft.com/en-us/research/?p=381074).
- Expected squared error under a known noise-power distribution (statistical CSI) drives power and channel-use allocation in Cui et al. — [MSR](https://www.microsoft.com/en-us/research/?p=166043).
- Water-filling solves power allocation under MSE in progressive pseudo-analog transmission — [MSR](https://microsoft.com/en-us/research/?p=381044).
- Noisy CSI: "Video coding for OFDM systems with imperfect CSI: A hybrid digital–analog approach," Signal Processing: Image Communication (2020), DOI 10.1016/j.image.2020.115903 (authors not captured). It uses a water-filling-type power allocation that includes a CSI noise model — [datalearner record](https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/257/paper-detail/40785).
- An analog multicast paper allocates power per packet by maximizing a sum of log-rates (a capacity-type surrogate) after subchannel allocation — [arXiv 2003.03092](https://arxiv.org/pdf/2003.03092).
- For SoftCast, PSNR is linear in channel SNR (dB), and the same linear relation holds for MOS except at high CSNR, where quality saturates — [HAL, Cagnazzo et al.](https://hal.univ-grenoble-alpes.fr/LTCI/hal-02997050v1).
- EESM, for contrast: it compresses per-subcarrier SNRs into one effective SNR using an exponential average with a β that depends on the modulation and coding scheme (MCS); it is used in LTE — [HAL hal-01578885](https://hal.archives-ouvertes.fr/hal-01578885). For *uncoded* QAM no β values exist and β must be trained by link-level simulation. Uncoded BER averaged over subcarriers is a related abstraction — [US patent 8416881](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8416881). These are about digital BER/BLER, not analog MSE.

### Inferences
These are my derivations and should be checked; they are not claims from the sources.
- **Our phi is SoftCast's per-chunk LLSE error.** Take a component of variance λ, sent with gain g over noise variance n and decoded by LLSE. Its error is D = λ·n/(g²λ + n), which rearranges to (1/g²)·n/(1 + n/(g²λ)). That is exactly C·rho·phi(n) with phi(n) = n/(1 + n/κ), rho ∝ 1/g² and κ = g²λ, the component's received power.
  - A single κ shared by all components means equal received power per symbol. Deep JSCC's codeword power normalization comes close to this, so for a learned code the "importance" sits in the decoder sensitivity rho rather than in the power split.
  - So the *form* of [A] is SoftCast's/ParCast's by construction. What is new is that a learned nonlinear code obeys it with measured, per-image weights.
- **Why MIESM/EESM should fail for analog-like codes.** At high SNR, phi(n) ≈ n, so distortion is a weighted *arithmetic* mean of noise variances, dominated by the worst slots. A mean-dB rule averages log-SNR (a geometric mean of noise), and MIESM averages log(1+SNR); both underweight deep fades. At very low SNR phi saturates at κ: a phase in a deep fade costs at most its full weight, like a chunk that SoftCast drops (from memory, unverified).
  - EESM/MIESM are calibrated to the cliff of a capacity-approaching block code, and an analog code has no cliff. The analog literature itself does not make this argument, which is why our explicit statement and evidence ("MIESM fails for the learned code") look new in this corpus.
- The closest "abstraction feeding a resource allocator" in this literature is Cui et al.'s expected distortion over a noise distribution, used for multicast allocation. It feeds a static allocation, not a scheduler.

### Gaps
- No source found that names EESM, MIESM or effective SNR in an analog/HDA context. The query "effective SNR mapping EESM analog uncoded transmission distortion OFDM subcarriers" returned only digital sources. The Fujihashi survey was not read and might discuss system-level evaluation.
- Not verified: whether ParCast+'s or the A-HDAVT paper's distortion model is used across users, which would make it a multi-user abstraction.

## 3. Multi-user / multicast analog video over fading channels: user selection, opportunistic scheduling, OFDMA allocation, quality targets, fairness; analog transmission with outdated or imperfect CSI

### Takeaway
"Multi-user" in the analog literature almost always means **multicast or broadcast of one stream to heterogeneous receivers**: SoftCast, WaveCast, DCast, LayerCast, Analog Coded SoftCast, multicast allocation under a noise distribution (Cui et al.), and NOMA multicast (Supcast, in SURVEY). Imperfect CSI is handled for one link or one group, through statistical CSI, channel prediction, CSI-noise-aware water-filling or precoder-mismatch analysis.
- No analog/HDA work was found on opportunistic *unicast* user selection, whether PF, max-weight or index policies.
- None was found on OFDMA user-to-subcarrier assignment with per-user quality targets or fairness.
- None was found that schedules on outdated per-slot reports.

The closest digital counterparts with deadlines exist and are listed below.

### Cited Findings
- **Multicast by design.** SoftCast sends one stream, and each receiver's quality matches its own channel — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/72994). LayerCast serves receivers that differ in both SNR and bandwidth, with optimal power between layers and chunks — [HIT record](https://scholar.hit.edu.cn/en/publications/layered-soft-video-broadcast-for-heterogeneous-receivers/). WaveCast adapts to receivers with different channel conditions — [pkuscholar](https://pkuscholar.onrender.com/conf/vcip/FanXWZ12).
- **Multicast allocation from a noise distribution.** Cui et al. derive an optimal multicast resource allocation when the distribution of noise power is known, gaining 1.4 dB in average and 4 dB in outage PSNR over SoftCast — [MSR](https://www.microsoft.com/en-us/research/?p=166043).
- **Multiple users plus channel prediction.** A-HDAVT studies single- and multi-user cases and allocates power with channel-prediction-based power-distortion optimization — [IEEE Xplore](https://ieeexplore.ieee.org/document/7123183/authors).
- **Statistical CSI only.** Cui et al. (INFOCOM 2014) assume a sender with only statistical CSI in fast fading — [MSR](https://www.microsoft.com/en-us/research/?p=166043). He et al. give a near-optimal power and scheduling approximation when only statistical channel information is available — [MSR](https://microsoft.com/en-us/research/?p=381044).
- **Noisy CSI.** HDA video over OFDM uses water-filling with a CSI noise model — [datalearner record](https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/257/paper-detail/40785).
- **Mismatched channel.** Zheng, Cagnazzo and Kieffer analyse the effect of mismatched channel conditions on optimal linear-video precoders — [SPIC record](https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/248/paper-detail/40576).
- **Uplink, multiple users, relay choice.** Fujihashi et al. apply soft delivery to crowdsourced multi-view video uploads, with clustering and relay selection to fix weak direct links and a 4D-DCT across cameras — [IEICE author listing](https://global.ieice.org/en_transactions/Author/a_name=Takuya%20FUJIHASHI) (title not captured).
- **Analog JSCC with nonlinear mappings over a fading MAC.** Single-user mappings plus a suitable channel-access scheme reportedly approach the optimal distortion-cost trade-off. Analog JSCC must track channel fluctuations, but only the encoder parameters need updating — [EURASIP thesis page](https://theses.eurasip.org/document/analog-joint-source-channel-coding-for-wireless-communications/), [EUSIPCO 2015 paper](https://new.eurasip.org/Proceedings/Eusipco/Eusipco2015/papers/1570103369.pdf).
- **Theory on time-varying channels.** I. Estella Aguerri, D. Gündüz, "Joint Source-Channel Coding with Time-Varying Channel and Side-Information," arXiv:1312.0932 — [arXiv](https://arxiv.org/pdf/1312.0932). Only the title was verified. From memory (unverified), it studies expected distortion of layered and HDA schemes on time-varying channels.
- **Digital counterparts with deadlines** (for positioning [C]):
  - Liebl, Kalman, Girod (ICME 2005) delay the streams of users with good channels until their decoding deadlines approach, which helps poorly connected users under overload — [PDF](https://web.stanford.edu/~bgirod/pdfs/LieblICME05.pdf).
  - Zhao and Lin (INFOCOM 2014) study rate control and multi-channel scheduling for wireless live streaming with stringent deadlines, using threshold policies (digital SVC) — [PDF](https://jhc.sjtu.edu.cn/~shizhenzhao/infocom2014.pdf).
  - "Delay-Constrained Video Transmission: Quality-driven Resource Allocation and Scheduling" finds large scheduling gains, but enforcing equal video quality cuts the number of users supported, because cell-edge users consume more resources — [arXiv 1311.5921](https://arxiv.org/pdf/1311.5921).
  - Fairness- and QoS-guaranteed MU-MIMO broadcast scheduling (ICIP 2011) gets video packets out before they expire and gives leftover capacity to data users — [Buffalo record](https://researchconnect.buffalo.edu/en/publications/fairness-and-qos-guaranteed-user-scheduling-for-multi-user-mimo-b/).
  - Ni et al. (IEEE TWC 2014) schedule MU-MIMO users by priorities that weigh throughput against video quality; delay is 35% lower than prior work — [UTS record](https://opus.lib.uts.edu.au/handle/10453/118940).

### Inferences
- The analog literature's answer to user heterogeneity is to **broadcast one stream and let each receiver's SNR set its quality**. That removes the need for user selection, which explains why opportunistic unicast scheduling of analog codes is missing. Our setting (K unicast images, one slot per user per time) is the case broadcast cannot cover, so [C] has no analog precedent in what was found.
- On CSI, the analog works make the *encoder* robust to an unknown channel: statistical CSI, prediction, noisy-CSI water-filling. Our transmitter instead keeps the code fixed and makes the *scheduling decision* robust to a stale report through the report-to-channel kernel in the per-user dynamic program. These are complementary. A-HDAVT's prediction is the nearest idea to our staleness-aware transmitter.
- Quality targets do appear in analog work, but only as outage PSNR (Cui et al.), never as a per-user target in a scheduler. Our "share reaching a PSNR target" objective is closer to the digital deadline-scheduling literature (Liebl et al.; arXiv 1311.5921).

### Gaps
- Not checked, for lack of search budget: Fujihashi et al.'s multi-user MIMO/NOMA soft-delivery papers beyond Supcast; analog/HDA with HARQ or retransmission (a He et al.-style progressive retransmission may exist); multi-user OFDMA analog unicast in Chinese-language venues; and A-HDAVT's multi-user model (whether users are scheduled or multicast).
- The Estella Aguerri–Gündüz paper's content was not verified.

## 4. Comparisons of analog/HDA against digital (H.264/HEVC + LDPC/turbo + adaptive MCS, with or without HARQ) under CSI errors, mobility and outdated CSI: margins and baselines

### Takeaway
The margins reported over digital are large: about 5–8 dB in PSNR. Almost all of them come from **broadcast/multicast to heterogeneous receivers**, where digital must pick one rate, or from **mobility traces** against H.264/SVC over 802.11 rate adaptation. No analog/HDA paper found compares against HEVC with CQI-driven link adaptation and HARQ, or models CQI-report errors or staleness on the digital side.

### Cited Findings
- **SoftCast.** Up to 5.5 dB higher average PSNR across broadcast receivers than conventional single- or multi-layer video (20-node USRP testbed). For one receiver, it eliminates video glitches from mobility and is an order of magnitude more robust to packet loss — [MIT thesis record](https://dspace.mit.edu/handle/1721.1/66006?show=full), [MIT DSpace](https://dspace.mit.edu/handle/1721.1/72994).
- **WaveCast.** Up to 8 dB over an H.264-based broadcast framework; about 2 dB over SoftCast at low SNR — [pkuscholar](https://pkuscholar.onrender.com/conf/vcip/FanXWZ12).
- **LayerCast.** Up to 8 dB over an H.264-based framework in broadcast; up to 4 dB over SoftCast — [HIT record](https://scholar.hit.edu.cn/en/publications/layered-soft-video-broadcast-for-heterogeneous-receivers/).
- **Cui et al., JSTSP 2015.** Up to 5.3 dB over a digital scheme with **robust rate adaptation + H.264 SVC** (720p); up to 3.3 dB over SoftCast — [MSR](https://www.microsoft.com/en-us/research/?p=168039).
- **He et al. (progressive pseudo-analog).** Beats state-of-the-art digital and pseudo-analog schemes in trace-driven emulation (journal) and on a testbed (VCIP 2015). The abstracts give no number ("notable margin") — [MSR TMM record](https://microsoft.com/en-us/research/?p=381044), [MSR VCIP record](https://microsoft.com/en-us/research/?p=238252).
- **ParCast+.** Up to 6.4 dB PSNR over competing schemes in challenging scenarios. The baselines are not specified in the abstract — [MSR](https://www.microsoft.com/en-us/research/?p=168037).
- **FlexCast.** About 6 dB over prior graceful video systems and near-optimal — [PDF](https://web.stanford.edu/~skatti/pubs/mobicom11-flexcast.pdf).
- **Against SoftCast only.** Cactus +4.7 dB — [MSR](https://www.microsoft.com/en-us/research/?p=166044); Lan et al. HDA up to +5.0 dB — [MSR](https://microsoft.com/en-us/research/?p=381074); Cui et al. INFOCOM 2014 +2 dB average and +5 dB outage (unicast, fast fading, statistical CSI) — [MSR](https://www.microsoft.com/en-us/research/?p=166043).
- **SoftCast in LTE.** Z. Ding, J. Wu, W. Yu, Y. Han, X. Chen, "Pseudo Analog Video Transmission Based on LTE Physical Layer," IEEE/CIC ICCC 2016 (Chengdu), runs SoftCast inside the LTE frame structure on an SDR platform. It reports 3–5 dB over plain SoftCast; quantized coefficients and random mapping reduce OFDM PAPR. There is no comparison against an LTE digital stack in the summary — [BIT Pure](https://pure.bit.edu.cn/en/publications/pseudo-analog-video-transmission-based-on-lte-physical-layer/).
- **Deep JSCC video, for reference.** DeepWiVe outperforms H.264 + LDPC in all channel conditions, by up to 0.0485 MS-SSIM on average — [Imperial PDF](https://t4awstest.cc.ic.ac.uk/media/imperial-college/research-centres-and-groups/ipc-lab/DeepWiVe-1.pdf). From memory, its citation is T.-Y. Tung, D. Gündüz, IEEE JSAC 2022 (venue unverified here).

### Inferences
- Our 1.3–1.7 dB margin over HEVC intra + LTE CQI + HARQ with stale reports or CQI errors is much smaller than the analog literature's 5–8 dB. It was measured in a harder and fairer setting:
  - unicast with per-user link adaptation, which removes the common-rate penalty of broadcast;
  - HARQ with combining, which softens the digital cliff;
  - a modern codec.
- When a reviewer asks why the gain is not SoftCast-sized, this is the answer. Positioning: the analog literature's headline gains are mostly about the *common-rate* problem of broadcast, while ours isolate the *CSI-error* problem in unicast.
- The closest analog baseline to ours is Cui et al.'s (JSTSP 2015) "robust rate adaptation + H.264 SVC", up to 5.3 dB. "Up to" margins are typically measured at the most favourable SNR or trace; our numbers are averages.

### Gaps
- No full texts were opened, so for none of these works do I know the MCS sets, the presence of HARQ, the CSI delay, or whether margins are averages or maxima.
- Not found: any analog/HDA comparison with HEVC, with LTE/NR CQI reporting, or with HARQ. The query "SoftCast vs HEVC LDPC adaptive modulation" was planned but not run (search budget).

## 5. Has any work treated learned (deep JSCC) codes as linear-analog-like for system design, or combined analog/HDA codes with cellular schedulers (PF, LTE)?

### Takeaway
No work was found that treats a deep JSCC code as "linear-analog-like" for system design: an MMSE-sum abstraction, effective-SNR replacement or scheduling. In the analog literature, learning appears **inside** pseudo-analog pipelines: DNN-predicted power/energy metadata and perception-oriented pseudo-analog with deep learning. The only cellular integrations found are at the PHY (SoftCast in the LTE frame on an SDR; LTE-like OFDM grids in simulation). None includes a PF or CQI-driven MAC scheduler.

### Cited Findings
- **SoftCast in the LTE frame.** SoftCast was implemented in the LTE physical-layer frame structure on an SDR, with 3–5 dB over SoftCast (ICCC 2016) — [BIT Pure](https://pure.bit.edu.cn/en/publications/pseudo-analog-video-transmission-based-on-lte-physical-layer/).
- **Perception-oriented pseudo-analog with deep learning.** "Human-Perception-Oriented Pseudo Analog Video Transmissions with Deep Learning" (arXiv:2005.09302; authors not captured; the scheme is called ROIC-Cast in one summary) maps coefficients onto a 1.4 MHz, 64-subcarrier LTE-style OFDM grid. It picks the MCS from a target BLER and compares against three other pseudo-analog schemes in simulation — [arXiv](https://arxiv.org/pdf/2005.09302).
- **A pseudo-analog SDR system.** "A Design of SDR-based Pseudo-Analog Wireless Video Transmission System" (Tang and Huang, Tongji; arXiv:2005.04558) builds OFDM pseudo-analog video on GNU Radio/USRP without quantization or channel coding — [arXiv](https://arxiv.org/abs/2005.04558v1).
- **DNN-estimated power metadata.** A MERL paper argues that the fitting functions SoftCast uses for power allocation suffer model mismatch on real single- and multi-view signals. It proposes a DNN to estimate the power values, keeping metadata overhead near zero — [MERL TR2019-133 PDF](https://shadow.merl.com/publications/docs/TR2019-133.pdf) (title not captured).
- **A possible future scope.** The soft-delivery survey's abstract says soft delivery could become a standard for delivering DNN models and tactile information — [MERL TR2023-102](https://merl.com/publications/TR2023-102).
- **DeepJSCC on a real PHY.** DeepStream prototypes DeepJSCC over OFDM and notes that DeepJSCC has largely stayed in simulation — [arXiv 2509.05971](https://arxiv.org/html/2509.05971v1) (already in SURVEY F4).

### Inferences
- The link we draw (a learned nonlinear JSCC obeying SoftCast's per-component MMSE form, only when trained for slot-varying SNR) does not appear in either community in what was found. Analog works use learning to *improve* a linear code; deep JSCC works compare against SoftCast at most as a baseline.
- For the "Analog symbols are not NR-compatible" objection, Ding et al. (ICCC 2016) shows a pseudo-analog payload can live inside an LTE PHY frame on real hardware. It is a useful citation, though it says nothing about MAC scheduling.

### Gaps
- No head-to-head deep JSCC versus SoftCast benchmark was located. A search summary mentioned a CNN-based soft broadcast with up to 1.5 dB over SoftCast, but the underlying paper could not be identified, so it is not cited.
- Analog/HDA combined with PF or LTE MAC scheduling: none found, from only two queries.

## 6. Which analog/HDA works are closest to ours, and what is the overlap verdict per contribution?

### Takeaway
There are five closest works.
1. **ParCast/ParCast+** (TMM 2014): the per-component MMSE objective and the matching of important components to strong subchannels.
2. **He et al., progressive pseudo-analog** (TMM 2017 / VCIP 2015): a progressive, prefix-like analog stream with scheduling and water-filling under SNR and bandwidth variation, plus a statistical-CSI approximation.
3. **Cui et al.** (INFOCOM 2014; JSTSP 2015): distortion-optimal power and channel-use allocation under statistical CSI for unicast and multicast, with outage PSNR, and the closest digital baseline (rate adaptation + H.264 SVC).
4. **A-HDAVT** (IEEE, ~2015–16, details unverified): power allocation from channel prediction, with single- and multi-user cases.
5. **Zheng, Cagnazzo and Kieffer** (SPIC 2019): MSE-optimal precoding over unequal subchannels, with mismatched-channel analysis.

LayerCast (TCSVT 2015) and Lee–Petersen (1976) are honourable mentions for layered/prefix decoding and for the theory of [A] and [B].

Together these works pre-empt **the form of our abstraction** and **the principle that important parts go to good channels**. None of them:
- schedules several unicast users over time to a deadline;
- uses stale per-slot reports inside a scheduler;
- trains the code on the channel the scheduler creates;
- compares against HEVC + CQI + HARQ.

### Cited Findings
- ParCast+: important source components go to the highest-gain channel components, with joint power allocation, in single-user MIMO-OFDM unicast — [MSR](https://www.microsoft.com/en-us/research/?p=168037).
- Progressive pseudo-analog: scheduling with optimal water-filling power under SNR and bandwidth variation, with a statistical-CSI approximation — [MSR](https://microsoft.com/en-us/research/?p=381044), [MSR](https://microsoft.com/en-us/research/?p=238252).
- Cui et al.: power and channel-use reallocation under statistical CSI (unicast and multicast), and outage PSNR — [MSR](https://www.microsoft.com/en-us/research/?p=166043). The digital baseline with robust rate adaptation + H.264 SVC trails by up to 5.3 dB — [MSR](https://www.microsoft.com/en-us/research/?p=168039).
- A-HDAVT: channel-prediction-based power-distortion optimization, single- and multi-user — [IEEE Xplore](https://ieeexplore.ieee.org/document/7123183/authors).
- Zheng, Cagnazzo and Kieffer: MSE-optimal precoders over parallel subchannels and mismatch analysis — [SPIC record](https://www.datalearner.com/academic/journal-papers/0923-5965/volumes-and-issues/248/paper-detail/40576).
- LayerCast: users decode a prefix of layers according to their bandwidth, with optimal power between layers and chunks — [HIT record](https://scholar.hit.edu.cn/en/publications/layered-soft-video-broadcast-for-heterogeneous-receivers/).
- Lee–Petersen: the optimal linear vector code pairs the largest source eigenvalues with the channel noise eigenvalues in order — [UNM](https://digitalrepository.unm.edu/ece_etds/436).

**Verdict per contribution.** The overlap levels are my judgement from the findings above.

| Our contribution | Closest analog/HDA precedent(s) | Overlap | What remains ours |
|---|---|---|---|
| [A] Additive, sensitivity-weighted per-phase MMSE abstraction | SoftCast (by construction), ParCast+, Xiong et al. PDO, Lee–Petersen, Zheng et al., Cui et al. (expected distortion) | **HIGH for the form** (cite it; do not claim it). phi(n) = n/(1+n/κ) is algebraically SoftCast's per-chunk LLSE error (section 2 Inferences) | A *learned nonlinear* code obeys the form, with weights measured per image, only when trained for slot-varying SNR; it is used as a **link-to-system abstraction for a scheduler**; MIESM/EESM shown to fail. No analog paper discusses ESMs at all |
| [B] Important phases in good slots | ParCast+ (frequency/space pairing), Lee–Petersen, Cui et al. (diversity for high-priority data) | **MED-HIGH for the principle** | Done in **time**, online and causally, across competing users with reports, and ex post through the scheduler instead of a static pairing |
| [C] Multi-user deadline scheduling (finite-horizon RMAB, Lagrangian bound, Whittle/Lagrangian indices, closed-form rule) | None in analog/HDA. Nearest: He et al. (single-stream share scheduling); A-HDAVT (multi-user, details unknown). Digital: Liebl et al. 2005; Zhao and Lin 2014; Hosseini and Panwar (SURVEY F2) | **LOW from analog** | Essentially all of it, within the analog/HDA corpus searched |
| [D] Stale/outdated CSI reports | Cui et al. (statistical CSI); A-HDAVT (channel prediction); SPIC 2020 (CSI-noise water-filling); Zheng et al. (mismatch) | **LOW-MED** (encoder-side robustness, single link or group) | Staleness handled in the **scheduler** (report kernel inside per-user programs; the bound still holds for report-only schedulers) |
| [E] Comparison with HEVC + LTE CQI + HARQ, with report errors | Cui et al. JSTSP 2015 (robust rate adaptation + H.264 SVC, up to 5.3 dB); SoftCast/WaveCast/LayerCast vs H.264/SVC broadcast (5.5–8 dB) | **LOW** | First comparison found against a CQI-driven, HARQ-equipped modern digital stack under report errors; margin smaller (1.3–1.7 dB) because the setting removes broadcast's common-rate penalty |
| [F] Delay | He et al. (live/mobile streaming, progressive); digital deadline schedulers | **LOW** | Delay as the latency cost of opportunism, measured within frames and with arrivals |
| [G] Training the code on the scheduler-induced SNR distribution | None (analog codes are not trained). The nearest idea is adapting power to a predicted or statistical channel (A-HDAVT, Cui et al.) | **LOW** | All of it |
| [H] Prefix-decodable successive-refinement phases | He et al. progressive pseudo-analog; LayerCast (layer prefixes); Analog Coded SoftCast (repetition/expansion) | **MED** (progressive or layered analog streams exist) | Learned code with phases whose quality depends on each phase's slot SNR, scheduled across users |
| [I] Mean PSNR at the deadline / share reaching a PSNR target | Cui et al. outage PSNR; digital equal-quality scheduling (arXiv 1311.5921) | **LOW-MED** | Quality targets as scheduler objectives for analog-like codes |

### Inferences
- The defensible novelty claim against the analog/HDA literature is the combination: a learned code shown to obey the analog MMSE-sum abstraction (and not MIESM), used as the reward model of a multi-user deadline scheduler with stale reports, plus code–scheduler co-training. Each single piece of [A] or [B] has a strong analog precedent and should be cited as such. The PAPER.md wording ("The MMSE form is SoftCast's ... yes for linear analog codes, by construction") matches what was found.
- The two works a reviewer is most likely to raise are He et al. (TMM 2017), for "progressive analog plus scheduling plus water-filling under fading", and ParCast+, for "important parts on good channels". Both should be cited and distinguished explicitly: single user, no deadline across users, no report staleness, and a linear code.

### Gaps
- A-HDAVT and He et al. (TMM 2017) were judged from abstracts only. Their full texts could show multi-user time scheduling and should be read before the novelty claim for [C] is finalized. The Fujihashi survey (ACM CSUR 2023, arXiv 2111.08189) should also be skimmed for a "user scheduling" or "resource allocation" section.
- The search budget ended before these were checked: SoftCast's exact power rule; ParCast's MobiCom 2012 DOI; Supcast's scheduling details; any analog/HDA + HARQ, MU-MIMO user selection or OFDMA unicast paper; and deep JSCC vs SoftCast head-to-head results.
