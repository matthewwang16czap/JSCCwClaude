# Rate-distortion-optimized and content-aware multimedia scheduling (prior work for deadline scheduling of learned image codes)

*How these notes were made (2026-10-20).* The shared web-search budget for this session ran out after the first round of searches. WebFetch could not resolve most academic hosts (IEEE Xplore, arXiv, dblp, EPFL, Stanford, EURASIP, HAL, KOPS); only microsoft.com resolved. Each item is therefore tagged with how it was verified:
**[read]** full text read this session. Only RaDiO, from the Microsoft Research PDF, extracted with pdftotext.
**[meta]** title, authors and venue, plus an abstract-level summary, verified through a search result this session.
**[bib]** the authors, title and venue are printed in RaDiO's reference list (a primary source), but the content was not read.
**[survey]** taken from `jscc/docs/SURVEY.md` and not re-checked this session.
Works I know of but could not verify are listed only under **Gaps**, as *unverified leads*.

Labels for our components, used throughout:
K1 is the prefix-decodable successive-refinement learned analog code (up to 6 phases).
K2 is the calibrated additive, sensitivity-weighted per-phase MMSE link abstraction, D = Dsrc(L) + C(L)·Σρ_L[j]φ(n_j). Capacity-averaging MIESM/EESM fails for this code.
K3 is the finite-horizon restless bandit. A Lagrangian relaxation of the per-slot budget gives one dynamic program per user and an upper bound on every causal scheduler.
K4 is the pair of indices from the relaxation: the Whittle index and the Lagrangian (LP-priority, bid-price-like) index at the bound's per-slot prices.
K5 is the closed-form advantage index (gain this slot minus gain in a typical slot at the mean SNR), plus equal-slope phase quotas.
K6 is the PSNR-target objective: admission of the smallest remaining needs, and online stopping.
K7 covers stale CSI, the digital HEVC + CQI + HARQ baseline, and delay.
K8 is training the code on the SNR distribution that the scheduler induces.

## Q1. RaDiO and its successors: how close is per-data-unit MDP + Lagrange multiplier + deadlines to "one DP per user plus a price on the shared resource"? Does any version extend it to multiple users sharing a fading channel?

### Takeaway
The machinery is very close; the arms and the channel are not. RaDiO decomposes deadline-constrained streaming into one finite-horizon MDP per data unit, priced by a Lagrange multiplier on rate, and re-solves at every transmission opportunity. Its rate-controlled version sends the one data unit with the largest *threshold multiplier* at each opportunity: an unnamed indifference-price (Whittle-type) index policy. But its arms are data units of one stream, sent over an erasure/delay channel with ACK feedback. There is no per-user channel state to exploit, units are coupled through "sensitivities" by a heuristic (locally optimal ISA), and its bound is a channel-capacity bound, not a relaxation bound. The only verified multi-stream extension is Chakareski & Frossard (TMM 2006): senders share a (wireless-LAN) channel through importance-proportional bandwidth shares. No verified RaDiO extension schedules multiple users over a block-fading channel from per-slot CSI.

### Cited Findings
**Chou & Miao, RaDiO [read]**
- Citation: P. A. Chou and Z. Miao, "Rate-distortion optimized streaming of packetized media," *IEEE Trans. Multimedia*, vol. 8, no. 2, pp. 390–404, Apr. 2006, DOI 10.1109/TMM.2005.864313. The manuscript was received Mar. 2001; the earlier version is tech. rep. MSR-TR-2001-35, Feb. 2001. — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Setting: a server sends prestored media to one client. The network "loses (i.e., drops) or corrupts packets at random, and delivers those packets that it does not lose after a random delay." A back channel is optional and carries ACKs (sender-driven) or requests (receiver-driven). Every data unit has a delivery deadline. The objective is to "minimize the expected distortion subject to constraint on the expected rate, where the expectations are taken over channel realizations." Distortion is "incrementally additive." Data units depend on each other through a DAG. — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Single data unit ("Scenario E", sender-driven with ACKs): a finite-horizon MDP over the unit's transmission opportunities before its deadline. The actions are send / not send, the observation is ACK / no ACK, and a state is the action–observation history; an ACK ends the process. The policy minimises error + λ'·cost, where error is the late-or-lost probability and cost is the expected transmissions per source byte. It is solved by dynamic programming or branch and bound; the paper calls this the "transmit-one (X1)" algorithm. Each λ' traces one vertex of the convex hull of the operational error–cost function. — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Group of units ("Iterative Sensitivity Adjustment", ISA): because of the DAG, the expected distortion "cannot be split into a sum of terms each depending on only a single" policy. ISA minimises one unit at a time with the others fixed. Each unit then faces a single-unit problem weighted by its *sensitivity*: "the amount by which the expected distortion will increase if data unit l cannot be recovered at the receiver, given the current transmission policies for the other data units", i.e. a partial derivative. The paper gives two guarantees. Convergence: the objective is "non-increasing and bounded below." A Proposition: a necessary condition for optimality is that each unit's policy minimises its own Lagrangian for some multiplier. — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- With feedback, RaDiO re-runs ISA at every transmission opportunity on quantities conditioned on what has been fed back ("stepwise rate-distortion optimization"). In the authors' words this "factors the full state space into a product of state spaces, one for each data unit (or agent)… the coupling is loose." The truly optimal joint solution has a state space "that grows exponentially in the number of data units as well as the number of transmission opportunities": Podolsky et al. found it "intractable when there are more than two data units and more than two transmission opportunities." The authors call the factorisation "one of the main contributions of our work." — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Deadlines enter through *window control*: a unit may be transmitted only in a window of opportunities before its delivery deadline, and the windows advance monotonically. — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- **The rate-controlled RaDiO is an index policy.** λ "is adjusted so that exactly one data unit is selected for transmission at each transmission opportunity… creating a list of Lagrange multipliers λ_l, where for each data unit λ_l is the threshold for λ above which the data unit is not transmitted, and below which the data unit is transmitted… the data unit to transmit is the data unit with the largest λ_l." λ_l is set to 0 in three cases: the unit is within a forward trip time of its deadline, it has been ACKed, or it was sent within the last RTT. Otherwise λ_l is a closed-form importance-based estimate, reduced by a fixed factor for each unacknowledged transmission (the exact symbols were lost in text extraction). The authors report "very little penalty (a fraction of a dB)" for this constant-rate algorithm, and a Java server using "about one percent of the CPU on a 700 MHz Pentium III, for a 40 Kbps audio stream." — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- The experiments use a prefix code. A scalable WMA audio codec produces "an embedded bit string for each group of frames (GOF)" of about 0.75 s. Twelve 500-byte data units per GOF "are sequentially dependent", and "each data unit is labeled by the decrease in the perceptually weighted squared error if the data unit is decoded on time and all of its predecessors in the same GOF are decoded on time." RaDiO gains 2–6 dB over systems that are not rate-distortion optimised. — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- The bound: "System 5" is the operational D(R) of the source at the capacity of the packet-erasure channel, R(1−ε). The paper concludes that "no streaming system can achieve substantially better than the systems presented." — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Conference versions, all on Microsoft Research pages: "Rate-distortion optimized sender-driven streaming over best-effort networks" (MMSP 2001; reports 3–7 dB); "Rate-distortion optimized receiver-driven streaming over best-effort networks"; and "Rate-distortion optimized streaming from the edge of the network" (Chakareski, Chou, Girod, MMSP 2002, pp. 49–52 per RaDiO ref. [66]). — [MSR p=155215](https://www.microsoft.com/en-us/research/?p=155215); [MSR p=155221](https://www.microsoft.com/en-us/research/?p=155221); [MSR p=155207](https://www.microsoft.com/en-us/research/?p=155207)

**RaDiO for a wireless client (single user)**
- J. Chakareski and P. A. Chou, "Application layer error correction coding for rate-distortion optimized streaming to wireless clients," Proc. IEEE ICASSP 2002, vol. 3, pp. 2513–2516. It streams "packetized media over a lossy packet network to a wireless client" and uses incremental-redundancy coding that handles both packet loss and bit errors end-to-end, with no help from the base station. Packets are chosen to minimise mean end-to-end distortion under an average-rate constraint. It gains up to 8 dB over systems that are not rate-distortion optimised at BER up to 1e-2. [meta] — [MSR page](https://www.microsoft.com/en-us/research/publication/application-layer-error-correction-coding-for-rate-distortion-optimized-streaming-to-wireless-clients/)
- RaDiO's bibliography also lists a journal version in *IEEE Trans. Commun.*, pp. 1675–1687; the year and volume are garbled in our extraction. It also lists J. Chakareski, P. A. Chou and B. Aazhang, "Computing rate-distortion optimized policies for streaming media to wireless clients," DCC 2002, pp. 53–62. [bib] — [PDF refs 56–59](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)

**Multiple streams sharing one channel**
- J. Chakareski and P. Frossard, "Rate-distortion optimized distributed packet scheduling of multiple video streams over shared communication resources," *IEEE Trans. Multimedia*, vol. 8, no. 2, pp. 207–217, Apr. 2006 [meta].
  - Setting: multiple senders share a channel and coordinate their transmission schedules; the focus is wireless-LAN scenarios.
  - Method: each packet is described by two numbers, its size and its *importance* to reconstruction quality. Each sender allocates bandwidth to its own packets in proportion to their importance, which trades rate against distortion across streams.
  - Results: up to 6 dB gain for bandwidth adaptation and 10 dB for loss adaptation, against a conventional scheduler that ignores importance. A 2005 conference version reports up to 8 dB.
  - Sources: [NJIT record](https://researchwith.njit.edu/en/publications/rate-distortion-optimized-distributed-packet-scheduling-of-multip/); [EPFL PDF](https://infoscience.epfl.ch/record/87338/files/tmm2006.pdf?version=1); [2005 version](https://researchwith.njit.edu/en/publications/distributed-packet-scheduling-of-multiple-video-streams-over-shar/); related ICME 2005 abstract "Rate-distortion optimized bandwidth adaptation for distributed media delivery" ([UCI ICME05](https://cecs.uci.edu/~papers/icme05/defevent/html/abstract/a1452.htm)).

**Successors and refinements**
- M. Röder, J. Cardinal, R. Hamzaoui, "Efficient rate-distortion optimized media streaming for tree-structured packet dependencies," *IEEE Trans. Multimedia*, vol. 9, no. 6, pp. 1259–1272, 2007, DOI 10.1109/TMM.2007.902872 [meta]. The title is as I recall it; one search summary rendered it without "media", so the DOI is the reliable identifier.
  - Content: when the dependency DAG reduces to a tree (MPEG-1 frame-level packetization; H.264 allows it), dynamic programming finds the RD-optimal policies, much faster than earlier exact methods for arbitrary DAGs.
  - Companion papers: SPIE MMCN 2006 (DOI 10.1117/12.651229); ICASSP 2006, "Constructing dependency trees for rate-distortion optimized media streaming" (DOI 10.1109/ICASSP.2006.1661290); ICIP 2005, "Dynamic programming algorithm for rate-distortion optimized media streaming", which treats optimal single-packet policies (DOI 10.1109/ICIP.2005.1530018).
  - Sources: [KOPS](https://kops.uni-konstanz.de/entities/publication/866013b9-902f-42fb-9874-af574f473d43); [SPIE](https://ebooks.spiedigitallibrary.org/conference-proceedings-of-spie/6071/607104/Efficient-rate-distortion-optimized-media-streaming-for-tree-reducible-packet/10.1117/12.651229.full)
- RaDiO cites Röder, Cardinal, Hamzaoui, "On the complexity of rate-distortion optimal streaming of packetized media," DCC 2004, for the branch-and-bound X1. [bib] — [PDF ref 94](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Kalman, Girod and colleagues [bib]: M. Kalman, E. Steinbach, B. Girod, "R-D optimized media streaming enhanced with adaptive media playout," ICME 2002, pp. 869–872; the same authors, "Rate-distortion optimized video streaming with adaptive playout," ICIP 2002, vol. 3, pp. 189–192; M. Kalman, P. Ramanathan, B. Girod, "Rate-distortion optimized video streaming with multiple deadlines," ICIP 2003. — [PDF refs 69–71](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Chakareski and colleagues [bib]: J. Chakareski and B. Girod, "Rate-distortion optimized packet scheduling and routing for media streaming with path diversity," DCC 2003, pp. 203–212; "Server diversity in rate-distortion optimized streaming of multimedia," ICIP 2003; "Rate-distortion optimized video streaming with rich acknowledgments," SPIE VCIP 2004. J. Chakareski, P. A. Chou, B. Girod, "Computing rate-distortion optimized policies for hybrid receiver/sender driven streaming of multimedia," Asilomar 2002. J. Chakareski and P. A. Chou, "RaDiO Edge: rate-distortion optimized proxy-driven streaming from the network edge," *IEEE/ACM Trans. Networking*, listed as "submitted." — [PDF refs 60, 63, 66–68, 72](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Other RaDiO-framework papers [bib]: A. Sehgal and P. A. Chou, "Cost-distortion optimized streaming media over DiffServ networks," ICME 2002; A. C. Begen, Y. Altunbasak, M. A. Begen, "Rate-distortion optimized on-demand media streaming with server diversity," ICIP 2003; G. Cheung and W.-T. Tan, "Directed acyclic graph based source modeling for data unit selection of streaming media over QoS networks," ICME 2002; R. Zhang, S. L. Regunathan, K. Rose, "End-to-end distortion estimation for RD-based robust delivery of pre-compressed video," Asilomar 2001, and "Optimized video streaming over lossy networks with real-time estimation of end-to-end distortion," ICME 2002; B. Girod, M. Kalman, Y. Liang, R. Zhang, "Advances in channel-adaptive video streaming," *Wireless Commun. Mobile Comput.*, vol. 2, no. 6, 2002. — [PDF refs 62, 64, 73–75, 78](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Predecessors [bib]:
  - Z. Miao and A. Ortega, "Optimal scheduling for streaming of scalable media," Asilomar 2000, pp. 1357–1362.
  - Z. Miao and A. Ortega, "Expected run-time distortion based scheduling for delivery of scalable media," Packet Video Workshop 2002.
  - Z. Miao, PhD thesis, USC, 2002.
  - M. Podolsky, S. McCanne, M. Vetterli, "Soft ARQ for layered streaming media," *J. VLSI Signal Process.*, vol. 27, no. 1–2, pp. 81–97, 2001. This is the full joint MDP that RaDiO calls intractable beyond two units and two opportunities.
  - D. A. Turner and K. W. Ross, "Optimal streaming of layer-encoded multimedia presentations," ICME 2000.
  - Source: [PDF refs 3–5, 7–8, 14](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)

### Inferences
- **How RaDiO maps onto our frame.**
  - RaDiO's data unit corresponds to one phase of one user, and a transmission opportunity to a slot.
  - The rate multiplier λ corresponds to a per-slot price.
  - The single-unit MDP (X1) corresponds to a user's dynamic program. RaDiO's program is per unit; ours is per user, with state (phases held, accumulated weighted noise).
  - The sensitivity-weighted distortion increment ΔD_l·S_l corresponds to C(L)·ρ_L[j]·(distortion reduction of phase j).
  - The threshold multiplier λ_l, with "largest λ_l is sent", is our Whittle index: a user's indifference price for this slot.
  - "Exactly one unit per opportunity" corresponds to B resource units per slot.
  - Re-running ISA every opportunity corresponds to re-solving each slot.
  - The closed-form λ_l approximation, which costs "a fraction of a dB", corresponds to our closed-form advantage index.
  - Together, these make RaDiO the earliest verified multimedia ancestor of K3 and K4. Our paper should cite it explicitly as one, or a reviewer will.
- **What RaDiO lacks against ours.**
  - (i) No channel state per opportunity. Losses are i.i.d. and the value of waiting comes only from ACK timing and deadlines, so it has none of the *opportunism* that drives our advantage index (gain now versus gain in a typical slot).
  - (ii) Arrival is binary: a unit is delivered or not. Our phases degrade gracefully, with φ(n) of the slot's noise.
  - (iii) A single client. The resource is the sender's rate, not a per-slot capacity contested by K users.
  - (iv) Arms are coupled through the DAG and ISA, giving a local optimum. In our frame, users are independent once priced, and the within-user coupling sits inside the user's DP state.
  - (v) Its "bound" is an information-theoretic capacity bound, valid for every system and blind to deadlines and causality. It is not the Lagrangian relaxation bound on every causal scheduler that K3 provides.
  - (vi) No quality-target or share-of-users objective (K6), no stale channel reports (K7), and no co-design of the code (K8).
- The multi-user, fading version of RaDiO appears to have moved to a different line of work: the content-aware gradient schedulers of Q2 and the price-coordinated multi-user MDPs of Q3. Those papers dropped RaDiO's per-unit lookahead (gradient rules) or the per-slot index (price coordination). Our K3/K4 recombines the two.

### Gaps
- I could not read Chakareski & Frossard (TMM 2006) in full. Still open: how the "proportional to importance" shares are computed, whether fading or CSI is modelled, and whether a centralised RaDiO optimum or a bound is compared.
- Unverified leads; check before citing:
  - Röder, Cardinal, Hamzaoui, "Branch and bound algorithms for rate-distortion optimized media streaming," *IEEE TMM* 8(1), 2006.
  - The final venue of "RaDiO Edge": I recall *IEEE/ACM ToN* 14(6), 2006.
  - The final venue, volume and year of the Chakareski & Chou *IEEE Trans. Commun.* paper: I recall vol. 52, no. 10, 2004.
  - Kalman & Girod's follow-ups on conditional packet-delay distributions and multiple deadlines.
  - Chakareski, Apostolopoulos, Wee, Tan, Girod, "Rate-distortion hint tracks for adaptive video streaming," *IEEE TCSVT* 2005.
  - Y. Li, A. Markopoulou, J. Apostolopoulos, N. Bambos, "Content-aware playout and packet scheduling for video streaming over wireless links," *IEEE TMM* 2008. I recall it covers multiple streams on a shared wireless link, and it may be the closest RaDiO-style multi-user wireless paper.
  - S. H. Kang and A. Zakhor, "Packet scheduling algorithm for wireless video streaming," Packet Video 2002.
  - G. Liebl, M. Kalman, B. Girod, "Deadline-aware scheduling for wireless video streaming," ICME 2005.
- Not verified: whether any RaDiO paper treats a Markov-modulated (Gilbert–Elliott) or Rayleigh block-fading channel with CSI at the sender, and multiple users.

## Q2. Content-aware multi-user video scheduling over wireless downlinks: do these schedulers weigh marginal distortion reduction against channel quality, like our advantage index? Do they handle deadlines or quality targets, or certify near-optimality?

### Takeaway
They do weigh content against channel. The canonical design is a *gradient (utility-gradient) scheduler*, which serves the user maximising (content-derived weight) × (instantaneous achievable rate). The weight comes from the distortion of the received video. In the SVC/OFDM version it also includes deadline urgency and past transmission results. These are myopic per-slot rules. Deadlines enter as urgency weights or frame-delivery constraints, not as a finite-horizon program. No verified paper in this line optimises the share of users reaching a quality target, or bounds its gap to the optimal causal schedule.

### Cited Findings
- **Pahalawatta, Berry, Pappas, Katsaggelos [meta].** P. V. Pahalawatta, R. Berry, T. N. Pappas, A. K. Katsaggelos, "Content-aware resource allocation and packet scheduling for video transmission over wireless networks," *IEEE J. Sel. Areas Commun.*, vol. 25, no. 4, pp. 749–759, 2007 — [dblp](https://dblp.org/pid/29/4430). The precursor is "A content-aware scheduling scheme for video streaming to multiple users over wireless networks" (EUSIPCO 2006), "a gradient-based scheduler whose utility functions account for the distortion of the received video." — [EUSIPCO 2006 PDF](https://new.eurasip.org/Proceedings/Eusipco/Eusipco2006/papers/1568982331.pdf)
- **Huang, Li, Chiang, Katsaggelos [meta].** J. Huang, Z. Li, M. Chiang, A. K. Katsaggelos, "Joint source adaptation and resource allocation for multi-user wireless video streaming," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 18, no. 5, pp. 582–594, May 2008 (IEEE article no. 4455569). The algorithm, JARS, does three things. It "allocates the communication resource based on the video users' quality of service", it "adapts video sources based on smart summarization", and it "schedules the transmissions to meet the frame delivery deadlines." The paper reports near-full use of network resources and all frame deadlines met. Its keywords are optimization decomposition, pricing control and rate-distortion modeling. — [Princeton portal](https://collaborate.princeton.edu/en/publications/joint-source-adaptation-and-resource-allocation-for-multi-user-wi/)
- **Ji, Huang, Chiang, Lafruit, Catthoor [meta].** X. Ji, J. Huang, M. Chiang, G. Lafruit, F. Catthoor, "Scheduling and resource allocation for SVC streaming over OFDM downlink systems," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 19, no. 10, pp. 1549–1555, 2009.
  - Setting and method: multi-user SVC video over a downlink OFDM channel. A gradient-based scheduler weights users by content, deadlines and past transmission results.
  - Result: up to 6 dB average-PSNR gain over content-blind and deadline-blind schedulers when the network is congested.
  - Other versions: the conference paper "Downlink OFDM scheduling and resource allocation for delay constraint SVC streaming" (2008), and a 2011 book chapter.
  - Sources: [Princeton portal](https://collaborate.princeton.edu/en/publications/scheduling-and-resource-allocation-for-svc-streaming-over-ofdm-do/); [imec record](https://imec-publications.be/handle/20.500.12860/15540); [conference version](https://collaborate.princeton.edu/en/publications/downlink-ofdm-scheduling-and-resource-allocation-for-delay-constr/)
- **Titles verified, content not checked [meta, title only]:**
  - "Adaptive resource allocation and frame scheduling for wireless multi-user video streaming" (TUM portal) — [TUM](https://portal.fis.tum.de/de/publications/adaptive-resource-allocation-and-frame-scheduling-for-wireless-mu/)
  - "Content-aware packet scheduling for multi-session video streaming" (BUAA portal) — [BUAA](https://research.buaa.edu.cn/zh/publications/content-aware-packet-scheduling-for-multi-session-video-streaming/)
  - "An intelligent fuzzy logic based content and channel aware downlink …" (Kingston portal) — [Kingston](https://researchinnovation.kingston.ac.uk/en/publications/an-intelligent-fuzzy-logic-based-content-and-channel-aware-downli-2/)
  - "Multi-user video streaming over multiple heterogeneous wireless networks: a distributed, cross-layer design paradigm" — [HAL](https://hal-ensta-paris.archives-ouvertes.fr/hal-01229092)
- Already in the survey: PSNR-aware OFDMA video scheduling (PMC4764368) and SVC scheduling over OFDM (Ji et al., now re-checked above). [survey] — [SURVEY.md §F4](/home/user/JSCCwClaude/jscc/docs/SURVEY.md)

### Inferences
- **Our advantage index (K5) compared with gradient scheduling.** Both rank users by a content term × a channel term. Our index subtracts the gain the user would get from a typical slot at its mean SNR. That normalises each user against its own channel distribution, much as proportional fairness divides by average rate, so the index is opportunistic as well as content-aware. A plain "marginal distortion reduction × instantaneous rate" gradient rule lacks that baseline. Ji et al.'s "past transmission results" weight may play a similar role (not verified).
- **Deadlines.** In this family they are handled myopically, through urgency weights (Ji et al.) or by meeting frame deadlines inside a per-period optimisation (Huang et al.). Our per-user DP plans over the T-slot horizon instead. That is the RaDiO-style lookahead these schedulers gave up.
- **Prices.** Huang et al.'s "optimization decomposition, pricing control" means a price on the shared resource also appears there. It is a static, convex network-utility-maximisation price per scheduling period, not the per-slot price of a finite-horizon relaxation (K3/K4).
- **Certification and targets.** No verified paper in this family gives an upper bound to measure its gap against, or optimises the *share* of users reaching a PSNR target (K6). Those two remain differentiators, unless the unverified leads under Gaps show otherwise.
- **Distortion model (K2).** These schedulers use packet-loss and concealment-based expected distortion for a digital codec, a binary per-packet outcome. Our abstraction maps per-phase noise to distortion for an analog code. The additive form is shared; the physical meaning is not.

### Gaps
- Full texts were unavailable. Still unknown: Pahalawatta's exact utility-gradient formula (per-packet distortion weights, error concealment, packet-deadline handling) and Ji et al.'s exact weights.
- Unverified leads (classic; check before citing):
  - E. Maani, P. Pahalawatta, R. Berry, T. Pappas, A. Katsaggelos, "Resource allocation for downlink multiuser video transmission over wireless lossy networks," *IEEE TIP* 2008.
  - P. Pahalawatta and A. Katsaggelos, a review of content-aware resource allocation for wireless video, *Wireless Commun. Mobile Comput.* 2007.
  - X. Zhu and B. Girod, "Distributed media-aware rate allocation for wireless video streaming," *IEEE TCSVT* 2010.
  - S. Khan, Y. Peng, E. Steinbach, M. Sgroi, W. Kellerer, "Application-driven cross-layer optimization for video streaming over wireless networks," *IEEE Commun. Mag.* 2006.
  - The gradient-scheduling theory anchors: Agrawal & Subramanian (Allerton 2002) and Stolyar (*Oper. Res.* 2005).
  - Y. Li et al., *IEEE TMM* 2008 (see Q1).

## Q3. Mihaela van der Schaar's group: cross-layer MDPs for delay-sensitive multimedia, multi-user problems decomposed into per-user MDPs through priced resources. Exactly what do they decompose, and how?

### Takeaway
I verified this at abstract level only. Fu & van der Schaar (JSAC 2010) model multi-user wireless video as a multi-user MDP (MUMDP), with heterogeneous video traffic, time-varying channels and coupling between users. They decompose it by Lagrangian relaxation into per-user local MDPs. A network coordinator sets *resource prices*, each user solves its own cross-layer MDP against those prices, and online learning handles unknown statistics. This is the closest multimedia precedent for "one program per user + prices" (K3). As far as verified, it is a price-coordinated dual decomposition of a long-run problem. No index policy, finite-horizon image deadline frame, or causal upper bound is verified in it.

### Cited Findings
- **Fu & van der Schaar [meta].** F. Fu and M. van der Schaar, "A systematic framework for dynamically optimizing multi-user wireless video transmission," *IEEE J. Sel. Areas Commun.*, vol. 28, no. 3, pp. 308–320, 2010; arXiv 0903.0207 (Mar. 2009). The MUMDP model "accounts for differing video traffic, changing channel conditions, and coupling between users." It is split "into per-user local MDPs using Lagrangian relaxation, so each user can solve its own cross-layer optimization while a network coordinator sets resource prices." The abstract also describes "an online learning algorithm for cases where the environment statistics are unknown." Comparison: simulation against conventional approaches. — [arXiv listing](https://arxiv.org/abs/0903.0207); [dblp author page](https://dblp.uni-trier.de/pid/68/3683.html)

### Inferences
- **What is decomposed.** The coupling constraint is the shared wireless resource. Relaxing it with prices leaves each user one MDP over its own video and channel state. This is the weakly-coupled-MDP pattern behind our K3, applied to video by 2009–2010. The prices are set by a coordinator, presumably by iterating on the dual. In our frame, the relaxation's per-slot prices come out of one LP or DP solve. They are then used both as a *bound* and inside a Lagrangian (LP-priority) index that settles who gets each slot.
- **"Foresighted" against "myopic".** A foresighted user maximises long-run value (the MDP value function), a myopic user only the immediate reward. Our per-user DP is foresighted within a finite frame with a hard deadline. The advantage index is a one-step rule made foresighted by its certainty-equivalent baseline (the gain from a typical future slot).
- **Reviewer risk.** "Per-user MDPs coupled by resource prices for multi-user wireless video" is prior art (Fu & van der Schaar 2010), so our paper must not present the decomposition itself as new. The parts we can defend: the finite-horizon restless-bandit formulation for one image per user with a deadline; the upper bound on every causal scheduler, used as a certificate (our rule within ~0.26 dB at K = 8, and the gap shrinking with K); the index policies compared against that bound; and the reward from an analog learned code's calibrated abstraction.

### Gaps
- The full text was unreachable. Still unknown: whether the prices are per state/slot or uniform; whether the horizon is discounted infinite or finite; whether the dual value is reported as an upper bound; how per-slot conflicts are resolved; and what a user's state contains (I believe buffered packets with deadlines plus channel state).
- Unverified leads from this group (all need checking before citing):
  - F. Fu and M. van der Schaar, "Decomposition principles and online learning in cross-layer optimization for delay-sensitive applications," *IEEE Trans. Signal Process.* 2010. I recall a single user, with the MDP decomposed across protocol layers.
  - F. Fu and M. van der Schaar, "Structural solutions for dynamic scheduling in wireless multimedia transmission," *IEEE TCSVT* 2012. I recall DAG-structured data units with deadlines, priority graphs and post-decision value functions.
  - J. Xu, Y. Andreopoulos, Y. Xiao, M. van der Schaar, "Non-stationary resource allocation policies for delay-constrained video streaming: application to video over Internet-of-Things-enabled networks," *IEEE JSAC* 2014. I recall it turns deadlines into weights over a finite horizon and uses non-stationary (time-dependent) slot allocation, which would be close to our finite-horizon frame and is high priority to check.
  - H.-P. Shiang and M. van der Schaar, "Multi-user video streaming over multi-hop wireless networks: a distributed, cross-layer approach based on priority queuing," *IEEE JSAC* 2007.
  - F. Fu, T. M. Stoenescu, M. van der Schaar, "A pricing mechanism for resource allocation in wireless multimedia applications," *IEEE JSTSP* 2007.
  - N. Mastronarde and M. van der Schaar, "Online reinforcement learning for dynamic multimedia systems," *IEEE TIP* 2010, and "Fast reinforcement learning for energy-efficient wireless communication," *IEEE TSP* 2011.
  - M. van der Schaar and D. S. Turaga, "Cross-layer packetization and retransmission strategies for delay-sensitive wireless multimedia transmission," *IEEE TMM* 2007.

## Q4. Scalable/layered video and progressive images over fading or OFDMA channels with deadlines; utility-based scheduling of SVC layers; QoE targets (share of satisfied users); multi-user video in LTE/5G

### Takeaway
Scheduling layered or embedded media under deadlines is classical. SVC over an OFDM downlink with deadline-aware gradient scheduling is Ji et al. (2009). Restless-bandit SVC scheduling is Hosseini & Panwar (2016). For progressive images, the single-link literature allocates unequal error protection across an embedded bitstream; this is the static, digital, single-user ancestor of "put the important phases in the good slots." I could not verify any work in this session on (a) QoE-target or share-of-satisfied-users objectives, or (b) 2016–2026 LTE/5G multi-user video scheduling (search budget exhausted). Both remain open checks.

### Cited Findings
- **SVC over an OFDM downlink with deadlines:** Ji et al., *IEEE TCSVT* 2009 (details under Q2). [meta] — [Princeton portal](https://collaborate.princeton.edu/en/publications/scheduling-and-resource-allocation-for-svc-streaming-over-ofdm-do/)
- **SVC as a restless bandit:** Hosseini & Panwar, "Restless video bandits: optimal SVC streaming in a multi-user wireless network" (arXiv 1612.03985). It schedules SVC users with deadlines as a restless bandit, solved as an LP, and admits up to 2x more users at a given quality. [survey] — [SURVEY.md §F2](/home/user/JSCCwClaude/jscc/docs/SURVEY.md)
- **Related SVC multi-user paper (title only):** "SVC-based multi-user streamloading for wireless networks" (arXiv 1512.00137). [meta, title only] — [arXiv PDF](https://arxiv.org/pdf/1512.00137)
- **Scheduling scalable media by expected distortion:** the two Miao & Ortega papers (Asilomar 2000; Packet Video 2002), and Turner & Ross (ICME 2000) on layer-encoded presentations. [bib] — [PDF refs 3, 5, 14](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- **Embedded (prefix) code with deadlines in the RaDiO experiments:** 12 sequentially dependent units per GOF of a bitplane-coded audio stream, each labelled with its distortion decrement. [read] — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- **Progressive or embedded images over noisy, erasure or fading channels [bib]:**
  - V. Chande and N. Farvardin, "Progressive transmission of images over memoryless noisy channels," *IEEE JSAC* 18(6):850–860, 2000.
  - V. Chande, H. Jafarkhani, N. Farvardin, "Joint source-channel coding of images for channels with feedback," ITW 1998.
  - J. Lu, A. Nosratinia, B. Aazhang, "Progressive source-channel coding of images over bursty error channels," ICIP 1998.
  - M. J. Ruf and J. W. Modestino, "Operational rate-distortion performance for joint source and channel coding of images," *IEEE TIP* 8(3):305–320, 1999.
  - A. E. Mohr, E. A. Riskin, R. E. Ladner, "Unequal loss protection: graceful degradation of image quality over packet erasure channels through forward error correction," *IEEE JSAC* 18(6):819–829, 2000.
  - A. Albanese, J. Blömer, J. Edmonds, M. Luby, M. Sudan, "Priority encoding transmission," *IEEE TIT* 42:1737–1744, 1996.
  - V. Stankovic, R. Hamzaoui, Z. Xiong, "Packet loss protection of embedded data with fast local search," ICIP 2002.
  - S. Dumitrescu, X. Wu, Z. Wang, "Globally optimal uneven error-protected packetization of scalable code streams," *IEEE TMM* 6(2):230–239, 2004.
  - D. G. Sachs, R. Anand, K. Ramchandran, "Wireless image transmission using multiple-description based concatenated codes," SPIE VCIP 2000.
  - Source: [PDF refs 9, 21, 25–26, 32, 34–38](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- **Scalable or wireless video with error protection [bib]:**
  - U. Horn, K. Stuhlmüller, M. Link, B. Girod, "Robust Internet video transmission based on scalable coding and unequal error protection," *Signal Process.: Image Commun.* 15(1–2):77–94, 1999.
  - F. Zhai, R. Berry, T. N. Pappas, A. K. Katsaggelos, "A rate-distortion optimized error control scheme for scalable video streaming over the Internet," ICME 2003.
  - T. Stockhammer, H. Jenkac, G. Kuhn, "Streaming video over variable bit-rate wireless channels," *IEEE TMM* 6(2):268–277, 2004.
  - P.-C. Hu, Z.-L. Zhang, M. Kaveh, "Channel condition ARQ rate control for real-time wireless video under buffer constraints," ICIP 2000.
  - Source: [PDF refs 20, 24, 80, 83](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)

### Inferences
- **K1 (a prefix code scheduled under deadlines) is not new as a framing.** Embedded bitstreams (RaDiO's audio; progressive images; SVC layers) have been scheduled under deadlines since about 2000. What is new is the code type: a learned *analog* code whose phases degrade gracefully with each slot's SNR. With digital layers, a layer decodes or it does not, and link adaptation is per layer.
- **K2 relative to progressive-image JSCC.** UEP work allocates channel-code protection across an embedded stream for one link with a known channel law. It decides "how much protection each segment gets." Our scheduler decides "which slot (of random SNR) each phase gets, across users, online." The sensitivity-ordered matching is shared, but our version is dynamic, multi-user and analog.
- **K6 (share of users reaching a target).** Among verified multimedia works, only Hosseini & Panwar's "users admitted at a quality" is close (from the survey). Admitting the smallest remaining needs and stopping online once the realised SNRs predict the target has no verified multimedia precedent here.

### Gaps
- **Not verified, and should be searched in a follow-up:**
  - QoE-target, "video capacity" or share-of-satisfied-users objectives in LTE. Recalled leads: Singh, Oyman, Papathanassiou, Chatterjee, Andrews, "Video capacity and QoE enhancements over LTE," ICC 2012; Ramamurthi & Oyman, "Video-QoE aware radio resource allocation for HTTP adaptive streaming," ICC 2014; El Essaili et al., QoE-based resource management for adaptive HTTP video in LTE, *IEEE TCSVT* 2015.
  - Multi-user streaming via Lyapunov or network-utility maximisation: Bethanabhotla, Caire, Neely, *IEEE Trans. Commun.* 2015; Joseph & de Veciana, "NOVA," INFOCOM 2014.
  - SVC over OFDMA: Cicalò & Tralli, distortion-fair SVC over OFDMA, *IEEE TMM* 2014; Li, Chen, Tan, SVC over multi-user MIMO-OFDM, *IEEE TMM* 2013.
  - Progressive images over fading channels: Stankovic, Hamzaoui, Xiong, "Real-time error protection of embedded codes for packet erasure and fading channels," *IEEE TCSVT* 2004; Hamzaoui, Stankovic, Xiong, "Optimized error protection of scalable image bit streams," *IEEE SPM* 2005; Sherwood & Zeger, *IEEE SPL* 1997; Nosratinia, Lu, Aazhang, *IEEE Trans. Commun.* 2003.
- **2016–2026 follow-ups:** none verified this session (DRL multi-user video or QoE scheduling in 5G, semantic-aware video scheduling). This is a known hole in these notes.
- **Authors and venue of arXiv 1512.00137** are not verified. I believe it is from the Panwar/de Veciana groups.

## Q5. Has any multimedia scheduling paper explicitly used restless bandits, Whittle indices or Lagrangian bounds for multi-user video/image delivery with deadlines?

### Takeaway
Explicitly, yes, but only one paper verified: Hosseini & Panwar's "Restless Video Bandits" (SVC, deadlines, an LP; already in the survey). Implicitly, two classic lines use the same machinery under other names. RaDiO (2001/2006) uses per-unit finite-horizon MDPs priced by λ and sends the unit with the largest threshold multiplier, a Whittle-type index for one stream. Fu & van der Schaar (2010) relax a multi-user MDP into per-user MDPs coordinated by resource prices. No verified multimedia paper reports a Lagrangian upper bound on *every causal scheduler* for multi-user image or video delivery with deadlines, or a Whittle index for codes that degrade gracefully with SNR.

### Cited Findings
- Hosseini & Panwar (arXiv 1612.03985): SVC users with deadlines as a restless bandit, solved as an LP. Up to 2x users admitted at a quality. [survey] — [SURVEY.md §F2](/home/user/JSCCwClaude/jscc/docs/SURVEY.md)
- RaDiO's rate-controlled variant: per-unit threshold multipliers λ_l, with the largest λ_l transmitted at each opportunity and closed-form approximations of λ_l. Its performance bound is the operational D(R) at the erasure capacity. [read] — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- Fu & van der Schaar (JSAC 2010): Lagrangian relaxation of a multi-user MDP into local per-user MDPs, with resource prices from a network coordinator. [meta] — [arXiv listing](https://arxiv.org/abs/0903.0207)
- The survey already covers the general deadline and opportunistic restless-bandit theory (Yu–Xu–Tong, IEEE TAC 2018; Borkar et al., IEEE TCNS 2018) and the semantic-communication restless bandits. [survey] — [SURVEY.md §F2, §F5](/home/user/JSCCwClaude/jscc/docs/SURVEY.md)

### Inferences
- K3 and K4 should be framed as importing restless-bandit theory into a problem the multimedia community has solved with decomposition and price heuristics since RaDiO. Most of that theory is from operations research: finite-horizon index policies are asymptotically optimal against the Lagrangian bound (Hu & Frazier; Brown & Smith; Gast et al.; all in the survey). What our work adds on top is the certificate (the bound), the tie-breaking and conflict rule (Lagrangian index, our index), and the analog-code reward.
- One risk to check: an LP solution of a restless bandit, as in Hosseini & Panwar, is itself the Whittle relaxation and so yields an upper bound. If that paper reports the LP value as a bound, it overlaps K3's "bound on every causal scheduler" more than the survey entry suggests. Read it in full before claiming the bound as new for layered media.

### Gaps
- **Unverified leads**, possibly multimedia-adjacent restless-bandit or Lagrangian works with deadlines:
  - V. Raghunathan, V. Borkar, M. Cao, P. R. Kumar, "Index policies for real-time multicast scheduling for wireless broadcast systems," INFOCOM 2008 (Whittle index; deadlines).
  - I.-H. Hou and P. R. Kumar, "Utility-optimal scheduling in time-varying wireless networks with delay constraints," MobiHoc 2010. I recall a Lagrangian dual turned into per-client "bids."
  - I.-H. Hou, V. Borkar, P. R. Kumar, "A theory of QoS for wireless," INFOCOM 2009.
  - Xu et al., JSAC 2014 (see Q3).
- **2016–2026 searches** for "Whittle index video streaming", "restless bandit QoE", "deadline-aware semantic image scheduling" and "index policy adaptive bitrate cellular" were not run (budget exhausted).

## Q6. The 3–5 closest works, and an overlap verdict for each of our contributions

### Takeaway
The closest works are RaDiO, Fu & van der Schaar (2010), Hosseini & Panwar (2016), the content-aware gradient downlink schedulers (Ji et al. 2009; Pahalawatta et al. 2007), and Chakareski & Frossard (2006).
- **Already in the multimedia literature:** decomposition by prices (K3) and an indifference-price index (K4).
- **Partly precedented:** the additive, sensitivity-weighted distortion model (K2) and the content × channel rule (K5).
- **No verified multimedia precedent:** the causal upper bound used as a certificate; per-phase SNR-graded rewards for an analog learned code; the target-share admission and stopping rule (K6); and training the code on the channel the scheduler induces (K8).

### Cited Findings
- **(1) Chou & Miao, RaDiO, IEEE TMM 2006 [read].** One finite-horizon MDP per data unit under a Lagrange rate price, with deadlines (windows). It re-solves each opportunity, sends the largest threshold multiplier when one unit fits per opportunity, and is tested on an embedded (prefix) code against a capacity bound. — [PDF](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ChouM06.pdf)
- **(2) Fu & van der Schaar, IEEE JSAC 2010 [meta].** A multi-user video MDP, relaxed by a Lagrangian into per-user MDPs with coordinator-set resource prices, plus online learning. — [arXiv listing](https://arxiv.org/abs/0903.0207)
- **(3) Hosseini & Panwar, arXiv 1612.03985 [survey].** A restless bandit for multi-user SVC with deadlines, solved as an LP; reports users admitted at a quality. — [SURVEY.md](/home/user/JSCCwClaude/jscc/docs/SURVEY.md)
- **(4) Ji, Huang, Chiang, Lafruit, Catthoor, IEEE TCSVT 2009; Pahalawatta, Berry, Pappas, Katsaggelos, IEEE JSAC 2007 [meta].** Content- and channel-aware (gradient) multi-user downlink schedulers with distortion-based utilities. Ji et al. adds OFDM, SVC and deadline-aware weights. — [Ji et al.](https://collaborate.princeton.edu/en/publications/scheduling-and-resource-allocation-for-svc-streaming-over-ofdm-do/); [Pahalawatta et al. (dblp)](https://dblp.org/pid/29/4430)
- **(5) Chakareski & Frossard, IEEE TMM 2006 [meta].** RD-optimised scheduling of multiple video streams on a shared (wireless-LAN) channel, from per-packet size and importance, distributed with importance-proportional bandwidth shares. — [NJIT record](https://researchwith.njit.edu/en/publications/rate-distortion-optimized-distributed-packet-scheduling-of-multip/)

### Inferences
**Verdict for each of our contributions.** Y means present in the work, P partial, – absent, ? unknown; all judgments are against the verified material above.

| ours | RaDiO (TMM 2006) | Fu–vdS (JSAC 2010) | Hosseini–Panwar (2016) | gradient schedulers (Ji 2009 / Pahalawatta 2007) | Chakareski–Frossard (TMM 2006) | verdict |
|---|---|---|---|---|---|---|
| K1 prefix code under deadlines | Y (embedded audio, 12 sequential units/GOF) | ? | Y (SVC layers) | Y (SVC in Ji) | – (non-scalable video emphasised) | Framing not new; a learned analog code with SNR-graded phases is new |
| K2 additive, sensitivity-weighted distortion abstraction | P (additive ΔD with sensitivities; binary arrival) | ? | – (digital layers) | P (expected distortion with loss / concealment) | P (per-packet importance) | Additive, sensitivity-weighted form is precedented; a per-phase noise-to-distortion map for an analog learned code, calibrated, with MIESM/EESM shown to fail, is new |
| K3 per-user DP via Lagrangian relaxation + causal upper bound | P (per-unit DP + λ; capacity bound, not a relaxation bound; one stream) | Y for the decomposition (per-user MDPs + prices); bound ? | P/Y (restless bandit solved as LP; bound ?) | – (myopic) | – | Decomposition is prior art. The finite-horizon per-user image frame and the bound *as a certificate* are the defensible part; check whether Hosseini–Panwar and Fu–vdS report their relaxations as bounds |
| K4 Whittle / Lagrangian (bid-price) index | P (largest threshold multiplier λ_l wins the opportunity: an unnamed Whittle-type index) | P (prices, no index verified) | ? (index not verified) | – | – | Indifference-price index has a media-streaming ancestor (RaDiO); the Lagrangian index at relaxation prices has none verified |
| K5 closed-form advantage index + equal-slope quotas | P (closed-form λ_l approximation, "a fraction of a dB") | – | – | P (content × channel product; deadline / past-service weights) | P (importance-proportional shares ≈ equal-slope across streams; inference) | The family (content × channel, myopic) is old. The certainty-equivalent "gain now minus gain at a typical slot" form and its measured gap to the bound are not found in verified works |
| K6 share at a PSNR target, admission + online stopping | – | – | P (users admitted at a quality) | – | – | No verified multimedia precedent for the rule; check the LTE "video capacity" literature (Gaps) |
| K7 stale CSI, digital CQI / HARQ baseline, delay | P (feedback delay through RTT; ACK-based, no CSI) | ? | ? | P (deadlines; CSI assumed) | ? | Positioning, not a novelty claim |
| K8 code trained on the scheduler-induced channel | – | – | – | – (Huang et al. 2008 adapts sources jointly with scheduling, not codec training) | – | No verified precedent; Huang et al.'s joint source adaptation + scheduling is the nearest cousin to cite |

**Writing advice that follows from the verdicts:**
- Cite RaDiO in the scheduling section as the media-streaming origin of "decompose into per-unit programs under a price; serve the largest threshold price." Then state our three departures: users with their own fading channels (opportunism), graceful analog phases, and a relaxation bound that certifies the rule.
- Cite Fu & van der Schaar (2010) as the multi-user per-user-MDP-with-prices precedent. Do not claim the decomposition as new.
- Cite Ji et al. (2009) and Pahalawatta et al. (2007) as the myopic content × channel ancestors of the advantage index, and Chakareski & Frossard (2006) for content-aware sharing of one channel across streams.

### Gaps
- Of the five closest works, only RaDiO was read in full. Fu & van der Schaar (2010), Hosseini & Panwar (2016), Ji et al. (2009), Pahalawatta et al. (2007) and Chakareski & Frossard (2006) were judged from abstracts or summaries. Their cells marked "?" need a full read before the paper's related-work section is finalised. Priority order: Fu–vdS (is the dual reported as a bound? which prices?), then Hosseini–Panwar (the LP value as a bound? which index?), then Xu et al. JSAC 2014 (an unverified lead, possibly a finite-horizon deadline video frame), then Y. Li et al. TMM 2008 (an unverified lead; multi-stream content-aware wireless scheduling).
- The search budget ran out, so a follow-up session should run targeted searches:
  - the unverified leads listed under Q1–Q5;
  - "Whittle index video" and "restless bandit QoE";
  - "share of satisfied users video LTE";
  - 2016–2026 multi-user video and image scheduling with deadlines in 5G.
