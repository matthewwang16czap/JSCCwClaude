# Restless-bandit and index-policy scheduling in wireless networks (prior work for deadline scheduling of learned image codes)

*Verification status, read first.* "Verified" means that during this session the work's existence and the setting or result quoted here appeared in page content returned by search: abstracts, repository records, author pages, slides, or the reference lists of citing papers. WebFetch could not resolve any host in this environment (DNS failures on arxiv.org, ar5iv, dblp.org, lancaster.ac.uk, netlab.tkk.fi and netlify.com). The turn's shared web-search budget then ran out after this researcher's 23rd search. As a result: (i) no full text was read, so index formulas and theorem conditions come from abstracts; (ii) several named lines could only be listed as **unverified candidates** in the Gaps subsections: gradient/PF analysis, OFDMA, feedback delay, Kalman prediction, EDF variants, Jaramillo–Srikant and "Kang, Eryilmaz". Do not cite those without checking. Works already in SURVEY.md F2/F5 are marked "(SURVEY)" and were not re-checked unless a note says so.

*Labels for our components* (used in the overlap verdicts):
- **[M]** The model: a finite-horizon (T-slot deadline) restless bandit. Each user is owed one image sent as up to 6 prefix-code phases, and each slot has 1 or B units. The reward is terminal: concave in accumulated service (deadline PSNR from an additive MMSE model of the SNRs the phases got), or all-or-nothing for reaching a PSNR target.
- **[DP]** The Lagrangian relaxation of the per-slot budget gives one DP per user, with state (phases held, accumulated weighted noise).
- **[UB]** That relaxation is an upper bound on every causal scheduler, with a measured gap of 0.005 dB per user at K = 128, B = K/8.
- **[WI]** The Whittle index, taken as the top of the indifference range.
- **[LI]** The Lagrangian index at the bound's per-slot prices.
- **[AI]** The closed-form advantage index: predicted gain from this slot minus the gain from a typical mean-SNR slot, with per-user quotas. It is the certainty-equivalent form of the above.
- **[SR]** Stale or noisy reports. Each arm's program decides on the report through a report-to-channel kernel (r = J0(2π f_D τ)), and the bound holds for report-based schedulers.
- **[SC]** The many-users test with B = K/8 units per slot.
- **[BL]** Baselines: PF, max-SNR, ID-guided.

## 1. Index policies for opportunistic and flow-level scheduling with time-varying channels and finite jobs: what is the arm's state and reward, and does any use a terminal or concave reward of accumulated service?

### Takeaway
In every verified paper of this line (Ayesta–Erausquin–Jacko 2010, Jacko 2011, Aalto–Lassila–Osti 2015/2016, Aalto–Lassila–Taboada 2019, Ayesta–Jacko–Novak 2011, Borkar et al. 2018), the arm is a user (a flow or a queue). Its state is its remaining work or queue content plus its channel state. The cost is a holding cost per slot until the job finishes, plus abandonment or energy costs, over an infinite horizon. Whittle's relaxation of the one-user-per-slot constraint gives closed-form indices, and these contain the classical opportunistic rules as special cases. **None of the verified flow-level papers has a hard per-frame deadline, or a terminal reward that is concave in channel-dependent accumulated service.** The nearest analogues are completion-based costs (our all-or-nothing target without a deadline) and random-deadline abandonment penalties. Two Kumar-group papers do combine deadlines with Whittle indices: Raghunathan–Borkar–Cao–Kumar 2008 (multicast) and Singh–Guo–Kumar (inter-delivery regularity over fading).

### Cited Findings
- **[A1] U. Ayesta, M. Erausquin, P. Jacko, "A modeling framework for optimizing the flow-level scheduling with time-varying channels," *Performance Evaluation*, 2010.** Volume, pages and DOI were not seen.
  - Setting: wireless downlink scheduling of a finite number of finite-length jobs whose available service rate varies over time because of fading. The objective is to minimise the mean number of users in the system or the mean waiting time.
  - Method: a priority index from a restless-bandit-type analysis, the "Potential Improvement" (PI) rule. The index is a ratio: the current available service rate divided by the expected potential improvement of that rate.
  - Results: for certain objectives the index reduces to the known "Relatively Best" and "Proportionally Best" rules, which the authors present as a formal justification of opportunistic rules for flow-level performance. With constant rates PI reduces to the cμ-rule, and PI achieves the maximum stability region.
  - Sources: [Lancaster eprint record](https://eprints.lancs.ac.uk/id/eprint/65375/); [author PDF](https://www.lancaster.ac.uk/staff/jacko/publications/2010_AyestaEtal_PI.pdf)
- **[A2] P. Jacko, "Value of information in optimal flow-level scheduling of users with Markovian time-varying channels," *Performance Evaluation*, 2011.** Volume and pages not seen.
  - Setting: A1's model extended to Gilbert–Elliott (two-state Markov) channels.
  - Result: the index for a user in the bad channel state uses both a one-period and a steady-state potential improvement. The rule is reported near-optimal in most instances.
  - Sources: [Lancaster eprint](https://eprints.lancs.ac.uk/id/eprint/65374/); [author PDF](https://lancaster.ac.uk/staff/jacko/publications/2011_Jacko_wirelessMarkovian.pdf); [IMDEA talk page](https://networks.imdea.org/whatsnew/events-agenda/value-of-information-in-optimal-flow-level-scheduling-of-users-with-markovian-time-varying-channels/)
- **[A3] S. Aalto, P. Lassila, P. Osti, "Whittle index approach to size-aware scheduling with time-varying channels," *Proc. ACM SIGMETRICS 2015*, pp. 57–69, DOI 10.1145/2745844.2745851.** The DOI comes from an OpenAlex record that appeared among the search results.
  - Setting: downlink cellular scheduling of elastic flows (for example TCP file transfers) with known file sizes and known channel states.
  - Method: a size-aware Whittle index policy, compared against proportional fair (PF) and PI.
  - Sources: [author slides](https://www.netlab.tkk.fi/~samuli/Presentations/Conferences/aalto-sigmetrics-2015.pdf); [OpenAlex record](https://api.openalex.org/works/doi:10.1145%2F2745844.2745851); [Aalto's wireless-scheduling page](https://www.netlab.tkk.fi/~samuli/wscheduling.html)
- **[A4] S. Aalto, P. Lassila, P. Osti, "Whittle index approach to size-aware scheduling for time-varying channels with multiple states," *Queueing Systems* 83 (2016) 195–225, DOI 10.1007/s11134-016-9484-z.**
  - Extends A3 to channels with more than two states.
  - The problem is indexable when file sizes follow a Pascal distribution, and the Whittle index is derived.
  - Sources: [Aalto research portal](https://research.aalto.fi/en/publications/whittle-index-approach-to-size-aware-scheduling-for-time-varying-/); [Unpaywall DOI record](https://unpaywall.org/10.1007%2FS11134-016-9484-Z)
- **[A5] S. Aalto, P. Lassila, I. Taboada, *Performance Evaluation* 136 (2019).** Whittle index with partial channel information; details under question 3.
- **[A6] U. Ayesta, P. Jacko, V. Novak, "A nearly-optimal index rule for scheduling of users with abandonment," *IEEE INFOCOM 2011*, pp. 2849–2857, DOI 10.1109/INFCOM.2011.5935122.**
  - Setting: job scheduling in which users may abandon before they are served. The objective is holding costs plus abandonment penalties.
  - Method: the one- and two-user cases are solved exactly. For larger systems a heuristic index derived from restless bandits ("AJN") is proposed. It reduces to the cμ-rule without abandonment and to the cμ/θ-rule under some conditions.
  - Result: in a computational study AJN is usually at least as good as competing rules and often optimal. The slides add that it is occasionally slightly suboptimal.
  - Sources: [Lancaster eprint](https://eprints.lancs.ac.uk/id/eprint/71356); [author PDF](https://lancaster.ac.uk/staff/jacko/publications/2011_AyestaEtal_abandonment.pdf)
- **[A7] V. S. Borkar, G. S. Kasbekar, S. Pattathil, P. Y. Shetty, "Opportunistic scheduling as restless bandits," *IEEE Trans. Control of Network Systems* 5(4):1952–1961, 2018; arXiv:1706.09778.** (SURVEY; re-checked.)
  - Setting: users with finite queues and a packet holding cost. The energy spent per transmission depends on a time-varying channel quality that differs across users. Average-cost MDP.
  - Result: the problem is shown indexable, and the user with the lowest index is scheduled. In simulations this beats max-weight and weighted fair scheduling on average cost.
  - **Citing caveat:** the IAS repository abstract says "Whittle-like indexable" where the arXiv abstract says "Whittle indexable".
  - Sources: [arXiv](https://arxiv.org/abs/1706.09778v1); [IAS repository](https://repository.ias.ac.in/135152)
- **[A8] V. Raghunathan, V. Borkar, M. Cao, P. R. Kumar, "Index policies for real-time multicast scheduling for wireless broadcast systems," *IEEE INFOCOM 2008*.** Citing reference lists disagree on the pages (1570–1578 vs 2243–2251), and the primary text was not seen.
  - Setting, as described by citing works: an access point serves N latency-sensitive multicast flows, and each packet must reach its multicast group by a deadline. Both periodic and one-shot arrivals are considered.
  - Method: restless bandit with Whittle's relaxation, indexable under complete receiver feedback; the flow with the largest Whittle index is served.
  - A statistical-estimation mechanism replaces full feedback, needing only one ACK per transmission.
  - Sources: secondary only, via citing papers surfaced by search, e.g. [ResearchGate author listing](https://www.researchgate.net/scientific-contributions/Vivek-Raghunathan-22329655)
- **[A9] R. Singh, X. Guo, P. R. Kumar, "Index policies for optimal mean-variance trade-off of inter-delivery times in real-time sensor networks," arXiv:1501.06446.** Venue not verified.
  - Setting: an access point schedules real-time packets to nodes over a fading wireless channel, motivated by automotive sensor networks.
  - Result: indexable, with explicit Whittle indices. The index policy is optimal in certain cases.
  - Source: [arXiv](https://arxiv.org/pdf/1501.06446)
- **Other wireless restless-bandit titles surfaced by search** (authors and venues not verified; I recall Borkar as a co-author but did not confirm it): "Scheduling in Wireless Networks with Spatial Reuse of Spectrum as Restless Bandits" ([arXiv:1910.04402](https://arxiv.org/pdf/1910.04402)); "User Association in Dense mmWave Networks as Restless Bandits" ([arXiv:2107.09153](https://arxiv.org/pdf/2107.09153)).

### Inferences
- **A1/A2 (the PI family): what overlaps and what is missing.**
  - Overlap: they are the published ancestors of [AI]. PI compares the current rate with an expected potential improvement, and Jacko's index compares a one-period with a steady-state improvement. Both are ratio forms of our difference "gain from this slot minus gain from a typical mean-SNR slot". They are also the precedent for reading relative-rate schedulers (RB/PB) as approximate restless-bandit indices.
  - Missing:
    - No deadline. The cost is holding time until completion, not a terminal reward.
    - Service is bits at the current rate, which add linearly. Ours is an SNR-dependent MMSE term whose value depends on the phase it lands on.
    - One user per slot.
    - No Lagrangian-bound certificate. As far as the abstracts say, near-optimality is checked against optimal policies on small instances.
    - No stale reports. A2 exploits channel memory, but the current state is observed.
- **A3/A4: overlap and gap.**
  - Overlap: the arm's state carries remaining work. Our "phases held" is that analogue; size-aware Whittle indices are opportunistic. The fact that A1's index depends only on rates while A3 calls itself "size-aware" suggests the earlier models had memoryless job sizes (an inference, not checked).
  - Missing: deadline, non-linear accumulation (our "accumulated weighted noise" has no counterpart), terminal reward, partial observation, and B > 1.
- **A6: overlap and gap.**
  - Overlap: abandonment is a random deadline with an all-or-nothing penalty. That is the closest flow-level analogue of our PSNR-target objective, and A6 also has exactly solved small cases.
  - Missing: the abstract mentions no channel variation, and there is no deterministic deadline and no concave reward.
- **A7: overlap and gap.**
  - Overlap: user-specific fading in the arm's state, with a Whittle index for wireless.
  - Missing: average cost with no deadline, no content-quality reward, no partial observation, one user per slot, no bound.
- **A8: overlap and gap.**
  - Overlap: deadline, Whittle index, wireless, and reduced feedback with estimation. It is the earliest wireless deadline restless bandit found in this session.
  - Missing: packets are delivered or not, so there is no SNR-dependent quality, no concave reward and no reported Lagrangian bound.
- **A9: overlap and gap.**
  - Overlap: a Whittle index over fading channels for a real-time objective.
  - Missing: the objective is regularity of deliveries, not a per-frame terminal quality.
- **Answer to "terminal or concave reward of accumulated service?"**
  - No verified flow-level paper uses one.
  - The concave partial-credit form exists in the deadline restless-bandit line: Yu–Xu–Tong's *convex non-completion penalty* is a concave reward of the service received by the deadline (question 2). There, however, a slot's service is a deterministic unit, not a channel-dependent quality increment.
  - Our arm is that arm with a stochastic, user-specific per-slot quality increment. That is why its state needs a second coordinate (the accumulated weighted noise).

### Gaps
- A1/A2: volume, pages and DOI are not verified, and the exact index formulas were not read (full texts unreachable).
- The Aalto–Lassila–Taboada VALUETOOLS 2019 conference version ("Indexability of an opportunistic scheduling problem with partial channel information") may have DOI 10.1145/3306309.3306324. A link with that DOI appeared in the results, but the mapping is uncertain.
- **Unverified candidates** (from memory; check before citing):
  - I. M. Verloop, "Asymptotically optimal priority policies for indexable and nonindexable restless bandits," *Ann. Appl. Probab.* 26(4), 2016.
  - M. Larrañaga, U. Ayesta, I. M. Verloop, "Dynamic control of birth-and-death restless bandits: application to resource-allocation problems," *IEEE/ACM ToN*, 2016; also their convex-holding-cost abandonment-queue papers.
  - I. Taboada et al. (with Jacko, Ayesta, Liberal): size-aware opportunistic scheduling for time-varying channels, *Performance Evaluation*, about 2014.
  - B. Sadiq, G. de Veciana, "Balancing SRPT prioritization vs opportunistic gain in wireless systems with flow dynamics," ITC 2010.
  - U. Ayesta, M. K. Gupta, I. M. Verloop, "On the computation of Whittle's index for Markovian restless bandits," *Math. Methods Oper. Res.*, 2021.
- I did not find a flow-level paper with a hard deadline and partial credit, but further searching was impossible, so absence is not established.

## 2. Deadline-constrained and real-time scheduling: does any certify near-optimality with a Lagrangian upper bound?

### Takeaway
Yes, but by proof (asymptotically, or for an average-constrained relaxation) rather than by reporting a numerical gap on the instance.
- **Yu–Xu–Tong (TAC 2018)** bound the Whittle policy's gap to optimality, and the bound vanishes with scale.
- **Hao–Xu–Tong (TAC 2022)**, new relative to SURVEY, prove a "Lagrangian priority policy" asymptotically optimal. Its priority is "the highest tax under which activating an arm remains optimal", which is our [WI] definition, and it is built from the relaxed single-arm problems as in [DP]/[LI].
- **Singh–Kumar** decompose a deadline-constrained wireless network through the Lagrangian of its constrained MDP into per-packet problems with transmission prices. The resulting policies depend only on packet age.
- **Kumar's frame model (Hou–Borkar–Kumar 2009)** has exactly our frame and deadline structure: packets not delivered by the frame's end are dropped. But its requirements are long-run timely-throughput targets per client, served by debt-based priorities. There is no per-frame terminal quality and no fading-dependent quality.

### Cited Findings
- **[B1] I-H. Hou, V. S. Borkar, P. R. Kumar, "A theory of QoS for wireless," *IEEE INFOCOM 2009*;** UIUC technical report of September 2008.
  - Framework: it handles delay, delivery ratio and channel reliability together, and covers admission control and scheduling.
  - Model: packets that miss their frame deadline are dropped, and the per-frame deliveries that meet deadlines define "timely throughput".
  - Policies: debt-based "largest-debt-first" policies give priority to clients with the larger debt. They meet every feasible set of client requirements.
  - Admission control: feasibility can be checked in nearly linear time, which gives a tractable admission test.
  - Sources: [UIUC IDEALS record](https://www.ideals.illinois.edu/items/105587); [Kumar talk abstract, LCCC 2010](https://lccc.control.lth.se/abstractID=12.html); [slides](https://lccc.control.lth.se/uploads/LCCC2010/WorkshopFebruary2010/PRKumar.pdf); [pathwise-debt follow-up, arXiv:1309.3798](https://arxiv.org/pdf/1309.3798)
- **[B2] I-H. Hou, P. R. Kumar, "Utility maximization for delay constrained QoS in wireless," arXiv:0908.0362** (submitted to INFOCOM 2010; the published version was not confirmed this session).
  - Model: each client's utility is a function of the delivery ratio it obtains, and the access point assigns delivery ratios to maximise total utility.
  - Method: the problem decomposes into a client-side problem and an access-point problem; the client side is a bidding game for service time.
  - Result: the selfish equilibrium maximises total utility, and the proposed AP policy reaches the optimum with some fairness.
  - Source: [arXiv](https://arxiv.org/abs/0908.0362)
- **[B3] Hou and Kumar companions** (titles verified, contents not read):
  - "Scheduling Heterogeneous Real-Time Traffic over Fading Wireless Channels" ([arXiv:0908.0587](https://arxiv.org/pdf/0908.0587)); I recall the authors as Hou and Kumar.
  - The book *Packets with Deadlines: A Framework for Real-Time Wireless Networks* ([Springer](https://link.springer.com/978-3-031-79256-4)).
- **[B4] R. Singh, P. R. Kumar, "Throughput optimal decentralized scheduling of multi-hop networks with end-to-end deadline constraints: Unreliable links," arXiv:1606.01608.** The journal venue was not confirmed this session.
  - Model: packets that miss their end-to-end deadline are dropped. The aim is routing and scheduling policies that optimise weighted averages of timely throughputs under nodal power constraints.
  - Method: the decomposition is "taken from the Lagrangian of the constrained network-wide MDP rather than from a fluid model". Scheduling, power and routing decisions depend only on packet age, not on network state or queue lengths.
  - Result: a distributed policy reaches any point on the Pareto frontier under average-power constraints.
  - Companions: Part II on wireless networks with interference ([arXiv:1709.01672](https://arxiv.org/pdf/1709.01672)). The earlier "Decentralized throughput maximizing policies for deadline-constrained wireless networks" ([arXiv:1509.05293](https://arxiv.org/pdf/1509.05293.pdf)) introduces Lagrange multipliers as "prices for attempting packet transmissions" to get a decentralized policy.
  - Source: [arXiv:1606.01608](https://arxiv.org/pdf/1606.01608)
- **[B5] Z. Yu, Y. Xu, L. Tong, "Deadline scheduling as restless bandits," *IEEE Trans. Automatic Control* 63 (2018) 2343–2358; arXiv:1610.00399.** (SURVEY; re-checked.)
  - Model: "Jobs arrive randomly at a service center with stochastic job sizes, rewards, and completion deadlines. The service provider faces random processing costs, convex non-completion penalties, and a capacity constraint that limits the simultaneous processing of jobs."
  - Results: the problem is indexable, and the Whittle index has a closed form when processing costs are constant. An upper bound on the Whittle policy's gap to optimality "converges to zero as the job arrival rate and the number of available processors increase simultaneously to infinity".
  - **Citing caveat:** the arXiv v4 abstract words this as converging "super exponentially as the number of available processors increases". Check the final version.
  - Sources: [arXiv v4](https://www.arxiv.org/abs/1610.00399v4); [ar5iv](https://ar5iv.labs.arxiv.org/html/1610.00399)
- **[B6] L. Hao, Y. Xu, L. Tong, "Asymptotically optimal Lagrangian priority policy for deadline scheduling with processing rate limits," *IEEE Trans. Automatic Control* 67(1):236–250, Jan. 2022, DOI 10.1109/TAC.2021.3049340.** New relative to SURVEY.
  - Setting: deferrable jobs arrive randomly, with a time-varying cap on the total processing rate in each stage. Keywords include restless bandits, dynamic programming and electric-vehicle charging.
  - Method: the restless bandit is relaxed into independent single-arm problems. A job's "Lagrangian priority value" is the highest tax under which activating it remains optimal, and jobs are processed in order of these values.
  - Results: asymptotic optimality as the system grows. Numerically, the policy earns 22–49% higher average reward than the classical Whittle index policy, which ignores processing-rate limits.
  - Sources: [IEEE Xplore](https://ieeexplore.ieee.org/document/9314189); [NSF PAR](https://par.nsf.gov/biblio/10250769); [SUFE abstract](https://academicnewsletter.sufe.edu.cn/info/391167)
- **[B7] Video and real-time restless bandits.**
  - Hosseini and Panwar, "Restless video bandits" (SURVEY: layered video with deadlines, solved as an LP; **not re-checked** this session): [arXiv:1612.03985](https://arxiv.org/abs/1612.03985).
  - Title surfaced, contents and authors not verified: "Optimal Decentralized Dynamic Policies for Video Streaming over Wireless Channels" ([arXiv:1902.07418](https://arxiv.org/pdf/1902.07418)).
- **[B8] Other titles surfaced, contents not verified:**
  - "Near-Optimal Packet Scheduling in Multihop Networks with End-to-End Deadline Constraints" ([arXiv:2310.19077](https://ar5iv.labs.arxiv.org/html/2310.19077)).
  - "Throughput-Optimal Scheduling Design with Regular Service Guarantees in Wireless Networks" ([arXiv:1405.0625](https://arxiv.org/pdf/1405.0625)).
- **[B9] Tripathi and Modiano** (SURVEY): Whittle-index scheduling of real-time sources over a wireless broadcast network to minimise functions of age of information. The index policy is exactly optimal for two sources over reliable channels. Source: [arXiv:1908.10438](https://arxiv.org/pdf/1908.10438)

### Inferences
- **B5 (Yu–Xu–Tong) is the structural twin of [M].**
  - Same structure: a deadline per job, B "processors" per slot, and a convex non-completion penalty, which is the same thing as a concave terminal reward of accumulated service. Their Whittle index has a closed form, and the gap vanishes with scale.
  - What theirs lacks: service per slot is a deterministic unit. As I recall, randomness enters through a processing cost that is common to all jobs (not verified). So there is no user-specific channel, no quality that depends on which phase got which SNR, and no reports.
- **B6 (Hao–Xu–Tong) is the method twin of [DP]+[WI]+[LI].**
  - Their "highest tax under which activation remains optimal" is our definition of the Whittle index as the top of the indifference range.
  - Their "Lagrangian priority" from relaxed single-arm problems is the family of our Lagrangian index. Whether they evaluate it at the relaxation's optimal multipliers, as our [LI] does, was not verified.
  - They also show the 22–49% gain over the plain Whittle index when per-stage capacity varies. That parallels our finding that [LI] beats [WI] at the PSNR target.
  - **Reviewer risk:** the name "Lagrangian priority/index" is taken. Cite B6 and state the differences: wireless channel, partial observation, quality reward.
  - What theirs lacks: wireless fading, user-specific stochastic service, reports, image-quality reward.
- **B4 (Singh–Kumar) is the wireless instance of "Lagrangian relaxation of per-slot constraints, then per-entity programs with prices".**
  - Their entities are packets with state = age, and their objective is long-run timely throughput under average constraints.
  - Ours are users with state (phases, weighted noise) and a per-frame terminal quality under hard per-slot budgets. Our bound is the value of the same kind of relaxation.
- **B1 (Hou–Borkar–Kumar) has our frame structure.**
  - Same: frames of T slots, deliveries counted per frame, undelivered packets dropped.
  - Different: the debt is a running dual variable across frames, enforcing long-run delivery ratios. Our prices are within-frame, slot-indexed Lagrange multipliers of a finite-horizon relaxation. The channels are Bernoulli success, not SNR-dependent quality.
  - B2 adds concave utilities and a dual-type decomposition, again on long-run ratios.
- **Answer on certification:** the verified works certify near-optimality by theorems:
  - B5: a gap bound.
  - B6: asymptotic optimality.
  - Ouyang et al. 2016: approaches the relaxed-constraint optimum (question 3).
  - B4: optimal for the average-constrained relaxation.
  
  None of them was seen reporting a numerical relaxation-bound gap at moderate N on its actual problem. Our measured certificate (0.26 dB at K = 8; 0.005 dB per user at K = 128 with B = K/8) is therefore new *evidence*, not a new *method*. This rests on an incomplete search.

### Gaps
- **Unverified candidates** (from memory):
  - J. J. Jaramillo, R. Srikant, "Optimal scheduling for fair resource allocation in ad hoc networks with elastic and inelastic traffic," IEEE/ACM ToN, about 2011. Deadline-constrained inelastic flows with utility, deficit/dual-based.
  - X. Kang, W. Wang, J. J. Jaramillo, L. Ying, "On the performance of largest-deficit-first for scheduling real-time traffic in wireless networks," IEEE/ACM ToN, about 2016.
  - R. Li, A. Eryilmaz, "Scheduling for end-to-end deadline-constrained traffic with reliability requirements in multihop networks," IEEE/ACM ToN, about 2012.
  - S. Shakkottai, R. Srikant, "Scheduling real-time traffic with deadlines over a wireless channel," *Wireless Networks*, about 2002. This is the EDF-with-channel line.
  - Single-user deadline programs over fading, which are cousins of our per-user DP: A. Fu, E. Modiano, J. N. Tsitsiklis (IEEE TWC 2006); J. Lee, N. Jindal (IEEE TWC 2009); M. Zafer, E. Modiano (IEEE TIT 2008 / ToN 2009).
- **"Kang, Eryilmaz et al." could not be identified.** Two arXiv titles on scheduling over time-correlated channels with imperfect CSI or ARQ feedback appeared without visible authors (question 3, C5). Check whether one of them is the intended paper.
- Not re-checked: Hosseini–Panwar's details (LP vs Whittle index, reward). Not verified: how Hao–Xu–Tong compute the multipliers.

## 3. Partially observed or outdated channel state

### Takeaway
Partially observed arms are well established in wireless restless bandits, but under a different observation model from ours:
- Gilbert–Elliott channels whose state is revealed only for the scheduled user, through ARQ. Ouyang–Murugesan–Eryilmaz–Shroff prove Whittle indexability with a closed-form index. Ouyang–Eryilmaz–Shroff show the Whittle policy approaches the relaxed-constraint optimum with many users.
- "Partial channel information" in flow-level scheduling (Aalto–Lassila–Taboada 2019: indexability in parts of the parameter space, explicit index).

Not found among verified works: every user reporting every slot, i.i.d. Rayleigh fading across slots, a Gaussian report-to-channel kernel with r = J0(2π f_D τ), and a finite horizon with a terminal quality reward. In that model the belief is a one-step Gaussian posterior rather than a Markov belief.

### Cited Findings
- **[C1] W. Ouyang, S. Murugesan, A. Eryilmaz, N. B. Shroff, "Exploiting channel memory for joint estimation and scheduling in downlink networks," *IEEE INFOCOM 2011*, pp. 3056–3064; journal version "…—A Whittle's indexability analysis," *IEEE Trans. Inf. Theory* 61(4):1702–1719, 2015; arXiv:1009.3959.**
  - Method: downlink scheduling with imperfect channel state information is modelled as a POMDP and mapped to a restless bandit.
  - Result: the problem is Whittle indexable, the index has a closed form, and numerically the policy is near-optimal. The authors note that indexability is "traditionally known to be hard to establish".
  - Sources: [arXiv](https://arxiv.org/pdf/1009.3959); [ar5iv](https://ar5iv.arxiv.org/html/1009.3959); [DBLP author page](https://dblp.org/pid/52/7909)
- **[C2] W. Ouyang, A. Eryilmaz, N. B. Shroff, "Downlink scheduling over Markovian fading channels," *IEEE/ACM Trans. Networking*, 2016 (volume and pages not seen); conference version "Asymptotically optimal downlink scheduling over Markovian fading channels," *IEEE INFOCOM 2012*, pp. 1224–1232; arXiv:1108.3768.**
  - Setting: a time-slotted downlink with N users whose heterogeneous channels are two-state ON/OFF Markov chains. Channel states are learned through ARQ feedback, revealed at the end of the scheduled users' transmissions only.
  - Results: the index policy's performance can get arbitrarily close to that of the relaxed-constraint optimal policy when the initial state is near a carefully selected state. Under a numerically verifiable recurrence assumption, Whittle's index policy is globally optimal in the many-users limit.
  - Sources: [arXiv](https://arxiv.org/pdf/1108.3768); [Emergent Mind summary](https://www.emergentmind.com/papers/1108.3768)
- **[C3] S. Aalto, P. Lassila, I. Taboada, "Whittle index approach to opportunistic scheduling with partial channel information," *Performance Evaluation* 136 (2019) 102052, DOI 10.1016/j.peva.2019.102052.**
  - Setting: downlink scheduling where the scheduler knows the channel states only imperfectly, minimising flow-level holding costs. The authors note that earlier Whittle work assumed exact channel information.
  - Results: indexability is proven in parts of the parameter space and an explicit Whittle index formula is derived. Where indexability holds, the relaxed problem's optimal policy is of threshold type.
  - A VALUETOOLS 2019 version gives sufficient conditions for indexability, the explicit index, and an evaluation.
  - Sources: [Aaltodoc record](https://aaltodoc.aalto.fi/items/4f9064e8-d4b0-4b81-a048-e1a3180d5e40); [UPV/EHU record](https://research.science.eus/documentos/5f1cdcec29995265e44d611e)
- **[C4] Earlier entries that also apply:**
  - Jacko 2011 [A2]: Gilbert–Elliott channel memory, and the "value of information" in the index. [Lancaster eprint](https://eprints.lancs.ac.uk/id/eprint/65374/)
  - Raghunathan et al. 2008 [A8]: full receiver feedback replaced by estimation from one ACK per transmission. [secondary](https://www.researchgate.net/scientific-contributions/Vivek-Raghunathan-22329655)
- **[C5] Titles surfaced, authors and contents not verified:**
  - "Low-complexity Optimal Scheduling over Correlated Fading Channels with ARQ Feedback" ([arXiv:1206.1634](https://arxiv.org/pdf/1206.1634)).
  - "Scheduling in Time-correlated Wireless Networks with Imperfect CSI and Stringent Constraint" ([arXiv:1403.7773](https://arxiv.org/pdf/1403.7773)).
- **[C6] Already in SURVEY F5:**
  - The opportunistic spectrum access restless-bandit lineage: Zhao–Krishnamachari–Liu; Ahmad et al.; Liu–Zhao (indexability of multichannel access); Tekin–Liu.
  - Gornet and Sinopoli: Kalman-predicted rewards in a learning restless bandit ([arXiv:2405.09584](https://arxiv.org/abs/2405.09584)).

### Inferences
- **What [SR] shares with C1–C3, and what differs.**
  - Shared: in all of them the arm's program acts on an information state (a belief or a report), not on the true channel.
  - Different in our model:
    - Every user reports every slot, so there is no exploration value: reporting does not depend on being scheduled, unlike ARQ-only feedback.
    - Fading is i.i.d. across slots, so the information state is a one-step kernel (h | report ~ CN(r·h_rep, 1 − r²), normalised), not a belief that evolves through a Markov chain.
    - The horizon is finite and the reward is terminal.
  - So the *concept* is established. The *model*, together with the result that the Lagrangian bound stays valid for every report-measurable scheduler, was not found. That result follows the same logic as C2's relaxed-constraint optimum, a relaxation in the information-state space, so present it as an application rather than a theorem of its own.
- **Effect of outdated reports on multiuser diversity** (derived from the Gaussian kernel the project already uses; I could not verify a classical citation for it).
  - Take unit-mean Rayleigh fading. Selecting the largest of K reports gives an expected *actual* gain of r²·H_K + (1 − r²), where H_K = Σ_{k≤K} 1/k ≈ ln K.
  - So the selection gain over a random user (H_K − 1) is scaled by r²: 0.64 at r = 0.8, and nothing at r = 0.
  - This is the quantitative form of the SURVEY's Grunewalder–Khaleghi principle: opportunism gains only what the information says about the current slot. Our measured "keeps 81% of the gain over PF at r = 0.8" is consistent with only part of the gain being opportunistic.
- **Kalman-predicted CSI.** Prediction across slots matters only when fading is correlated from slot to slot. With our i.i.d.-across-slots model the one-step kernel is the whole posterior. A Gauss–Markov extension would make the arm's state include the last report and its age. Then C1/C2-style beliefs, and Kalman-belief index policies (unverified candidates below), become directly relevant.

### Gaps
- **Unverified candidates** (from memory; check before citing):
  - S. Murugesan, P. Schniter, N. B. Shroff, "Multiuser scheduling in a Markov-modeled downlink using randomly delayed ARQ feedback," IEEE TIT, about 2012.
  - K. Jagannathan, S. Mannor, I. Menache, E. Modiano, "A state action frequency approach to throughput maximization over uncertain wireless channels," INFOCOM 2011 / *Internet Mathematics*. As I recall, it gives LP (state-action frequency) upper bounds under partial observation. If verified, it is the closest precedent for [UB] with [SR].
  - C.-P. Li, M. J. Neely, network utility maximization over partially observable Markovian channels (*Performance Evaluation*, about 2013).
  - Q. Ma, C. Tepedelenlioğlu, "Practical multiuser diversity with outdated channel feedback," IEEE TVT, about 2005. This would be the classical citation for the r² scaling above.
  - J. Niño-Mora, "An index policy for dynamic fading-channel allocation to heterogeneous mobile users with partial observations" (NGI 2008) and "A restless bandit marginal productivity index for opportunistic spectrum access with sensing errors" (NET-COOP 2009).
  - Kalman-belief index policies: J. Le Ny, E. Feron, M. Dahleh, "Scheduling continuous-time Kalman filters" (IEEE TAC, about 2011); C. Dance, T. Silander, "Optimal policies for observing time series and related restless bandit problems" (JMLR 2019); J. Niño-Mora, S. Villar, multitarget tracking via restless-bandit marginal productivity indices and the Kalman filter (CDC 2009).
  - Throughput-optimal scheduling under channel-state uncertainty: Pantelidou, Ephremides and Tits.
  - Fading-channel prediction for adaptive transmission: A. Duel-Hallen (Proc. IEEE 2007).
- C2: whether the many-users regime schedules a fixed fraction of users per slot (the Weber–Weiss setting) was not verified.

## 4. Gradient, marginal-utility and score-based schedulers as myopic or certainty-equivalent indices, and finite-horizon versions

### Takeaway
Only one link in this question was verified this session. Ayesta–Erausquin–Jacko derive the relative-rate rules ("Relatively Best", "Proportionally Best") and the cμ-rule as special cases of their PI index. Relative-SNR and score-like schedulers therefore already have a published index-policy reading for flow-level costs. The gradient and PF analyses (Stolyar; Agrawal–Subramanian; Kushner–Whiting), Bonald's score-based scheduler and M-LWDF could **not** be re-verified. No verified work had a finite-horizon (deadline) gradient or marginal-utility scheduler with a terminal concave utility. The verified decompositions with concave utilities (Hou–Kumar 2010) and with prices (Singh–Kumar) work on long-run averages.

### Cited Findings
- The PI index contains "Relatively Best" and "Proportionally Best" as special cases for certain objectives, and the cμ-rule when rates are constant. The authors present this as a formal justification for opportunistic rules under flow-level objectives. — [Lancaster eprint](https://eprints.lancs.ac.uk/id/eprint/65375/)
- Aalto–Lassila–Osti benchmark their size-aware Whittle index against proportional fair and PI. — [slides](https://www.netlab.tkk.fi/~samuli/Presentations/Conferences/aalto-sigmetrics-2015.pdf)
- Borkar et al.'s Whittle policy beats max-weight and weighted fair scheduling on average cost. — [arXiv](https://arxiv.org/abs/1706.09778v1)
- Hou–Kumar maximise concave utilities of delivery ratios through a client/AP decomposition (a bidding game for service time). This is the verified "marginal utility with deadlines" scheme, but on long-run ratios. — [arXiv](https://arxiv.org/abs/0908.0362)
- Singh–Kumar's Lagrange multipliers are "prices for attempting packet transmissions". — [arXiv:1509.05293](https://arxiv.org/pdf/1509.05293.pdf)

### Inferences
- **How [AI] relates to these rules.**
  - PF and gradient rules serve the user with the largest marginal utility × instantaneous rate, where the utility is a concave function of *long-run* throughput. They are myopic: a slot is valued by its immediate contribution to that utility.
  - [AI] values a slot by the predicted gain in *terminal* quality, minus the gain from a typical (mean-SNR) slot. Subtracting a typical-slot comparator is a certainty-equivalent estimate of the option value of waiting, which is what a deadline creates and PF lacks.
  - That makes [AI] the difference-form sibling of PI/RB (current rate relative to expected or potential rate). The per-user quotas play the role of per-user service targets or prices: in the spirit of Kumar's debts, but within one frame.
- **Finite horizon.** PF's averaging window is its only "horizon". With a hard deadline the marginal value of a slot depends on the slots left, so a finite-horizon gradient rule needs a value-to-go. In our programs that is [DP], [WI] and [LI]; [AI] approximates it with the mean-SNR comparator.
- **Suggested extra baseline.** Reviewers from the flow-level community may ask for a PI-type ratio baseline: predicted gain now ÷ expected gain from the best future slot. It is cheap to add next to PF and max-SNR.

### Gaps
- **Unverified candidates** (from memory; the project's PROBLEM.md already names Stolyar and Agrawal–Subramanian):
  - A. L. Stolyar, "On the asymptotic optimality of the gradient scheduling algorithm for multiuser throughput allocation," *Operations Research* 53(1), 2005.
  - R. Agrawal, V. Subramanian, "Optimality of certain channel aware scheduling policies," Allerton 2002.
  - H. J. Kushner, P. A. Whiting, "Convergence of proportional-fair sharing algorithms under general conditions," IEEE TWC 3(4), 2004.
  - T. Bonald, "A score-based opportunistic scheduler for fading radio channels," European Wireless 2004.
  - M. Andrews et al., "Providing quality of service over a shared wireless link," IEEE Commun. Mag. 2001 (M-LWDF).
  - X. Liu, E. K. P. Chong, N. B. Shroff, "A framework for opportunistic scheduling in wireless networks," *Computer Networks* 2003.
  - S. Borst, "User-level performance of channel-aware scheduling algorithms in wireless data networks," IEEE/ACM ToN 2005.
  - B. Sadiq, S. Baek, G. de Veciana, "Delay-optimal opportunistic scheduling and approximations: the log rule," IEEE/ACM ToN 2011.
  - J. Niño-Mora, "Dynamic priority allocation via restless bandit marginal productivity indices," TOP 2007. This is the "marginal productivity index" framing of marginal-utility indices.
  - R. Margolies et al., "Exploiting mobility in proportional fair cellular scheduling: measurements and algorithms," IEEE/ACM ToN 2016. This is PF with predicted future rates, the nearest "finite-window PF" I know of.
- No finite-horizon gradient scheduler with a terminal concave utility was found. That is not a verified absence: the search was cut short.

## 5. Multi-unit-per-slot (OFDMA, many resource blocks) index scheduling, its asymptotic optimality, and LP or fluid priority policies in wireless

### Takeaway
Asymptotic optimality of Whittle or Lagrangian priority policies, when users and per-slot capacity grow together, is proven in deadline scheduling: Yu–Xu–Tong, where the gap vanishes as arrivals and processors grow, and Hao–Xu–Tong, where the Lagrangian priority policy is asymptotically optimal under time-varying rate caps. In wireless downlink scheduling over Markov channels with ARQ observation, Ouyang–Eryilmaz–Shroff show that the Whittle policy approaches the relaxed-constraint optimum in the many-users limit. The general theory is in SURVEY F5 (Hu–Frazier, Zhang–Frazier, Gast–Gaujal–Yan, Brown–Smith, Hong et al.). **No OFDMA-specific index-policy paper, with several resource blocks per user per slot, could be verified this session.**

### Cited Findings
- Yu–Xu–Tong: "a capacity constraint that limits the simultaneous processing of jobs". The gap bound vanishes "as the job arrival rate and the number of available processors increase simultaneously". — [arXiv v4](https://www.arxiv.org/abs/1610.00399v4)
- Hao–Xu–Tong: a time-varying cap on total processing rate per stage, a Lagrangian priority policy that is asymptotically optimal as the system grows, and +22–49% over the Whittle index policy. — [IEEE Xplore](https://ieeexplore.ieee.org/document/9314189); [SUFE abstract](https://academicnewsletter.sufe.edu.cn/info/391167)
- Ouyang–Eryilmaz–Shroff: arbitrarily close to the relaxed-constraint optimum locally, and globally optimal in the many-users limit under a recurrence assumption. — [arXiv:1108.3768](https://arxiv.org/pdf/1108.3768)
- "Exponential asymptotic optimality of Whittle index policy," *Queueing Systems*, 2023, DOI 10.1007/s11134-023-09875-x. The title and venue appeared in search; the authors were not shown (I recall Gast, Gaujal and Yan, unverified). — [Springer](https://link.springer.com/article/10.1007/s11134-023-09875-x)
- "Conditions for indexability of restless bandits and an O(K³) algorithm to compute Whittle index," *Advances in Applied Probability*. Title and venue only; authors not shown. Relevant to computing [WI]. — [Cambridge](https://resolve-he.cambridge.org/core/journals/advances-in-applied-probability/article/abs/conditions-for-indexability-of-restless-bandits-and-an-mathcaloleftk3right-algorithm-to-compute-whittle-index/05749A3BD36DAC46166F96F9BEF8B9D7)
- The theory side is in SURVEY F5, with Gast–Gaujal–Yan's LP-based policies at [arXiv:2106.10067](https://arxiv.org/abs/2106.10067) and Hu–Frazier at [arXiv:1707.00205](https://arxiv.org/abs/1707.00205).

### Inferences
- **[SC] is the standard regime.** Our test (B = K/8 units per slot; the Lagrangian index's gap falls from 0.21 dB at K = 8 to 0.005 dB at K = 128) is the Weber–Weiss / Hu–Frazier regime, in which a fixed fraction of the arms is activated each slot. The same kind of result is proven for deadline jobs (B5, B6) and for downlink scheduling with Markov channels (C2). The scaling *phenomenon* is therefore not new. What is new is the finite-K numerical evidence on this reward, and the fact that the gap falls faster than 1/√K. That agrees with the O(1) and exponential refinements in SURVEY F5 (Zhang–Frazier; Gast et al.).
- **When a user takes several units in one slot, the arm becomes multi-action.** Our "B units per slot" serves B distinct users, one phase each, so it stays a standard two-action restless bandit. If a user could take several resource blocks in one slot (true OFDMA), the arm would need multi-action theory: weakly coupled MDPs as in Hong et al. (SURVEY), or the unverified multi-action candidates below.

### Gaps
- **Unverified candidates** (from memory):
  - Y. Cui, V. K. N. Lau, "Distributive stochastic learning for delay-optimal OFDMA power and subband allocation," IEEE TSP 2010. It uses per-user value-function approximation, a decomposition close to [DP].
  - D. Bertsimas, J. Niño-Mora, "Restless bandits, linear programming relaxations, and a primal-dual index heuristic," *Operations Research* 2000.
  - G. Zayas-Cabán, S. Jasin, G. Wang, "An asymptotically optimal heuristic for general nonstationary finite-horizon restless multi-armed, multi-action bandits," *Adv. Appl. Probab.* 2019.
  - G. Xiong, J. Li, R. Singh, "Reinforcement learning augmented asymptotically optimal index policy for finite-horizon restless bandits," AAAI 2022.
  - Verloop (AAP 2016; under question 1).
- I could not find, and could not search for, a wireless paper that applies LP or fluid priority policies (Bertsimas–Niño-Mora style) to OFDMA scheduling with deadlines.

## 6. The five closest works and an overlap verdict per contribution

### Takeaway
1. **Hao–Xu–Tong 2022** is the method twin: Lagrangian relaxation into per-arm programs, priority defined as the top of the indifference range, and asymptotic optimality.
2. **Yu–Xu–Tong 2018** is the problem twin: deadlines, a concave terminal reward, and B units per slot.
3. **Hosseini–Panwar** is the application twin: layered video with deadlines over wireless, as a restless bandit solved by LP.
4. **Ouyang, Murugesan, Eryilmaz and Shroff 2015/2016** is the twin for stale reports and many users: partially observed arms in downlink scheduling, near the relaxed optimum.
5. **The Ayesta–Erausquin–Jacko PI family**, with Jacko 2011 and Aalto–Lassila–Osti, is the twin for the closed-form index: opportunistic finite jobs, and an index comparing this slot with the expected or potential alternative.

No verified work combines all of: a deadline frame, a terminal concave quality reward driven by the SNR each phase received, fading-dependent per-user service, stale reports through a Gaussian report kernel, and a numerically tight Lagrangian certificate. Every one of these parts appears separately.

### Cited Findings
- (1) Hao, Xu, Tong, IEEE TAC 67(1):236–250, 2022, DOI 10.1109/TAC.2021.3049340: the Lagrangian priority value is the highest tax under which activating an arm remains optimal; asymptotically optimal; +22–49% over the Whittle index. — [IEEE Xplore](https://ieeexplore.ieee.org/document/9314189)
- (2) Yu, Xu, Tong, IEEE TAC 63 (2018) 2343–2358, arXiv:1610.00399: convex non-completion penalties and a capacity constraint; indexable; closed-form index for constant costs; the gap vanishes with scale. — [arXiv](https://www.arxiv.org/abs/1610.00399v4)
- (3) Hosseini, Panwar, "Restless Video Bandits: Optimal SVC Streaming in a Multi-user Wireless Network," arXiv:1612.03985: layered video with deadlines, solved by LP, up to 2× the users admitted at a quality. This is per SURVEY F2 and was **not re-checked** this session. — [arXiv](https://arxiv.org/abs/1612.03985)
- (4) Ouyang, Murugesan, Eryilmaz, Shroff, IEEE TIT 61(4):1702–1719, 2015 (Whittle indexable with a closed form under ARQ-only observation), and Ouyang, Eryilmaz, Shroff, IEEE/ACM ToN 2016 (near the relaxed-constraint optimum; many-users optimality). — [arXiv:1009.3959](https://arxiv.org/pdf/1009.3959); [arXiv:1108.3768](https://arxiv.org/pdf/1108.3768)
- (5) Ayesta, Erausquin, Jacko, *Perf. Eval.* 2010 (PI = current rate ÷ expected potential improvement; RB/PB/cμ as special cases), with Jacko, *Perf. Eval.* 2011 (one-period vs steady-state improvement) and Aalto, Lassila, Osti, SIGMETRICS 2015 / QUESTA 2016 (size-aware Whittle). — [eprint 65375](https://eprints.lancs.ac.uk/id/eprint/65375/); [eprint 65374](https://eprints.lancs.ac.uk/id/eprint/65374/); [slides](https://www.netlab.tkk.fi/~samuli/Presentations/Conferences/aalto-sigmetrics-2015.pdf)
- Runners-up:
  - Singh–Kumar: Lagrangian per-packet decomposition with transmission prices in deadline-constrained wireless networks. [arXiv:1606.01608](https://arxiv.org/pdf/1606.01608)
  - Aalto–Lassila–Taboada 2019: Whittle index with partial channel information. [Aaltodoc](https://aaltodoc.aalto.fi/items/4f9064e8-d4b0-4b81-a048-e1a3180d5e40)
  - Raghunathan–Borkar–Cao–Kumar 2008: deadlines, Whittle index, estimation from one ACK. [secondary](https://www.researchgate.net/scientific-contributions/Vivek-Raghunathan-22329655)
  - Hou–Borkar–Kumar 2009: frame-deadline model with debt priorities. [IDEALS](https://www.ideals.illinois.edu/items/105587)
  - Ayesta–Jacko–Novak 2011: abandonment as a random deadline. [eprint](https://eprints.lancs.ac.uk/id/eprint/71356)

### Inferences
| Our contribution | Closest prior work | Overlap verdict |
|---|---|---|
| [M] Deadline-frame restless bandit with a terminal concave PSNR reward (or a target) driven by channel-dependent phase service | Yu–Xu–Tong (convex non-completion penalty = concave terminal reward; B units); Hao–Xu–Tong; Hosseini–Panwar (layered video, deadlines); Ayesta–Jacko–Novak (random-deadline all-or-nothing) | **New instance, known skeleton.** Deadline restless bandits with concave terminal rewards exist. What is new is that per-slot service is a fading-dependent quality increment of a learned code, so the arm state needs (phases, accumulated weighted noise). |
| [DP] One program per user from the Lagrangian relaxation | Whittle/Hawkins/Adelman–Mersereau/Brown–Smith (SURVEY); Singh–Kumar (wireless, per-packet programs with prices); Hao–Xu–Tong | **Known technique**; the instantiation is ours. |
| [UB] Bound on every causal scheduler; measured gaps of 0.26 dB (K 8) to 0.005 dB per user (K 128) | The relaxation bound is textbook. Ouyang 2016 (relaxed-constraint optimum, wireless) and Yu–Xu–Tong (gap bound) prove asymptotic closeness. | **Bound known; the numerical certificate on a real codec reward is evidence not seen elsewhere**, although the search was incomplete. Jagannathan et al. (unverified) may have LP bounds under partial observation: check it. |
| [WI] Whittle index as the top of the indifference range | Hao–Xu–Tong define their priority as "the highest tax under which activating an arm remains optimal"; Brown–Smith on ties (SURVEY) | **Matched in definition.** Cite Hao–Xu–Tong. Our contribution is the finite-horizon computation on the two-coordinate state and the stale-report version. |
| [LI] Lagrangian index at the bound's per-slot prices | Hu–Frazier / Brown–Smith / Gast et al. (SURVEY); Hao–Xu–Tong's "Lagrangian priority policy" | **Known, and the name collides with Hao–Xu–Tong.** New only as applied to wireless deadline scheduling with quality rewards and stale reports. Whether Hao–Xu–Tong use the relaxation's optimal multipliers is not verified. |
| [AI] Closed-form advantage index (certainty-equivalent; gain now minus a typical slot's gain; quotas) | PI (Ayesta–Erausquin–Jacko); Jacko's one-period vs steady-state improvement; RB/PB; PF/gradient (unverified) | **Partially new.** The "current vs typical/potential" structure is published, in ratio form, for flow-level costs. A difference of predicted *terminal-quality* gains with per-user quotas in a deadline frame was not found. Present it as the certainty-equivalent of [WI]/[LI], in the PI tradition. |
| [SR] Report-to-channel kernel inside each arm's program; the bound holds for report-based schedulers | Ouyang et al. 2011/2015/2016 (belief arms, ARQ); Aalto–Lassila–Taboada 2019 (partial channel information); Jacko 2011; Raghunathan 2008 (estimation) | **Concept known; model new.** All users report every slot, the kernel is Gaussian with Jakes correlation, fading is i.i.d. across slots, and the reward is a finite-horizon terminal quality. The bound's validity follows the standard relaxation-in-information-space argument. |
| [SC] Gap shrinking as K grows with B = K/8 | Yu–Xu–Tong; Hao–Xu–Tong; Ouyang 2016; Gast et al. (SURVEY) | **Known phenomenon.** The new part is the empirical rate on this problem (faster than 1/√K). |
| [BL] PF, max-SNR, ID-guided | Aalto–Lassila–Osti compare against PF and PI | Standard. **Consider adding a PI-type baseline.** |

- **Positioning sentence the report writer can use.** Deadline restless bandits with concave terminal penalties (Yu–Xu–Tong) and Lagrangian priority rules (Hao–Xu–Tong) are established, as are partially observed wireless arms (Ouyang et al.; Aalto–Lassila–Taboada) and opportunistic flow-level indices (Ayesta–Erausquin–Jacko, Aalto–Lassila–Osti). Our contribution is to bring them together for a learned prefix code, whose arm state and reward come from an MMSE quality model of the SNRs each phase received, and to certify the result numerically against the Lagrangian bound under fresh and stale reports.

### Gaps
- Ranking caveat: the gradient/PF, OFDMA, feedback-delay and Kalman lines could not be verified (question 4/5/3 Gaps). An unverified work there could displace a pick, notably Jagannathan et al. (LP bounds under uncertain channels) or a finite-window PF.
- Not re-checked this session: Hosseini–Panwar's exact formulation, and the Hong et al. CMU blog (SURVEY F5).
- **Recommended follow-up verification, in priority order:**
  1. Hao–Xu–Tong full text: how the multipliers are chosen, and the exact definition of the priority value.
  2. Jagannathan–Mannor–Menache–Modiano.
  3. Murugesan–Schniter–Shroff.
  4. Stolyar 2005 and Agrawal–Subramanian 2002, for the gradient interpretation.
  5. Ma–Tepedelenlioğlu, for the r² scaling.
  6. Niño-Mora NGI 2008.
  7. Jaramillo–Srikant, and Kang et al. (LDF).
  8. Hosseini–Panwar re-read.
