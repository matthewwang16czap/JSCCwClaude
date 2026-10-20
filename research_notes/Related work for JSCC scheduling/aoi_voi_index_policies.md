# Index-policy scheduling for freshness, estimation error and information value (AoI, AoII, QAoI, VoI, remote estimation, semantics) and learned Whittle indices: prior work for finite-horizon deadline scheduling of learned image codes

> **Evidence base and limits (read first).** Compiled 2026-10-20 from about 33 targeted web searches. Each search returned abstracts, repository metadata and bibliographic records. No full texts could be opened. arXiv, HAL, dblp, NSF-PAR, Crossref, OpenAlex and Semantic Scholar were unreachable from this sandbox: WebFetch had no DNS, and curl got proxy 403s. The session's shared web-search budget (200 calls across all agents) also ran out mid-task. As a result, the planned searches on these topics were never run:
> - semantic and goal-oriented index policies;
> - learning Whittle indices;
> - finite-horizon and deadline AoI;
> - finite-horizon asymptotics.
>
> Every bullet under "Cited Findings" rests on a record that a search returned. **No index formula was visible in any record, so none is reported.** Items marked **UNVERIFIED** come from background knowledge only and must be checked before citing. References already in `jscc/docs/SURVEY.md` (sections F2 and F5) are pointed to, not repeated.
>
> **Our components, as used in the overlap tables below:**
> - **C1:** finite horizon T with a terminal reward (decoded PSNR at the deadline).
> - **C2:** MMSE link abstraction. The arm state is (phases held n, accumulated sensitivity-weighted noise Y). The error only decreases, and a passive arm is frozen.
> - **C3:** i.i.d. Rayleigh slot SNR with reports that may be stale, handled through a report-to-channel kernel inside the arm's program.
> - **C4:** B ≥ 1 identical units per slot, with at most one phase per user per slot.
> - **C5:** Lagrangian relaxation gives one DP per user and an upper bound, which serves as a certificate.
> - **C6:** Whittle index, and a Lagrangian index at per-slot prices.
> - **C7:** closed-form certainty-equivalent marginal-gain index: this slot's gain minus a typical slot's gain.
> - **C8:** all-or-nothing reward for reaching a PSNR target.
> - **C9:** continuous traffic. Each request has its own deadline and is run as a Whittle index.
>
> Table marks: Y = has it; ~ = partial or analogous; – = absent; ? = not visible in the records retrieved.

## 1. Whittle and Lagrangian index policies for AoI and its variants: what their indices look like, and whether any has a finite horizon, a terminal reward, or several units per slot

### Takeaway
The AoI Whittle-index line (Kadota et al. → Hsu et al. → Tripathi and Modiano → Maatouk et al. and Kriouile et al. → Zou et al.) works over an infinite horizon with time-averaged cost. Each arm's state is its age, or for AoII a belief, and passive arms get worse over time. The index is a closed-form function of age (or of an age-cost function) and of channel reliability. Several of these works allow M > 1 channels per slot and unreliable channels. The works that know the channel state (Talak et al.; Tang, Wang, Song and Song) do not use a Whittle index. Instead they use age-based or virtual-queue rules, or a single-user constrained MDP obtained by Lagrangian relaxation, which has a threshold in the channel state and is then truncated to fit the hard constraint. **None of the retrieved records has a finite horizon or a terminal reward.** The closest *method* to ours in this family is Tang, Wang, Song and Song: relax the per-slot constraint, solve one user's program with a Markov channel state, truncate, and prove asymptotic optimality.

### Cited Findings
**Broadcast AoI, one transmission per slot**
- I. Kadota, A. Sinha, E. Uysal-Biyikoglu, R. Singh, E. Modiano, "Scheduling Policies for Minimizing Age of Information in Broadcast Wireless Networks," *IEEE/ACM Trans. Networking* 26(6):2637–2650, 2018, DOI 10.1109/TNET.2018.2873606, arXiv 1801.01803.
  - A Greedy policy is optimal in symmetric networks.
  - For general networks the paper builds a randomized policy, a Max-Weight policy and a Whittle's-index policy, with performance guarantees that depend on the network configuration.
  - Max-Weight and Whittle beat the other policies in every simulated configuration and are near-optimal.
  - Sources: [ACM DL](https://dl.acm.org/doi/10.1109/TNET.2018.2873606); [arXiv](https://arxiv.org/pdf/1801.01803).
  - The conference version is Kadota, Uysal-Biyikoglu, Singh, Modiano, "Minimizing the Age of Information in Broadcast Wireless Networks," Allerton 2016, pp. 844–851. These details come from reference lists of later AoI papers that the search returned, e.g. [arXiv 2001.03096](https://arxiv.org/pdf/2001.03096).
- Kadota, Sinha and Modiano, "Optimizing Age of Information in Wireless Networks with Throughput Constraints" (INFOCOM 2018). Its randomized, Max-Weight and Whittle's-index policies are guaranteed to be within factors of 2, 4 and 8 of the minimum AoI. The record confirms the title and these factors; the author list and venue are from background knowledge. — [MIT DSpace](https://dspace.mit.edu/entities/publication/80e8d05d-d02e-4654-b0f9-1132222bfbad)
- Kadota and Modiano, "Minimizing the Age of Information in Wireless Networks with Stochastic Arrivals," arXiv 1905.07020. Content not retrieved. — [Semantic Scholar](https://www.semanticscholar.org/paper/Minimizing-the-Age-of-Information-in-Wireless-with-Kadota-Modiano/68acc0bfba74ea3eb394d296d18f71c68f7dacd7); [arXiv](https://arxiv.org/pdf/1905.07020)

**Random arrivals**
- Y.-P. Hsu, E. Modiano, L. Duan, "Scheduling Algorithms for Minimizing Age of Information in Wireless Broadcast Networks with Random Arrivals" (the arXiv title adds "The No-Buffer Case"), *IEEE Trans. Mobile Comput.* 19(12), 2020, presented in part at ISIT 2017 and ISIT 2018, arXiv 1712.07419.
  - Setting: a base station serves many users, information arrives at random, and transmission capacity is limited.
  - When arrival statistics are known (offline case), the paper uses MDP methods plus Whittle's method.
  - Structural results cut the MDP's computation and give a **closed-form Whittle index**.
  - In the no-buffer case, the index policy is age-optimal when all users' arrival processes are stochastically identical.
  - Sources: [arXiv](https://arxiv.org/pdf/1712.07419); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/134024).
- Y.-P. Hsu, "Age of Information: Whittle Index for Scheduling Stochastic Arrivals," ISIT 2018, arXiv 1801.03422. In numerical studies it nearly minimizes average age, but optimality is not proven in general. — [arXiv](https://arxiv.org/pdf/1801.03422)

**General age penalties, and the link to estimation error**
- V. Tripathi, E. Modiano, "A Whittle Index Approach to Minimizing Functions of Age of Information," Allerton 2019, arXiv 1908.10438.
  - Setting: several sources update a monitor over a single-hop broadcast network. The goal is to minimize the time average of **general non-decreasing cost functions of AoI**.
  - Method: a restless-bandit formulation, a proof of indexability, and a Whittle index policy.
  - The policy is exactly optimal for two sources with reliable channels, even when their cost functions differ.
  - Structural properties of optimal policies suggest the Whittle policy is near-optimal in general.
  - The published version also covers unreliable channels.
  - Minimizing **monitoring error for LTI systems and for symmetric Markov chains is equivalent to minimizing suitably chosen monotone functions of AoI**.
  - Constant-factor optimality is left open.
  - Sources: [arXiv](https://arxiv.org/abs/1908.10438); [NSF PAR](https://par.nsf.gov/servlets/purl/10197848); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/137830).
- Z. Tang, Z. Sun, N. Yang, X. Zhou, "Whittle Index-Based Scheduling Policy for Minimizing the Cost of Age of Information," *IEEE Commun. Lett.* 26(1), 2022, arXiv 2109.05869. Content not retrieved. — [ANU repository](https://openresearch-repository.anu.edu.au/items/70842812-f03e-4738-befc-488a37ecd588); [arXiv](https://arxiv.org/pdf/2109.05869)

**M of N channels, and the many-users limit**
- A. Maatouk, S. Kriouile, M. Assaad, A. Ephremides, "On the Optimality of the Whittle's Index Policy for Minimizing the Age of Information," *IEEE Trans. Wireless Commun.* 20(2):1263–1277, 2021, arXiv 2001.03096.
  - Setting: a central entity schedules **M of N users** over **unreliable channels**.
  - Each user's index depends on its age and its channel statistics; the M users with the highest indices are scheduled.
  - The policy is **asymptotically optimal in the many-users regime**. The proof first shows optimality near a specific system state, then globally under a recurrence assumption that is verified numerically.
  - An earlier result by the same authors: Greedy is optimal when all users have identical channel statistics.
  - Sources: [arXiv](https://arxiv.org/pdf/2001.03096); [HAL](https://hal.archives-ouvertes.fr/hal-03106667).
- S. Kriouile, M. Assaad, A. Maatouk, "On the Global Optimality of Whittle's Index Policy for Minimizing the Age of Information," *IEEE Trans. Inf. Theory* 68(1):572–600, 2022, arXiv 2102.02528. It proves the same asymptotic optimality under a less stringent condition. — [arXiv](https://arxiv.org/pdf/2102.02528); [HAL](https://hal.archives-ouvertes.fr/hal-03518032)

**Fading and channel-state knowledge, HARQ, heterogeneous channels**
- R. Talak, S. Karaman, E. Modiano, "Optimizing Age of Information in Wireless Networks with Perfect Channel State Information," WiOpt 2018 (IFIP), arXiv 1803.06471. The journal version is "Improving Age of Information in Wireless Networks With Perfect Channel State Information," *IEEE/ACM Trans. Networking* 28(4):1765–1778, 2020.
  - Setting: general interference constraints and **time-varying channels whose state the scheduler knows in every slot**.
  - A virtual-queue policy is optimal up to an additive constant. An **age-based policy is within a factor 4** of optimal.
  - Compared with the no-CSI case, age improves roughly 4-fold.
  - Sources: [arXiv](https://arxiv.org/pdf/1803.06471); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/137845); [journal record](https://biblio.cerist.dz/show/ar/00000000000000621371000000).
- H. Tang, J. Wang, L. Song, J. Song, "Minimizing Age of Information with Power Constraints: Multi-User Opportunistic Scheduling in Multi-State Time-Varying Channels," arXiv 1912.05947, presented in part at Allerton 2019. A precursor is arXiv 1908.01334, "Scheduling to Minimize Age of Information in Multi-State Time-Varying Networks with Power Constraints."
  - Setting: power-limited sensors, **Markov time-varying channel states**, and a hard bandwidth constraint.
  - Method: the hard bandwidth constraint is relaxed, giving a **single-sensor constrained MDP**. Its optimal policy has a **threshold structure** and is found by linear programming.
  - An **asymptotically optimal truncated policy** satisfies the hard constraint.
  - In simulations the scheduler exploits good channels.
  - Sources: [arXiv](https://arxiv.org/abs/1912.05947v3); [arXiv 1908.01334](https://arxiv.org/pdf/1908.01334); [CityU record](https://scholars.cityu.edu.hk/en/publications/scheduling-to-minimize-age-of-information-in-multi-state-time-var/).
- E. T. Ceran, D. Gündüz, A. György, "A Reinforcement Learning Approach to Age of Information in Multi-User Networks With HARQ," *IEEE JSAC* 39(5):1412–1426, 2021, DOI 10.1109/JSAC.2021.3065057, arXiv 2102.09774.
  - Setting: one source updates several users over unreliable channels. One user is served per slot, with ACK/NACK feedback, under a cap on average transmissions.
  - With known channel statistics, the paper derives optimal policies for **ARQ and HARQ**.
  - Without them, it uses reinforcement learning: SARSA with linear function approximation, UCRL2 and DQN.
  - "Whittle index" appears as an index term, but how it is used was not visible.
  - An earlier version without HARQ is arXiv 1806.00336.
  - Sources: [arXiv](https://arxiv.org/pdf/2102.09774); [Imperial PDF](https://spiral.imperial.ac.uk/bitstream/10044/1/87916/2/CGG_JSAC21.pdf); [arXiv 1806.00336](https://arxiv.org/pdf/1806.00336).
- Y. Zou, K. T. Kim, X. Lin, M. Chiang, "Minimizing Age-of-Information in Heterogeneous Multi-Channel Systems: A New Partial-Index Approach," ACM MobiHoc 2021, DOI 10.1145/3466772.3467030.
  - Setting: **multiple heterogeneous, unreliable channels**.
  - The Whittle index is not defined here because **indexability is ill-defined when there are several dual costs**.
  - The paper introduces **partial indexability and a partial index**, each defined with respect to one channel's cost given the other channels' costs.
  - The SWIM policy combines partial indices with max-weight matching and iteratively updates the dual costs.
  - SWIM is asymptotically optimal under a global-attractor condition. The paper won the best paper award, and the ToN version adds S. Zhou as an author.
  - Sources: [Purdue PDF](https://engineering.purdue.edu/~linx/paper/mobihoc21-partial-index.pdf); [NSF PAR](https://par.nsf.gov/biblio/10300528-minimizing-age-information-heterogeneous-multi-channel-systems-new-partial-index-approach); [Purdue news](https://engineering.purdue.edu/ECE/News/2021/best-paper-award-at-ACM-MobiHOC-goes-to-ECE-phd-student); [ToN tech report](https://staff.ie.cuhk.edu.hk/~xjlin/paper/ton25-partial-index-tech.pdf).

**Age of Incorrect Information (AoII) and query AoI (QAoI)**
- A. Maatouk, S. Kriouile, M. Assaad, A. Ephremides, "The Age of Incorrect Information: A New Performance Metric for Status Updates," *IEEE/ACM Trans. Networking* 28(5):2215–2228, 2020, DOI 10.1109/TNET.2020.3005549, arXiv 1907.06604.
  - AoII counts only updates that bring new, correct information to the monitor. It addresses the limits of both AoI and conventional error-penalty functions.
  - The paper formulates an MDP for one transmitter–receiver pair over an unreliable channel.
  - Sources: [arXiv](https://arxiv.org/pdf/1907.06604); [HAL](https://hal.archives-ouvertes.fr/hal-03106628).
- S. Kriouile, M. Assaad, "Minimizing the Age of Incorrect Information for Real-time Tracking of Markov Remote Sources," ISIT 2021, pp. 2978–2983, arXiv 2102.03245.
  - Setting: several Markov sources. The scheduler cannot see source states, so the problem is a **POMDP over beliefs**.
  - Method: **Lagrangian relaxation**. The dual problem has an optimal **threshold** policy, the problem is shown to be indexable, and **closed-form Whittle indices** are derived.
  - The policy beats an AoI-based baseline.
  - Follow-up: "Minimizing the Age of Incorrect Information for Unknown Markovian Source," arXiv 2210.09681, described as by the same authors.
  - Sources: [arXiv](https://arxiv.org/pdf/2102.03245); [arXiv 2210.09681](https://arxiv.org/pdf/2210.09681).
- Y. Chen, A. Ephremides, "Scheduling to Minimize Age of Incorrect Information with Imperfect Channel State Information," *Entropy* 23(12):1572, 2021, DOI 10.3390/e23121572.
  - Setting: a base station updates several users, only some of whom can be served per slot, with **imperfect CSI**.
  - Method: an MDP, then a relaxation that yields a **Whittle index policy when the problem is indexable** and an **"indexed priority policy" when it is not**.
  - The authors note that **with CSI a Whittle index policy is generally infeasible**.
  - Sources: [MDPI](https://www.mdpi.com/1099-4300/23/12/1572); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8699940/).
- F. Chiariotti, J. Holm, A. E. Kalør, B. Soret, S. K. Jensen, T. B. Pedersen, P. Popovski, "Query Age of Information: Freshness in Pull-Based Communication," *IEEE Trans. Commun.* 70(3):1606–1622, 2022, DOI 10.1109/TCOMM.2022.3141786, arXiv 2105.06845.
  - Age matters only when a receiver queries the data, so QAoI **samples age at the query instants**.
  - Optimizing QAoI can lead to different decisions than push-based AoI. It lowers both average and worst-case perceived age for periodic and stochastic queries.
  - The ICC precursor is "Freshness on Demand," arXiv 2011.00917.
  - Sources: [DeepAI](https://deepai.org/publication/query-age-of-information-freshness-in-pull-based-communication); [arXiv](https://arxiv.org/pdf/2105.06845.pdf); [AAU](https://vbn.aau.dk/da/publications/query-age-of-information-freshness-in-pull-based-communication); [arXiv 2011.00917](https://arxiv.org/pdf/2011.00917).

### Inferences
- **What the indices value is not what ours values.** In every verified AoI or AoII index, a passive arm gets worse on its own (age grows, or the belief drifts). The index therefore measures how costly the staleness is now, scaled by how reliable the channel is. In Maatouk et al.'s words, the index is based on "age and channel statistics".
  - In ours a passive arm is frozen. The state (n, Y) changes only when the user is served. The arm is "restless" only through the channel report and the deadline clock.
  - Our index therefore values *this slot's channel relative to the remaining slots before the deadline*. It can be zero or negative (wait) even for a user who is far from done. An AoI index is monotone in age; ours is not monotone in n, because the MMSE terms give diminishing returns.
- **Finite horizon and terminal reward.** None of the retrieved AoI works has either. The nearest analogue of evaluating at a deadline is QAoI (age sampled at query instants), but the verified record shows no index policy there. Finite-horizon index theory lives in the general RMAB literature (SURVEY F5) and in deadline RMABs (SURVEY F2), not in AoI.
- **Several units per slot.** This is common:
  - M of N in Maatouk et al. and Kriouile et al.;
  - heterogeneous channels in Zou et al.;
  - general interference constraints in Talak et al.;
  - a bandwidth constraint in Tang, Wang, Song and Song.

  Zou et al.'s result matters if our B units were frequency-selective, i.e. one SNR per user per unit. One price per slot would then become a price vector per slot, and plain Whittle indexability would no longer be well defined. With identical units, as in our model, a single per-slot price suffices.
- **The closest method in this family** is Tang, Wang, Song and Song: relax the hard per-slot constraint, solve one user's program (a CMDP solved by LP) with a Markov channel state, obtain a threshold in the channel state, truncate to meet the hard constraint, and prove asymptotic optimality. Our steps are the finite-horizon, MMSE-reward counterparts: per-slot prices, a per-user DP, a Lagrangian index, and a check of the bound gap.
- **Stale reports.** Chen and Ephremides found that with (imperfect) CSI a Whittle policy is "generally infeasible" and fell back on a priority index. This parallels our choice to define the index as the top of the price range over which serving and waiting are worth the same, and to use the Lagrangian index, which does not need indexability. Kriouile and Assaad's belief-state Whittle index is the AoII counterpart of an arm program that conditions on a stale report.
- **HARQ** (Ceran et al.) is the AoI counterpart of information accumulating over several transmissions. Its reward is all-or-nothing (decode or not), so it resembles our PSNR-target variant (C8) more than our smooth MMSE reward (C2).

| Work | Overlaps (C#) | What it lacks relative to ours |
|---|---|---|
| Kadota et al. ToN 2018 | C6 Y; C3 ~ (unreliable links; the abstract's wording was recalled, not re-read); C7 ~ (Max-Weight is a myopic drift rule) | C1, C2, C4, C5-bound, CSI per slot, C8, C9 |
| Hsu, Modiano, Duan TMC 2020 | C6 Y (closed form); C9 ~ (random arrivals, no deadlines) | C1, C2, C3 (fading), C5 |
| Tripathi and Modiano 2019 | C6 Y; C2 ~ (estimation error ↔ age penalty); C3 ~ | C1; partial refresh (each delivery is a full refresh); C4 ?; C5 |
| Maatouk et al. 2021; Kriouile et al. 2022 | C4 Y; C6 Y; many-user asymptotics | C1, C2, CSI per slot, C5-bound certificate |
| Talak, Karaman, Modiano ToN 2020 | C3 Y (CSI each slot); C4 ~ | C6 (no Whittle index), C1, C2, C5 |
| Tang, Wang, Song, Song 2020 | C3 Y (Markov channel state); C4 Y; C5 Y (relaxed single-user CMDP); C6 ~ (threshold, truncated policy) | C1, C2, C8, stale reports |
| Ceran, Gündüz, György JSAC 2021 | C3 ~; C8 ~ (HARQ decode-or-not); C5 ~ (constrained MDP) | C1, C2 (smooth MMSE), CSI per slot, C4 |
| Zou et al. MobiHoc 2021 | C4 Y (heterogeneous); C5 ~ (dual costs); C6 Y (partial index) | C1, C2, CSI per slot |
| Kriouile and Assaad ISIT 2021 | C5 Y (Lagrangian, threshold dual); C6 Y; belief state ~ C3 | C1, C2 (MMSE of a static source), fading |
| Chen and Ephremides 2021 | C3 Y (imperfect CSI); C4 Y; C6 Y/~ (priority fallback) | C1, C2, C5-bound |
| QAoI (Chiariotti et al. 2022) | C1 ~ (evaluation at query instants) | No index policy visible; C2–C6 |

### Gaps
- **Index formulas were not retrieved** for Kadota et al., Hsu et al., Tripathi and Modiano, or Maatouk et al., because arXiv and the publisher pages were unreachable. From background knowledge (UNVERIFIED): Kadota et al.'s index for linear AoI is quadratic in age and proportional to weight × success probability. Check this in arXiv 1801.01803 before quoting any form.
- **Not verified:**
  - whether Hsu et al. or Tripathi and Modiano allow more than one channel per slot;
  - the journal venue of Tang, Wang, Song and Song (believed to be *IEEE JSAC* 2020; UNVERIFIED);
  - how Ceran et al. use the Whittle index.
- **Finite-horizon or deadline AoI index policies.** The planned searches did not run because the budget ran out, so it is unknown whether such work exists. **Candidates (UNVERIFIED):**
  - N. Lu, B. Ji, B. Li, "Age-based Scheduling: Improving Data Freshness for Wireless Real-Time Traffic," MobiHoc 2018 (deadline traffic plus AoI);
  - V. Raghunathan, V. Borkar, M. Cao, P. R. Kumar, "Index policies for real-time multicast scheduling for wireless broadcast systems," INFOCOM 2008;
  - Z. Jiang, B. Krishnamachari, X. Zheng, S. Zhou, Z. Niu, "Timely status update in wireless uplinks: analytical solutions with asymptotic optimality," *IEEE IoT J.* 2019;
  - Sombabu, Mate, Manjunath, Moharir, "Whittle index for AoI-aware scheduling," COMSNETS 2020.
- **Titles seen in search listings but not read:**
  - "Whittle's index-based age-of-information minimization in multi-energy harvesting source networks" ([arXiv 2408.02570](https://arxiv.org/pdf/2408.02570));
  - "Minimizing Age of Information via Scheduling over Heterogeneous Channels" ([NSF PAR 10284912](https://par.nsf.gov/biblio/10284912-minimizing-age-information-heterogeneous-multi-channel-systems-new-partial-index-approach); the URL slug in the listing differs, so treat as uncertain).

## 2. Remote estimation and sensor scheduling with estimation-error rewards (Kalman MSE, MMSE) and Whittle or threshold indices: how close are they to our "accumulated weighted noise" state and MMSE reward?

### Takeaway
The estimation-error restless-bandit line uses a posterior error variance as the arm state, and the index is a function of that variance. This is the closest in *vocabulary* to our MMSE reward. The main works are:
- Le Ny, Feron and Dahleh: Kalman covariance, M sensors, a relaxation bound that splits into per-system problems, and an analytical Whittle index for scalar systems.
- Niño-Mora and Villar: a Kalman-filter marginal-productivity index, near-optimal as targets grow with the number of radars per target held fixed.
- Dance and Silander: a threshold on posterior variance, giving a Whittle index with closed forms.
- Ornee and Sun: an exact Whittle index for the MSE of Gauss–Markov sources over parallel channels.

All of these are steady-state, discounted or time-average. Their sources have process noise, so a passive arm's error grows. The structure is therefore not ours: our image is a *static* source sent as distinct, importance-ordered phases. Each phase's MMSE term is set by its slot's SNR and then frozen, and the reward is terminal. The control-oriented works add fading (Gatsis et al.; Liu et al.) and price-based coordination (Molin and Hirche; Gatsis et al.), but no finite-horizon index policies.

### Cited Findings
- J. Le Ny, E. Feron, M. A. Dahleh, "Scheduling Continuous-Time Kalman Filters," *IEEE Trans. Autom. Control* 56(6):1381–1394, 2011, DOI 10.1109/TAC.2010.2095970. The ACC 2009 version, "Scheduling Kalman Filters in Continuous Time," is arXiv 0810.5148.
  - Setting: **N independent linear systems observed by M sensors**. The goal is a steady-state causal estimate with a measurement cost; sensors switch instantly between systems, and there are extra resource constraints.
  - A **relaxation gives a performance bound**. It is computed from a convex program with linear matrix inequalities, which **splits into smaller coupled problems**.
  - In the scalar case with identical sensors, there is an **analytical Whittle index**, and **indexability is guaranteed**.
  - In general, open-loop periodic switching policies **approach the bound arbitrarily closely**.
  - Sources: [arXiv](https://arxiv.org/abs/0810.5148); [author PDF](https://dahleh.lids.mit.edu/wp-content/uploads/2011/09/LeNy.IEEE_.09.Scheduling.Kalman.pdf); [IEEE via Unpaywall](https://unpaywall.org/10.1109%2FTAC.2010.2095970); [MIT DSpace](https://dspace.mit.edu/handle/1721.1/62007).
- J. Niño-Mora, S. S. Villar, "Multitarget tracking via restless bandit marginal productivity indices and Kalman filter in discrete time," IEEE CDC 2009. A 2010 UC3M working paper (ws101506) also exists.
  - Setting: phased-array radars track targets with independent scalar Gauss–Markov dynamics, using a Kalman filter. The objective is discounted or average, and covers tracking-error variance plus measurement costs.
  - The problem is treated as a restless bandit with real-valued states, using Whittle's marginal-productivity index.
  - The index policy beats the myopic policies advocated in earlier work.
  - A small-scale study suggests it is **nearly optimal as the number of targets grows, with radars per target held fixed**.
  - Sources: [UC3M e-archivo](https://e-archivo.uc3m.es/entities/publication/2bf8059f-5424-46b3-9a48-11a316c2846e); [RePEc](https://ideas.repec.org/p/cte/wsrepe/ws101506.html).
- C. R. Dance, T. Silander, "Optimal Policies for Observing Time Series and Related Restless Bandit Problems," *JMLR* 20(35):1–93, 2019, arXiv 1703.10010.
  - Setting: when to make noisy, costly observations of a Gaussian random walk.
  - A **threshold policy is optimal**: observe when the posterior variance exceeds a level. The result extends to many other cost functions.
  - The proof uses Niño-Mora's verification theorem and the link between Christoffel words and discontinuous nonlinear maps.
  - The result carries over to LQG control with costly observations.
  - The restless bandit of observing **several such series has a well-defined Whittle index**. The paper gives **closed-form formulae** for it and compares it with heuristics.
  - Sources: [JMLR](https://jmlr.org/papers/v20/17-185.html); [arXiv](https://arxiv.org/pdf/1703.10010).
- T. Z. Ornee, Y. Sun, "A Whittle Index Policy for the Remote Estimation of Multiple Continuous Gauss-Markov Processes over Parallel Channels," ACM MobiHoc 2023. The arXiv v4 (2305.04809) is titled "Remote Estimation of Gauss-Markov Processes over Multiple Channels: A Whittle Index Policy."
  - Setting: a scheduler decides when to sample several continuous-time Gauss–Markov processes and send the samples over **multiple channels**. Transmission times are i.i.d. across samples and channels.
  - The goal is to minimize a **weighted sum of time-averaged expected estimation errors**.
  - The problem is a continuous-time restless bandit with a continuous state. The paper proves indexability and gives an **exact Whittle index**.
  - It also gives a signal-agnostic (AoI) variant over parallel channels.
  - With one source and one channel, the optimal policy is both threshold sampling and a Whittle index rule: **the index is zero exactly when the channel is idle and the estimation error equals the sampling threshold**.
  - Sources: [arXiv](https://arxiv.org/html/2305.04809v4); [NSF PAR](https://par.nsf.gov/biblio/10497737-whittle-index-policy-remote-estimation-multiple-continuous-gauss-markov-processes-over-parallel-channels).
- K. Gatsis, M. Pajic, A. Ribeiro, G. J. Pappas, "Opportunistic Control Over Shared Wireless Channels," *IEEE Trans. Autom. Control* 60(12):3140–3155, 2015, DOI 10.1109/TAC.2015.2416922. The ICCPS 2014 version is "Opportunistic Scheduling of Control Tasks Over Shared Wireless Channels," pp. 48–59, DOI 10.1109/ICCPS.2014.6843710.
  - Setting: the scheduler **observes random channel conditions** and picks which control systems transmit on **non-overlapping frequencies**. Each system's power adapts to its channel and sets its probability of success.
  - The objective is to minimize total power subject to an **expected Lyapunov decrease per system**.
  - The paper gives an offline algorithm for the optimal design and an online protocol that converges almost surely to it.
  - Sources: [DukeSpace](https://dukespace.lib.duke.edu/items/11c602bd-610d-4dfa-bc1c-6811ff821e11); [Duke Scholars](https://scholars.duke.edu/individual/pub1080247); [ICCPS PDF](https://cpsl.pratt.duke.edu/sites/cpsl.pratt.duke.edu/files/images/gatsis_iccps14.pdf).
- W. Liu, D. E. Quevedo, K. H. Johansson, B. Vucetic, Y. Li, "Stability Conditions for Remote State Estimation of Multiple Systems over Multiple Markov Fading Channels," *IEEE Trans. Autom. Control* 68(7):4273–4280, 2023, DOI 10.1109/TAC.2022.3201172, arXiv 2104.04181.
  - It gives a necessary and sufficient stability condition in terms of the system parameters and the channel statistics, plus a sensor scheduling policy that meets it.
  - A companion paper on semi-Markov channels reports that fast fading can be preferable to slow fading for stability.
  - Sources: [KTH PDF](https://people.kth.se/~kallej/papers/Remote_State_Estimation_TAC2023_Liu.pdf); [record](https://academicnewsletter.sufe.edu.cn/info/389802); [arXiv 2104.04181](https://export.arxiv.org/abs/2104.04181v1); [arXiv 2203.16826](https://arxiv.org/abs/2203.16826v3).
- A. Molin, S. Hirche, "Price-based Adaptive Scheduling in Multi-Loop Control Systems with Resource Constraints," *IEEE Trans. Autom. Control* 59(12):3282–3295, 2014. Secondary descriptions say the work uses a **distributed price-exchange mechanism**, in which event triggers **adapt their thresholds to the resource constraint**. — [TUM mediaTUM](https://mediatum.ub.tum.de/1285321); [CPS-VO paper](https://cps-vo.org/sites/cps-vo.org/files/cpsvo_file_nodes/Resource-Aware_Control_for_Cyber-Physical_Systems_paper.pdf)
- A. Molin, H. Esen, K. H. Johansson, "Scheduling networked state estimators based on Value of Information," *Automatica* 110:108578, 2019, DOI 10.1016/j.automatica.2019.108578. Per a secondary description (a 2020 IEEE TCNS paper by Vasconcelos et al., per the search summary), it proposes a **dynamic priority scheme** for real-time data over a shared network, **based on Value of Information**. Its abstract was not retrieved. — [Stuttgart PUMA record](https://puma.ub.uni-stuttgart.de/bibtex/1694dafbc846f5aa65e85853470e033bd); [NSF PAR copy of the secondary source (uncertain match)](https://par.nsf.gov/servlets/purl/10213784)
- A. S. Leong, A. Ramaswamy, D. E. Quevedo, H. Karl, L. Shi, "Deep reinforcement learning for wireless sensor scheduling in cyber–physical systems," *Automatica* 113:108759, 2020, DOI 10.1016/j.automatica.2019.108759, arXiv 1809.05149. It formulates sensor scheduling as an MDP and solves it with a Deep Q-Network. It significantly outperforms round-robin and reduced-waiting-time scheduling in many examples. — [Paderborn RIS](https://ris.uni-paderborn.de/record/15741); [PDF](https://ei.uni-paderborn.de/fileadmin-eim/elektrotechnik/fg/rat/other/test/pdf-article/leoram20a.pdf)
- **Lagrangian sensor management.**
  - J. L. Williams' MIT work handles a communication-cost constraint in leader-node sensor management by **Lagrangian relaxation**. His PhD thesis (MIT, 2007) combines Lagrangian relaxation with heuristic approximations for sensor-network tracking. — [CSAIL abstract](https://publications.csail.mit.edu/abstracts/abstracts06/jlwil/jlwil.html); [DSpace](https://dspace.mit.edu/handle/1721.1/38534)
  - A talk abstract by D. Castañón poses sensor control as a POMDP with combinatorially large action and state spaces. A relaxation gives a new POMDP whose value **bounds the optimal performance**. — [NYU event](https://engineering.nyu.edu/events/2012/04/19/active-sensing-theories-and-algorithms)

### Inferences
- **How our state maps onto a Kalman state.** For a Gaussian source sent through linear MMSE links, the error of each received component is its variance divided by (1 + SNR). Our Y is the sum of these sensitivity-weighted per-phase errors, plus the full variance of the phases not yet received. That is a posterior error, as in Kalman scheduling, with three differences:
  1. **No process noise:** a passive arm's error stays put instead of growing toward its prior.
  2. **Each phase is a distinct component sent once:** it is not a repeated measurement of the same state.
  3. **Measurement quality is the random slot SNR:** it is not a fixed sensor noise.

  Points 1 and 2 make our arm resemble an ordered assignment of items with SNR-dependent values (Derman–Lieberman–Ross; SURVEY F2) more than a Kalman bandit. In a Kalman bandit, the urgency of an index comes from error growth; in ours, it comes from the deadline and the channel.
- **Overlap of structure.** The overlap is real at three levels:
  - **Bound:** Le Ny et al.'s decomposable relaxation bound, approached by constructed policies, is the closest precedent for our Lagrangian bound used as a certificate.
  - **Index as a function of an error state:** Le Ny; Niño-Mora and Villar; Dance and Silander; Ornee and Sun.
  - **Thresholds ↔ index:** Ornee and Sun's "index zero exactly at the threshold" is the steady-state version of our Lagrangian index compared with a price.
- **Overlap of reward.** Partial. Their reward is a running (time-average or discounted) error. Ours is a terminal PSNR, a log of the final weighted MMSE, which is concave and saturating. It is computed from a calibrated model of a learned code, not from a linear-Gaussian model.
- **Fading.** Gatsis et al. and Liu et al. show that channel-aware estimation and control scheduling over fading channels is established. Gatsis et al. is also an opportunistic "pick systems for non-overlapping frequencies given observed channels" problem with a dual (price) design: C3 + C4 + C5 in a control setting. Neither is an index policy over a finite horizon.
- **Value of Information.** VoI priority (Molin et al. 2019) and price-adapted triggers (Molin and Hirche 2014) are the control-side precedents for our C7 marginal-gain index and our per-slot prices. From what was retrieved, VoI is a "gain from sending now" rule. It lacks our subtraction of a typical later slot's gain, which comes from the deadline and the fading.

| Work | Overlaps (C#) | What it lacks relative to ours |
|---|---|---|
| Le Ny, Feron, Dahleh TAC 2011 | C2 ~ (Kalman variance); C4 Y (M sensors); C5 Y (decomposable relaxation bound); C6 Y (scalar Whittle) | C1 (steady state), C3 (no fading), static source, C8, C9 |
| Niño-Mora and Villar CDC 2009 | C2 ~; C4 Y (radars); C6 Y (marginal-productivity index); many-target regime | C1, C3, C5-certificate (?) |
| Dance and Silander JMLR 2019 | C2 ~ (posterior variance); C6 Y (closed form) | C1, C3, C4 (?), C5 |
| Ornee and Sun MobiHoc 2023 | C2 ~ (MSE); C4 Y (parallel channels); C6 Y (exact) | C1, C3 (random delays, not SNR-dependent quality), static source, C5 |
| Gatsis et al. TAC 2015 | C3 Y; C4 Y; C5 ~ (offline optimum plus online protocol) | C1, C2 (Lyapunov decrease, not MMSE), C6 |
| Liu et al. TAC 2023 | C2 ~; C3 Y (Markov fading); C4 Y | C1, C5, C6 (stability, not an optimal index) |
| Molin and Hirche TAC 2014 | C5 ~ (prices); C6 ~ (thresholds adapted to prices) | C1, C2, C3 |
| Molin, Esen, Johansson 2019 | C2 ~; C7 ~ (value-of-information priority) | C1, C3, C5 (?) |
| Leong et al. 2020 | C2 ~; learned scheduler | C1, C5, C6 |

### Gaps
- Full texts were not read, so the shape of the scalar Whittle index in Le Ny et al. and the closed forms in Dance and Silander are not reported. Whether Le Ny's relaxation is literally a Lagrangian dual is also unconfirmed: the record says "relaxation … convex program … decomposed into smaller coupled problems".
- **Candidates not checked (UNVERIFIED):**
  - D. A. Castañón, "Approximate dynamic programming for sensor management," IEEE CDC 1997. From memory it decouples a finite-horizon classification problem with a terminal cost by Lagrangian relaxation of per-stage sensor constraints. **If confirmed, it is the closest classical precedent for our terminal-reward Lagrangian decomposition.** One search for it returned only related work.
  - J. L. Williams, J. W. Fisher III, A. S. Willsky, "Approximate dynamic programming for communication-constrained sensor network management," *IEEE Trans. Signal Process.* 2007.
  - J. Chakravorty, A. Mahajan, threshold remote-estimation papers (IEEE TAC 2017, 2020).
  - X. Ren, J. Wu, K. H. Johansson, G. Shi, L. Shi, transmission power control for remote estimation over fading channels (IEEE TAC 2018).
  - S. Wu, X. Ren, S. Dey, L. Shi, "Optimal scheduling of multiple sensors over shared channels with packet transmission constraint," *Automatica* 2018.
  - V. Gupta, T. H. Chung, B. Hassibi, R. M. Murray, stochastic sensor selection (*Automatica* 2006).
  - T. Soleymani, J. S. Baras, S. Hirche, K. H. Johansson, value of information as a difference of value functions (IEEE TAC, around 2023).
  - O. Ayan, M. Vilgelm, M. Klügel, S. Hirche, W. Kellerer, "Age-of-information vs. value-of-information scheduling for cellular networked control systems," ICCPS 2019.
  - Y. Sun, Y. Polyanskiy, E. Uysal, Wiener-process sampling with random delay (IEEE TIT 2020).
  - J. Pan, Y. Sun, N. B. Shroff, Wiener-process estimation over an unreliable channel (SIGMETRICS 2023).
  - V. Krishnamurthy, R. J. Evans, hidden Markov model multi-armed bandits for beam scheduling (IEEE TSP 2001).
  - Estimating a *static* Gaussian parameter with SNR-dependent measurements as a restless bandit (the special case closest to ours). No such work was found or searched.
- **VoI titles seen but not read:**
  - "Consistency of Value of Information: Effects of Packet Loss and Time Delay in Networked Control Systems Tasks" ([arXiv 2403.11932](https://arxiv.org/pdf/2403.11932));
  - "Value of information in remote estimation subject to delay and packet loss" ([TUM portal](https://portal.fis.tum.de/en/publications/value-of-information-in-remote-estimation-subject-to-delay-and-pa/));
  - "Scheduling Policy for Value-of-Information (VoI) in Trajectory Estimation for Digital Twins" ([arXiv 2301.10962](https://ar5iv.labs.arxiv.org/html/2301.10962)).
- Mamduhi, Molin, Tolić, Hirche, *Automatica* 81:209–216, 2017 extends the price-based line, according to the search summary for Molin and Hirche. Its title was not seen.

## 3. Scheduling for the timeliness of semantic or task information with index policies

### Takeaway
The verified "semantic freshness" metrics that come with index policies are:
- **AoII**, with a belief-state Whittle index (Kriouile and Assaad) and an imperfect-CSI variant with a priority fallback (Chen and Ephremides);
- **signal-aware estimation** (Ornee and Sun), together with Tripathi and Modiano's equivalence between monitoring error and functions of age.

QAoI moves the evaluation instant to query times, which is pull-based and goal-oriented. None of the retrieved works uses the decoded quality of content (e.g., PSNR from a learned code) as the reward, and none has a finite horizon. Our reward is "semantic" in that it is task quality rather than age, but the index machinery we use is generic.

### Cited Findings
- AoII penalizes only time spent with incorrect information. It was introduced as a metric beyond both AoI and conventional error penalties. — [arXiv 1907.06604](https://arxiv.org/pdf/1907.06604)
- A UMD thesis frames AoII as "a new performance metric in semantic communications." The thesis author is not shown in the record seen. — [UMD DRUM](https://drum.lib.umd.edu/items/b6245cea-0aec-4ee4-a9fe-4be544d51e2d)
- For several Markov sources, AoII gets a closed-form Whittle index via a belief POMDP and Lagrangian relaxation. It beats an AoI-based baseline. — [arXiv 2102.03245](https://arxiv.org/pdf/2102.03245)
- With imperfect CSI, AoII gets a Whittle index when indexable and a priority index otherwise. A Whittle policy with CSI is "generally infeasible." — [MDPI](https://www.mdpi.com/1099-4300/23/12/1572)
- Signal-aware and signal-agnostic Whittle indices exist for Gauss–Markov estimation over parallel channels. In the single-source case, the signal-aware index ties to threshold sampling. — [arXiv 2305.04809](https://arxiv.org/html/2305.04809v4)
- Monitoring error of LTI systems and of symmetric Markov chains is equivalent to monotone functions of AoI, so AoI-function Whittle indices cover these estimation tasks. — [arXiv 1908.10438](https://arxiv.org/abs/1908.10438)
- QAoI evaluates age only at query instants, and optimizing it gives different decisions than push-based AoI. — [IEEE TCOM 2022 via DeepAI](https://deepai.org/publication/query-age-of-information-freshness-in-pull-based-communication)
- Already in our survey (F2): an uncertainty-of-information RMAB (arXiv 2102.06384) and significance-driven semantic communication framed as an RMAB (arXiv 2608.28441). The survey notes that semantic RMABs so far use abstract rewards, not a model of decoded image quality. — [SURVEY.md F2](/home/user/JSCCwClaude/jscc/docs/SURVEY.md)

### Inferences
- **Signal-aware vs signal-agnostic** (Ornee and Sun) maps onto our *content-aware* index, which uses per-image sensitivity weights and calibrated PSNR, versus a content-agnostic rule that counts phases. Their finding that the signal-aware policy is the better one supports making the content-aware index a contribution only as an application. The general principle is established.
- **AoII** maps loosely onto our all-or-nothing target reward (C8): a user is "incorrect" until its PSNR target is met. AoII's belief-state Whittle index, built from a Lagrangian threshold dual, is the methodological cousin of our C8 arm program.
- **Tripathi and Modiano's equivalence does not carry over.** In AoI, each delivery is a full refresh, so error is a function of age. In ours, each delivery is a *partial refinement* (one phase), so the error depends on (n, Y), not on any age. An age-penalty Whittle index therefore cannot represent our reward; this distinction is worth stating in a paper.
- **QAoI** is the AoI-side justification for scoring at a deadline. A request's quality matters only when the image is consumed, which is our terminal reward.

### Gaps
- The searches on semantic and goal-oriented Whittle policies were blocked by the budget. **Candidates (UNVERIFIED):**
  - M. K. C. Shisher, B. Ji, I-H. Hou, Y. Sun, "Learning and Communications Co-Design for Remote Inference Systems: Feature Length Selection and Transmission Scheduling," *IEEE JSAIT* 2023. Choosing a feature length is the closest semantic analogue of choosing how many phases a user gets. **High priority to verify.**
  - M. K. C. Shisher, Y. Sun, "How does data freshness affect real-time supervised learning?" MobiHoc 2022.
  - A. Maatouk, M. Assaad, A. Ephremides, "The Age of Incorrect Information: an Enabler of Semantics-Empowered Communication," *IEEE TWC* 2023.
  - M. Kountouris, N. Pappas, "Semantics-Empowered Communication for Networked Intelligent Systems," *IEEE Commun. Mag.* 2021.
  - E. Uysal et al., "Semantic Communications in Networked Systems: A Data Significance Perspective," *IEEE Network* 2022.
  - J. Luo, N. Pappas, semantic-aware remote estimation of multiple Markov sources (arXiv 2024).
  - E. Fountoulakis, N. Pappas, M. Kountouris, "Goal-oriented policies for cost of actuation error minimization," *IEEE Commun. Lett.* 2023.
- It is unknown whether any work schedules learned semantic or JSCC payloads with an AoI- or AoII-style index. SURVEY F2 (2026-10-13) found none for deep JSCC with a learned quality reward; this session could not re-check.

## 4. Learning Whittle indices: what do learning approaches add when the model is known, as ours is?

### Takeaway
The learning-based schedulers verified in this session learn because something is unknown:
- channel statistics: Ceran et al. use SARSA, UCRL2 and DQN for AoI with HARQ;
- source dynamics: Kriouile and Assaad's AoII paper with an unknown Markov source;
- or a state space too large for dynamic programming: Leong et al. use DQN for sensor scheduling.

NeurWIN (already in SURVEY F2) learns an index from a simulator of one arm. Our arm model is known (calibrated codec, known fading law and report kernel) and its program is small, so learning adds nothing to the nominal problem. It can still add robustness to model mismatch and online adaptation, but at the cost of the Lagrangian certificate.

### Cited Findings
- With known channel statistics, Ceran et al. derive optimal ARQ and HARQ AoI policies. Without them, they compare SARSA with linear function approximation, UCRL2 and DQN, which need no channel statistics. — [arXiv 2102.09774](https://arxiv.org/pdf/2102.09774)
- Leong et al. pose wireless sensor scheduling for remote estimation as an MDP and solve it with a model-free Deep Q-Network, beating round-robin and reduced-waiting-time baselines. — [Paderborn RIS](https://ris.uni-paderborn.de/record/15741)
- A follow-up on minimizing AoII for an *unknown* Markovian source exists. Its content was not retrieved. — [arXiv 2210.09681](https://arxiv.org/pdf/2210.09681)
- Our survey distinguishes **planning** restless bandits (known dynamics, every arm observed) from **learning** ones (unknown payoffs, bandit feedback), and places our problem in the planning class: full channel reports and a calibrated quality model, so nothing has to be explored. It lists NeurWIN (NeurIPS 2021) as the learned-index route "if the index has to be learned (CQI errors, correlated fading)." — [SURVEY.md F2, F5](/home/user/JSCCwClaude/jscc/docs/SURVEY.md)

### Inferences
- **What learning would add for us:**
  1. Robustness when the channel or report model is wrong, e.g. correlated fading or a stale-report law that differs from the kernel. This is Ceran et al.'s motivation.
  2. Adaptation if the codec's quality model drifts, or across image distributions.
  3. Scale, if the arm state grew too large for exact DP (Leong et al.'s motivation). For us that would mean many more phases, fine Y grids, or long report memories. That is not our regime: 6 phases and T slots.
  4. Learning a joint priority under continuous traffic where per-request deadlines interact.
- **What learning would cost us:**
  - The exact per-user DP already yields the Whittle and Lagrangian indices and the upper bound. A learned index offers no certificate unless the bound is still computed from the model.
  - NeurWIN-style training also presupposes indexability, which our Lagrangian index avoids.
- **The reviewer-proof framing:** "our model is known, so we compute indices and bounds exactly; learned-index methods target unknown models and are complementary." Optionally add a small model-mismatch experiment that shows how far the computed index degrades under mismatch.

### Gaps
- The planned searches on learning Whittle indices did not run. **Candidates (UNVERIFIED; check venue, year and claims before citing):**
  - K. Avrachenkov, V. S. Borkar, "Whittle index based Q-learning for restless bandits with average reward," *Automatica* 2022 (arXiv 2004.14427): two-timescale Q-learning with a slower subsidy iterate that converges to the Whittle index.
  - J. Fu, Y. Nazarathy, S. Moka, P. G. Taylor, "Towards Q-learning the Whittle index for restless bandits," ANZCC 2019.
  - F. Robledo, V. Borkar, U. Ayesta, K. Avrachenkov, "QWI: Q-learning with Whittle Index," *ACM SIGMETRICS PER* 2021/22.
  - K. Nakhleh, S. Ganji, P.-C. Hsieh, I-H. Hou, S. Shakkottai, "NeurWIN," NeurIPS 2021 (already cited in SURVEY F2).
  - K. Nakhleh, I-H. Hou, "DeepTOP: Deep Threshold-Optimal Policy for MDPs and RMABs," NeurIPS 2022.
  - G. Xiong, J. Li, R. Singh, "Reinforcement Learning Augmented Asymptotically Optimal Index Policy for Finite-Horizon Restless Bandits," AAAI 2022. It is **finite-horizon** and learns an LP-based (Hu–Frazier-type) index, so it is **the most relevant learning paper for us if confirmed**.
  - G. Xiong, J. Li, finite-time analysis of Whittle-index Q-learning with neural function approximation, NeurIPS 2023.
  - K. Wang, L. Xu, A. Taneja, M. Tambe, "Optimistic Whittle Index Policy: Online Learning for Restless Bandits," AAAI 2023.
  - J. A. Killian, A. Biswas, S. Shah, M. Tambe, "Q-Learning Lagrange Policies for Multi-Action Restless Bandits," KDD 2021.
  - A. Biswas, G. Aggarwal, P. Varakantham, M. Tambe, "Learn to Intervene," IJCAI 2021.
  - E. U. Atay, I. Kadota, E. Modiano, "Aging Wireless Bandits: Regret Analysis and Order-Optimal Learning Algorithm," WiOpt 2021.
  - V. Tripathi, E. Modiano, an online-learning approach to time-varying AoI costs, MobiHoc 2021.
  - N. Akbarzadeh, A. Mahajan, learning Whittle index policies with scalable regret (arXiv 2022).
- No "Whittle-index DQN for AoI" paper was verified. Leong et al. is a DQN scheduler, not a learned index.

## 5. Asymptotic optimality with many users and a fixed fraction served per slot, and Lagrangian bounds used to certify policies

### Takeaway
In the AoI line, **asymptotic optimality of Whittle-type policies is proven only for infinite-horizon, time-average cost**, and only under extra conditions:
- recurrence or attractor-type conditions with M of N users (Maatouk et al. 2021; Kriouile et al. 2022 under a weaker condition);
- a global attractor for heterogeneous channels (Zou et al. 2021);
- for the truncated relaxed-CMDP policy over Markov channels (Tang, Wang, Song, Song).

Otherwise, certificates are constant-factor or additive bounds: factors 2, 4 and 8 in Kadota et al.; a factor of 4 and an additive constant in Talak et al. The closest analogue of **certifying a policy with a decomposed relaxation bound** is Le Ny, Feron and Dahleh, where constructed policies approach the bound arbitrarily closely. In the finite-horizon regime, which needs no attractor condition, the references remain those already in SURVEY F5: Hu–Frazier, Brown–Smith, Zhang–Frazier, Gast et al. and Hong et al.

### Cited Findings
- Whittle for AoI, M of N users over unreliable channels: asymptotically optimal in the many-users regime. Local optimality comes first; global optimality holds under a recurrence assumption verified numerically. — [arXiv 2001.03096](https://arxiv.org/pdf/2001.03096)
- The same result under a less stringent condition (IEEE TIT 2022). — [arXiv 2102.02528](https://arxiv.org/pdf/2102.02528)
- SWIM (partial index plus matching) is asymptotically optimal under a global-attractor condition, for heterogeneous multi-channel AoI. — [Purdue PDF](https://engineering.purdue.edu/~linx/paper/mobihoc21-partial-index.pdf)
- Relaxing the hard bandwidth constraint to a single-sensor CMDP, solved by LP, then truncating it gives an "asymptotic optimal truncated policy" over Markov channels. — [arXiv 1912.05947](https://arxiv.org/abs/1912.05947v3)
- Factor guarantees: randomized, Max-Weight and Whittle are within 2, 4 and 8 of optimal AoI (throughput-constrained paper). — [MIT DSpace](https://dspace.mit.edu/entities/publication/80e8d05d-d02e-4654-b0f9-1132222bfbad)
- With CSI: a virtual-queue policy within an additive constant, and an age-based policy within a factor 4. — [arXiv 1803.06471](https://arxiv.org/pdf/1803.06471)
- A relaxation bound from a convex LMI program that splits into smaller coupled problems, approached arbitrarily closely by periodic switching policies (Kalman scheduling). — [arXiv 0810.5148](https://arxiv.org/abs/0810.5148)
- Kalman marginal-productivity index: nearly optimal as targets grow with radars per target fixed (small-scale study). — [UC3M](https://e-archivo.uc3m.es/entities/publication/2bf8059f-5424-46b3-9a48-11a316c2846e)
- The finite-horizon theory already in our survey: Hu and Frazier; Brown and Smith; Zhang and Frazier; Gast, Gaujal and Yan. Index policies from the single-arm programs at optimal multipliers lose O(√N) without indexability; tie-breaking matters. — [SURVEY.md F5](/home/user/JSCCwClaude/jscc/docs/SURVEY.md)

### Inferences
- **None of the AoI asymptotic results applies to our main frame** (K = 8, one unit per slot, finite horizon). They are average-cost results that need recurrence or attractor conditions. For our scaled frame (B = K/8 units), the right theory is the finite-horizon line in F5, which needs no such conditions. The AoI results are worth citing only as "the wireless-scheduling analogue".
- **Kinds of certificate.**
  - AoI certificates are mostly sample-path lower bounds that give constant factors.
  - The estimation side (Le Ny et al.) uses a decomposable relaxation bound and shows that achievable policies approach it. That is what we do with the Lagrangian bound.
  - Neither line certifies a policy under *stale channel reports*. Our claim that "the Lagrangian bound remains a bound on every report-based scheduler" (SURVEY F5, "What changes in our frame") was not found in the retrieved works.
- **The fixed-fraction regime** appears in Maatouk et al. and Kriouile et al. (M of N) and in Niño-Mora and Villar (radars per target fixed). Our rmab_scale experiment (B = K/8) is the finite-horizon counterpart.

### Gaps
- **Candidates (UNVERIFIED):**
  - I. M. Verloop, "Asymptotically optimal priority policies for indexable and nonindexable restless bandits," *Ann. Appl. Probab.* 2016.
  - G. Zayas-Cabán, S. Jasin, G. Wang, "An asymptotically optimal heuristic for general nonstationary finite-horizon restless multi-armed, multi-action bandits," *Adv. Appl. Probab.* 2019.
  - D. B. Brown, J. Zhang, "Dynamic Programs with Shared Resources and Signals: Dynamic Fluid Policies and Asymptotic Optimality," *Operations Research* 2022. Its exogenous shared signals correspond to our channel reports, so it is **high priority to verify**.
  - N. Gast, B. Gaujal, C. Yan, "Exponential asymptotic optimality of Whittle index policy," *Queueing Systems* 2023.
  - A. Ghosh, D. Nagaraj, M. Jain, M. Tambe, "Indexability is Not Enough for Whittle," AAMAS 2023.
- How Kadota et al.'s lower bound is built, and whether it is a Lagrangian bound, was not visible.

## 6. The five closest works in this lineage, and an overlap verdict per contribution

### Takeaway
No retrieved work combines all of the following:
- a finite horizon with a terminal quality reward;
- an MMSE-of-received-SNRs arm state for a static source;
- Rayleigh fading with (stale) per-slot reports;
- B units per slot;
- a Lagrangian bound used together with Whittle and Lagrangian indices.

Every single ingredient has a precedent. Among the works found in *this* lineage (excluding SURVEY F2/F5 items such as Yu–Xu–Tong, Hosseini–Panwar, Hu–Frazier and Brown–Smith, which remain the closest overall), the five closest are:
1. Le Ny, Feron and Dahleh (2011);
2. Tang, Wang, Song and Song (2020);
3. Ornee and Sun (2023);
4. Chen and Ephremides (2021), with Kriouile and Assaad (2021) as companion;
5. Ceran, Gündüz and György (2021).

### Cited Findings
1. **Le Ny, Feron, Dahleh, *IEEE TAC* 56(6), 2011.** Kalman-covariance arms, M sensors, a decomposable relaxation bound approached by constructed policies, and an analytical scalar Whittle index with guaranteed indexability. — [arXiv 0810.5148](https://arxiv.org/abs/0810.5148)
2. **H. Tang, J. Wang, L. Song, J. Song, arXiv 1912.05947 (Allerton 2019 in part).** Markov multi-state channels known to the scheduler, a hard bandwidth constraint relaxed into a single-sensor CMDP with threshold structure solved by LP, and an asymptotically optimal truncated policy. — [arXiv](https://arxiv.org/abs/1912.05947v3)
3. **Ornee, Sun, ACM MobiHoc 2023 (arXiv 2305.04809).** Weighted time-average MSE of Gauss–Markov sources, parallel channels, exact Whittle index, and the index ↔ threshold correspondence. — [arXiv](https://arxiv.org/html/2305.04809v4)
4. **Chen, Ephremides, *Entropy* 23:1572, 2021**, with **Kriouile, Assaad, ISIT 2021.** Imperfect CSI, several users per slot, Whittle when indexable and a priority index otherwise. Companion: a belief-state POMDP with a Lagrangian threshold dual and closed-form Whittle indices. — [MDPI](https://www.mdpi.com/1099-4300/23/12/1572); [arXiv 2102.03245](https://arxiv.org/pdf/2102.03245)
5. **Ceran, Gündüz, György, *IEEE JSAC* 39(5), 2021.** HARQ accumulation over retransmissions on unreliable channels, a transmission-budget CMDP solved exactly with known statistics, and reinforcement learning without them. — [arXiv 2102.09774](https://arxiv.org/pdf/2102.09774)

Runners-up:
- Dance and Silander, JMLR 2019 (posterior-variance threshold and Whittle closed forms) — [JMLR](https://jmlr.org/papers/v20/17-185.html)
- Niño-Mora and Villar, CDC 2009 (Kalman marginal-productivity index, fixed radars per target) — [UC3M](https://e-archivo.uc3m.es/entities/publication/2bf8059f-5424-46b3-9a48-11a316c2846e)
- Maatouk et al., TWC 2021 and Kriouile et al., TIT 2022 (M of N asymptotics) — [arXiv 2001.03096](https://arxiv.org/pdf/2001.03096)
- Zou et al., MobiHoc 2021 (partial index for heterogeneous units) — [Purdue PDF](https://engineering.purdue.edu/~linx/paper/mobihoc21-partial-index.pdf)
- Molin and Hirche, TAC 2014 (price-based thresholds) — [mediaTUM](https://mediatum.ub.tum.de/1285321)
- Molin, Esen and Johansson, Automatica 2019 (value-of-information priority) — [PUMA](https://puma.ub.uni-stuttgart.de/bibtex/1694dafbc846f5aa65e85853470e033bd)
- Tripathi and Modiano 2019 (estimation error ↔ age penalties) — [arXiv](https://arxiv.org/abs/1908.10438)
- QAoI, TCOM 2022 (evaluation at query instants) — [DeepAI](https://deepai.org/publication/query-age-of-information-freshness-in-pull-based-communication)

### Inferences
**What each of the five lacks relative to ours:**
1. **Le Ny et al.:**
   - steady state, not a finite horizon or terminal reward;
   - no fading and no channel reports;
   - process noise makes passive arms deteriorate, whereas our arms freeze;
   - continuous time.

   It is the strongest precedent for "error-state arms + decomposable relaxation bound + Whittle index + several units".
2. **Tang et al.:**
   - an AoI reward, not MMSE or PSNR;
   - average cost, not a finite horizon;
   - no stale reports or terminal target.

   It is the strongest precedent for "relax the per-slot constraint → one user's program with a channel state → threshold → truncate to the hard constraint → asymptotic optimality".
3. **Ornee and Sun:**
   - time-average MSE;
   - transmissions are whole samples with random delays, not SNR-dependent partial refinements;
   - no fading reports, no finite horizon, no bound certificate.
4. **Chen and Ephremides / Kriouile and Assaad:**
   - AoII rewards;
   - infinite horizon;
   - no static-source MMSE.

   Closest to our handling of stale reports and to our fallback when indexability is murky.
5. **Ceran et al.:**
   - AoI reward with decode-or-not success, closest to C8 rather than C2;
   - no per-slot CSI;
   - one user per slot;
   - infinite horizon.

**Overlap verdict per contribution:**

| Our contribution | Overlap in this lineage | Verdict and what to cite |
|---|---|---|
| C1 Finite-horizon RMAB with terminal (deadline) reward | None in the AoI, VoI or estimation records retrieved; QAoI only conceptually | **LOW here.** The overlap is HIGH with deadline-RMAB theory already in SURVEY F2/F5 (Yu–Xu–Tong; Hu–Frazier; Brown–Smith). Claim the formulation and application, not the theory. |
| C2 MMSE link abstraction (n, Y) as the arm state | Error-state arms: Le Ny; Niño-Mora and Villar; Dance and Silander; Ornee and Sun. Error ↔ age: Tripathi and Modiano. | **MEDIUM.** The vocabulary is shared, the structure is not: static source, distinct importance-ordered phases, SNR-set and frozen per-phase MMSE, a learned code's calibrated sensitivities, PSNR reward. Cite the Kalman RMABs and state the difference. |
| C3 Fading with (stale) reports inside the arm program | CSI-aware AoI (Talak; Tang et al.); imperfect-CSI AoII (Chen and Ephremides); belief-state AoII (Kriouile and Assaad); opportunistic control (Gatsis) | **MEDIUM.** The report kernel inside a finite-horizon arm DP, *with the bound still valid for report-based schedulers*, was not found. That is the defensible part. |
| C4 B units per slot | Maatouk; Kriouile; Zou (heterogeneous); Ornee and Sun; Le Ny; Niño-Mora and Villar; Gatsis | **HIGH.** Standard; make no claim. Note Zou's partial index in case units become frequency-selective. |
| C5 Lagrangian relaxation → per-user DP → upper bound as certificate | Le Ny (decomposable relaxation bound); Tang (relaxed CMDP); Kriouile and Assaad (Lagrangian threshold dual); Molin and Hirche and Gatsis (prices or duals); generic theory in F5 | **MEDIUM–HIGH.** The method is established. Claim only the measured bound gap on this problem, and the validity of the bound under stale reports. |
| C6 Whittle index + Lagrangian index at per-slot prices | Whittle everywhere above; truncated Lagrangian (Tang); dual-cost iteration (Zou); priority fallback (Chen and Ephremides); F5 theory | **HIGH (method).** No claim beyond the application. The "top of the indifference price range" definition is a detail, akin to Chen and Ephremides' fallback. |
| C7 Closed-form certainty-equivalent marginal-gain index (this slot's gain − a typical slot's) | Myopic marginal-gain rules: Max-Weight (Kadota), VoI priority (Molin et al. 2019), price-adapted thresholds (Molin and Hirche) | **LOW–MEDIUM.** The opportunity-cost term relative to a typical future slot, motivated by the deadline and the fading, was not seen in the retrieved abstracts. A modest claim is possible after checking Talak's CSI age-based rule and the VoI papers in full. |
| C8 All-or-nothing PSNR-target reward | Binary successes: HARQ decode (Ceran); AoII correct or incorrect (Kriouile and Assaad) | **LOW–MEDIUM.** No finite-horizon target reward was found in this lineage. |
| C9 Continuous traffic, a Whittle index per request with its own deadline | Arrivals without deadlines: Hsu et al.; Kadota and Modiano | **LOW here.** HIGH versus Yu–Xu–Tong (SURVEY F2), which is already cited. |
| (Learning) | Ceran et al.; Leong et al.; NeurWIN | **N/A.** Our model is known. Position learned indices as complementary, for model mismatch. |

**Bottom line for a TWC-style related-work paragraph:**
- Our scheduler is an estimation-error restless bandit in the Le Ny, Niño-Mora and Villar, and Ornee and Sun tradition, solved with the finite-horizon Lagrangian machinery of F5 (with Tang et al. as the wireless precedent for relax-then-truncate over channel states).
- It is distinguished by three things: (i) a static, progressively refined source whose per-slot gain is set by fading SNR through a learned code's MMSE model; (ii) a terminal deadline reward; (iii) stale reports inside each arm's program, with the bound kept valid.
- The novelty is the combination and the application, not the index method.

### Gaps
- The verdicts rest on abstracts and metadata only. Before submission, read the full texts of the five closest works to confirm three things:
  - that no finite-horizon or terminal variants are hidden in extensions (journal versions often add these);
  - Le Ny et al.'s bound construction;
  - Tang et al.'s channel-threshold structure and truncation.
- These UNVERIFIED candidates could change the verdict, so check them first:
  - Castañón (CDC 1997): finite-horizon, terminal cost, Lagrangian decoupling of sensor constraints. It could raise C1 and C5 to HIGH.
  - Shisher, Ji, Hou and Sun (JSAIT 2023): feature-length selection, which could raise C2 and C3.
  - Xiong, Li and Singh (AAAI 2022): a learned finite-horizon index.
  - Brown and Zhang (Operations Research 2022): shared signals, which could raise C3.
  - Lu, Ji and Li (MobiHoc 2018): AoI with deadlines.
