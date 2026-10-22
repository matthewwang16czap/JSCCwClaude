# OR theory for coupled dynamic programs: weakly coupled MDPs, Lagrangian/LP relaxations, finite-horizon restless bandits, bid prices and certainty-equivalent heuristics (methodological prior work for deadline scheduling of learned image codes)

> **Verification status (read first).** This session ran 10 web searches before the shared per-turn search budget ran out. WebFetch failed with DNS errors for every host tried (arXiv, INFORMS, HAL, Dartmouth, doi.org), and direct downloads were refused by the egress policy (403). As a result, **no paper was read in full in this session.**
>
> Tags used below:
> * **[V]**: existence, bibliographic data and abstract-level claims verified in this session from search results. Anything beyond the abstract, such as theorem constants or exact definitions, is NOT verified.
> * **[S]**: taken from the project's own `jscc/docs/SURVEY.md` (sections F2/F5, checked 2026-10-13/18). Not re-verified.
> * **[U]**: from the researcher's background knowledge only, NOT verified in this session. [U] items appear only in the Gaps subsections, with a citation-confidence note, and must be checked before citing.
>
> A follow-up turn with a fresh search budget could close the [U] items.
>
> **Notation for our model.** K users (arms); T slots; B units per slot, each lost independently with probability p, so the realised supply is B_t ~ Bin(B, 1-p). This supply is common to all users. Each user takes a binary action a_{k,t} per slot, subject to sum_k a_{k,t} <= B_t. The user state is x = (n, Y) plus the slot SNR g (or a stale report of g). The reward is terminal, f_k(x_{k,T}): either PSNR at the deadline, or 1{PSNR >= target}. The relaxation prices are lambda_t. W_k(lambda) is user k's priced program value. The per-user numbers quoted in the Inferences come from `jscc/docs/PROBLEM.md`, section 3, item 6b.

## 1. Weakly coupled MDPs and their relaxations: what is proven and which policy is recommended

### Takeaway
Component (a) is the standard Lagrangian relaxation (decomposition) of a weakly coupled stochastic DP with time-indexed multipliers (Hawkins 2003; Adelman and Mersereau 2008; Brown and Smith 2020). Two recent results bear directly on it:
* **The bound is already as tight as an ALP.** With binary actions and unimodular linking constraints, the ALP bound equals the Lagrangian bound (Brown and Zhang, technical note). Our per-slot cardinality constraint satisfies both conditions.
* **A shared random budget can loosen the bound.** When an exogenous signal shared by all arms drives the problem, the multipliers must depend on the signal history to be asymptotically tight (Brown and Zhang 2022). Our lost units are exactly such a shared signal.

Every policy this literature recommends follows the same pattern: decompose, then recombine with a priority rule built from the subproblem values. Examples are Hawkins's per-period integer program, Bertsimas and Niño-Mora's primal-dual index, Brown and Smith's Lagrangian index with careful tie-breaking, and Brown and Zhang's fluid, re-optimised and budget-balancing policies. None of these works recommends a myopic closed-form rule.

### Cited Findings
- **Hawkins (2003) [V].** J. T. Hawkins, *A Langrangian decomposition approach to weakly coupled dynamic optimization problems and its applications*, Ph.D. thesis, Operations Research Center, MIT, 2003. The misspelling "Langrangian" is in the catalogue record itself.
  - Advisor: D. Bertsimas. The Mathematics Genealogy Project also lists G. Perakis.
  - Abstract-level content:
    - a Lagrangian approach that decouples weakly coupled dynamic optimisation problems, for both finite and infinite horizons;
    - bounds on the optimal value from low-dimensional DPs;
    - integer programming used to combine the subproblem solutions into feasible policies;
    - applications to multi-armed and restless bandits, queueing networks, serial supply chains, linear control and on-line auctions.
  - Corresponds to our (a), and to (c) as the "combine" step.
  - Sources: [DSpace@MIT record](https://dspace.mit.edu/handle/1721.1/29599); [Mathematics Genealogy](https://www.mathgenealogy.com/id.php?id=99685)
- **Adelman and Mersereau (2008) [V].** D. Adelman and A. J. Mersereau, "Relaxations of weakly coupled stochastic dynamic programs," *Operations Research* 56(3):712–727, 2008, DOI 10.1287/opre.1070.0445.
  - Covers a broad class of stochastic DPs that can be relaxed by decomposition. Keywords: approximate DP, Lagrangian optimization, linear programming, column generation.
  - It is the standard comparison of the Lagrangian relaxation with the approximate-LP (ALP) approach. Its own theorem statements could not be retrieved here; see the Brown and Zhang note below for the comparison.
  - Corresponds to (a).
  - Sources: [INFORMS](https://pubsonline.informs.org/doi/10.1287/opre.1070.0445); [Adelman's page](https://faculty.chicagobooth.edu/daniel-adelman/publications-and-research/relaxations-of-weakly-coupled-stochastic-dynamic-programs)
- **Brown and Zhang, technical note [V].** D. B. Brown and J. Zhang, "Technical Note—On the strength of relaxations of weakly coupled stochastic dynamic programs," *Operations Research*, DOI 10.1287/opre.2022.2287; preprint SSRN 3849573, 2021.
  - Per the preprint abstract:
    - the ALP gives a tighter upper bound than the Lagrangian relaxation (in the initial state), but the two are often close;
    - the gap between them is controlled by the integrality gap of the ALP's separation problems;
    - the gap is **zero when the subproblem actions are binary and the linking constraints are unimodular.**
  - Corresponds to (a): it settles how tight our bound is.
  - Sources: [SSRN](https://doi.org/10.2139/ssrn.3849573); [INFORMS reference page](https://pubsonline.informs.org/doi/references/10.1287/opre.2022.2287); [Semantic Scholar](https://www.semanticscholar.org/paper/On-the-Strength-of-Relaxations-of-Weakly-Coupled-Brown-Zhang/3e436e7641b87ee657ef6062e66c16f578a5591a)
- **Bertsimas and Niño-Mora (2000) [V].** D. Bertsimas and J. Niño-Mora, "Restless bandits, linear programming relaxations, and a primal-dual index heuristic," *Operations Research* 48(1):80–90, 2000, DOI 10.1287/opre.48.1.80.12444. Earlier version: UPF Economics working paper 301, 1994, revised 1997.
  - A hierarchy of increasingly strong LP relaxations of the PSPACE-hard restless bandit. The last one is exact (the exponential-size MDP LP); the lower-order ones are efficiently solvable.
  - From the first-order relaxation they build a priority-index policy whose indices are defined from the optimal dual variables (the "primal-dual" heuristic). This gives a policy together with a suboptimality guarantee (the bound certifies it).
  - Computationally near-optimal; the second-order relaxation gives strong bounds.
  - Corresponds to (c): the dual-variable index of the first-order LP is the LP form of our Lagrangian index. Also to (a): the first-order relaxation.
  - Sources: [IDEAS/RePEc](https://ideas.repec.org/a/inm/oropre/v48y2000i1p80-90.html); [UPF WP 301](https://econ-papers.upf.edu/en/onepaper.php?id=301); [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/opre.48.1.80.12444)
- **Gocgun and Ghate (2012) [V; authors not shown in the snippets].** "Lagrangian relaxation and constraint generation for allocation and advanced scheduling," *Computers & Operations Research* 39(10):2323–2336, October 2012.
  - Approximate DP for scheduling renewable resources to randomly arriving jobs. Two MDP classes ("allocation" and "advanced" scheduling) are shown to be weakly coupled.
  - The Lagrangian relaxation is solved by constraint generation.
  - Lagrangian decisions beat myopic decisions by a statistically significant margin, by more for advanced scheduling.
  - Corresponds to (a): an exact LP alternative to subgradient. Also to (d): evidence against myopic rules.
  - Sources: [UMN Experts](https://experts.umn.edu/en/publications/lagrangian-relaxation-and-constraint-generation-for-allocation-an/); [IAOR](https://iaorifors.com/paper/75485); [Medipol listing](https://avesis.medipol.edu.tr/yayin/e7b619d5-ee51-4b4e-a291-1967aba48e31/lagrangian-relaxation-and-constraint-generation-for-allocation-and-advanced-scheduling)
- **Brown and Smith (2020) [V].** D. B. Brown and J. E. Smith, "Index policies and performance bounds for dynamic selection problems," *Management Science* 66(7):3029–3050, 2020, DOI 10.1287/mnsc.2019.3342. Free version on Optimization Online (2017/2018).
  - Problem: a decision maker repeatedly selects a subset of items, with a cap on how many can be selected.
  - Index policies and upper bounds are built from the Lagrangian relaxation that drops the cap.
  - The Lagrangian index policy and the bound are asymptotically optimal with many items, under mild conditions.
  - **Tie-breaking and mixed policies matter a great deal for performance.**
  - Applications: dynamic assortment with demand learning, and applicant screening. Keywords include restless bandits, Lagrangian relaxations, Gittins index and Whittle index.
  - Corresponds to (a), (b) and (c): the closest single reference for our pipeline.
  - Sources: [INFORMS PDF](https://pubsonline.informs.org/doi/pdf/10.1287/mnsc.2019.3342); [Optimization Online](https://optimization-online.org/?p=14804); [Duke Scholars](https://scholars.duke.edu/publication/1456481)
- **Brown and Zhang (2022) [V].** D. B. Brown and J. Zhang, "Dynamic programs with shared resources and signals: Dynamic fluid policies and asymptotic optimality," *Operations Research* 70(5):3015–3033, 2022, DOI 10.1287/opre.2021.2181.
  - Setting: an exogenous finite-state Markov signal is observed, then a limited shared resource is allocated across projects. This generalises Markovian bandits.
  - A Lagrangian relaxation and a DP formulation of a fluid relaxation (the "dynamic fluid" relaxation) give upper bounds and a feasible policy, computed by an iterative primal-dual algorithm.
  - Under mild conditions, the bound and the policy are asymptotically optimal as the number of projects grows.
  - **Key finding:** the Lagrange multipliers must depend on the history of past signals. Restricting them, for example to depend only on the current signal, gives a gap that grows linearly in the number of projects.
  - Applications: dynamic capital budgeting, and multi-location inventory with limited production capacity and demand correlated through a market state.
  - Corresponds to (a) under lost units (a shared random supply).
  - Sources: [IDEAS](https://ideas.repec.org/a/inm/oropre/v70y2022i5p3015-3033.html); [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/opre.2021.2181); [Zhang's Duke dissertation](https://dukespace.lib.duke.edu/items/7ac31600-b116-4387-af59-6a642a144c5d)
- **Brown and Zhang (2025) [V].** D. B. Brown and J. Zhang, "Fluid policies, reoptimization, and performance guarantees in dynamic resource allocation," *Operations Research* 73(2):1029–1045, 2025, DOI 10.1287/opre.2022.0601. Three results, with N the number of subproblems:
  - **Fluid policies:** easy-to-implement feasible fluid policies are within O(sqrt N) of optimal. This holds for heterogeneous subproblems and several shared resource constraints.
  - **Re-optimisation:** a policy acting on a re-optimised fluid value function is also within O(sqrt N). The authors say this is the first guarantee for re-optimisation in this general setting.
  - **Fluid-budget balancing:** these policies stay close to the fluid-optimal actions while using as much of the shared resources as possible. They are within **O(1)** under particular non-degeneracy assumptions.
  - The motivating example is the restless bandit with a budget constraint.
  - Corresponds to (c), and to our rule that every free unit is used.
  - Sources: [Duke Scholars](https://scholars.duke.edu/publication/1559343); [IDEAS](https://ideas.repec.org/a/inm/oropre/v73y2025i2p1029-1045.html); [INFORMS](https://pubsonline.informs.org/doi/fpi/10.1287/opre.2022.0601)
- **Information relaxations for weakly coupled DPs [V, abstract level].**
  - Ye, Zhu and Zhou, "Weakly coupled dynamic program: Information and Lagrangian relaxations," arXiv 1405.3363; IEEE Trans. Automatic Control 2018, per the search record. Adding an information relaxation with a penalty to the Lagrangian relaxation gives a tighter dual bound. Sources: [arXiv](https://arxiv.org/html/1405.3363v1); [Semantic Scholar](https://www.semanticscholar.org/paper/Weakly-Coupled-Dynamic-Program:-Information-and-Ye-Zhu/a397a09dbd04809e6a002e4961809d6ba0bba20b)
  - Brown, Smith and Sun, "Information relaxations and duality in stochastic dynamic programs," *Operations Research*, DOI 10.1287/opre.1090.0796. Only the title and DOI are verified. Source: [INFORMS](https://pubsonline.informs.org/doi/10.1287/opre.1090.0796)
  - Corresponds to (f): the clairvoyant optimum is the zero-penalty information relaxation.

### Inferences
- **(a) in standard form.**
  - Put multipliers lambda_t >= 0 on the per-slot linking constraints. For any causal policy, E[sum_k f_k] <= E[sum_k f_k] + sum_t lambda_t E[B_t - sum_k a_{k,t}] <= (1-p) B sum_t lambda_t + sum_k W_k(lambda).
  - L(lambda) is a pointwise maximum of affine functions of lambda, so it is convex and piecewise linear.
  - A subgradient in lambda_t is (1-p)B - sum_k Pr{user k takes slot t under its lambda-optimal policy}. Our subgradient steps are therefore textbook Lagrangian-dual descent.
  - A discretised arm is a finite MDP, so by LP duality min_lambda L(lambda) equals the first-order (occupation-measure, or "fluid") LP value. It can be computed exactly by LP or by column/constraint generation (Adelman and Mersereau keywords; Gocgun and Ghate). This removes price-grid or step-size error from the bound and gives the LP-index policy of Gast et al. directly (section 2).
- **Tightness.**
  - Our actions are binary, and the per-slot row sum_k a_{k,t} <= B_t is a single all-ones cardinality row, which is totally unimodular. So, by the Brown and Zhang technical note, the Adelman–Mersereau ALP bound coincides with our Lagrangian bound: no first-order relaxation is tighter.
  - A tighter causal bound must come from one of three routes:
    - multipliers that depend on the shared-signal history (Brown and Zhang 2022);
    - information relaxations with Lagrangian-based penalties (Ye, Zhu and Zhou; Brown and Smith);
    - higher-order LP relaxations over pairs of users (Bertsimas and Niño-Mora).
- **Lost units are a shared exogenous signal.**
  - With B = 1 (the K = 8 frame) a lost slot is lost for every user. After a run of lost slots all users are behind together, so later slots are worth more. Time-only prices lambda_t cannot express this correlation.
  - Brown and Zhang (2022) show that history-blind multipliers can leave a gap linear in N, in their Markov-signal setting.
  - In our scaling runs, losses are i.i.d. per unit and B = K/8 grows. The available fraction then concentrates (relative s.d. sqrt(p/((1-p)B))), so the effect should vanish as K grows.
  - At K = 8 with lost slots, however, part of the bound-to-policy gap may be looseness of the bound rather than loss of the policy.
  - Check: recompute the bound with prices conditioned on the number of slots lost so far, or with the dynamic fluid relaxation.
- **Which policy the literature recommends, mapped onto ours.**
  - Hawkins's per-period integer program over subproblem values: with one cardinality constraint per slot it is solved by sorting the gains, so our Lagrangian index policy (c) is Hawkins's combine step for this coupling.
  - Bertsimas and Niño-Mora's dual-variable (reduced-cost) index: the same object in LP form.
  - Brown and Smith: the Lagrangian index, with the tie-breaking stressed.
  - Brown and Zhang (2025): fluid, re-optimised, and budget-balancing policies. Our "every free unit is used" fill rule is budget balancing in spirit.
  - Gocgun and Ghate measure the loss of myopic rules against Lagrangian ones.
- **Stale CSI.** The relaxation remains a bound on report-based schedulers because each user's program may condition on more information (its own true Y) than any transmitter has. This is an information-relaxation argument in the sense of Brown, Smith and Sun.

### Gaps
- **Not read in full** (access blocked):
  - Adelman and Mersereau's own ALP-versus-Lagrangian theorems and their recommended policy;
  - Brown and Smith's exact definition of the Lagrangian index, their tie-breaking and mixing rule, any explicit rate, and whether they add information-relaxation bounds;
  - Hawkins's chapters.
- **Unconfirmed details:**
  - the Gocgun and Ghate authorship (Y. Gocgun and A. Ghate per the assignment) is not printed in the retrieved snippets;
  - the publication year of the Brown and Zhang technical note (2022 or 2023) is not confirmed;
  - the Ye, Zhu and Zhou first names and the TAC volume are not confirmed.
- **[U] Bertsimas and Mišić**, "Decomposable Markov decision processes: A fluid optimization approach," *Operations Research* 64(6), 2016 (citation confidence high). Believed to show that a fluid LP formulation, re-solved at each epoch, bounds the optimum more tightly than the ALP and two Lagrangian relaxations.
- **[U] Meuleau, Hauskrecht, Kim, Peshkin, Kaelbling, Dean and Boutilier**, "Solving very large weakly coupled Markov decision processes," AAAI 1998 (confidence high). Believed to coin the term "weakly coupled MDP", to solve per-task MDPs as functions of the resource, and to allocate resources online greedily by marginal value, which is close to our quotas.
- **[U] Brown, Smith and Sun, *Operations Research* 58(4):785–801, 2010** (confidence high). Believed content: the zero-penalty perfect-information bound, and penalties that tighten it, tight with the ideal penalty.
- **[U] Standard references for subgradient optimisation of Lagrangian duals** (confidence high): Fisher, "The Lagrangian relaxation method for solving integer programming problems," *Management Science* 27(1), 1981; Held, Wolfe and Crowder, "Validation of subgradient optimization," *Mathematical Programming* 6, 1974.

## 2. Finite-horizon restless bandits: what is proven, at which rate per arm in K, under which conditions

### Takeaway
In a finite horizon, index policies derived from the Lagrangian or LP relaxation at the optimal multipliers are asymptotically optimal. They need neither indexability nor any attractor condition. Measured as the gap to the bound per arm, against the number of arms K (with a fixed share pulled per period), the rates are:

| condition | per-arm gap | source |
|---|---|---|
| all models | O(1/sqrt K) | Gast, Gaujal and Yan |
| non-degenerate | O(1/K), i.e. O(1) in total, for "fluid-priority" policies, which include most index policies | Zhang and Frazier; Brown and Zhang 2025 (fluid-budget balancing) |
| non-degenerate | exponentially small, for the LP-index policy | Gast, Gaujal and Yan |

Re-solving the LP (LP-update) is more robust when the model is misestimated. Our Lagrangian index's measured gap fits the non-degenerate regime: 0.207 dB per user at K = 8 and 0.005 dB at K = 128, i.e. about O(1) in total. The 30 dB target objective does not fit: its constant 1.5-point gap needs explaining.

### Cited Findings
- **Hu and Frazier [V].** W. Hu and P. I. Frazier, "An asymptotically optimal index policy for finite-horizon restless bandits," arXiv:1707.00205, July 2017.
  - Finite horizon, several pulls per period.
  - A Lagrangian relaxation splits the problem into single-arm problems, and an index policy is built from their solutions.
  - Proven asymptotically optimal as the number of arms grows; the abstract states no rate.
  - In simulations it beats state-of-the-art heuristics.
  - Corresponds to (c).
  - Sources: [arXiv](https://arxiv.org/abs/1707.00205v1); [ar5iv](https://ar5iv.labs.arxiv.org/html/1707.00205)
- **Xiong et al., AAAI 2022 [V, title only].** "Reinforcement learning augmented asymptotically optimal index policy for finite-horizon restless bandits." The record indicates a multi-action / learning extension of the Hu and Frazier policy. This is the learned-index route, in case the model is unknown. Source: [mlanthology](https://mlanthology.org/aaai/2022/xiong2022aaai-reinforcement)
- **Zhang and Frazier [V].** X. Zhang and P. I. Frazier, "Restless bandits with many arms: Beating the central limit theorem," arXiv:2107.11911, 25 July 2021.
  - Finite horizon. Intuition from the central limit theorem, and the tightest earlier bounds, suggest an O(sqrt N) gap.
  - Under a **non-degeneracy condition**, a wide class of practically computable **fluid-priority policies**, including most widely used index policies, has an **O(1) gap**.
  - Without non-degeneracy, fluid-priority policies still have O(sqrt N).
  - Corresponds to (b) and (c), and explains our measured rate.
  - Sources: [arXiv](https://arxiv.org/abs/2107.11911v1); [DeepAI mirror](https://deepai.org/publication/restless-bandits-with-many-arms-beating-the-central-limit-theorem)
  - The infinite-horizon companion is "Near-optimality for infinite-horizon restless bandits with many arms," arXiv:2203.15853 (title verified only). Source: [arXiv](https://arxiv.org/pdf/2203.15853)
- **Gast, Gaujal and Yan [V].** N. Gast, B. Gaujal and C. Yan, "Linear program-based policies for restless bandits: Necessary and sufficient conditions for (exponentially fast) asymptotic optimality," *Mathematics of Operations Research* 49(4):2468–2491, 2024, DOI 10.1287/moor.2022.0101; arXiv:2106.10067, June 2021.
  - As N goes to infinity, the optimal value converges to the value of an LP.
  - They give necessary and sufficient conditions for a policy to be (i) asymptotically optimal, (ii) optimal at the square-root rate, and (iii) optimal at an exponential rate.
  - The **LP-index policy** has the square-root rate on all models. It has an **exponential rate if the model is non-degenerate (finite horizon)**, or satisfies a uniform global attractor property (infinite horizon).
  - The **LP-update** policy re-solves the LP at every epoch. It usually beats LP-index, and can do significantly better when the transition matrices are misestimated.
  - Corresponds to (a) and (c), and to stale reports.
  - Sources: [IDEAS](https://ideas.repec.org/a/inm/ormoor/v49y2024i4p2468-2491.html); [HAL](https://hal.umontpellier.fr/LIG-SRCPR-POLARIS/hal-03262307v4); [arXiv](https://arxiv.org/abs/2106.10067v3)
- **Brown and Zhang (2025) [V]** and **Brown and Smith (2020) [V]**: see section 1. Fluid and re-optimised policies achieve O(sqrt N) in total; fluid-budget balancing achieves O(1) under non-degeneracy. Lagrangian index policies are asymptotically optimal, and ties and mixing matter.
  - Sources: [Brown and Zhang 2025](https://scholars.duke.edu/publication/1559343); [Brown and Smith 2020](https://optimization-online.org/?p=14804)
- **Average-reward line [S]**, already in SURVEY F5:
  - Hong, Xie, Chen and Wang, *Math. Oper. Res.* 2025: the ID policy is O(tau^4/sqrt N) under unichain and aperiodicity alone.
  - Random tie-breaking fails.
  - Index policies need a global attractor.
  - Hong et al., NeurIPS 2023: Follow-the-Virtual-Advice (FTVA) policy, O(1/sqrt N) under a synchronisation assumption.
  - Zhang, Hong and Wang, NeurIPS 2025: heterogeneous weakly coupled MDPs.
  - Sources: [SURVEY.md F5](/home/user/JSCCwClaude/jscc/docs/SURVEY.md); [arXiv 2402.05689](https://arxiv.org/abs/2402.05689); [arXiv 2306.00196](https://arxiv.org/abs/2306.00196); [arXiv 2502.06072](https://arxiv.org/abs/2502.06072); [CMU blog](https://www.cs.cmu.edu/~csd-phd-blog/2026/restless-bandits/)
- **Deadline RMAB [S].** Yu, Xu and Tong, "Deadline scheduling as restless bandits," IEEE TAC 2018: indexable, a closed-form Whittle index for constant costs, and a gap bound that vanishes with scale. Sources: [SURVEY.md F2](/home/user/JSCCwClaude/jscc/docs/SURVEY.md); [arXiv 1610.00399](https://arxiv.org/abs/1610.00399)
- **Review [V title; S].** Niño-Mora, "Markovian restless bandits and index policies: A review" (*Mathematics* 11:1639, 2023 per SURVEY). Source: [arXiv 2601.13045](https://arxiv.org/pdf/2601.13045)

### Inferences
- **Per-user rates for our Lagrangian index**, from the measured gap per user at K = 8/16/32/64/128 (0.207/0.090/0.039/0.009/0.005 dB, `jscc/docs/PROBLEM.md` 3.6b):

  | K | gap per user (dB) | K × gap, i.e. total gap (dB) | sqrt K × gap |
  |---|---|---|---|
  | 8 | 0.207 | 1.66 | 0.59 |
  | 16 | 0.090 | 1.44 | 0.36 |
  | 32 | 0.039 | 1.25 | 0.22 |
  | 64 | 0.009 | 0.58 | 0.07 |
  | 128 | 0.005 | 0.64 | 0.06 |

  - The overall log-log slope is -1.34, since 0.207/0.005 = 41 over a 16x increase in K.
  - Since sqrt K × gap falls, the rate is faster than 1/sqrt K. The total gap is bounded and falling, which is the **O(1)-total non-degenerate regime** (Zhang and Frazier; Brown and Zhang 2025). The exponential claim for LP-index (Gast et al.) can be neither confirmed nor excluded: the flattening between K = 64 and 128 looks like a numerical floor (grid and price discretisation of the bound, Monte Carlo CI).
  - Better tests: plot K × gap with CIs, compute the bound exactly by LP on the discretised arms, and run LP-update.
- **The Whittle policy (b)** goes from 0.265 to 0.016 dB per user, i.e. a total of 2.12 and 2.05 dB: also O(1) in total. This fits Zhang and Frazier's statement that their fluid-priority class contains most widely used index policies.
- **Our closed form (d)** goes from 0.267 to 0.155 dB per user, i.e. a total of 2.1 and 19.8 dB: the total grows about linearly in K. PF behaves the same way (0.71 to 0.52 per user). Neither is asymptotically optimal, which is what the theory predicts for policies that are not priority rules of the relaxation (section 4).
- **The constant gap at the 30 dB target is inconsistent with the theory as stated.** At that target the Lagrangian index stays about 1.5 points from the bound at every K. For a finite-state model, the optimal value converges to the LP value, and the LP-index loses only O(1/sqrt K) per arm on all models (Gast et al.). A per-user gap that does not shrink therefore signals a mismatch, for example:
  - the computed bound is not the LP value of the model the policy uses: grid interpolation of a 0/1 terminal reward can make the programs optimistic, or the prices may be unconverged;
  - the bound's arm model differs from the simulator, for instance in how losses are modelled;
  - the target objective has many exact ties at lambda* (flat value regions), and the tie-break does not reproduce the LP's fractional allocations.

  This is worth checking before claiming a target-objective result.
- **What our model satisfies, and what remains to check.** Our state is continuous ((n, Y) and the SNR) but is discretised on grids (ny points in Y, ng fading quantiles), so the theorems apply to the discretised finite MDP. Heterogeneous users (images, mean SNRs) are covered by Brown and Zhang (2025); images drawn with replacement from a fixed pool also fit the usual "fixed arm-type mix" scaling.
  - Non-degeneracy means, roughly, that the LP solution randomises in a way that lets the priority rule absorb O(sqrt K) fluctuations (the exact definitions are [U]; see Gaps). With continuous fading, exact indifference at lambda* has probability near zero, which favours non-degeneracy.
  - The indifference regions, by contrast, are degenerate configurations: a stale report with r = 0, no deadline pressure yet, or 0/1 rewards far from the target. In these regions many users have zero gain at lambda*, tie-breaking decides, and the rate may revert to O(1/sqrt K). Our "top of the indifference range" convention is a tie-breaking device for exactly this case.
- **Finite versus infinite horizon.** The average-reward guarantees (Hong et al.; and [U] Weber and Weiss, Verloop) need attractor or unichain conditions; the finite-horizon guarantees (Hu and Frazier, Zhang and Frazier, Gast et al. finite horizon, Brown and Smith, Brown and Zhang) need none. Our frame is finite-horizon, so the second set applies.
- **Indexability** is needed only for (b). (c) needs none (Hu and Frazier; Gast et al.; Brown and Smith).
- **Stale reports.** If, as the `alloc/whittle.py` docstring states, the Whittle index is the indifference price under one constant price for all slots, while (c) uses the optimal time-varying lambda*_t, then (b) prices the future inconsistently with the optimal multipliers. That is a plausible reason for (b) trailing (c) at every K (0.016 vs 0.005 dB at K = 128), and for (b) degrading on stale reports. Gast et al.'s LP-update finding suggests a remedy: re-solve lambda*_t during the frame when the model, or the report kernel, is uncertain.

### Gaps
- **Not read in full:**
  - the exact non-degeneracy definitions of Zhang and Frazier and of Gast et al., and whether they hold for a discretised, continuous-SNR arm;
  - Hu and Frazier's exact index definition and tie-breaking, and any rate;
  - whether Gast et al.'s exponential rate is for the per-arm normalised gap, and what initial-state assumptions it requires.
- **[U] Weber and Weiss**, "On an index policy for restless bandits," *J. Appl. Probab.* 27(3):637–648, 1990 (confidence high). Believed to prove that the Whittle index policy is asymptotically optimal (average reward, fixed fraction active) when a global attractor exists, with a counterexample otherwise. Listed in SURVEY F2 as an anchor.
- **[U] Verloop**, "Asymptotically optimal priority policies for indexable and nonindexable restless bandits," *Ann. Appl. Probab.* 26(4):1947–1995, 2016 (confidence high). Believed to give a class of LP-derived priority policies, defined without indexability, that are asymptotically optimal under a global-attractor assumption (average reward). It names the "LP-priority" idea.
- **[U] Ghosh, Nagaraj, Jain and Tambe**, "Indexability is not enough for Whittle: Improved, near-optimal algorithms for restless bandits," AAMAS 2023 (title confidence high; arXiv id not verified). Believed to show that Whittle index policies can be far from optimal even for indexable arms, and to propose LP / mean-field-based planning with near-optimality guarantees as N grows. Whether and how they treat the finite horizon is unverified.
- **[U] Gast, Gaujal and Yan**, "Exponential asymptotic optimality of Whittle index policy," *Queueing Systems*, 2023 (confidence medium-high). The arXiv id 2211.01961 appeared in a search-result link list without a title, so the match is unverified.
- **[U] Niño-Mora**, "Restless bandits, partial conservation laws and indexability," *Adv. Appl. Probab.* 33(1), 2001 (confidence high): PCL-indexability, the standard route to proving indexability.
- **[U] Whittle**, "Restless bandits: activity allocation in a changing world," *J. Appl. Probab.* 25A, 1988 [S].
- **No source defines the index as the top of the indifference range.** The search budget was exhausted before an OR source could be sought that defines the finite-horizon Whittle index this way, i.e. as the largest price at which taking the slot is still optimal. This may be a new detail; Brown and Smith's emphasis on ties and mixing is the nearest anchor.

## 3. Bid-price controls: is lambda*_t a bid price, and is the Lagrangian index a bid-price policy?

### Takeaway
* **lambda*_t is a bid price.** It is the optimal Lagrange multiplier, equivalently the LP dual price, of slot t's perishable capacity in the decomposed relaxation. In revenue-management language it is an open-loop, time-indexed bid price for slot t.
* **The Lagrangian index is not a pure bid-price control.** The "bid" is each user's endogenous, state-dependent gain in its own priced program (the DP-decomposition style of bid pricing, not the deterministic-LP style). The B units are rationed by ranking and always filled, so the policy is a bid-price policy with priority rationing: an LP-priority / Lagrangian index policy.

The revenue-management theorems that would anchor this mapping could not be verified in this session, so they are listed as [U] under Gaps. The dual-price interpretation itself rests on the verified items below.

### Cited Findings
- **Bertsimas and Niño-Mora's priority index** is defined from the optimal dual variables of the first-order LP relaxation, i.e. the duals of the per-period activation constraint are the prices. [IDEAS/RePEc](https://ideas.repec.org/a/inm/oropre/v48y2000i1p80-90.html)
- **Brown and Zhang (2022)** treat the Lagrange multipliers on a shared per-period resource as prices, and show they must depend on the signal history to be asymptotically tight. [IDEAS](https://ideas.repec.org/a/inm/oropre/v70y2022i5p3015-3033.html)
- **Brown and Zhang (2025)**: "fluid-budget balancing" policies stay close to the fluid-optimal actions while using as much of the shared resources as possible, and are O(1) under non-degeneracy. This is the analogue of "rank, then fill all units". [Duke Scholars](https://scholars.duke.edu/publication/1559343)
- **Gast, Gaujal and Yan**: LP-update re-solves the LP at every epoch (re-solved prices) and helps most under model error. [IDEAS](https://ideas.repec.org/a/inm/ormoor/v49y2024i4p2468-2491.html)
- **Hawkins (2003)** lists on-line auctions among the applications of Lagrangian decomposition. [DSpace@MIT](https://dspace.mit.edu/handle/1721.1/29599)

### Inferences
- **lambda*_t as a bid price.**
  - In network revenue management, a bid price is the opportunity cost of one unit of a resource. A request is accepted if its revenue is at least the sum of the bid prices of the resources it uses. Static bid prices are typically the duals of the capacity rows of a deterministic LP ([U] Talluri and van Ryzin).
  - Our lambda*_t is exactly the dual of slot t's capacity row in the first-order (fluid) LP, i.e. the minimiser of the Lagrangian dual. So it is a static (open-loop), time-indexed bid price for slot t.
  - The closest RM analogue of a perishable per-slot capacity is a resource that exists only for one period, e.g. a hotel room-night. The analogy is imperfect: hotel guests book multi-night bundles in advance, whereas our users consume slots sequentially and decide at the time of use.
- **The "bid" is endogenous.**
  - User k's willingness to pay for slot t is d_k = Q_k(t, x_k, take; lambda*) - Q_k(t, x_k, skip; lambda*) + lambda*_t. This is computed in its own priced DP and already nets out the future prices lambda*_{t+1..T}, i.e. the user's own opportunity cost of using up deadline or phases now.
  - This matches the "DP-decomposition" bid prices of RM ([U] Topaloglu 2009; Liu and van Ryzin), where single-resource DPs priced by Lagrange multipliers yield state-dependent marginal values. It does not match the plain deterministic-LP rule "fare at least bid price", where the fare is exogenous.
- **Accepting at the bid price versus ranking.**
  - A pure bid-price control serves user k if d_k >= lambda*_t and can leave units idle. By complementary slackness, the expected number of users with d_k > lambda*_t equals the expected supply (1-p)B when lambda*_t > 0, but realisations fluctuate by O(sqrt K).
  - Our policy serves the top B by d_k. It rations when demand exceeds supply, and fills units with the next-best users (gain below lambda*_t) when demand falls short.
  - The asymptotic RMAB analyses quantify exactly the cost of these fluctuations: O(sqrt K) in total generally, O(1) under non-degeneracy (Zhang and Frazier; Brown and Zhang 2025; Gast et al.).
  - In RM terms, (c) is therefore "bid prices plus priority rationing plus budget balancing", not a bid-price control in the strict accept/reject sense.
- **Static, re-solved, or exact.**
  - Re-solving lambda* during the frame is the RMAB analogue of re-solved bid prices (Gast et al.'s LP-update; [U] the RM re-solving literature).
  - A "dynamic" bid price in the strict sense would be the exact marginal value of slot capacity in the joint DP, which depends on all users' states. lambda*_t is its relaxation-based, state-independent approximation.
  - Our Whittle index is a different object: the price at which a user is indifferent, computed with one constant price for every slot. It is a reservation price under a stationary price path, not a time-varying bid price.

### Gaps
None of the following were verified in this session; all are [U], with citation confidence noted.
- **Talluri and van Ryzin**, "An analysis of bid-price controls for network revenue management," *Management Science* 44(11):1577–1593, 1998 (confidence high). Believed to show that bid-price controls are not optimal in general but that LP-based bid prices are asymptotically optimal under proportional scaling of capacity and demand. Also Talluri and van Ryzin, *The Theory and Practice of Revenue Management* (book, 2004).
- **Adelman**, "Dynamic bid prices in revenue management," *Operations Research* 55(4):647–661, 2007 (confidence high). Believed: an ALP with an affine value-function approximation yields time-varying bid prices and a bound tighter than the deterministic LP.
- **Topaloglu**, "Using Lagrangian relaxation to compute capacity-dependent bid prices in network revenue management," *Operations Research* 57(3):637–649, 2009 (confidence high). Believed: relaxing the constraints that tie acceptance decisions across legs, with time- and product-indexed multipliers, gives single-leg DPs, capacity-dependent and time-varying bid prices, and a bound tighter than the deterministic LP. This is the closest RM analogue of our (a)+(c).
- **Re-solving rates** (confidence medium-high):
  - Cooper, *Operations Research* 50(4), 2002: re-solving can hurt.
  - Reiman and Wang, *Math. Oper. Res.* 33(2), 2008: asymptotically optimal re-solving.
  - Jasin and Kumar, *Math. Oper. Res.* 37(2), 2012: bounded (O(1)) revenue loss with re-solving under non-degeneracy.
  - Bumpensanti and Wang, *Management Science* 66(7), 2020: uniformly bounded loss without non-degeneracy, using infrequent re-solving.
  - Vera and Banerjee, "The Bayesian prophet," *Management Science* 67(3), 2021: O(1) regret against the hindsight (prophet) benchmark.

  These parallel the O(sqrt N) to O(1) results for restless bandits; the parallel is unverified here.

## 4. Certainty-equivalent control, open-loop feedback, rollout and one-step lookahead; marginal analysis for the quotas

### Takeaway
Component (d) has two parts:
* **The quotas** are a certainty-equivalent (CE) plan. Equal-slope quotas solve the deterministic mean-SNR problem, a separable concave resource allocation that greedy marginal analysis solves exactly (Gross; Fox).
* **The advantage index** is a one-step-lookahead timing rule whose continuation (the option value of waiting) is the CE value of one typical slot.

Such rules have no general optimality guarantee. Their known guarantees are only relative ([U]): open-loop feedback control (OLFC) is at least as good as the optimal open-loop policy, rollout is at least as good as its base policy, and deterministic-LP / CE policies are asymptotically optimal in regimes where the CE problem coincides with the fluid limit. Ours is not such a regime. The fluid relaxation averages over users but keeps each user's fading, whereas CE removes the fading. This is consistent with the measured plateau of the closed form (0.155 dB per user at K = 128).

### Cited Findings
- **Myopic rules lose to Lagrangian ones.** In weakly coupled scheduling MDPs, decisions from the Lagrangian relaxation beat myopic decisions by a statistically significant margin (Gocgun and Ghate 2012). [UMN Experts](https://experts.umn.edu/en/publications/lagrangian-relaxation-and-constraint-generation-for-allocation-an/)
- **Fluid policies follow the relaxation, not a CE plan.** Brown and Zhang (2025) show that fluid policies, which follow the relaxation's (fluid) solution, are within O(sqrt N), and that budget balancing reaches O(1). [Duke Scholars](https://scholars.duke.edu/publication/1559343)
- **Our quotas are exact for the static problem.** The project's own analysis states that the greedy equal-slope rule is exact for separable allocation with concave q_k (marginal analysis: Gross 1956; Fox 1966). [PROBLEM.md, section 3, item 1](/home/user/JSCCwClaude/jscc/docs/PROBLEM.md)

### Inferences
- **Names.**
  - The quota step is certainty-equivalent control: replace the future slots' random SNRs by their mean, solve the deterministic allocation max sum_k q_k(L_k at mean SNR) s.t. sum_k L_k = (1-p)BT, and implement it. Its common multiplier (the equal slope) is Everett's generalised Lagrange multiplier ([U]). In RM terms it is a CE booking limit per user, plus a CE bid price.
  - The advantage index Delta_k(g_t) - Delta_k(g_mean) is a **one-step lookahead with a CE cost-to-go**: the value of keeping the opportunity is approximated by the gain of one typical slot. It is the advantage function Q - V of reinforcement learning, with V approximated by CE. The phrase "gain now minus gain from a typical future unit" is the CE estimate of the user's own opportunity cost, i.e. a per-user CE bid price.
- **Why CE is not the fluid limit here.**
  - In RMAB asymptotics, the fluid problem is the relaxation's primal: per-user randomised policies with the budget met in expectation. It keeps each user's full stochastic dynamics, including the selection (option) value of waiting for good slots under fading.
  - CE discards both that randomness and that selection value. The CE continuation differs from the user's true continuation in two ways:
    - Jensen effects: the PSNR at the mean SNR differs from the mean PSNR over SNRs;
    - selection: a scheduler that picks a user's good slots gets better-than-mean slots.

    It also ignores competition, which is the price lambda*_t.
  - Hence (d) is not a priority rule of the fluid LP, and nothing makes its per-user gap vanish as K grows: total 2.1 dB at K = 8, 19.8 dB at K = 128.
  - In network RM, CE does equal the fluid limit, because there scaling averages the very randomness (demand) that CE removes. In our K-scaling, each user's own fading never averages out.
- **What would make (d) asymptotically optimal.** Theory suggests replacing the CE continuation with lambda*_t, or with the user's own priced-program continuation. Either turns (d) into (c). Alternatively, use (d) as the base policy of a rollout, as `tools/headroom.py` did: rollout gained +0.05 dB at K = 8, and the rollout cost-improvement property ([U]) guarantees it is no worse than (d).
- **Quotas versus prices.** For the stochastic problem, the fluid LP replaces fixed quotas by per-slot prices: counts become random and respond to realised SNRs. Fixed CE quotas therefore act as a partitioned booking limit. They are optimal for the static (equal-SNR) problem when every q_k is concave in the number of slots (Gross; Fox), but not for the dynamic problem.

### Gaps
None of the following were verified in this session; all are [U], with citation confidence noted.
- **Bertsekas, *Dynamic Programming and Optimal Control*, Vol. I** (4th ed., Athena Scientific, 2017), chapters on suboptimal control and approximate DP (confidence high). Believed content:
  - definitions of certainty-equivalent control (CEC), open-loop feedback control (OLFC), limited lookahead and rollout;
  - OLFC performs at least as well as the optimal open-loop policy;
  - CEC is optimal in linear-quadratic problems (certainty equivalence) but carries no general guarantee;
  - rollout's cost-improvement property: a rollout policy is no worse than its base heuristic (for sequentially consistent or improving heuristics in the deterministic case; policy improvement in the stochastic case).
- **Rollout papers** (confidence high): Bertsekas, Tsitsiklis and Wu, "Rollout algorithms for combinatorial optimization," *J. Heuristics* 3, 1997; Bertsekas and Castañón, "Rollout algorithms for stochastic scheduling problems," *J. Heuristics* 5, 1999.
- **Marginal analysis and multipliers:**
  - Gross, "A class of discrete-type minimization problems," RAND RM-1644, 1956 (confidence medium-high);
  - Fox, "Discrete optimization via marginal analysis," *Management Science* 13(3):210–216, 1966 (confidence high);
  - Everett, "Generalized Lagrange multiplier method for solving problems of optimum allocation of resources," *Operations Research* 11(3):399–417, 1963 (confidence high);
  - Ibaraki and Katoh, *Resource Allocation Problems: Algorithmic Approaches*, MIT Press, 1988 (confidence high);
  - Shoham and Gersho, bit allocation (1988), already in SURVEY F.
- **Fluid-heuristic guarantees in pricing and RM** (confidence high): Gallego and van Ryzin, *Management Science* 40(8), 1994. Believed: the fixed price from the deterministic problem is asymptotically optimal.
- **Unread title.** A search result listed arXiv 2308.13166, "Certainty Equivalence Control-Based Heuristics in Multi-Stage Convex Stochastic Optimization Problems." It was not read, so it is not used.

## 5. Dynamic/stochastic knapsack with deadlines, sequential stochastic assignment, and OR applications with terminal concave or threshold rewards

### Takeaway
Within the relaxation, each user faces a priced, finite-deadline, sequential accept/reject problem. Structurally this is a dynamic and stochastic knapsack with a deadline: the capacity is the phases left, the items are slots with random SNR, and the policy is a threshold against the price. Unlike the knapsack, the reward is terminal and non-additive (concave, or 0/1). The phase-to-slot mapping is a Derman–Lieberman–Ross sequential stochastic assignment.

Terminal rewards do not take us outside the restless-bandit theory: they telescope into per-step rewards. Threshold rewards, however, create flat value regions, i.e. degenerate, tie-heavy instances.

Verified OR applications in this session's evidence that use the same Lagrangian/index machinery are applicant screening and dynamic assortment (Brown and Smith), capital budgeting and multi-location inventory (Brown and Zhang), job scheduling (Gocgun and Ghate), and deadline scheduling as a restless bandit ([S] Yu, Xu and Tong). The closest terminal-reward analogues are known to the researcher but unverified [U]: sensor management for classification, budgeted learning / ranking-and-selection, and guaranteed-delivery advertising / online convex allocation.

### Cited Findings
- **Brown and Smith (2020)** apply Lagrangian index policies and bounds to dynamic assortment with demand learning and to an applicant screening problem. The reward timing of the screening model (whether it is terminal) was not visible in the snippets. [INFORMS PDF](https://pubsonline.informs.org/doi/pdf/10.1287/mnsc.2019.3342)
- **Brown and Zhang (2022)** apply Lagrangian and dynamic-fluid relaxations to dynamic capital budgeting, and to multi-location inventory with limited production capacity under a shared market-state signal. [IDEAS](https://ideas.repec.org/a/inm/oropre/v70y2022i5p3015-3033.html)
- **Gocgun and Ghate (2012)** treat allocation and advanced scheduling of renewable resources to random job arrivals as weakly coupled MDPs. [IAOR](https://iaorifors.com/paper/75485)
- **Hawkins (2003)** applies Lagrangian decomposition to restless bandits, queueing networks, serial supply chains, linear control and on-line auctions. [DSpace@MIT](https://dspace.mit.edu/handle/1721.1/29599)
- **[S]** Yu, Xu and Tong (IEEE TAC 2018): jobs with deadlines as a restless bandit, a closed-form Whittle index, and asymptotic optimality. Hosseini and Panwar: scalable video with deadlines as a restless bandit solved by LP. Derman, Lieberman and Ross (*Management Science* 1972) is listed in SURVEY as the anchor for phase order. Sources: [SURVEY.md F2](/home/user/JSCCwClaude/jscc/docs/SURVEY.md); [arXiv 1610.00399](https://arxiv.org/abs/1610.00399); [arXiv 1612.03985](https://arxiv.org/abs/1612.03985)

### Inferences
- **Terminal rewards are covered by the theory.** Write r_t = f(x_{t+1}) - f(x_t). A terminal reward f(x_T) then becomes a sum of per-step rewards, so the bounds, indices and asymptotic results of sections 1 and 2 apply unchanged. Nothing in them requires rewards to be paid per period.
  - For concave terminal rewards, what changes is that the gain is an increment of a concave function of accumulated service, so it is state-dependent through both n and Y. A user's marginal value falls as it accumulates service, and that is what makes the per-slot price meaningful.
- **Threshold (0/1) rewards are the degenerate case.** Users far below or far above the target have zero marginal value, so many users have zero gain at lambda*. Tie-breaking and admission then decide the outcome (Brown and Smith on ties; the CMU blog on random tie-breaking failing for all-or-nothing rewards [S]). Expect O(1/sqrt K) rather than O(1) behaviour, and sensitivity to grids.
- **The per-user priced program is a stochastic knapsack with a deadline.** Capacity is P phases; items are slots arriving one per period with i.i.d. random "size-adjusted value" (via the SNR); the decision is accept/reject against the price lambda_t. Its optimal policy should be a state-dependent SNR threshold.
  - The classical dynamic and stochastic knapsack results are for additive rewards ([U]). With a concave terminal reward the threshold also depends on Y.
- **The online-allocation view gives a complementary asymptotic regime.** Per-slot SNR vectors are i.i.d. across slots, and the mean-PSNR objective is a sum over users of a function of each user's cumulative service. So the problem is also an instance of online stochastic allocation with a concave objective of cumulative allocations ([U] Agrawal and Devanur 2015; Balseiro, Lu and Mirrokni 2021/2023).
  - That theory is asymptotic in the horizon T: Fenchel-dual or gradient rules with O(sqrt T) regret, which is the same family as the gradient scheduler already cited in PROBLEM.md (Stolyar; Agrawal and Subramanian).
  - Our regime (short deadline T, many users K) is the restless-bandit regime instead.
  - Both views agree that the right per-slot quantity is the dual price of slot capacity.

### Gaps
None of the following were verified in this session; all are [U], with what each is believed to show.
- **Sequential stochastic assignment** (confidence high for DLR):
  - Derman, Lieberman and Ross, "A sequential stochastic assignment problem," *Management Science* 18(7):349–355, 1972: optimal thresholds that do not depend on the workers' values, by the rearrangement inequality.
  - Albright and Derman, *Management Science* 19(1), 1972: asymptotics.
  - Albright, *Management Science* 21(1), 1974: random arrival times.
- **Dynamic and stochastic knapsack:**
  - Papastavrou, Rajagopalan and Kleywegt, "The dynamic and stochastic knapsack problem with deadlines," *Management Science* 42(12):1706–1718, 1996 (confidence high);
  - Kleywegt and Papastavrou, "The dynamic and stochastic knapsack problem," *Operations Research* 46(1):17–35, 1998 (confidence high), with an optimal threshold (opportunity-cost) acceptance policy;
  - Kleywegt and Papastavrou, random-sized items, *Operations Research* 49(1), 2001 (confidence medium-high).
- **Perishable resources with deadlines:** Graczová and Jacko, "Generalized restless bandits and the knapsack problem for perishable inventories," *Operations Research* 62(3):696–711, 2014 (confidence medium-high): finite-horizon perishable items, with an index-knapsack heuristic.
- **Sensor management** (the closest terminal-reward Lagrangian analogue):
  - Castañón, "Approximate dynamic programming for sensor management," Proc. IEEE CDC 1997 (confidence medium-high). Believed: Lagrangian relaxation of per-time sensor-resource constraints across objects, with per-object POMDPs whose cost is a terminal classification error.
  - Williams, Fisher and Willsky, *IEEE Trans. Signal Process.* 55(8), 2007 (confidence medium): Lagrangian relaxation under a communication constraint.
- **Budgeted learning and ranking-and-selection** (terminal reward equals the quality of the final decision):
  - Guha and Munagala, STOC 2007 (budgeted learning, LP/Lagrangian approximations), and Guha, Munagala and Shi, "Approximation algorithms for restless bandit problems," *J. ACM* 58(1), 2010 (confidence medium-high): constant-factor approximations from LP relaxations;
  - Frazier, Powell and Dayanik, knowledge-gradient policy, *SIAM J. Control Optim.* 47(5), 2008 (confidence high): a one-step lookahead on a terminal reward;
  - Chen, Lin and Zhou, *JMLR* 16, 2015 (confidence medium-high): budget allocation in crowd labelling, with terminal accuracy and an optimistic knowledge gradient.
- **Clinical trials:** Villar, Bowden and Wason, "Multi-armed bandit models for the optimal design of clinical trials: benefits and challenges," *Statistical Science* 30(2), 2015 (confidence high). Finite-horizon Gittins/Whittle-type indices; whether any reward is terminal is uncertain.
- **Advertising with delivery goals:**
  - Vee, Vassilvitskii and Shanmugasundaram, "Optimal online assignment with forecasts," ACM EC 2010 (confidence high): dual-price allocation for guaranteed display contracts;
  - Balseiro, Lu and Mirrokni, "Regularized online allocation problems: Fairness and beyond," ICML 2021 (concave regularisers of cumulative consumption), and "The best of many worlds: Dual mirror descent for online allocation problems," *Operations Research* 71(1), 2023 (confidence high);
  - Agrawal and Devanur, "Fast algorithms for online stochastic convex programming," SODA 2015 (confidence high).
- **Seasonal assortment:** Caro and Gallien, "Dynamic assortment with demand learning for seasonal consumer goods," *Management Science* 53(2), 2007 (confidence high): a finite-horizon index derived from a Lagrangian relaxation.
- **No closer match found in the time available.** No verified OR paper was found whose arms have a terminal threshold reward on accumulated noisy service and which is solved with a Lagrangian/LP-priority index and measured as K grows. This could not be searched further, so it is a gap, not evidence of novelty.

## 6. Mapping: our component → established OR name → key references → what is new in our use

### Takeaway
Every component of our method has an established OR name:
* (a) is the Lagrangian (first-order LP / fluid) relaxation of a weakly coupled DP with time-indexed multipliers.
* (b) is the finite-horizon Whittle index.
* (c) is the Lagrangian index / LP-priority / fluid-priority policy, which is equivalently a bid-price policy with priority rationing.
* (d) is certainty-equivalent planning (marginal-analysis quotas) plus a one-step lookahead with a CE cost-to-go.
* (e) is the ID policy of the average-reward theory.
* (f) is the perfect-information (zero-penalty information) relaxation, computed by column generation on a configuration LP.

What is new in our use is mainly the arm: a learned-codec quality state with a terminal concave or threshold reward and stale-report kernels. The other new contributions are empirical: the measured O(1)-total rate on that arm, the CE-versus-fluid diagnosis of the closed form, and the "top of the indifference range" tie convention. No new theory is claimed.

### Cited Findings
- Sections 1–5 above carry the sources for each row ([V] and [S] links). The [U] references in the table are flagged and listed in the Gaps of sections 1–5.

### Gaps
- Several rows rely on [U] references (the RM bid-price literature, Bertsekas, marginal analysis, Derman–Lieberman–Ross, Gilmore–Gomory and Dantzig–Wolfe for column generation). They must be verified before the mapping is used in a paper.
- **(f), [U]:** Gilmore and Gomory, "A linear programming approach to the cutting-stock problem," *Operations Research* 9, 1961, and Dantzig and Wolfe, "Decomposition principle for linear programs," *Operations Research* 8, 1960 (confidence high). These are the standard origins of configuration LPs and column generation. Vera and Banerjee (2021) is a possible reference for the "prophet / hindsight" benchmark terminology.

### Inferences: the mapping table

| Our component | Established OR name | Key references (status) | What is new in our use |
|---|---|---|---|
| **(a)** Per-slot prices lambda_t; bound (1-p)B sum_t lambda_t + sum_k W_k(lambda); minimised by subgradient steps | Lagrangian relaxation (decomposition) of a weakly coupled stochastic DP with time-indexed multipliers. Equivalently, the dual of the first-order LP ("fluid" / occupation-measure) relaxation; in finite-horizon form, the Whittle relaxation with time-varying subsidies. Its subgradient is the expected budget slack. | Hawkins 2003 [V]; Adelman and Mersereau 2008 [V]; Bertsimas and Niño-Mora 2000 [V]; Brown and Smith 2020 [V]; Hu and Frazier 2017 [V]; Gast, Gaujal and Yan 2024 [V]; Brown and Zhang technical note [V] (ALP = Lagrangian bound for binary actions and unimodular links, which is our case); Brown and Zhang 2022 [V] (history-dependent multipliers under shared signals); Gocgun and Ghate 2012 [V] (exact solution by constraint generation); Fisher 1981 [U] (subgradient) | The arm is a learned-codec quality model (n, Y) under continuous fading. The reward is terminal: concave PSNR or a 0/1 target. A report kernel makes the bound valid for schedulers that see only stale CSI (an information-relaxation argument). Lost units form a shared random supply, handled with time-only prices; Brown and Zhang suggest signal-dependent prices would be tighter at small B. |
| **(b)** Whittle index, taken at the top of the indifference range | Finite-horizon (time-dependent) Whittle index: the arm's indifference subsidy, computed under one constant price path. Requires indexability. A non-unique indifference price is a tie / degeneracy issue. | Whittle 1988 [S]; Brown and Smith 2020 [V] (ties and mixing matter); Zhang and Frazier 2021 [V] (index policies are fluid-priority policies: O(1) total under non-degeneracy); Yu, Xu and Tong 2018 [S]; Weber and Weiss 1990, Verloop 2016, Niño-Mora 2001, Ghosh et al. 2023 [U] | The convention "largest price at which taking is still optimal" as a principled tie-break. At r = 0 it reduces to the expected marginal gain. No OR source was found that defines it this way. Measured: O(1) total gap (2.1 dB at K = 8 and K = 128), but it degrades on stale reports. |
| **(c)** Lagrangian / LP-priority index: gain at lambda*_t, serve the top B, ties broken by Whittle, all units filled | Lagrangian index policy (Brown and Smith). Index policy from the Lagrangian relaxation (Hu and Frazier). LP-index / LP-priority policy (Gast et al.; Verloop [U]). Fluid-priority policy (Zhang and Frazier). Primal-dual (reduced-cost) index (Bertsimas and Niño-Mora). Hawkins's per-period integer program, which reduces here to sorting. Fluid-budget balancing (Brown and Zhang 2025). In RM terms: a bid-price policy with priority rationing (Talluri and van Ryzin; Topaloglu [U]). | Rates, per user: O(1/sqrt K) on all models (Gast et al. [V]; Brown and Zhang 2025 [V]); O(1/K) under non-degeneracy (Zhang and Frazier [V]); exponential for LP-index under non-degeneracy (Gast et al. [V]); LP-update is more robust under model error [V] | First measurement on a learned-codec scheduling problem: 0.207 to 0.005 dB per user for K = 8 to 128, i.e. a total gap of 1.66 to 0.64 dB, the non-degenerate O(1)-total regime. The users are heterogeneous (images, mean SNRs), and the policy is tested on stale reports. Open issue: the 30 dB target gap stays at about 1.5 points for every K, which the theory does not allow if the bound is the policy's LP; check discretisation, prices and ties. |
| **(d)** Advantage index (this slot's gain minus the gain of one mean-SNR slot) with equal-slope quotas | Certainty-equivalent control: a mean-value plan of a separable concave resource allocation, solved exactly by greedy marginal analysis / equal slopes (Everett multiplier). Executed with a one-step lookahead whose CE cost-to-go is the option value of one typical slot (the RL "advantage" Q - V). In RM terms: CE booking limits plus a CE bid price. | Bertsekas DP&OC (CEC, OLFC, lookahead, rollout) [U]; Gross 1956, Fox 1966 [U; cited in PROBLEM.md]; Everett 1963 [U]; Gocgun and Ghate 2012 [V] (myopic rules lose to Lagrangian ones); Brown and Zhang 2025 [V] (fluid policies, not CE plans, are asymptotically optimal) | A closed form computed directly from the learned quality model, with no DP. The diagnosis is new and consistent with the theory: CE is not the fluid limit when K grows, because each user's own fading never averages out. Hence the plateau: 0.155 dB per user, with the total gap rising from 2.1 dB (K = 8) to 19.8 dB (K = 128). Remedy: use lambda*_t, or the program's continuation, as the reference, which turns (d) into (c). |
| **(e)** ID-guided policy and the random-order baseline (CMU blog) | ID policy: the relaxation's ideal actions, with conflicts resolved by a fixed priority order of arms (persistency). Random tie-breaking is the known failing baseline. Related: Follow-the-Virtual-Advice (FTVA); weakly coupled MDP extensions. | Hong, Xie, Chen and Wang, MOR 2025 [S]; Hong et al., NeurIPS 2023 [S]; Zhang, Hong and Wang, NeurIPS 2025 [S]; CMU blog 2026 [S] | Run in a finite horizon with terminal rewards, where index policies need no attractor condition. Measured: ID beats random at the target (persistency matters for all-or-nothing rewards); the index policies beat both on mean PSNR (ranking by value matters for smooth rewards). |
| **(f)** Clairvoyant optimum by column generation on a configuration LP | Perfect-information (hindsight / "prophet") relaxation, i.e. the zero-penalty information relaxation. Computed as a Dantzig–Wolfe / configuration (set-partitioning) LP with column generation; the pricing problem is each user's best slot set. | Brown, Smith and Sun 2010 [V title and DOI only]; Ye, Zhu and Zhou 2018 [V] (information plus Lagrangian relaxations give tighter bounds); Gilmore and Gomory 1961, Dantzig and Wolfe 1960 [U]; Vera and Banerjee 2021 [U] | Used as a second, non-causal bound bracketing (a). A natural next step from Ye, Zhu and Zhou and Brown and Smith: add Lagrangian-based penalties to obtain a causal bound tighter than (a). |
| Phase-to-slot order with fixed quotas (supporting rule) | Sequential stochastic assignment: thresholds independent of the workers' values | Derman, Lieberman and Ross 1972 [S/U]; Albright and Derman 1972 [U] | Exact here because the distortion is linear in the per-phase noise terms. Prefix order is kept when lengths are online or slots can be lost. |
