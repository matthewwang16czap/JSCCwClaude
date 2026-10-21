# The problem: scheduling encode-once token prefixes over a fading multi-user downlink

2026-10-09. Chosen after reading PADC (IEEE TWC 2023, with its code) and JSCCformer-f (IEEE
TWC 2024) in full: `docs/ASSESSMENT.md` 1f explains why the earlier framing ("one model for
every rate + per-image allocation") is no longer new. This note states a problem in which the
properties of this tree's token codec are REQUIRED, not merely convenient, and the experiments
that decide whether it carries a TWC paper.

## 1. System

* A base station serves K users in a frame of T slots (the deadline). User k is owed one image
  x_k (a camera frame for remote monitoring, a map tile, an AR asset): all are due at the end
  of the frame.
* Each slot carries S channel uses and is given to one user (TDMA; the OFDMA version gives
  resource blocks).
* Block fading: user k's slot-t gain h_k(t) ~ CN(0, beta_k), independent over slots, with mean
  SNRs that differ per user (path loss). The scheduled receiver knows h and equalises, so the
  slot is an AWGN channel at SNR g_k(t) = |h_k(t)|^2 SNR_k. The base station learns g_k(t) at
  the start of slot t (CQI, as in every opportunistic scheduler); future slots are unknown
  beyond their statistics.
* Codec: every image is encoded ONCE, before the frame, into its token sequence (SNR- and
  rate-agnostic encoder). In a slot the scheduled user receives the next chunk of its prefix:
  the next S / (sym x tiles) tokens of every tile. At the deadline user k holds L_k tokens per
  tile that arrived over its slots' SNRs, and decodes once.
* Objective: maximise the mean (or an alpha-fair function) of the users' quality at the
  deadline; or, the QoS form (PADC's own objective, made multi-user): the share of users whose
  image reaches a target quality by the deadline; variant: the slots needed to bring every
  user to a target (makespan).

## 2. Why this codec, and not the published ones

| requirement of the problem | this tree | PADC (TWC 2023) | JSCCformer-f (TWC 2024) | DeepJSCC-MIMO / SwinJSCC per CBR |
|---|---|---|---|---|
| the code exists before the slot SNRs are known | yes: SNR- and rate-agnostic encoder | no: the encoder's attention blocks take the SNR | no: the encoder re-encodes every block from feedback | per-rate models trained over random SNR: yes |
| the length is decided online, slot by slot | yes: any prefix decodes | prefix code (channel prefix), but the length is chosen before sending | yes, block by block | no: one length per model |
| a prefix whose chunks had different SNRs decodes | yes after `--snr-chunks` training (P-A); predicted per image within 0.08 dB (P-B) | decoder conditioned on ONE SNR for the whole code | yes (feedback carries it) | n/a |
| no feedback beyond CQI | yes | yes | no: channel-output feedback of every symbol, a decoder copy per user at the transmitter | yes |
| content-aware marginal value of a slot, before sending | per-image curves (alloc/) | OraNet (per image, per SNR, per rate) | from the decoder copy | no |
| many users | yes | per-image rule, no shared budget | single link (and a 2-user broadcast) | n/a |

The properties that make the problem work, together: encode-once tokens, any-length prefixes,
decoding of mixed-SNR prefixes, transmitter-side utility curves, and only CQI feedback. No
published scheme has all of them (searches on 2026-10-09 found no opportunistic multi-user
scheduler over progressive or prefix DeepJSCC streams with a semantic utility; the closest are
DRJSCC (single link, re-encodes the remaining blocks when the channel changes, arXiv
2311.15309) and a scalable multi-user DeepJSCC for diverse bandwidths (multicast layers)).

## 3. The scheduler

* Utility model (gate P-B, passed 2026-10-10, `docs/NOTES.md` P1 and P1 step 2): NOT an
  effective SNR of the chunks -- the code is ordered and early tokens are 2-20x more
  noise-sensitive -- but an additive noise-sensitivity model, valid once the decoder is trained
  on piecewise SNRs (`--snr-chunks`):
  D_k(L; n_1..n_L) = Dsrc_k(L) + C_k(L) sum_j rho_L[j] phi(n_j), phi(n) = n / (1 + n / kappa)
  (`alloc/utility.py`, fitted by `tools/utility_fit.py`): per image and phase, held-out
  profiles within 0.03-0.07 dB and decoded schedules within 0.08 dB (p95 0.29), no bias.
  Separable over slots, position- and SNR-aware: the value of a slot to a user is a
  closed-form function of what it already holds.
* Mean quality. On the real Kodak utility PF takes the multi-user diversity gain (+1.1 dB
  over round robin) and content-aware LENGTHS add only +0.05 dB (equal-slope, fixed or
  re-planned); the myopic marginal-utility greedy loses 0.23 dB. What PF misses is how much a
  slot is worth to the user in QUALITY: the ADVANTAGE index serves the user whose predicted
  PSNR gain from this slot most exceeds its gain from a slot at its mean SNR (`slope_adv`):
  +0.476 dB over PF, +0.465 decoded by the codec (`docs/NOTES.md`, P1 step 3), +0.46 dB on the
  5th-percentile user. On a stand-in about 0.3 dB of it is valuing slots in PSNR (users near
  their noise-free quality take the bad slots) and 0.2 dB the position (sensitive head phases
  wait for peaks).
* Quality target. The share of users reaching Q dB. `online`: every slot, admit the users with
  the smallest remaining need (phases at the mean SNR, from what they already hold) that fit
  in the slots left, open spare slots to all, serve by the advantage index, and stop a user as
  soon as the model says its realised SNRs got it there. Against PADC's rule (per-image length
  at the mean SNR, fixed; given shortest-first admission and PF timing): +9.4 points of users
  at 30 dB (K = 8, T = 24), +10.0 decoded, +6.9 to +10.4 over targets, frame lengths, K = 16
  and identical images. Its parts: per-image sizing (+5.0 points; the curves), online stopping
  (+3.9; encode-once prefix + mixed-SNR utility, no feedback beyond CQI), advantage timing
  (+3.2 to +3.4; the utility), admission (+2.0, +11 in tight frames).
* Optimality, layer by layer (what is exact, what is not; `tools/headroom.py` measures each):
  1. Slot COUNTS with a channel that does not change within the frame (every slot alike for
     a user): max sum_k q_k(L_k) s.t. sum_k L_k = T, L_k <= P is a separable allocation, exact
     by a dynamic program over users (O(K T P)); the greedy equal-slope rule (our quotas) is
     exact when every q_k is concave (marginal analysis: Gross 1956, Fox 1966). For a target
     with known needs, shortest-first admission maximises the number served (cardinality
     knapsack; Moore-Hodgson with one deadline).
  2. WHICH slots, with every SNR of the frame known (clairvoyant): a combinatorial
     assignment (8^24 ~ 5e21 raw assignments at K 8, T 24) whose user values are non-linear in
     the set of slots and their order. Our policies do not solve it. It is solved offline for
     evaluation: column generation on the configuration LP (each user one set of up to P
     slots, each slot to at most one user; every set is priced, so its value bounds every
     schedule from above, causal or not) and an integer program over the generated columns;
     equal to brute force on small frames, bound - optimum ~0.001 dB at K 8, T 24 on the
     stand-in (certified).
  3. ONLINE (the real problem: a slot's SNR is known when it starts): a finite-horizon
     restless multi-armed bandit (`alloc/whittle.py`). With the nested model (one master
     sensitivity profile m for every prefix length, `tools/utility_fit.py --nested`) a
     user's state is two numbers, n phases held and Y = sum_j m_j phi(n_j), and its quality
     at the deadline is -10 log10(Dsrc(n) + C(n) Y / sum(m[:n])). The joint program (2K
     continuous states) is intractable, but the users are coupled only by one user per
     slot: pricing a slot at lam decomposes it into K single-user programs on (t, n, Y)
     (Whittle; Hawkins; Adelman and Mersereau), which gives (a) an upper bound on EVERY
     causal scheduler, (1 - p) T lam + sum_k W_k(lam), minimised over the price; (b) the
     Whittle index, user k's indifference price for this slot, and the index policy (serve
     the largest); (c) the advantage index as its certainty-equivalent approximation (the
     continuation valued at a typical, mean-SNR slot instead of the program's). Checked
     against exact expectimax on small frames: bound >= best causal >= Whittle. Stand-in,
     K 8, T 24: the Whittle policy ties our rules (-0.02 dB [-0.04, +0.01] mean PSNR; -0.4
     points at 30 dB), rollout +0.05 dB / +3.1 points, causal bound +0.20 dB / +5.6 points,
     clairvoyant optimum +0.19 / +6.3: best causal - our rule is within [+0.05, +0.19] dB
     for the mean (certified), [+3.1, +5.6] points at 30 dB, where a better causal policy
     has room. The gradient-scheduler link (Stolyar; Agrawal and Subramanian) stays as the
     long-horizon motivation.
  4. PHASE ORDER: when a user's slot count is fixed before the frame (fixed quotas, no lost
     slots) the phases need not go in prefix order. For a fixed set of slots the best mapping
     puts the most sensitive phase in the best slot (rearrangement inequality); online, with
     the user's remaining slots i.i.d., the optimal causal mapping for the expected
     distortion is the threshold rule of the sequential stochastic assignment problem
     (Derman, Lieberman and Ross 1972), which applies exactly because the distortion is
     linear in the per-phase noise terms. Stand-in: +0.12 dB ex post on our rule's own slots,
     +0.27 for the clairvoyant optimum with any order (prefix order +0.19). With online
     lengths or lost slots the prefix order is what keeps a cut-short code decodable.
  5. The price of prefix coding is the nesting penalty (0.17-0.47 dB, mostly recoverable at
     2-4 phases by a decoder per length), which the scheduling gain has to exceed.
  6. Re-examined against the restless-bandit literature (2026-10-18; `docs/SURVEY.md` F5;
     COMMANDS 18). Our frame is a PLANNING restless bandit (known dynamics, every arm's state
     observed): every user reports its channel whatever is scheduled, and the quality model is
     calibrated at the transmitter. The learning-RMAB papers (unknown pay-off processes, only
     the played arm observed: Grunewalder and Khaleghi, JMLR 2019; Gornet and Sinopoli 2024)
     answer a question we do not have, but they point at the two assumptions that matter:
     a. STALE REPORTS. A report is taken before the slot: h_rep = r h + sqrt(1 - r^2) e, r =
        |J0(2 pi f_D tau)| (3.5 GHz, 5 ms-old reports: 0.98 at 3 km/h, 0.76 at 10, 0.50 at 15,
        ~0 at 24, below 0.3 beyond; 1 ms-old: 0.91 at 30 km/h, 0.66 at 60). The arm is then
        partially observed, but nothing else changes: given the report, h ~ CN(r h_rep, 1 - r^2)
        (the Kalman predictor of a Gauss-Markov channel), so the user's program decides on the
        report's fading bin and draws the slot's true gain from a report-to-channel kernel
        (`alloc/csi.py`, `Arm(cqi_corr=r)`, checked against simulation: the program's value
        matches the simulated value at r = 1 / 0.8 / 0.4). The decomposition and the bound
        survive: (1 - p) T lam + sum_k W_k(lam; r) bounds every scheduler that sees only such
        reports (the user's program may even know its own true Y). This is the standard
        belief-state relaxation of a partially observed restless bandit, not a new kind of
        bound: Ouyang, Eryilmaz and Shroff bound every policy the same way for ARQ-revealed
        Markov channels and plot the bound against N (read in full 2026-10-20); ours is new in
        the observation model (every user reports every slot, no exploration value) and the
        finite horizon. What a scheduler gains by switching between users is bounded by what its
        information says about the current slot (Grunewalder and Khaleghi, Prop. 9), so as r
        falls the opportunistic part of every gain and the lift of the channel the scheduler
        makes must vanish, and only the static, content part (how many phases each user gets)
        remains. An "aware" transmitter decides on each slot's posterior (JSCC: the SNR of its
        expected distortion, exact for the additive model's MSE; digital: the rate of best
        expected goodput); a "naive" one takes the reports as exact, as schedulers do today.
        Measured (COMMANDS 18.1-18.2, `docs/NOTES.md` 5 and 6b): as predicted, the opportunistic
        part fades and at K 8, T 24 it is all there is (the content part is 0.000 at every r):
        slope_adv - PF +0.475 dB at r = 1, +0.385 / +0.256 at r = 0.8 / 0.6 (aware +0.475 /
        +0.342), ~0 at r = 0 (the counterpart of Ouyang et al.'s memoryless channel, where
        feedback is worthless and their index policy is uniform random). The lift of the
        scheduler-made channel (+5.0 / +4.5 / +3.6 dB, phases 1-3) keeps 65% of itself at r =
        0.8, 8% at 0.3. Unlike the stand-in suggested, the aware transmitter is worth something:
        it keeps the whole gain over PF to r = 0.8 and saves the quality target (stopping on the
        reports breaks: a user served at a reported peak looks done although its slot was worse;
        aware plan_adv keeps +8 points over PADC at r = 0.8). JSCC's lead over digital grows
        with report age, the physical axis of C5: +0.3 dB at r = 1, +1.7 dB / +15.5 points at
        0.8, +1.7 / +14 at 0.6 against digital with incremental-redundancy HARQ (without it:
        +2.5 / +19, +3.3 / +20; COMMANDS 20.2), and the JSCC side holds decoded (20.1). The
        bound falls with r (31.68 -> 30.18 dB, r 1 -> 0.4) and the gap of every policy to it
        shrinks: less information, less to gain.
     b. MANY USERS PER SLOT. The theory of index policies (Hu and Frazier 2017; Brown and Smith
        2020; Zhang and Frazier 2021; Hao, Xu and Tong, IEEE TAC, read in full 2026-10-20: our
        Lagrangian index is their Lagrangian priority policy; Gast, Gaujal and Yan 2024 and
        Zhang and Frazier, both read in full 2026-10-21: our Lagrangian index is Gast et al.'s
        LP index) is asymptotic in the number of arms with a fixed share served per period:
        there, Lagrangian (LP-priority) index policies lose O(1/sqrt(N)) per arm against the
        Lagrangian bound, without indexability, provided they reproduce the relaxation's
        activations (LP-compatible; Gast et al. Thm 2); in total, O(1) when every slot has a
        mixed state in the relaxation (non-degenerate) and O(sqrt(N)) otherwise (Zhang and
        Frazier). A downlink TTI has many resource units (subband groups): B units per slot,
        each one phase, a user at most one per slot, is exactly that regime, with the bound's
        supply term (1 - p) B T lam. Our K 8, one-unit frame is its hardest point.
        `tools/rmab_scale.py` (COMMANDS 18.3) measures the gap per user for K = 8 ... 128 with B
        = K/8, for PF, our rule, the Whittle index and the Lagrangian index (each user's gain at
        the bound's per-slot prices, ties broken by the Whittle index). Measured (mean PSNR, gap
        to the bound per user, K 8 / 16 / 32 / 64 / 128): Lagrangian 0.207 / 0.090 / 0.039 /
        0.009 / 0.005 dB, faster than 1/sqrt(K); Whittle 0.265 -> 0.016; our closed form 0.267
        -> 0.155, a plateau (its continuation is a typical slot, not the price); PF 0.71 ->
        0.52. In total, the Lagrangian index's gap stays bounded (1.66 -> 0.64 dB): Zhang and
        Frazier's non-degenerate O(1). At 30 dB the Lagrangian index stays ~1.5 points from the
        bound at every K, which the theory does not predict. Two leads (2026-10-20): Hao, Xu and
        Tong's Figs. 3-4 show the same flat gap when ties are not broken by the relaxation's own
        randomized policy (ours go to the Whittle index), and McDowell et al. (read in full;
        citing Chakraborty, Sun, Hou and Eryilmaz 2026) report the same for the multi-action
        per-slot knapsack, where an LP-priority tie rule restores asymptotic optimality; and
        alloc/whittle.lagrangian_bound's subgradient step grows with the unit imbalance; at B =
        4 and 8 the per-slot prices never improved on the best constant price (smoke stand-in:
        0.007 dB per user for the mean; the stand-in's target does not bind). One-unit results
        are unaffected. Batch 5 of the full texts (2026-10-21) makes it specific: a flat
        per-user gap is a total gap linear in K, which no LP-compatible policy has (even a
        degenerate problem gives 1/sqrt(K) per user), and no priority rule is LP-compatible when
        several users are indifferent at once (Gast et al. Thm 5), as a 0/1 target makes likely;
        ours ranks the indifferent users by the Whittle index. Prices matter little: on the
        stand-in with a binding 29 dB target (T 10), proper per-slot prices lower the bound by
        0.07 / 0.15 points per user at K 8 / 32. Diagnosis owed: docs/PAPER.md, owed item 9 (the
        relaxation's tie rule first). The lost-slot runs (--preempt) are outside these theorems
        (a budget shared at random by all users; Zhang and Frazier list it as open). So the
        theory's regime holds, and at scale the programs' index is the scheduler to use; the
        closed form is its cheap approximation (one unit per slot: a tie). On stale reports (K
        8) the Lagrangian index stays best and the Whittle index degrades (-0.15 dB against our
        rule at r = 0.4): its price-range convention is fragile when the reports say little.
     c. THE INDEX WHERE NOW AND LATER ARE ALIKE. When a later slot is as good as this one
        (useless reports, or no deadline pressure yet) taking and leaving the slot are worth the
        same over a whole range of prices and the indifference price is not unique; the index is
        defined as the top of that range (the largest price at which taking it is still
        optimal), located by the gain's slope -1 past it. At r = 0 it is then the user's
        expected marginal gain, falling with the phases held (what a static allocation needs);
        with fresh reports nothing changes (identical to the previous definition on the exact
        small frames). Taking the bottom of the range, as a plain sign test does, lets the
        programs' interpolation error pick the users (seen on the stand-in: very unequal counts,
        -0.5 dB). The convention is ours: Yu, Xu and Tong's Whittle index (TAC 2018, Def. 2) is
        the smallest subsidy at which resting is optimal, the bottom of the range, and so are Le
        Ny, Feron and Dahleh's (Def. 4) and Ouyang et al.'s (eq. 10); Hao, Xu and Tong define
        their priority as a supremum, but with the future prices fixed the indifference point is
        unique there (full texts, 2026-10-20). A published rule in the same spirit: McDowell et
        al.'s LP-priority tie rule keeps the larger codelength among tied maximisers, resolving
        indifference toward the more active action.
     d. ONE PROGRAM PER USER, THEN A CONFLICT RULE (the CMU blog, Hong 2026, read in full;
        SURVEY F5). Every policy here has the same two steps: the relaxation gives each user an
        ideal action (take or leave this slot at the price lambda*_t), and a rule decides who is
        served when more users want the slot than it has units. Whittle / Lagrangian indices
        rank by value; our rule ranks by its certainty-equivalent value among users below their
        quotas; the blog's ID policy takes the wanting users in a fixed order (persistency),
        which it proves near-optimal for average reward without the global attractor index
        policies need. In our finite-horizon frame no attractor condition is needed for index
        policies, but the blog's lesson still applies where rewards are all-or-nothing: at a
        quality target, spreading slots over too many users wastes them (its random tie-breaking
        failure), which our admission rules (shortest need first) already prevent.
        `tools/rmab_scale.py` runs the ID-guided policy and its random-order baseline beside the
        indices (COMMANDS 18.3). Measured: at 30 dB ID beats random by +0.5 to +0.7 points at
        every K (persistency, modest here: the stand-in's +9 was four frames), but both trail
        the Lagrangian index (2.6 / 3.1 vs 1.5 points from the bound at K 128); for the mean
        value ranking wins (0.005 vs 0.09-0.11 dB at K 128). On stale reports (r = 0.7) the
        guided policies collapse at the target (ID 35.6%, random 46.3%, the best policy 59.9%):
        with flat ideal actions the order alone decides. The conflict rule matters because of
        the deadline, not in general: for age of information, Tang, Wang, Song and Song (read in
        full 2026-10-21) pick uniformly at random among the sensors whose relaxed policy wants
        the slot and prove that O(1/sqrt(N)) from the optimum, because a deferred sensor waits
        one more slot at a bounded cost; in a frame with a terminal all-or-nothing target a
        deferred user can lose its last good slot.
  7. DELAY (2026-10-18; measured in COMMANDS 19, 2026-10-19, `docs/NOTES.md` 6c). A frame has
     one deadline, so delay can mean three things, and they do not all favour the same rule
     (snrc4, calibrated transmitter, K 8, 2000 frames; the Whittle index on the nested utility):
     a. THE LATENCY BUDGET: the shortest frame that reaches a quality (`tools/latency.py`: each
        policy's quality against T, inverted). The cleanest delay claim, the same quality with a
        shorter deadline. For slope_adv's mean PSNR at T = 16 / 24 PF needs 17.6 / 29.5 slots:
        9.3% [8.7, 9.9] / 18.6% [16.3, 20.7] fewer for ours; for online's share at 30 dB at
        T = 16 (66.8%) PADC needs 26.0 slots (38.6% [34.9, 41.9] fewer), and online's share at
        T = 24 (74.0%) no other policy reaches within 40 slots (PADC saturates near 70%).
     b. TIME TO THE TARGET within a frame (`schedule_sim.py --delay`): the share of users at the
        target by every slot, a CDF over all users (a miss never counts). T 24: online 18.3 /
        42.1 / 61.9 / 74.0% by slots 6 / 12 / 18 / 24, PADC 19.5 / 48.9 / 61.6 / 65.6, PF 16.2 /
        33.3 / 53.6 / 63.5: online trails PADC in the first half and overtakes in the last
        quarter, 0.36 slots later on average with misses counted and 1.8 later on the users
        both bring there (its deadline guard waits). The mean slot over the users who get
        there is NOT the comparison: ours brings more, and harder, users there. The WHITTLE
        INDEX is ahead at every slot: 38.4 / 60.0 / 69.3 / 72.8% (PADC 20.1 / 48.4 / 60.8 /
        64.6 on the same frames), 2.9 slots [2.7, 3.1] before PADC on the same users, at
        online's final share. With the target reward the index of a user with n phases to go is
        about its success gain per remaining phase (the top of its indifference range), so it
        serves the shortest remaining need first, weighted by the channel --
        shortest-remaining-processing-time, the rule that minimises completion times, which
        the closed form (this slot against a typical one) does not know.
     c. CONTINUOUS TRAFFIC (`tools/stream_sim.py`): requests arrive over time (Poisson), each
        with an image, a mean SNR and its own deadline D; a request leaves as soon as its rule
        is done with it, so the slots a rule saves (online stopping, peak timing) serve other
        requests; with one deadline for all it is the frame exactly (checked slot by slot). This
        is Yu, Xu and Tong's deadline restless bandit (SURVEY F2) with our quality reward. The
        measures: the share of requests at the target within d slots of arriving, for every d
        up to D (the reliability-latency curve of URLLC); the capacity, the arrival rate at which
        a given share still reach the target (Hosseini and Panwar count users admitted at a
        quality); and the delay on the requests two rules both bring there. 30 dB, D 24, at the
        frame's load (~0.33 requests per slot): online 70-71%, PADC 46%, stop_adv 66-68%, PF
        timing with stopping 58-62%. Online's share RISES with load (66% at 0.1 requests per
        slot, when a request is mostly alone and every slot is its: no choice of slot), PADC
        stays near 45%; capacity at 60%: online >= 0.45 per slot at every D, PADC never (about
        a third of requests cannot reach 30 dB at any load). But online's delay is long: its
        p90 is the deadline at high load, and on the same requests it is 2.9 to 7 slots slower
        than PADC. The WHITTLE INDEX does not have this flaw: each request's program (horizon =
        its own deadline) weighs how urgent it is against how good its slot is. D 24, 0.3 / 0.4
        requests per slot: 68.3 / 68.8% at 30 dB (online 70.5 / 69.3), 57.3 / 54.1% within D/4
        (online 39.2 / 27.3), mean delay 4.2 / 4.8 slots (online 8.0 / 11.1), and 0.6 / 1.0
        slots FASTER than PADC on the same requests. With deadlines that differ the program's
        index is what delivers the delay advantage, at the cost of one table per image and
        mean-SNR class (computed once).
     What does not favour our rule, and is reported with the rest: the first decodable image
     comes later (round robin and PF serve every user once first, slot (K + 1) / 2 on average;
     slope_adv's at slot 6.7, PF's at 4.5), and the PSNR averaged over the frame's slots (a
     progressive display) is lower (25.0 vs 26.2 dB against PF): the deadline objective does not
     reward early coarse images. If an application does, the users' programs take a per-slot
     reward (the quality shown after each slot, or 1 per slot spent at the target), which gives
     a delay-aware Whittle index and its bound by the same decomposition (not implemented).

## 4. Baselines

1. Round robin with the same prefix codes (channel- and content-blind).
2. Max-SNR (opportunistic, content-blind) and proportional fair on log2(1 + g) (the LTE / NR
   scheduler).
3. PADC-style: each user's length fixed before the frame from predicted curves at its mean SNR
   (PADC's per-image rule, or the equal-slope rule), delivered with PF timing.
4. Fixed-length per-user codes (one model per CBR, as DeepJSCC-MIMO / SwinJSCC) with
   opportunistic timing: isolates what online LENGTH adds over online TIMING.
5. Digital: BPG (and a progressive codec) with capacity-achieving or LDPC codes, AMC per slot,
   the same schedulers; as run: HEIC with the LTE CQI table or a 3 dB-gap code, the backoff
   tuned per policy, and HARQ (a lost block retransmitted and combined, `--harq`).
6. JSCCformer-f's setting (K = 1, a quality target, feedback): the single-user special case,
   where this scheme stops on predicted quality with CQI only.

Metrics: mean PSNR / MS-SSIM / LPIPS at the deadline, 5th-percentile user, Jain's fairness,
slots to bring all users to a target. Kodak and CLIC; K in {4, 8, 16}; mean SNRs spread over
0-20 dB; T from tight to loose.

## 5. Gates

| gate | test | pass |
|---|---|---|
| P-A mixed-SNR decoding | `tools/mixed_snr.py` on the w1 ViT, then on a `--snr-chunks 4` fine-tune vs its constant-SNR control (COMMANDS.txt section 5) | a mixed prefix within ~0.2 dB of the model's prediction; constant-SNR quality within 0.1 dB of the control. **2026-10-10: passes with `--snr-chunks` training (additive within 0.06 dB on 2-chunk means; constant-SNR cost 0.06-0.10 dB); fails for constant-SNR decoders** |
| P-B utility model | the same runs, then per phase (COMMANDS.txt section 6) | MAE <= 0.2 dB per image on held-out profiles. **2026-10-10: passes for snrc4 (held-out 0.03-0.07 dB per image and phase, replayed schedules 0.08 dB); s1 0.12-0.29** |
| P-C1 mean quality | simulator on the real utility, then the codec on the schedules (section 7) | >= 0.3 dB mean PSNR over PF, decoded, K = 8, Rayleigh. **2026-10-10: content-aware lengths +0.05 (the first form fails); the advantage index passes: +0.476 simulated, +0.465 [0.39, 0.53] decoded; +0.34 to +0.49 over every scenario** |
| P-C2 quality target | the same, with `--target` | >= 5 points more users at the target than PADC's rule (with shortest-first admission and PF timing), decoded, K = 8, T = 24, 30 dB. **2026-10-10: passes: +9.4 simulated, +10.0 [7.3, 12.9] decoded (+9.6 with a 0.2 dB margin, 0.2% missed); +6.9 to +10.4 over the scenarios, +4.2 with 10% curve errors** |
| P-C3 realistic transmitter | the same with `--belief` (each image calibrated from 2P constant decodes; kappa, rho from other images), correlated fading (section 8) | P-C1 and P-C2 still pass. **2026-10-11: passes: decoded +0.464 dB [0.39, 0.53] and +11.0 points [8.3, 13.8]; calibration error 0.03-0.06 dB, as good as the full fit; under slot correlation up to 0.98 the leads are >= +0.34 dB and >= +5.6 points** |
| P-D separation | `tools/digital_rd.py` + the same schedulers on a digital utility (alloc/digital.py: HEIC, error-free CQI link adaptation, free file switching, up to 12 slots per user) | ahead of HEVC-intra + CQI link adaptation at low and mid SNR; honest against the capacity bound. **2026-10-11: ON PAR, not ahead: mean PSNR +0.13 dB (prefix) / +0.28 (per-CBR) at each one's best policy; target: +6 to +9 points in tight frames, -6 to +0.2 at 32 dB; a 3 dB-gap code is 0.5-0.7 dB ahead, capacity 1.7-1.9. At the equal per-user cap (6 slots, 2026-10-12): mean unchanged; target +0.5 / +5.0 / +8.4 points at 30 / 28 / T 16, -1.6 at 32 (prefix code), +0.8 to +10.5 with a decoder per length. Error-free CQI is the assumption left that favours digital: section 13 (`--cqi-error`)** |
| P-E one model per CBR | six specialists (1/48 ... 1/8), each trained like snrc4_s1 plus `--fixed-cbr`, one per-CBR utility; scheduled with `--fixed-codes` (spare slots planned before the frame) | the prefix code's best policy ahead of the specialists' best. **2026-10-11: with known slots per-CBR wins: mean +0.15 dB, target +2.0 to +6.0 points against `online` (much of it the slot planner, which the prefix code can use too: `plan_adv` / `replan_adv`, section 12). With lost slots the prefix code wins mean PSNR by +0.36 / +0.53 dB (10% / 25% lost) against per-CBR's best hedge. 2026-10-12, the same planner on both: per-CBR still +1.0 to +4.4 points with known slots (the nesting penalty itself); with 25% lost `replan_adv` +0.5; with a decoder per length (P-F) -0.9 to +1.2 known, +2.8 points / +0.73 dB at 25% lost. Decoded: the model and both schedulers hold on the per-CBR codes (MAE 0.06-0.08 dB)** |
| P-F decoder heads | decoder-only specialists and LoRA budget heads on the frozen snrc4 encoder | how much of the penalty a per-length decoder recovers. **2026-10-11: LoRA heads none (= more training alone); a decoder of its own ~40% at 1 and 6 phases. 2026-10-12, all six lengths: 79-101% of the penalty at 2-4 phases (57-100% beyond more training), 49-65% at 1, 5, 6: where schedules live the penalty is mostly a decoder matter, paid at the receiver. Encode once + a decoder per length is the recommended JSCC system; equal-budget check: section 13** |

G-A, G-B and G-C of `docs/ASSESSMENT.md` stay relevant: G-A picks the codec, G-B says how much
the content part of the utility can add, G-C prices the prefix.

## 6. Risks

* PF with prefix codes takes most of the mean-quality gain, and content-aware lengths add
  +0.05 dB on Kodak (measured): the mean-quality claim rests on the advantage index, the
  content claim on the quality target. Report the parts separately.
* For mean quality the prefix code itself is not required: equal shares with PF timing tie PF
  (what one model per CBR allows), and the advantage index works on any code split over slots
  once the decoder is trained piecewise (with uniform rho: about +0.3 of its +0.5 dB on the
  stand-in), so a per-CBR codec without the nesting penalty is the fair baseline there. What
  is prefix-specific: the position part of the index and, for the target, online stopping.
* The decoder may need a per-token noise map (DeepJSCC-MIMO's heatmap idea) for strongly mixed
  profiles; that is the next code change if P-A fails after `--snr-chunks` training.
* Correlated fading (Jakes) and OFDMA change the numbers, not the method; both belong in the
  paper's evaluation (`--fade-corr`: Gauss-Markov slot gains; on the stand-in a correlation of
  0.9 widens online's lead over padc, whose fixed lengths cannot react to a long fade).
* The transmitter's knowledge of each image's curves: 10% log-MSE errors halve the target
  lead (+4.2). The base station has the image and the codec, so it can calibrate each image
  from 2P constant-SNR decodes of its own (`tools/utility_fit.py --calib`), with kappa and rho
  measured once on other images: gate P-C3.
* Fairness across frames: the quality-target rules admit the smallest remaining needs, which
  maximises each frame's share but could refuse the same hard (cell-edge) users frame after
  frame if they persisted. Our frames redraw users, so the simulations cannot show it. The
  standard remedy is a per-user debt carried across frames (Hou, Borkar and Kumar, INFOCOM 2009:
  largest-debt-first is feasibility optimal for long-run delivery ratios), which can sit on top
  of any of our rules; to state as a limitation.
* The codec's competitiveness: at CBR 1/12 and 1 / 4 / 7 dB on Kodak this tree's w1 ViT gives
  28.94 / 30.26 / 31.35 dB, JSCCformer-f 30.84 / 32.05 / 33.16 with two-block feedback; after
  discounting ~1 dB of feedback gain, still ~0.8-0.9 dB behind, and about on par with PADC's
  CNN. The scheduling claims are relative (same codec under every policy), but a TWC reviewer
  will ask; the likely fix is data (JSCCformer-f and PADC train on ImageNet, this tree on the
  800 DIV2K images; `--trainset DIV2K Flickr2K CLIC` is the cheap first step).

## 7. The code's channel is made by the scheduler: code-scheduler co-design (2026-10-13)

Everything above designs the scheduler for a GIVEN code. But under a scheduler a code's phases
never see "the channel": each phase goes out in the slot the scheduler picked for it, so the SNR
every phase sees, and how many phases every user receives, are decisions of the MAC. The code's
channel is endogenous. Our index puts the first, most sensitive phases into fading peaks and lets
the tail take ordinary slots; snrc4 was trained on chunks at i.i.d. SNRs uniform over [-2, 22]
dB with random cuts and uniform lengths: another channel.

* The object (`tools/codesign.py bank`): the distribution p_pi(L, s_1..s_L) of the number of
  phases a user receives and the SNR of each, in prefix order, under policy pi, over the frames
  (users' mean SNRs, fades, images). On the stand-in utility (K 8, T 24, U[0, 15] dB, Rayleigh):
  slope_adv's first phase sits +3.3 dB above the user's mean SNR, its third +1.8, the tail ~0;
  blindly taken slots -1.9 dB (the fade's mean, -2.5 dB, less clipping at -2 dB); the per-phase
  mean SNR falls from 10.7 dB at phase 1 to 1.1 dB at phase 6 (the users who get long prefixes
  are the weak ones); replan_adv at 30 dB 10.8 -> 4.8 dB. And the profile differs by length:
  the receiver KNOWS how many phases arrived, so under an opportunistic scheduler the length is
  side information about the channel (stand-in: users with 2 phases saw ~11 dB on both, users
  with 6 saw 3.9 dB on the first and 1.1 on the last), which a length-conditioned decoder (FiLM
  on L) can learn and the transmitter never has to send.
* Two designers, one objective. The code designer picks the weights theta, the MAC designer the
  policy pi; both want J(theta, pi) = E[quality at the deadline] (mean PSNR, or the share at a
  target). The scheduler needs the code only through its utility u_theta (the additive model,
  measured and calibrated from 2P decodes per image: sections 3 and 5); the code needs the
  scheduler only through p_pi. Alternating best responses:
    code:      theta_{r+1} = argmin_theta E_{(L, s) ~ p_{pi_r}} E_x D_theta(x; L, s)
               (fine-tuning on a bank of pi_r's histories: `main.py --snr-profiles`)
    scheduler: pi_{r+1} = our policy on u_{theta_{r+1}} (measure, fit, calibrate, re-plan)
  This is a common-payoff (identical-interest) game; with exact best responses J cannot
  decrease, J(theta_{r+1}, pi_{r+1}) >= J(theta_{r+1}, pi_r) >= J(theta_r, pi_r), and its fixed
  points are codes trained for the channel their own scheduler makes. Ours are approximate:
  the scheduler step is within the certified gaps of section 3 (stand-in: within 0.05-0.19 dB
  of the best causal policy for the mean) and the model's error (0.03-0.09 dB); the code step
  is SGD on MSE while J is PSNR (a concave transform: E[PSNR] and -E[MSE] rank codes alike
  only approximately); and the bank draws histories independently of the training image,
  where pi decides with the image's own curve (a content-conditioned bank needs a quality
  predictor for training crops: not built).
* Why only this tree can close the loop: a round of the loop is cheap because the scheduler runs
  on the model (a bank of 32,000 histories is a CPU minute, no decoding), and the scheduler can
  re-plan for a new code from 2P decodes per image. Without a link abstraction every round
  would need the codec inside the scheduler.
* The test, round 1 (COMMANDS.txt section 14): four fine-tunes of snrc4 with the same recipe and
  budget (400 epochs), differing only in the channel: hf (the recipe so far, the control),
  blind (slope_adv's lengths, every phase in a blindly taken slot: the physical channel at the
  scheduler's lengths), sa (what slope_adv delivers), sat (what replan_adv delivers at 30 dB).
  Each decodes the ORIGINAL system's schedules (same users, images and slot SNRs: the code's
  best response alone), then its own re-planned schedules (the loop closed once). sa - blind
  isolates the scheduler's slot selection (the claim), blind - hf the lengths and the users'
  mean SNRs (matched training, the older idea). The scheduler's channel for the retrained
  codes against the one they trained on (`codesign.py stats --against`) says whether one round
  is near the fixed point.
* Round 1 measured (2026-10-14, docs/NOTES.md "P2, round 1"): on the real utility slope_adv's
  quotas give every user 3 phases at +3.2 / +2.7 / +1.8 dB over its mean SNR (blind slots
  -1.8); the code trained on that channel decodes the same slots +0.28 dB better than the
  same-budget control and +0.32 better than the code trained on blind slots, +0.09 under PF
  slots; at 30 dB nothing. Re-banked on the retrained code's utility the channel barely moves
  (99.7% of users keep their length): the fixed point is reached in one round, so the loop's
  second step adds nothing measurable. Round 2 (COMMANDS 15): a 3-phase specialist on the
  generic channel, a shuffled bank, 16 users, a PF-trained code.
* Round 2 measured (2026-10-14, COMMANDS 15): against a 3-phase specialist trained on the
  generic channel (the length slope_adv's quotas give everyone), the scheduler-trained code
  keeps +0.12 dB on the same slots (+0.15 with 16 users); permuting each user's phases over its
  own slots keeps two thirds of it. The codes learn their schedulers: kappa rises as deep fades
  get rarer, and the phase weights move to the phase the scheduler favours (PF: phase 2).
* How fresh the reports must be (2026-10-18; section 3, item 6a; COMMANDS 18.2): the channel a
  scheduler makes exists only as far as the scheduler sees the channel. With stale reports of
  correlation r the lift of every phase must shrink towards the blind channel (Grunewalder and
  Khaleghi: a policy's sampling times bias what it samples only through the dependence between
  its information and the pay-off). Measured (COMMANDS 18.2, banks of 4000 frames, slope_adv):
  the lift over random selection, +5.0 / +4.5 / +3.6 dB for phases 1 / 2 / 3 with fresh reports,
  keeps 91% of itself at r = 0.95, 65% at 0.8, 36% at 0.6 and 8% at 0.3. So scheduler-aware
  training is a low-mobility, fresh-report effect (3.5 GHz, 5 ms-old reports: r 0.98 at
  3 km/h, 0.76 at 10).
* Theory owed if it passes: (a) the game and its monotone improvement (above), with the two
  approximation gaps bounded by what section 3 measures; (b) the linear-Gaussian version
  (analog prefix code of a Gaussian source, per-phase power allocation under a per-phase SNR
  distribution): the matched allocation, and the loss of the i.i.d.-trained one, in closed form;
  (c) the information the received length carries about the channel under a given policy,
  and what an L-conditioned decoder gains from it.
