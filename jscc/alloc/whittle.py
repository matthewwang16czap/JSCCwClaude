"""The frame's scheduling problem as a finite-horizon restless bandit (docs/PROBLEM.md,
section 3): per-user dynamic programs, the Whittle index, and an upper bound on every
causal scheduler.

A frame: K users, T slots, at most one user served per slot, a slot's SNRs known when it
starts. With the NESTED utility model (alloc/utility.nested_rho: one master sensitivity
profile m for every prefix length) a user's quality at the deadline depends on two numbers:

    n   phases held
    Y   sum_j m_j phi(noise(snr_j)) over the phases held (their accumulated weighted noise)
    PSNR = -10 log10(Dsrc(n) + C(n) Y / sum(m[:n]))

so each user is a small Markov decision process: in slot t it sees its slot SNR g (its mean
SNR plus block fading, |h|^2 ~ Exp(1), i.i.d. over slots; the slot lost to other traffic
with probability p) and, if served, moves (n, Y) -> (n + 1, Y + m_{n+1} phi(noise(g))). The
users are coupled only by "one user per slot". Pricing a slot at lam (Whittle 1988; the
Lagrangian decomposition of weakly coupled dynamic programs, Hawkins 2003, Adelman and
Mersereau 2008) splits the frame into K single-user programs:

  bound   for any price lam >= 0, the expected objective (sum over users) of ANY causal
          scheduler is at most (1 - p) T lam + sum_k W_k(lam), W_k(lam) being user k's
          best expected quality when every slot it takes costs lam (each slot it takes
          is a slot no other user gets; the relaxation lets that hold only on average).
          `lagrangian_bound` minimises it over a constant price.
  index   user k's Whittle index in state (t, n, Y) with slot SNR g is the price at which
          it is indifferent between taking the slot and not; the index policy serves the
          largest (tools/schedule_sim.py --policies whittle). Valued at a typical
          (mean-SNR) slot instead of the program's continuation, it is the advantage index
          of tools/schedule_sim.py (its certainty-equivalent approximation).

The programs run on grids: Y on `ny` points (denser near 0, linear interpolation), the
fading on `ng` equiprobable quantiles of Exp(1), the price on a geometric grid. With the
--target objective the reward is 1 if the user's PSNR reaches the target, else 0.

Stale reports (cqi_corr = r < 1, alloc/csi.py): the transmitter decides on a report with
correlation r to the slot's channel, so the program decides on the report's fading bin and
the slot's true gain is drawn from the kernel P(true bin | report bin). The index then takes
the slot's report. The bound stays a bound on every causal scheduler that sees only such
reports: the user's program may even know its own true Y, more than any transmitter does.
"""

import numpy as np

from alloc.utility import noise, phi


def master_profile(util, i):
    """The master sensitivity profile of image i: the utility's own (a --nested fit stores
    it), else its longest prefix's rho (exact for a nested model). Returns (m, deviation):
    the largest |rho_L - nested(m, L)| over lengths (0 for a nested model)."""
    P = util.P
    rho = [r if r.ndim == 1 else r[i] for r in util.rho]
    m = np.asarray(util.meta.get("nested") or rho[P - 1], dtype=np.float64)
    m = m / m.sum()
    dev = max(float(np.abs(rho[L - 1] - m[:L] / m[:L].sum()).max()) for L in range(1, P + 1))
    return m, dev


def price_grid(target, n=24):
    """Slot prices: PSNR dB per slot for the mean objective, success probability for a target."""
    hi, lo = (1.0, 1e-4) if target is not None else (30.0, 1e-3)
    return np.concatenate([[0.0], np.geomspace(lo, hi, n - 1)])


class Arm:
    """One user's dynamic program over (slot, phases held, accumulated weighted noise)."""

    def __init__(self, util, i, mean_snr, T, target=None, p_lost=0.0, clip=(-2.0, 22.0),
                 lams=None, ny=128, ng=16, fade_db=None, fade_w=None, cqi_corr=1.0):
        self.P, self.T, self.target, self.p = util.P, int(T), target, float(p_lost)
        self.mean, self.clip, self.r = float(mean_snr), tuple(clip), float(cqi_corr)
        if self.r < 1.0 and fade_db is not None:
            raise ValueError("cqi_corr < 1 needs the default Rayleigh fading bins")
        self.kern = None                                        # report bin -> true bin
        if self.r < 1.0:
            from alloc.csi import kernel
            self.kern = kernel(ng, self.r)
        self.kappa = float(util.kappa if np.ndim(util.kappa) == 0 else util.kappa[i])
        self.m, self.deviation = master_profile(util, i)
        self.S = np.concatenate([[1.0], np.cumsum(self.m)])     # S[n] = sum(m[:n]); S[0] unused
        self.dsrc, self.csens = util.dsrc[i], util.csens[i]
        self.empty = util.psnr(i, [])
        if fade_db is None:                                     # equiprobable Exp(1) quantiles
            q = -np.log(1.0 - (np.arange(ng) + 0.5) / ng)
            fade_db, fade_w = 10.0 * np.log10(q), np.full(ng, 1.0 / ng)
        g = np.clip(float(mean_snr) + np.asarray(fade_db, dtype=np.float64), clip[0], clip[1])
        self.w = np.asarray(fade_w, dtype=np.float64) / np.sum(fade_w)
        self.f = phi(noise(g), self.kappa)                      # the slot's noise term, per bin
        self.ymax = self.S[self.P] * float(phi(noise(clip[0]), self.kappa)) * 1.0001
        self.y = self.ymax * np.linspace(0.0, 1.0, ny) ** 2     # denser where PSNR is steep
        self.lams = price_grid(target) if lams is None else np.asarray(lams, dtype=np.float64)
        self.W = self._solve(self.lams)                         # (T + 1, n_lam, P + 1, ny)

    def reward(self, n, Y):
        """Terminal reward of n phases with accumulated noise Y (array)."""
        Y = np.asarray(Y, dtype=np.float64)
        if n == 0:
            q = np.full(Y.shape, self.empty)
        else:
            D = self.dsrc[n - 1] + self.csens[n - 1] * Y / self.S[n]
            q = -10.0 * np.log10(np.maximum(D, 1e-12))
        return q if self.target is None else (q >= self.target).astype(np.float64)

    def _weights(self, Yq):
        """Grid cell and weight of linear interpolation at Yq (any shape)."""
        Yq = np.clip(Yq, 0.0, self.y[-1])
        j = np.clip(np.searchsorted(self.y, Yq, side="right") - 1, 0, len(self.y) - 2)
        return j, (Yq - self.y[j]) / (self.y[j + 1] - self.y[j])

    def _interp(self, V, Yq):
        """Linear interpolation of V (..., ny) on the Y grid at Yq (any shape)."""
        j, a = self._weights(Yq)
        return V[..., j] * (1.0 - a) + V[..., j + 1] * a

    def _by_report(self, v):
        """(L, ng true bins, ny) -> (L, ng report bins, ny) through the report kernel."""
        L, G, Y = v.shape
        return (self.kern @ v.transpose(1, 0, 2).reshape(G, L * Y)).reshape(G, L, Y).transpose(1, 0, 2)

    def _cells(self):
        if not hasattr(self, "_cell"):
            self._cell = [self._weights(self.y[None, :] + self.m[n] * self.f[:, None])
                          for n in range(self.P)]
        return self._cell

    def _solve(self, lams):
        """W[t, l, n, :]: the value at the start of slot t with n phases held, on the Y grid,
        when every slot taken costs lams[l] (lams (L,)), or lams[l, t] in slot t (lams (L, T))."""
        lams = np.asarray(lams, dtype=np.float64)
        per_slot = lams.ndim == 2
        L, P, ny = lams.shape[0], self.P, len(self.y)
        W = np.empty((self.T + 1, L, P + 1, ny))
        for n in range(P + 1):
            W[self.T, :, n, :] = self.reward(n, self.y)[None, :]
        cell = self._cells()
        for t in range(self.T - 1, -1, -1):
            lam = (lams[:, t] if per_slot else lams)[:, None, None]
            W[t, :, P, :] = W[t + 1, :, P, :]
            for n in range(P):
                stay = W[t + 1, :, n, :]                                       # (L, ny)
                V, (j, a) = W[t + 1, :, n + 1, :], cell[n]
                take = V[:, j] * (1.0 - a) + V[:, j + 1] * a - lam              # (L, ng, ny)
                if self.kern is not None:                  # decided on the report's bin
                    take = self._by_report(take)
                best = np.maximum(stay[:, None, :], take)
                W[t, :, n, :] = self.p * stay + (1.0 - self.p) * (best.transpose(0, 2, 1) @ self.w)
        return W

    def demand(self, prices, start=None, W=None):
        """With a price per slot (T,): the value from the start, and the expected number of
        slots t the user takes (T,) under its best response (a forward pass of its state
        distribution through the decisions of its program; the gradient of the value).
        `start` = (t0, n, Y): from that state at the start of slot t0 (LP-update: the slots
        before t0 take nothing, the value is the state's). W: priced(prices), if at hand."""
        prices = np.asarray(prices, dtype=np.float64)
        if W is None:
            W = self._solve(prices[None, :])[:, 0]              # (T + 1, P + 1, ny)
        t0, mass = self.start_mass(start)
        value = float(W[0, 0, 0]) if start is None else float((W[t0] * mass).sum())
        taken = np.zeros(self.T)
        for t in range(t0, self.T):
            mass, taken[t], _ = self._slot(W, float(prices[t]), t, mass)
        return value, taken

    def start_mass(self, start=None):
        """(t0, the state distribution (P + 1, ny) at the start of slot t0): the frame's
        start, or the state (n, Y) of `start` = (t0, n, Y), split between its two Y grid
        points as the programs interpolate."""
        mass = np.zeros((self.P + 1, len(self.y)))
        if start is None:
            mass[0, 0] = 1.0
            return 0, mass
        t0, n, Y = start
        j, a = self._weights(np.asarray(float(Y)))
        mass[int(n), int(j)] += 1.0 - float(a)
        mass[int(n), int(j) + 1] += float(a)
        return int(t0), mass

    def _slot(self, W, price, t, mass, theta=0.0, tol=0.0):
        """One slot of a forward pass: the state distribution `mass` (P + 1, ny) at the start
        of slot t under the decisions of the program with values W, priced `price` in slot t.
        A state whose gain from the slot net of its price is above tol takes it; one within
        tol of 0 (at the kink: indifferent, a later slot as good) takes it with probability
        theta, unless the slot is worth nothing to it at all (done or out of reach, which a
        zero price leaves at the kink too). Returns (the distribution at t + 1, the mass that
        takes the slot for sure, the indifferent mass). theta = tol = 0: the strict best
        response (bit for bit the earlier forward pass)."""
        new = mass * self.p                                     # the slot is lost
        new[self.P] += mass[self.P] * (1.0 - self.p)            # nothing left to take
        cell, ny = self._cells(), len(self.y)
        strict = mid_mass = 0.0
        for n in range(self.P):
            if not mass[n].any():
                continue
            j, a = cell[n]
            val = W[t + 1, n + 1, j] * (1.0 - a) + W[t + 1, n + 1, j + 1] * a   # (ng, ny)
            if self.kern is not None:                      # decided on the report's bin
                val = self.kern @ val
            stay = W[t + 1, n, :][None, :]
            d = val - price - stay                         # (the earlier forward pass's order)
            take = d > tol                                                    # (ng, ny)
            mv = mass[n][None, :] * self.w[:, None] * (1.0 - self.p)          # (ng, ny)
            strict += float((mv * take).sum())
            mid = (d >= -tol) & ~take & (val - stay > tol)   # indifferent, and worth something
            if theta > 0 and mid.any():                    # mixed at the kink
                mid_mass += float((mv * mid).sum())
                mt = mv * (take + theta * mid)
                new[n] += (mv - mt).sum(0)
                if self.kern is not None:
                    mt = self.kern.T @ mt
                new[n + 1] += (np.bincount(j.ravel(), (mt * (1.0 - a)).ravel(), ny)
                               + np.bincount((j + 1).ravel(), (mt * a).ravel(), ny))
                continue
            if tol > 0:
                mid_mass += float((mv * mid).sum())
            new[n] += (mv * ~take).sum(0)
            if self.kern is not None:                      # the true bin it lands in
                mt = self.kern.T @ (mv * take)
                np.add.at(new[n + 1], j.ravel(), (mt * (1.0 - a)).ravel())
                np.add.at(new[n + 1], (j + 1).ravel(), (mt * a).ravel())
                continue
            np.add.at(new[n + 1], j[take], (mv * (1.0 - a))[take])
            np.add.at(new[n + 1], j[take] + 1, (mv * a)[take])
        return new, strict, mid_mass

    def value(self, lam):
        """W(lam): the expected reward minus prices paid, from the start of the frame."""
        if np.any(np.isclose(self.lams, lam)):
            return float(self.W[0, int(np.argmin(np.abs(self.lams - lam))), 0, 0])
        return float(self._solve([lam])[0, 0, 0, 0])

    def state(self, snrs):
        """(n, Y) of a delivery history (one SNR per phase, in phase order)."""
        n = len(snrs)
        if n == 0:
            return 0, 0.0
        return n, float(np.dot(self.m[:n], phi(noise(np.asarray(snrs, dtype=np.float64)),
                                                self.kappa)))

    def gain(self, t, n, Y, snr):
        """The value of taking slot t at this SNR minus not taking it, at every grid price.
        With cqi_corr < 1, snr is the slot's REPORT (unclipped): the value of taking it is
        averaged over the true SNR's posterior (alloc/csi.py)."""
        if n >= self.P or t >= self.T:
            return np.full(len(self.lams), -np.inf)
        return (self._take(self.W[t + 1, :, n + 1, :], n, Y, snr)
                - self._interp(self.W[t + 1, :, n, :], Y) - self.lams)

    def _take(self, V, n, Y, snr):
        """V (L, ny): the next slot's values with n + 1 phases; their value after this slot
        carries phase n + 1 at this SNR (or, cqi_corr < 1, this report: the mean over the
        true SNR's posterior samples, alloc/csi.py; exact in the tail, where the program's
        own equiprobable bins would cap a very good slot at the top bin's value), (L,)."""
        if self.kern is None:
            return self._interp(V, Y + self.m[n] * float(phi(noise(snr), self.kappa)))
        from alloc.csi import posterior
        post = posterior(np.array([float(snr) - self.mean]), self.r)[0]           # (S,)
        g = np.clip(self.mean + 10.0 * np.log10(np.maximum(post, 1e-12)), *self.clip)
        return self._interp(V, Y + self.m[n] * phi(noise(g), self.kappa)).mean(-1)

    def priced(self, prices):
        """The program's values (T + 1, P + 1, ny) when slot t costs prices[t] (a constant
        or (T,)): for gain_at, the Lagrangian (LP-priority) index."""
        prices = np.broadcast_to(np.asarray(prices, dtype=np.float64), (self.T,))
        return self._solve(prices[None, :])[:, 0]

    def gain_at(self, Wp, prices, t, n, Y, snr):
        """The gain of taking slot t at this SNR (report) in the program priced at `prices`
        (Wp = priced(prices)): the Lagrangian index of Hu and Frazier / Brown and Smith, at the
        bound's prices Hao, Xu and Tong's Lagrangian priority value (minus lambda_t), well
        defined without indexability."""
        if n >= self.P or t >= self.T:
            return -np.inf
        lam = float(np.broadcast_to(np.asarray(prices, dtype=np.float64), (self.T,))[t])
        return float(self._take(Wp[t + 1, None, n + 1, :], n, Y, snr)[0]
                     - self._interp(Wp[t + 1, n, :], Y) - lam)

    def index(self, t, n, Y, snr):
        """The Whittle index: the price at which taking this slot and leaving it are worth
        the same (negative if the slot is not worth taking even for free). Where the two are
        worth the same over a range of prices (the user would buy a later slot as good as
        this one anyway: stale reports, or no deadline pressure yet), the top of the range:
        the largest price at which taking it is still (weakly) optimal. Past that top the
        gain falls with slope -1 (the user would not buy the phase at all), which locates
        the top between two grid prices. Gains within half the smallest grid price of 0
        count as 0 (the programs' interpolation error)."""
        d = self.gain(t, n, Y, snr)
        if not np.isfinite(d[0]):
            return -np.inf
        tol = 0.5 * float(self.lams[1]) if len(self.lams) > 1 and self.lams[1] > 0 else 1e-9
        if self.kern is not None:          # the slot's samples against the program's bins
            tol = max(tol, 10.0 * float(self.lams[1]))
        if d[0] < -tol:
            return float(d[0])
        j = int(np.flatnonzero(d >= -tol)[-1])
        if j == len(d) - 1:
            return float(self.lams[-1] + d[-1])
        if d[j] <= tol:                                   # flat up to here
            return float(max(self.lams[j], self.lams[j + 1] + d[j + 1]))
        return float(self.lams[j] + d[j] * (self.lams[j + 1] - self.lams[j]) / (d[j] - d[j + 1]))

    def indexability(self):
        """The share of (slot, phases, Y, SNR bin) states at which taking the slot stops and
        starts again as the price rises (0: the arm is indexable on the grid)."""
        bad = tot = 0
        for t in range(self.T):
            for n in range(self.P):
                Y2 = self.y[None, :] + self.m[n] * self.f[:, None]
                v = self._interp(self.W[t + 1, :, n + 1, :], Y2)                 # (n_lam, ng, ny)
                if self.kern is not None:                  # decided on the report's bin
                    v = self._by_report(v)
                d = v - self.W[t + 1, :, n, :][:, None, :] - self.lams[:, None, None]
                take = d >= 0                                          # (n_lam, ng, ny)
                bad += int(np.sum(np.any(~take[:-1] & take[1:], axis=0)))
                tot += take.shape[1] * take.shape[2]
        return bad / max(tot, 1)


def kink_demand(arms, prices, supply, tol, Ws=None, starts=None):
    """The relaxation's demand at a kink: the programs' forward passes in lockstep, priced
    `prices` (T,); in each slot every state whose gain is within tol of 0 (indifferent: the
    users who would take a later slot as well) takes the slot with one common probability
    theta_t, the one that brings the expected demand to the supply where it can (the
    relaxation's own randomisation: Brown and Smith's split in proportion to the occupation
    measure, Hao, Xu and Tong's phi*, water-filling across the mixed states). Ws: the
    programs priced at `prices`; starts: one (t0, n, Y) per arm (LP-update). Returns
    (demand (T,), theta (T,), the sure demand (T,), the indifferent mass (T,))."""
    T = arms[0].T
    prices = np.asarray(prices, dtype=np.float64)
    Ws = [a.priced(prices) for a in arms] if Ws is None else Ws
    starts = [None] * len(arms) if starts is None else starts
    init = [a.start_mass(s) for a, s in zip(arms, starts)]
    t0, mass = init[0][0], [m for _, m in init]
    dem, theta, sure, mid = (np.zeros(T) for _ in range(4))
    for t in range(t0, T):
        parts = [a._slot(W, float(prices[t]), t, m, 0.0, tol) for a, W, m in zip(arms, Ws, mass)]
        sure[t], mid[t] = sum(p[1] for p in parts), sum(p[2] for p in parts)
        th = float(np.clip((supply - sure[t]) / mid[t], 0.0, 1.0)) if mid[t] > 1e-12 else 0.0
        theta[t], dem[t] = th, sure[t] + th * mid[t]
        mass = ([a._slot(W, float(prices[t]), t, m, th, tol)[0]
                 for a, W, m in zip(arms, Ws, mass)] if th > 0 else [p[0] for p in parts])
    return dem, theta, sure, mid


def lagrangian_bound(arms, T, p_lost=0.0, refine=12, per_slot=0, units=1, scale="price",
                     kink=None, start=None, info=None):
    """min over prices lam >= 0 of (1 - p) B sum_t lam_t + sum_k W_k(lam): an upper bound on
    the expected objective (summed over the users) of every causal scheduler that serves at
    most B = `units` users per slot, each unit lost with probability p. First over a
    constant price (the grid minimum refined by golden-section search between its
    neighbours), then, with per_slot > 0, that many projected subgradient steps on a price
    per slot (the gradient: the slots' supply (1 - p) B minus the users' expected demand).
    Returns (bound, prices: a constant or (T,)).

    scale: the subgradient step. "price" (the default, every result before 2026-10-22):
    half the constant price over sqrt(step number), times the gradient in units, so it grows
    with B (at B = 4 and 8 the steps never improved on the constant price). "supply": the
    same divided by the supply (1 - p) B, a relative step at every B.
    kink: a tolerance; the gradient is then taken at the relaxation's mix of the indifferent
    states (kink_demand), not at their strict best response, which counts every indifferent
    state as not buying and so overstates the shortfall wherever many states are tied.
    start = (t0, [(n, Y) per arm]): the bound from slot t0 on, the users in those states (the
    LP-update policy re-solves its prices so; the slots before t0 keep price 0).
    info (a dict): receives the indifferent mass at the returned prices (kink_demand with
    tol = kink, or the programs' precision), per slot and as a share of the supply."""
    lams = arms[0].lams
    supply = (1.0 - p_lost) * units
    t0 = 0 if start is None else int(start[0])
    sts = [None] * len(arms) if start is None else [(t0, n, Y) for n, Y in start[1]]
    left = T - t0

    def at(a, V, st):                         # V (L, P + 1, ny) at slot t0 -> (L,)
        if st is None:
            return V[:, 0, 0]
        return a._interp(V[:, int(st[1]), :], float(st[2]))

    f = supply * left * lams + sum(at(a, a.W[t0], st) for a, st in zip(arms, sts))
    j = int(np.argmin(f))
    best, lam = float(f[j]), float(lams[j])
    lo, hi = lams[max(j - 1, 0)], lams[min(j + 1, len(lams) - 1)]

    def g(x):
        if start is None:
            return supply * T * x + sum(a.value(x) for a in arms)
        return supply * left * x + sum(float(at(a, a._solve([x])[t0], st)[0])
                                       for a, st in zip(arms, sts))

    for _ in range(refine if hi > lo else 0):
        x1, x2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        v1, v2 = g(x1), g(x2)
        if v1 < best:
            best, lam = v1, x1
        if v2 < best:
            best, lam = v2, x2
        if v1 < v2:
            hi = x2
        else:
            lo = x1
    tol = kink if kink is not None else (0.5 * float(lams[1]) if len(lams) > 1 else 1e-9)
    if per_slot <= 0 and start is None:
        if info is not None:
            _kink_info(info, arms, np.full(T, lam), supply, tol, sts if start else None)
        return best, lam
    live = np.arange(T) >= t0
    prices = np.where(live, lam, 0.0)
    step = max(lam, 1e-3) * 0.5 / (supply if scale == "supply" else 1.0)
    best_p = prices.copy()
    for it in range(per_slot):
        Ws = [a.priced(prices) for a in arms]
        if kink is None:
            vals, dem = zip(*(a.demand(prices, st, W) for a, st, W in zip(arms, sts, Ws)))
            dem = np.sum(dem, 0)
        else:                                     # the values need no forward pass
            vals = [float(W[0, 0, 0]) if st is None else float(at(a, W[t0][None], st)[0])
                    for a, st, W in zip(arms, sts, Ws)]
            dem = kink_demand(arms, prices, supply, kink, Ws, sts if start else None)[0]
        ub = supply * prices[live].sum() + sum(vals)
        if ub < best - 1e-12:
            best, best_p = ub, prices.copy()
        grad = np.where(live, supply - dem, 0.0)
        prices = np.maximum(prices - step / np.sqrt(it + 1) * grad, 0.0)
    if info is not None:
        _kink_info(info, arms, best_p, supply, tol, sts if start else None)
    return best, best_p


def _kink_info(info, arms, prices, supply, tol, starts):
    dem, theta, sure, mid = kink_demand(arms, prices, supply, tol, starts=starts)
    live = slice(0 if starts is None else int(starts[0][0]), None)
    info.update(mid=mid.tolist(), theta=theta.tolist(), sure=sure.tolist(),
                mid_share=float(mid[live].sum() / max(supply * len(mid[live]), 1e-12)),
                sure_share=float(sure[live].sum() / max(supply * len(sure[live]), 1e-12)))
