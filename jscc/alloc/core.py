"""Per-image budget allocation at a matched average rate: the arithmetic.

Arrays (numpy) used throughout:

    v      (N, K, M)  metric m of image i decoded with candidate budget units[k]
                      (units ascending, the same K candidates for every image)
    q      (N, K)     the objective, larger is better (signed, scaled metrics)
    cost   (N,)       symbols of image i per budget unit, relative to the mean
                      (its pixel count / the mean pixel count), so an average
                      budget is cost-weighted exactly like the CBR

The allocation is the equal-slope rule. Each image's points (cost_i*units_k,
q_i(k)) are reduced to their upper concave hull; hull segments of all images
are taken in order of decreasing slope until the symbol budget is spent. That
is the exact optimum of the Lagrangian relaxation of

    max sum_i q_i(k_i)   s.t.   sum_i cost_i * units[k_i] <= target * sum_i cost_i

and the one segment that does not fit is time-shared (image j takes the
larger budget with probability alpha), so every method meets the target
average rate exactly, in expectation, and is compared with uniform allocation
at the same average CBR. The target is always spent in full: once no segment
of positive slope is left (a saturated or noisy curve), the flattest remaining
ones are taken, since they cost the objective least.
"""

import numpy as np

SIGN = {"psnr": 1.0, "ssim": 1.0, "msssim": 1.0, "lpips": -1.0, "lpips_vgg": -1.0,
        "dists": -1.0, "cls_top1": 1.0, "cls_prob": 1.0, "det_f1": 1.0, "obj_psnr": 1.0,
        "bg_psnr": 1.0, "obj_lpips": -1.0, "dino_sim": 1.0}


def parse_objective(spec, available=None):
    """'lpips' or 'lpips:0.7,psnr:0.3' -> [(metric, weight)], weights summing to 1."""
    parts = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, w = tok.partition(":")
        name = name.strip()
        if name not in SIGN:
            raise ValueError(f"unknown metric {name!r} in objective {spec!r}")
        if available is not None and name not in available:
            raise ValueError(f"objective uses {name!r}, which these curves do not have "
                             f"(they have {list(available)})")
        parts.append((name, float(w) if w.strip() else 1.0))
    total = sum(w for _, w in parts)
    if not parts or total <= 0:
        raise ValueError(f"empty or non-positive objective {spec!r}")
    return [(n, w / total) for n, w in parts]


def describe(spec):
    return "+".join(n if abs(w - 1) < 1e-12 else f"{w:g}*{n}" for n, w in spec)


def isotonic(y):
    """Least-squares non-decreasing fit along the last axis (pool adjacent
    violators). Expected quality of a nested prefix does not fall when more of
    it arrives; a raw curve can, by channel noise alone, and an oracle given raw
    numbers would 'gain' by harvesting that noise."""
    y = np.asarray(y, dtype=np.float64)
    flat = y.reshape(-1, y.shape[-1])
    out = np.empty_like(flat)
    for r, row in enumerate(flat):
        vals, wts, cnt = [], [], []
        for v in row:
            vals.append(float(v))
            wts.append(1.0)
            cnt.append(1)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                w = wts[-2] + wts[-1]
                vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w]
                wts[-2:] = [w]
                cnt[-2:] = [cnt[-2] + cnt[-1]]
        out[r] = np.repeat(vals, cnt)
    return out.reshape(y.shape)


def metric_scales(v):
    """(M,): the typical spread of each metric over the budget range,
    mean_i |v_i(K) - v_i(1)|. Dividing by it puts dB and LPIPS on one axis."""
    s = np.nanmean(np.abs(v[:, -1, :] - v[:, 0, :]), axis=0)
    return np.where(np.isfinite(s) & (s > 1e-12), s, 1.0)


def objective(v, metrics, spec, scales):
    """(N, K, M) -> (N, K): sum_j w_j * sign_j * v_j / scale_j."""
    metrics = list(metrics)
    q = np.zeros(v.shape[:2])
    for name, w in spec:
        j = metrics.index(name)
        q += w * SIGN[name] * v[:, :, j] / scales[j]
    return q


# -- the allocation -----------------------------------------------------------------------

def segments(q, cost, units):
    """Hull segments of every image, in the order the greedy takes them.

    Returns (img, k_from, k_to, slope) arrays sorted by decreasing slope. Every
    hull segment is listed, including flat and falling ones, so that a target
    budget can always be spent in full."""
    N, K = q.shape
    units = np.asarray(units, dtype=np.float64)
    rows = []
    for i in range(N):
        x, y = cost[i] * units, q[i]
        hull = []
        for k in range(K):
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                if (y[b] - y[a]) * (x[k] - x[a]) <= (y[k] - y[a]) * (x[b] - x[a]):
                    hull.pop()      # b lies on or under the chord a -> k
                else:
                    break
            hull.append(k)
        for n, (a, b) in enumerate(zip(hull[:-1], hull[1:])):
            rows.append((-(y[b] - y[a]) / (x[b] - x[a]), i, n, a, b))
    rows.sort()                     # decreasing slope; per-image order kept on ties
    if not rows:
        e = np.zeros(0, dtype=np.int64)
        return e, e, e, np.zeros(0)
    arr = np.array(rows)
    return (arr[:, 1].astype(np.int64), arr[:, 3].astype(np.int64),
            arr[:, 4].astype(np.int64), -arr[:, 0])


def allocate(segs, cost, units, target):
    """Per-image budgets that spend `target` units on average (cost-weighted).

    Returns (weights (N, K), average units actually spent): row i holds the
    probability of each candidate for image i -- one-hot except for the single
    time-shared image."""
    img, ka, kb, _ = segs
    N, K = len(cost), len(units)
    units = np.asarray(units, dtype=np.float64)
    choice = np.zeros(N, dtype=np.int64)
    budget = float(target) * float(cost.sum())
    spent = float((cost * units[0]).sum())
    partial = None
    for i, a, b in zip(img, ka, kb):
        step = float(cost[i] * (units[b] - units[a]))
        if spent + step <= budget + 1e-9 * max(budget, 1.0):
            choice[i], spent = b, spent + step
        else:
            if budget > spent:
                partial = (i, a, b, (budget - spent) / step)
                spent = budget
            break
    w = np.zeros((N, K))
    w[np.arange(N), choice] = 1.0
    if partial is not None:
        i, a, b, alpha = partial
        w[i, a], w[i, b] = 1.0 - alpha, alpha
    return w, spent / float(cost.sum())


def uniform_weights(N, units, target):
    """Every image at the same budget (time-shared between neighbours when the
    target falls between two candidates)."""
    units = np.asarray(units, dtype=np.float64)
    K = len(units)
    w = np.zeros((N, K))
    t = float(np.clip(target, units[0], units[-1]))
    k = int(np.searchsorted(units, t))
    if k < K and abs(units[k] - t) < 1e-9:
        w[:, k] = 1.0
    else:
        alpha = (t - units[k - 1]) / (units[k] - units[k - 1])
        w[:, k - 1], w[:, k] = 1.0 - alpha, alpha
    return w


def realised(v, w):
    """Expected per-image metric values (N, M) under allocation weights (N, K)."""
    return np.einsum("nk,nkm->nm", w, v)


def path(segs, v, cost, units, signs):
    """Average units (P,) and signed average metrics (P, M) along the greedy
    path, one point per segment taken; linear in between (time-sharing)."""
    img, ka, kb, _ = segs
    N = v.shape[0]
    units = np.asarray(units, dtype=np.float64)
    avg = [float((cost * units[0]).sum() / cost.sum())]
    qual = [(signs * v[:, 0, :]).mean(0)]
    for i, a, b in zip(img, ka, kb):
        avg.append(avg[-1] + float(cost[i] * (units[b] - units[a]) / cost.sum()))
        qual.append(qual[-1] + signs * (v[i, b] - v[i, a]) / N)
    return np.array(avg), np.array(qual)


def units_to_reach(avg, qual, goal, at):
    """Average units at which a path (avg increasing, signed quality) matches
    `goal`, judged around the target `at`:

      at or above the goal at `at` -> the least u <= at from which it stays at or
                                      above the goal up to `at` (so a noisy
                                      stretch above `at` cannot move it)
      below the goal at `at`       -> the first u > at where it gets there
                                      (NaN if it never does)"""
    avg, qual = np.asarray(avg, dtype=np.float64), np.asarray(qual, dtype=np.float64)
    q_at = float(np.interp(at, avg, qual))
    # a path evaluated at its own goal (uniform against itself) must land in the
    # first branch whatever the rounding of the two ways the goal is computed
    goal = float(goal) - 1e-9 * max(1.0, abs(float(goal)))

    def cross(u0, q0, u1, q1):
        f = 0.0 if q1 == q0 else (goal - q0) / (q1 - q0)
        return float(u0 + f * (u1 - u0))

    if q_at >= goal:
        m = avg < at
        u = np.append(avg[m], at)
        q = np.append(qual[m], q_at)
        below = np.nonzero(q < goal)[0]
        if len(below) == 0:
            return float(u[0])
        p = int(below[-1])
        return cross(u[p], q[p], u[p + 1], q[p + 1])
    m = avg > at
    u = np.insert(avg[m], 0, at)
    q = np.insert(qual[m], 0, q_at)
    for p in range(1, len(u)):
        if q[p] >= goal:
            return cross(u[p - 1], q[p - 1], u[p], q[p])
    return float("nan")


# -- statistics ------------------------------------------------------------------------------

def bootstrap_ci(d, n=2000, seed=0, level=0.95):
    """Percentile CI of the mean of paired per-image differences."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def _ranks(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(len(a), dtype=np.float64)
    for val in np.unique(a):                 # average ranks for ties
        m = a == val
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# -- one comparison: uniform vs selectors, cross-fitted over channel draws ----------------

def compare(draws, cost, units, metrics, selectors, targets, scales, primary, folds=2):
    """Uniform allocation vs each selector at every target average budget.

    draws      (N, D, K, M) per-image metric values, one slice per noise draw
    selectors  {name: q (N, K)} fixed objectives that never saw these draws
               (a predictor's output, the transmitter's noiseless curve), or
               {name: ("oracle", spec)} to select on the measurements
               themselves. The oracle selects on one half of the draws
               (isotonic-smoothed) and is scored on the other half, then the
               halves swap; with a single draw it selects and scores on the
               same numbers and is optimistic (flagged by `crossfit` False)
    primary    metric whose equal-quality bandwidth saving is reported: 1 - (average
               budget at which the method first reaches uniform's quality at the
               target) / (the least uniform budget that reaches it). A method that
               never reaches it inside the swept range is charged the full range
               (a non-positive saving, flagged by "reached" < 1), never skipped:
               skipping would keep only the lucky SNR columns of a flat, noisy curve

    Returns {"uniform": {t: (N, M)}, name: {t: {"values": (N, M),
    "units": (N,), "avg": float, "saving": float, "reached": float}},
    "crossfit": bool}
    """
    metrics = list(metrics)
    units = np.asarray(units, dtype=np.float64)
    N, D = draws.shape[:2]
    signs = np.array([SIGN[m] for m in metrics])
    jp = metrics.index(primary)
    crossfit = D >= 2
    groups = [list(range(f, D, folds)) for f in range(min(folds, D))] if crossfit else [[0]]
    out = {"uniform": {t: 0.0 for t in targets}, "crossfit": crossfit}
    for name in selectors:
        out[name] = {t: {"values": 0.0, "units": 0.0, "avg": 0.0, "saving": 0.0, "reached": 0.0}
                     for t in targets}
    for g in groups:
        rest = [d for d in range(D) if d not in g] if crossfit else g
        ev = np.nanmean(draws[:, g], axis=1)
        sel = np.nanmean(draws[:, rest], axis=1)
        share = 1.0 / len(groups)
        for t in targets:
            out["uniform"][t] = out["uniform"][t] + share * realised(ev, uniform_weights(N, units, t))
        for name, how in selectors.items():
            if isinstance(how, tuple) and how[0] == "oracle":
                q = isotonic(objective(sel, metrics, how[1], scales))
            else:
                q = np.asarray(how, dtype=np.float64)
            segs = segments(q, cost, units)
            avg_path, qual_path = path(segs, ev, cost, units, signs)
            uni_path = (signs[jp] * ev[:, :, jp]).mean(0)          # uniform, budget by budget
            for t in targets:
                w, spent = allocate(segs, cost, units, t)
                goal = float((signs[jp] * realised(ev, uniform_weights(N, units, t))[:, jp]).mean())
                need = units_to_reach(avg_path, qual_path[:, jp], goal, t)
                # against the least uniform budget that already reaches the same
                # quality (== t unless uniform's own curve is flat below t)
                base = units_to_reach(units, uni_path, goal, t)
                reached = bool(np.isfinite(need))
                if not reached:
                    need = float(units[-1])     # charged the whole range: saving <= 0
                cell = out[name][t]
                cell["values"] = cell["values"] + share * realised(ev, w)
                cell["units"] = cell["units"] + share * (w @ units)
                cell["avg"] += share * spent
                cell["saving"] += share * (1.0 - need / base if base > 0 else float("nan"))
                cell["reached"] += share * float(reached)
    return out
