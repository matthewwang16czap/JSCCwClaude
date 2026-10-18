"""Continuous traffic (docs/PROBLEM.md, section 3.7): image requests arrive over time, each
with its own deadline, instead of a frame of K users that share one. CPU.

    python tools/stream_sim.py --utility results/utility_snrc4.json \\
        --belief results/belief_snrc4.json --target 30 --deadline 24 \\
        --rate 0.1 0.15 0.2 0.25 0.3 --slots 20000

The model. Requests arrive as a Poisson process, --rate per slot on average. Each brings one
image (drawn from the utility file's pool), its user's mean SNR (uniform over --mean-snr) and
a deadline: it may be served in the --deadline slots that start with the one it arrives in.
Every slot carries one phase of one request (--units B: of B requests, one phase each), in
prefix order. A request's slot SNR is its mean plus i.i.d. Rayleigh fading, clipped to
--snr-clip, known when the slot starts (CQI). A request leaves when its policy has sent what
it plans (a fixed length, or, stopping on the SNRs its phases had, once the model puts its
image at the target), when it holds all P phases, or at its deadline; its user then decodes
what it holds (--stay: it stays until its deadline and takes slots nobody else wants, as a
user of a frame does). Measured per request: whether its decoded image reaches the target,
its DELAY (the slots from its arrival to the end of the slot whose phase got it there), and
the phases it took. Every policy sees the same requests and the same slot SNRs (common random
numbers: a request's fades are drawn for its whole life when it arrives). Requests that arrive
in the first --warmup slots (the queue starts empty) or after the --slots that follow (the
end would drain) are simulated but not counted.

Policies, the frame policies of tools/schedule_sim.py per request:

    pf_len   content-blind link adaptation: the length the pool-average curve needs at the
             request's mean SNR (all P if it never gets there), PF timing
    padc     PADC's length for the image at its mean SNR. Admitted on arrival if every
             admitted request can still get its length by its deadline (earliest deadline
             first; the requests of one slot shortest first, as a frame's users are); PF
             timing. One not admitted, or out of reach at its mean SNR, gets only the slots
             no admitted request wants
    stop     PF timing; a request leaves once the model, from the SNRs its phases had, puts
             it at the target (online stopping)
    stop_adv stop with the advantage index for timing
    online   stop_adv plus admission every slot: the requests with the smallest remaining
             needs (phases at the mean SNR) that can all still make their deadlines are
             admitted; when their needs leave no slack before some deadline, only the
             admitted requests due by then compete, otherwise every request does. With one
             deadline for all (a frame) this is schedule_sim's online policy exactly
    whittle  (slow) each request's own program (alloc/whittle.Arm, horizon = the deadline,
             that of the nearest 0.5 dB mean SNR; exact for a nested utility), its Whittle
             index at the request's age; ties and the slots no request gains from go to PF.
             A request leaves once the model puts it at the target, like the stopping
             policies (schedule_sim's whittle keeps such a user, whose index the programs'
             grid can leave slightly positive: the one way --stay differs from a frame)

A slot that no rule of the policy claims goes, by the policy's own timing rule (PF for
whittle), to a request in the system with phases left (schedule_sim does the same).

Reported per deadline and arrival rate: the share of requests whose image reaches the target,
the delay of those that do (mean, 50th / 90th / 95th percentile), the share at the target
within D/4, D/2, 3D/4 and D slots of arriving (a CDF over all requests, failures never
counted), phases per request, the busy share of slots, the requests in the system, and
against --ref the difference in success and in mean delay with a 95% interval over batches of
arrival time. Then the capacity: per policy, the largest rate at which at least --success of
the requests reach the target (interpolated between the rates run).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from alloc.utility import load_utility  # noqa: E402
from tools.schedule_sim import need, rate  # noqa: E402

POLICIES = ("pf_len", "padc", "stop", "stop_adv", "online", "whittle")
DEFAULT = ("pf_len", "padc", "stop", "stop_adv", "online")
FIXED = ("pf_len", "padc")
ADV_TIMING = ("stop_adv", "online")


class Requests:
    """A request stream in order of arrival: arrival slot, image, mean SNR, deadline (the
    first slot it may no longer use) and `snr[r, a]`: its SNR at age a (clipped, dB)."""

    def __init__(self, arr, img, mean, fade, deadline, clip=(-2.0, 22.0)):
        self.arr = np.asarray(arr, dtype=np.int64)
        self.img = np.asarray(img, dtype=np.int64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.D = int(deadline)
        self.dl = self.arr + self.D
        fade = np.asarray(fade, dtype=np.float64).reshape(len(self.arr), -1)
        if fade.shape[1] < self.D:
            raise ValueError("every request needs a fade for each slot of its life")
        self.snr = np.clip(self.mean[:, None] + fade[:, :self.D], clip[0], clip[1])

    def __len__(self):
        return len(self.arr)


def draw_requests(n_images, rate_, slots, deadline, mean_snr, clip, seed, max_deadline=None):
    """Poisson arrivals (rate_ per slot) over `slots` slots. The same seed gives the same
    requests at any deadline up to max_deadline (their fades are drawn for that long)."""
    rng = np.random.default_rng(seed)
    arr = np.repeat(np.arange(slots), rng.poisson(rate_, slots))
    n = len(arr)
    img = rng.integers(n_images, size=n)
    mean = rng.uniform(mean_snr[0], mean_snr[1], n)
    fade = 10.0 * np.log10(rng.exponential(1.0, (n, int(max_deadline or deadline))))
    return Requests(arr, img, mean, fade, deadline, clip)


def admit(cands, rest, dl, t, units):
    """Admission by remaining need: `cands` (in deadline order) are taken smallest finite need
    first (ties: arrival), each kept if every admitted request can still get its need by its
    deadline (earliest deadline first, `units` per slot, one phase per request per slot).
    Returns the admitted requests and, if their needs leave less than `units` of slack before
    some deadline, the admitted ones due by the first such deadline (else None)."""
    if not cands:
        return [], None
    c = np.asarray(cands)
    cap = units * (dl[c] - t).astype(np.float64)          # slots up to each one's deadline
    cum = np.zeros(len(c))                                # admitted need due by then
    pos = {r: j for j, r in enumerate(cands)}
    adm = []
    for r in sorted((r for r in cands if np.isfinite(rest[r])), key=lambda r: (rest[r], r)):
        j, n = pos[r], rest[r]
        if n <= dl[r] - t and np.all(cum[j:] + n <= cap[j:] + 1e-9):
            cum[j:] += n
            adm.append(r)
    if not adm:
        return adm, None
    tight = np.flatnonzero(cap - cum < units - 1e-9)
    if len(tight) == 0:
        return adm, None
    due = dl[c[tight[0]]]
    return adm, [r for r in adm if dl[r] <= due]


_ARMS = {}


def arm_for(belief, i, mean, D, goal, clip):
    """The whittle policy's program for a request (cached by image and 0.5 dB of mean SNR)."""
    from alloc.whittle import Arm
    m = round(2.0 * float(mean)) / 2.0
    key = (id(belief), int(i), m, int(D), float(goal), tuple(clip))
    if key not in _ARMS:
        _ARMS[key] = Arm(belief, int(i), m, int(D), target=goal, clip=clip)
    return _ARMS[key]


def run_stream(util, belief, req, policy, target, margin=0.0, offset=0.0, units=1, stay=False,
               pf_window=None, t_end=None):
    """One policy on one request stream. Decisions use `belief`, the decoded quality `util`.
    Returns a dict of per-request arrays (decoded PSNR, the slot its decoded image first
    reached the target or -1, departure slot, phases held, the slots it was served in) and
    per-slot arrays (units used, requests in the system)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy}")
    P, n, D = belief.P, len(req), req.D
    goal = target + margin
    tau = float(pf_window or D)
    plan = req.mean + offset
    hist = [[] for _ in range(n)]
    when = [[] for _ in range(n)]
    srv = np.zeros(n)                                     # PF: served-rate average
    tgt = np.zeros(n, dtype=np.int64)                     # fixed lengths (0: not admitted)
    rest = np.full(n, np.inf)                             # stopping: phases still needed
    ref = np.full(n, np.nan)                              # advantage: PSNR with a typical slot
    reach = np.full(n, -1, dtype=np.int64)
    dep = np.full(n, -1, dtype=np.int64)
    q_out = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)
    avg = belief.average() if policy == "pf_len" else None
    T_end = int(t_end) if t_end is not None else (int(req.dl.max()) if n else 0)
    busy = np.zeros(T_end, dtype=np.int64)
    in_sys = np.zeros(T_end, dtype=np.int64)
    active, nxt = [], 0

    def leave(r, t):
        dep[r] = t
        q_out[r] = util.psnr(int(req.img[r]), hist[r])

    for t in range(T_end):
        if active and req.dl[active[0]] <= t:             # deadlines (arrival order = deadline order)
            keep = []
            for r in active:
                if req.dl[r] <= t:
                    leave(r, t)
                else:
                    keep.append(r)
            active = keep
        new = []
        while nxt < n and req.arr[nxt] == t:
            new.append(nxt)
            nxt += 1
        if new:
            if policy == "pf_len":
                for r in new:
                    L = need(avg, 0, [], plan[r], goal)
                    tgt[r] = P if not np.isfinite(L) else int(L)
            elif policy == "padc":
                L = {r: need(belief, int(req.img[r]), [], plan[r], goal) for r in new}
                owed = sum(int(tgt[k]) - len(hist[k]) for k in active if tgt[k] > len(hist[k]))
                for r in sorted(new, key=lambda r: (L[r], r)):     # all due last: one test
                    if np.isfinite(L[r]) and L[r] <= D and owed + L[r] <= units * D:
                        tgt[r] = int(L[r])
                        owed += int(L[r])
            else:
                for r in new:
                    rest[r] = need(belief, int(req.img[r]), [], plan[r], goal)
            active += new
        in_sys[t] = len(active)
        has = [r for r in active if len(hist[r]) < P]
        if not has:
            continue
        g = {r: float(req.snr[r, t - req.arr[r]]) for r in active}
        if policy in FIXED:
            ok = [r for r in has if len(hist[r]) < tgt[r]]
        else:
            ok = [r for r in has if rest[r] > 0]
        mr = float(np.mean([rate(g[r]) for r in active]))

        def pf(r):
            return float(rate(g[r]) / (srv[r] + 1e-3 * mr))

        def adv(r):
            i = int(req.img[r])
            if not np.isfinite(ref[r]):
                ref[r] = belief.psnr(i, hist[r] + [float(plan[r])])
            return belief.psnr(i, hist[r] + [g[r]]) - ref[r]

        score = adv if policy in ADV_TIMING else pf
        by_score = lambda r: (-score(r), r)                   # noqa: E731
        by_pf = lambda r: (-pf(r), r)                         # noqa: E731
        stages = [(ok, by_score), (has, by_score)]            # its rule, then leftover slots
        if policy == "online":
            _, group = admit(ok, rest, req.dl, t, units)
            if group:
                stages.insert(0, (group, by_score))           # no slack: those due go first
        elif policy == "whittle":
            idx = {}
            for r in ok:
                arm = arm_for(belief, req.img[r], req.mean[r], D, goal, req_clip(req))
                nn, Y = arm.state(hist[r])
                idx[r] = arm.index(t - int(req.arr[r]), nn, Y, g[r])
            top = max(idx.values()) if idx else -np.inf
            if top > 1e-12:                                   # the top index, ties by PF
                stages = [([r for r in ok if idx[r] >= top], by_pf),
                          (ok, lambda r: (-idx[r], -pf(r), r)), (has, by_pf)]
            else:                                             # nobody gains: leftover (PF)
                stages = [(has, by_pf)]
        chosen = []
        for pool, key in stages:
            if len(chosen) >= units:
                break
            chosen += sorted((r for r in pool if r not in chosen), key=key)[:units - len(chosen)]
        busy[t] = len(chosen)
        for r in active:
            srv[r] *= 1.0 - 1.0 / tau
        for r in chosen:
            i = int(req.img[r])
            hist[r].append(g[r])
            when[r].append(t)
            srv[r] += float(rate(g[r])) / tau
            ref[r] = np.nan
            if reach[r] < 0 and util.psnr(i, hist[r]) >= target:
                reach[r] = t
            if policy in FIXED:
                fin = len(hist[r]) >= tgt[r] > 0
            else:
                rest[r] = need(belief, i, hist[r], plan[r], goal)
                fin = rest[r] == 0
            if fin or len(hist[r]) >= P:
                done[r] = True
        if not stay:
            gone = [r for r in chosen if done[r]]
            if gone:
                for r in gone:
                    leave(r, t + 1)
                active = [r for r in active if not (done[r] and dep[r] >= 0)]
    for r in active:
        leave(r, T_end)
    return {"psnr": q_out, "reach": reach, "dep": dep, "phases": np.array([len(h) for h in hist]),
            "when": when, "hist": hist, "busy": busy, "in_system": in_sys}


def req_clip(req):
    """The clip range a stream's SNRs were drawn with (for the whittle programs)."""
    return getattr(req, "clip", (-2.0, 22.0))


def stats(req, out, target, D, warmup, slots, units, batches):
    """The counted requests' measures, overall and per batch of arrival time."""
    a = req.arr
    m = (a >= warmup) & (a < warmup + slots)
    q, reach = out["psnr"][m], out["reach"][m]
    ok = q >= target - 1e-9
    delay = np.where(ok & (reach >= 0), reach - a[m] + 1, -1)
    d = delay[delay > 0]
    within = [float(np.mean((delay > 0) & (delay <= x))) for x in (D / 4, D / 2, 3 * D / 4, D)]
    b = np.minimum(((a[m] - warmup) * batches) // max(slots, 1), batches - 1)
    per_b = []
    for j in range(batches):
        s = b == j
        dj = delay[s & (delay > 0)]
        per_b.append((float(ok[s].mean()) if s.any() else np.nan,
                      float(dj.mean()) if len(dj) else np.nan))
    w = slice(warmup, warmup + slots)
    pct = (lambda p: float(np.percentile(d, p))) if len(d) else (lambda p: float("nan"))
    return {"_delay": delay, "_batch": b,
            "requests": int(m.sum()), "success": float(ok.mean()) if m.any() else float("nan"),
            "delay_mean": float(d.mean()) if len(d) else float("nan"),
            "delay_p50": pct(50), "delay_p90": pct(90), "delay_p95": pct(95), "within": within,
            "phases": float(out["phases"][m].mean()) if m.any() else float("nan"),
            "sojourn": float((out["dep"][m] - a[m]).mean()) if m.any() else float("nan"),
            "busy": float(out["busy"][w].sum()) / max(units * slots, 1),
            "in_system": float(out["in_system"][w].mean()), "batches": per_b}


def common_delay(s, r, batches):
    """Delay of policy s minus policy r on the requests BOTH bring to the target (the same
    requests: no selection effect), per batch: (mean, 95% half-width, requests)."""
    both = (s["_delay"] > 0) & (r["_delay"] > 0)
    diff = (s["_delay"] - r["_delay"]).astype(np.float64)
    per_b = [float(diff[both & (s["_batch"] == j)].mean()) if np.any(both & (s["_batch"] == j))
             else np.nan for j in range(batches)]
    m, h = paired(per_b, np.zeros(batches))
    return m, h, int(both.sum())


def paired(x, y):
    """Mean and 95% half-width of the per-batch differences x - y (NaN batches dropped)."""
    d = np.array(x, dtype=np.float64) - np.array(y, dtype=np.float64)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return float("nan"), float("nan")
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


def capacity(rates, success, level):
    """The largest rate with success >= level, interpolated: (rate, '') inside the grid,
    (max rate, '>=') if it holds at every rate, (min rate, '<') if at none."""
    rs, ss = np.asarray(rates, dtype=np.float64), np.asarray(success, dtype=np.float64)
    o = np.argsort(rs)
    rs, ss = rs[o], ss[o]
    if not ss[0] >= level:
        return float(rs[0]), "<"
    for j in range(1, len(rs)):
        if ss[j] < level:
            return float(rs[j - 1] + (ss[j - 1] - level) / (ss[j - 1] - ss[j]) * (rs[j] - rs[j - 1])), ""
    return float(rs[-1]), ">="


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--utility", required=True, help="tools/utility_fit.py output")
    ap.add_argument("--belief", default=None, help="the transmitter's curves (default: the truth)")
    ap.add_argument("--target", type=float, default=30.0, help="PSNR target (dB)")
    ap.add_argument("--margin", type=float, default=0.0, help="policies aim at target + margin")
    ap.add_argument("--plan-offset", type=float, default=0.0,
                    help="future phases planned at the mean SNR + this (dB)")
    ap.add_argument("--deadline", nargs="+", type=int, default=[24],
                    help="slots a request may be served in, from its arrival on")
    ap.add_argument("--rate", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4],
                    help="mean arrivals per slot (Poisson)")
    ap.add_argument("--slots", type=int, default=20000, help="the counted window of arrivals")
    ap.add_argument("--warmup", type=int, default=None, help="default: 4 x the longest deadline")
    ap.add_argument("--units", type=int, default=1, help="requests served per slot (one phase each)")
    ap.add_argument("--mean-snr", nargs=2, type=float, default=[0.0, 15.0])
    ap.add_argument("--snr-clip", nargs=2, type=float, default=[-2.0, 22.0])
    ap.add_argument("--policies", nargs="+", default=list(DEFAULT), choices=POLICIES)
    ap.add_argument("--ref", default=None, help="compared against (default: padc if run)")
    ap.add_argument("--success", nargs="+", type=float, default=[0.6, 0.7, 0.8],
                    help="capacity: shares of requests that must reach the target")
    ap.add_argument("--pf-window", type=float, default=None,
                    help="PF averaging window in slots (default: the deadline)")
    ap.add_argument("--stay", action="store_true",
                    help="requests stay until their deadline and take the spare slots")
    ap.add_argument("--batches", type=int, default=20, help="batches of arrival time for the CIs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    util = load_utility(a.utility)
    belief = load_utility(a.belief) if a.belief else util
    if getattr(util, "kind", "") == "digital":
        ap.error("stream_sim schedules token prefixes (a digital utility needs its own stopping)")
    if belief.names != util.names:
        ap.error("--belief must describe the same images, in the same order, as --utility")
    ref = a.ref or ("padc" if "padc" in a.policies else a.policies[0])
    if ref not in a.policies:
        ap.error("--ref must be one of --policies")
    Dmax = max(a.deadline)
    W = a.warmup if a.warmup is not None else 4 * Dmax
    span = W + a.slots + Dmax                             # arrivals keep the load to the end
    res = {}
    for D in a.deadline:
        res[D] = {}
        for lam in a.rate:
            req = draw_requests(len(util.names), lam, span, D, a.mean_snr, tuple(a.snr_clip),
                                [a.seed, int(round(lam * 1e6))], Dmax)
            req.clip = tuple(a.snr_clip)
            out = {}
            for p in a.policies:
                o = run_stream(util, belief, req, p, a.target, a.margin, a.plan_offset, a.units,
                               a.stay, a.pf_window, t_end=span + D)
                out[p] = stats(req, o, a.target, D, W, a.slots, a.units, a.batches)
            res[D][lam] = out
            report_rate(a, D, lam, out, ref)
        report_capacity(a, D, res[D])
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"args": vars(a), "ref": ref,
                       "results": {str(D): {str(lam): {p: {k: v for k, v in st.items()
                                                           if not k.startswith("_")}
                                                       for p, st in r.items()}
                                            for lam, r in byD.items()}
                                   for D, byD in res.items()},
                       "capacity": {str(D): {p: {str(lv): capacity(a.rate, [res[D][lam][p]["success"]
                                                                          for lam in a.rate], lv)
                                                 for lv in a.success}
                                             for p in a.policies} for D in a.deadline}}, f,
                      indent=1)
        print(f"wrote {a.out}")


def report_rate(a, D, lam, out, ref):
    n = out[ref]["requests"]
    print(f"\ndeadline {D} slots, {lam:g} requests per slot ({lam * D:.2f} per deadline window, "
          f"{a.units} unit{'s' if a.units > 1 else ''} per slot), target {a.target:g} dB, "
          f"{n} requests counted{', requests stay to their deadline' if a.stay else ''}")
    print(f"{'policy':9s} {'on target':>9s} {'delay':>6s} {'p50':>4s} {'p90':>4s} {'p95':>4s}   "
          f"{'within D/4 D/2 3D/4 D (%)':>26s} {'phases':>6s} {'busy':>5s} {'in sys':>6s}   "
          f"{'vs ' + ref + ': success [95% CI]':>28s}  {'delay, same requests [95% CI]':>32s}")
    for p, s in out.items():
        line = (f"{p:9s} {100 * s['success']:8.1f}% {s['delay_mean']:6.2f} {s['delay_p50']:4.0f} "
                f"{s['delay_p90']:4.0f} {s['delay_p95']:4.0f}   "
                + " ".join(f"{100 * w:5.1f}" for w in s["within"])
                + f"  {s['phases']:6.2f} {100 * s['busy']:4.0f}% {s['in_system']:6.2f}")
        if p != ref:
            ds, hs = paired([b[0] for b in s["batches"]], [b[0] for b in out[ref]["batches"]])
            dd, hd, nb = common_delay(s, out[ref], a.batches)
            line += (f"   {100 * ds:+12.1f} [{100 * (ds - hs):+.1f}, {100 * (ds + hs):+.1f}]"
                     f"  {dd:+6.2f} [{dd - hd:+.2f}, {dd + hd:+.2f}] on {nb}")
        print(line)


def report_capacity(a, D, by_rate):
    rates = sorted(by_rate)
    print(f"\ncapacity at deadline {D}: the largest rate (requests per slot) at which this share "
          f"of requests reaches {a.target:g} dB (interpolated; < / >=: outside the rates run)")
    print(f"{'policy':9s} " + " ".join(f"{100 * lv:>8g}%" for lv in a.success))
    for p in a.policies:
        cells = []
        for lv in a.success:
            c, flag = capacity(rates, [by_rate[r][p]["success"] for r in rates], lv)
            cells.append(f"{flag + format(c, '.3f'):>9s}")
        print(f"{p:9s} " + " ".join(cells))


if __name__ == "__main__":
    main()
