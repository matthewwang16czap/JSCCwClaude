"""Synthetic inputs for tools/paper_figs.py --self-test: every results/ file the figures read,
in the real tools' formats, built in about half a minute on one CPU core. The numbers are
made up (shaped like the real results), so the figures drawn from them check the drawing code
and the layout and are stamped as synthetic; they are never results.

How each format is made:
    tools/schedule_sim.py tables      its own report() / report_delay() on synthetic summaries
    tools/utility_fit.py abstractions the real tool, on synthetic mixed-SNR runs
    tools/mixed_snr.py runs           its make_profiles() / oat_table() on a synthetic code
    tools/codesign.py banks           the real build_bank(), synthetic utility, tiny frames
    decoded replays, rmab_scale.py and latency.py JSON: synthetic rows with the tools' keys

    python tools/paper_figs_selftest.py [DIR]          # only the inputs (default: a temp dir)
"""

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

GRID = [1.0, 4.0, 7.0, 10.0, 13.0]
CBRS = ["1/48", "1/24", "1/16", "1/12", "5/48", "1/8"]
PER, P = 256, 6                                   # tokens per phase, phases
MASTER = np.array([0.574, 0.227, 0.131, 0.046, 0.013, 0.009])
R = [1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.0]
S = [0.0, 1.0, 2.0, 3.0]
NUB = dict(utility="results/utility_snrc4_nested.json", belief="results/belief_snrc4_nested.json")


# ------------------------------------------------------------------ codes and utilities ------
def code(N, seed, gain_db=None, kind="piecewise"):
    """A synthetic code: (psnr(L, profiles) -> (N, n) array). piecewise: sensitivity-weighted
    MMSE noise (the additive model); constant: the mean noise variance, equal weights."""
    from alloc.utility import nested_rho, noise, phi
    rng = np.random.default_rng(seed)
    base = 10 ** (-np.array([28.7, 31.0, 32.6, 33.4, 33.5, 33.5]) / 10)
    dsrc = np.stack([base * 10 ** (-rng.uniform(-3, 3) / 10) for _ in range(N)])
    if gain_db is not None:
        dsrc = dsrc * 10 ** (-np.asarray(gain_db) / 10)
    csens = dsrc * rng.uniform(2.0, 4.0, (N, 1)) * np.linspace(1.0, 0.6, P)

    def psnr(L, prof):
        n = noise(np.asarray(prof, dtype=np.float64))                     # (n, L)
        if kind == "piecewise":
            F = phi(n, 1.9) @ nested_rho(MASTER, L)
            sd = 0.04
        else:                                       # ordered too, but its slots add as noise
            F = phi(n @ nested_rho(MASTER, L), 40.0)
            sd = 0.12
        D = dsrc[:, L - 1:L] + csens[:, L - 1:L] * F[None, :]
        return -10 * np.log10(D) + rng.normal(0.0, sd, D.shape)
    return psnr


def mixed_run(path, psnr, Ls, names, seed):
    """A tools/mixed_snr.py --design full --chunks phase run (format 2)."""
    from tools.mixed_snr import make_profiles, oat_table
    rng = np.random.default_rng(seed)
    levels, summary = [], {}
    for L in Ls:
        prof, kinds = make_profiles("full", L, GRID, 7.0, 10, rng)
        sc = psnr(L, prof)
        levels.append({"cbr": CBRS[L - 1], "units": L * PER,
                       "cuts": list(range(PER, L * PER, PER)), "profiles": prof, "kinds": kinds,
                       "psnr": sc.tolist()})
        if L > 1:
            summary[CBRS[L - 1]] = {"oat": oat_table(sc, prof, kinds, GRID, 7.0)}
    with open(path, "w") as f:
        json.dump({"format": 2, "design": "full", "chunking": "phase", "checkpoint": "synthetic",
                   "run_name": "synthetic", "images": names, "grid": GRID, "base_snr": 7.0,
                   "per_phase": PER, "levels": levels, "summary": summary}, f)


def compare(files):
    """tools/utility_fit.py --compare-abstractions, as it prints."""
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "utility_fit.py"), "--mixed",
                          *files, "--compare-abstractions"], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout


# -------------------------------------------------------------- schedule_sim.py tables ------
def args(**kw):
    a = dict(users=8, slots=24, frames=500, mean_snr=[0.0, 15.0], fading="rayleigh",
             fade_corr=0.0, plan="mean", length_bonus=None, fixed_codes=False, preempt=0.0,
             preempt_quantile=None, same_image=False, harq=None, harq_max=4, pred_noise=0.0,
             alpha=0.0, target=None, margin=0.0, plan_offset=0.0, cqi_aware=False, delay=False)
    a.update(NUB)
    a.update(kw)
    return SimpleNamespace(**a)


JSCC = SimpleNamespace(P=P, kind="token", psnr=lambda i, s: 10.0)
DIGITAL = {link: SimpleNamespace(P=P, kind="digital", link=link, mode="step")
           for link in ("cqi", "shannon")}


def mean_summary(gains, level, rng, failed=0.0):
    """schedule_sim's summary for a mean run: gains = {policy: dB over the first}."""
    base = "pf" if "pf" in gains else next(iter(gains))
    out = {}
    for p, g in gains.items():
        d, h = g - gains[base], 0.025 + 0.02 * rng.random()
        out[p] = {"mean_psnr": level + g, "p5_psnr": level - 4.6 + 1.2 * g,
                  "worst_user": level - 3.4 + g, "deep_fade_share": max(0.0, 0.03 - 0.04 * g),
                  "phases_per_user": 3.0, "served_snr_lift": 2.2 + 2.0 * max(g, 0.0),
                  "failed_share": failed, f"vs_{base}": d, f"vs_{base}_ci": [d - h, d + h]}
    return out


def target_summary(shares, rng, T=24, delay=None):
    """... for a target run: shares = {policy: share at the target}; delay = {policy: shares
    by T/4, T/2, 3T/4} (the share at T is the share)."""
    pols = list(shares)
    base = "pf" if "pf" in pols else pols[0]
    out = {}
    for p, s in shares.items():
        d, h = s - shares[base], 0.012 + 0.006 * rng.random()
        e = {"mean_psnr": 30.4 + 2.0 * (s - 0.6), "p5_psnr": 25.0, "worst_user": 26.0,
             "deep_fade_share": 0.02, "phases_per_user": 3.0, "served_snr_lift": 1.5,
             "failed_share": 0.0, "satisfied": s, f"vs_{base}": d, f"vs_{base}_ci": [d - h, d + h]}
        if "padc" in shares:
            d = s - shares["padc"]
            e["vs_padc"], e["vs_padc_ci"] = d, [d - h, d + h]
        out[p] = e
    if delay:
        refs = list(dict.fromkeys([base] + (["padc"] if "padc" in pols else [])))
        for p, by in delay.items():
            knots = np.interp(np.arange(1, T + 1), [0, T / 4, T / 2, 3 * T / 4, T],
                              [0.0] + list(by) + [shares[p]])
            reach = float(np.interp(0.5 * shares[p], [0.0] + list(by) + [shares[p]],
                                    [0, T / 4, T / 2, 3 * T / 4, T]))
            out[p].update({"first_slot": 6.0 + 2 * by[0], "time_avg_psnr": 25.0,
                           "at_target_by": knots.tolist(), "reach_slot": reach,
                           "reach_p90": float(T - 2), "reach_all": reach + 3.0})
            for ref in refs:
                d = reach - float(np.interp(0.5 * shares[ref], [0.0] + list(delay[ref])
                                            + [shares[ref]], [0, T / 4, T / 2, 3 * T / 4, T]))
                out[p].update({f"reach_all_vs_{ref}": d, f"reach_all_vs_{ref}_ci": [d - .3, d + .3],
                               f"same_users_vs_{ref}": d,
                               f"same_users_vs_{ref}_ci": [d - .4, d + .4],
                               f"same_users_vs_{ref}_n": 3000})
    return out


def sim_file(path, runs):
    """runs: [(args, util, summary, sigma, corr)] -> the text schedule_sim.py prints."""
    import tools.schedule_sim as ss
    with open(path, "a") as f, contextlib.redirect_stdout(f):
        for a, util, summary, sigma, corr in runs:
            digital = getattr(util, "kind", "") == "digital"
            base = "pf" if "pf" in summary else next(iter(summary))
            backoff = ({p: 1.5 for p in summary} if digital and (sigma or corr < 1) else 0.0)
            ss.report(a, util, digital, summary, base, sigma, backoff, None, corr)
            if a.delay:
                ss.report_delay(a, util, summary, base)


def write_sims(d, rng):
    J = lambda name: os.path.join(d, name)                                      # noqa: E731
    m = ("pf", "slope_adv", "pi", "whittle", "lagrange", "lagrange_relax")
    sim_file(J("s22_main_mean.txt"), [
        (args(), JSCC, mean_summary(dict(zip(m, (0, .47, -.48, .50, .52, .52))), 31.1, rng), 0, 1),
        (args(), JSCC, mean_summary(dict(zip(m, (0, .44, -.52, .47, .49, .49))), 30.9, rng), 2, 1),
        (args(cqi_aware=True), JSCC,
         mean_summary(dict(zip(m, (0, .45, -.55, .47, .49, .49))), 30.6, rng), 0, 0.8)])
    m4 = ("pf", "slope_adv", "whittle", "lagrange")
    sim_file(J("s21_lag_mean_more.txt"), [
        (args(preempt=0.25), JSCC, mean_summary(dict(zip(m4, (0, .45, .46, .48))), 29.9, rng),
         0, 1),
        (args(cqi_aware=True), JSCC, mean_summary(dict(zip(m4, (0, .45, .47, .49))), 30.6, rng),
         0, 0.8)])
    t5 = ("padc", "online", "plan_adv", "whittle", "lagrange")
    sim_file(J("s21_lag_t16.txt"), [
        (args(slots=16), JSCC, mean_summary(dict(zip(m4, (0, .40, .43, .45))), 29.9, rng), 0, 1),
        (args(slots=16, target=30.0, delay=True), JSCC,
         target_summary(dict(zip(t5, (.55, .62, .625, .645, .64))), rng, 16,
                        dict(zip(t5, ([.03, .2, .42], [.02, .15, .4], [.03, .17, .42],
                                      [.2, .45, .6], [.12, .38, .58])))), 0, 1)])
    sim_file(J("s21_lag_tgt_more.txt"), [
        (args(preempt=0.25, target=30.0), JSCC,
         target_summary(dict(zip(("padc", "online", "replan_adv", "whittle", "lagrange"),
                                 (.52, .60, .605, .62, .61))), rng), 0, 1),
        (args(target=30.0, margin=0.5, cqi_aware=True), JSCC,
         target_summary(dict(zip(("padc", "plan_adv", "whittle", "lagrange"),
                                 (.55, .65, .60, .59))), rng), 0, 0.8)])
    t7 = ("padc", "online", "plan_adv", "whittle", "lagrange", "whittle_soft", "lagrange_soft")
    sim_file(J("s24_main_soft.txt"), [
        (args(target=30.0, delay=True), JSCC,
         target_summary(dict(zip(t7, (.656, .75, .756, .75, .767, .772, .783))), rng, 24,
                        dict(zip(t7, ([.05, .30, .55], [.03, .20, .50], [.04, .25, .55],
                                      [.25, .55, .72], [.15, .45, .70], [.24, .56, .74],
                                      [.14, .44, .71])))), 0, 1),
        (args(target=30.0), JSCC,
         target_summary(dict(zip(t7, (.62, .65, .703, .64, .636, .66, .681))), rng), 2, 1)])
    t6 = ("padc", "plan_adv", "whittle", "lagrange", "whittle_soft", "lagrange_soft")
    sim_file(J("s24_main_soft_r80.txt"), [
        (args(target=30.0, margin=0.5, cqi_aware=True), JSCC,
         target_summary(dict(zip(t6, (.55, .647, .60, .59, .61, .621))), rng), 0, 0.8)])
    t5s = ("padc", "online", "plan_adv", "whittle_soft", "lagrange_soft")
    sim_file(J("s26_tgt_lost.txt"), [
        (args(preempt=0.25, target=30.0), JSCC,
         target_summary(dict(zip(("padc", "online", "replan_adv", "whittle_soft",
                                  "lagrange_soft"), (.52, .60, .605, .63, .625))), rng), 0, 1)])
    sim_file(J("s26_tgt_t16.txt"), [
        (args(slots=16, target=30.0, delay=True), JSCC,
         target_summary(dict(zip(t5s, (.55, .62, .625, .655, .66))), rng, 16,
                        dict(zip(t5s, ([.03, .2, .42], [.02, .15, .4], [.03, .17, .42],
                                       [.2, .45, .6], [.12, .38, .58])))), 0, 1)])
    sim_file(J("s26_tgt_clic.txt"), [
        (args(target=30.0, utility="results/utility_clic_nested.json",
              belief="results/belief_clic_nested.json"), JSCC,
         target_summary(dict(zip(("pf",) + t5s, (.50, .58, .64, .645, .66, .67))), rng), 0, 1)])

    # C5: the single decoder, the per-length decoders, digital with HARQ
    sm_r = np.array([31.42, 31.30, 31.17, 30.90, 30.42, 30.09, 29.84])
    st_r = np.array([74.7, 73.3, 71.7, 68.3, 61.9, 56.9, 55.3]) / 100
    sm_s, st_s = np.array([31.42, 31.39, 31.27, 31.13]), np.array([74.7, 74.6, 73.4, 70.9]) / 100

    def jscc_runs(util_path, best_m_r, best_t_r, best_m_s, best_t_s, mean_p, tgt_p, naive=0.05):
        """(the runs over r, the runs over the CQI error), as COMMANDS 13, 18 and 23 ran them."""
        out, by_s = [], []
        for aware in (False, True):
            for r, b in zip(R, best_m_r):
                g = dict(zip(mean_p, np.linspace(-0.47, 0.0, len(mean_p))))
                b = b - (naive if not aware and r < 1 else 0.0)
                out.append((args(utility=util_path, frames=2000, cqi_aware=aware), JSCC,
                            mean_summary(g, b, rng), 0, r))
        for aware in (False, True):
            for r, b in zip(R, best_t_r):
                b = b - (0.04 if not aware and r < 1 else 0.0)
                out.append((args(utility=util_path, frames=2000, target=30.0, cqi_aware=aware),
                            JSCC, target_summary(dict(zip(tgt_p, np.linspace(b - .12, b,
                                                                             len(tgt_p)))), rng),
                            0, r))
        for s, b in zip(S, best_m_s):
            g = dict(zip(mean_p, np.linspace(-0.47, 0.0, len(mean_p))))
            by_s.append((args(utility=util_path, frames=2000), JSCC, mean_summary(g, b, rng), s, 1))
        for s, b in zip(S, best_t_s):
            by_s.append((args(utility=util_path, frames=2000, target=30.0), JSCC,
                         target_summary(dict(zip(tgt_p, np.linspace(b - .1, b, len(tgt_p)))),
                                        rng), s, 1))
        return out, by_s

    single_r, single_s = jscc_runs(
        NUB["utility"], sm_r, st_r, sm_s, st_s, ("pf", "fixed_uni", "fixed_slope", "slope_adv"),
        ("pf", "padc", "online", "plan_adv", "replan_adv"))
    sim_file(J("s23_c5_single.txt"), single_r + single_s)
    hd_r, hd_s = jscc_runs("results/utility_hd.json", sm_r + .19, st_r + .02, sm_s + .19,
                           st_s + .02, ("rr", "pf", "fixed_uni", "fixed_slope", "slope_adv"),
                           ("pf", "padc", "padc_adv", "online", "plan_adv", "replan_adv"))
    sim_file(J("s18_stale_jscc.txt"), hd_r)
    _, old_s = jscc_runs("results/utility_snrc4.json", sm_r, st_r, sm_s - .02, st_s,
                         ("pf", "fixed_uni", "slope_adv"), ("online", "plan_adv", "replan_adv"))
    _, pc_s = jscc_runs("results/utility_percbr.json", sm_r, st_r, sm_s + .1, st_s + .03,
                        ("fixed_uni", "slope_adv"), ("padc_adv",))
    sim_file(J("s13_cqi_jscc.txt"), old_s + hd_s + pc_s)

    def dig_runs(link, harq, best_m, xs, axis, aware, target=False):
        out = []
        for x, b in zip(xs, best_m):
            sig, cor = (x, 1.0) if axis == "cqi" else (0.0, x)
            a = args(utility="results/rd_heic.json", frames=2000, harq=harq, cqi_aware=aware,
                     target=30.0 if target else None, belief=None)
            if target:
                summ = target_summary(dict(zip(("padc", "stop", "online_pf"),
                                               (b - .08, b - .03, b))), rng)
            else:
                summ = mean_summary({"pf": 0.0, "greedy": 0.25}, b - 0.25, rng, failed=0.05)
            out.append((a, DIGITAL[link], summ, sig, cor))
        return out

    heic_r = sm_r - np.array([0.13, 1.11, 1.35, 1.54, 1.49, 1.31, 1.09])
    gap_r = sm_r - np.array([-0.71, 0.65, 0.98, 1.19, 1.23, 1.13, 0.96])
    heic_s = sm_s - np.array([0.13, 0.71, 1.12, 1.40])
    gap_s = sm_s - np.array([-0.71, 0.24, 0.74, 1.09])
    sim_file(J("s20_harq_mean.txt"),
             dig_runs("cqi", "ir", heic_r - .04, R, "corr", False)
             + dig_runs("cqi", "ir", heic_r, R, "corr", True)
             + dig_runs("shannon", "ir", gap_r - .04, R, "corr", False)
             + dig_runs("shannon", "ir", gap_r, R, "corr", True)
             + dig_runs("cqi", "ir", heic_s, S, "cqi", False)
             + dig_runs("shannon", "ir", gap_s, S, "cqi", False)
             + dig_runs("cqi", "chase", heic_r[[3, 4]] - .3, [0.8, 0.6], "corr", False)
             + dig_runs("cqi", "chase", heic_r[[3, 4]] - .2, [0.8, 0.6], "corr", True))
    ht_r = st_r - np.array([0.3, 9.7, 12.2, 13.5, 12.1, 10.1, 8.3]) / 100
    ht_s = st_s - np.array([0.3, 6.2, 10.7, 12.8]) / 100
    sim_file(J("s20_harq_tgt.txt"), dig_runs("cqi", "ir", ht_r - .01, R, "corr", False, True))
    sim_file(J("s20_harq_tgt_aware.txt"), dig_runs("cqi", "ir", ht_r, R, "corr", True, True))
    sim_file(J("s20_harq_tgt_cqi.txt"), dig_runs("cqi", "ir", ht_s, S, "cqi", False, True))
    return m, ("pf", "padc", "online", "plan_adv", "whittle", "lagrange", "lagrange_relax")


# ------------------------------------------------------------ rmab_scale.py, latency.py ------
def rmab_file(path, gaps, target, users, rng, **extra):
    """gaps = {policy: per-K gaps (dB, or shares with a target)}."""
    per = {}
    for j, K in enumerate(users):
        bound = (31.9 if target is None else 0.80) + 0.002 * j
        row = {"B": K // 8, "bound": bound, "sure_share": 0.93, "mid_share": 0.07}
        for p, g in gaps.items():
            h = (0.02 if target is None else 0.004) / np.sqrt(K / 8) * (1 + 0.2 * rng.random())
            row[p], row[p + "_gap"], row[p + "_gap_ci"] = bound - g[j], g[j], [g[j] - h, g[j] + h]
        per[str(K)] = row
    a = dict(utility=NUB["utility"], belief=NUB["belief"], users=list(users), frames=100,
             slots=24, target=target, margin=0.0, soft_sigma="auto", policies=list(gaps))
    a.update(extra)
    with open(path, "w") as f:
        json.dump({"args": a, "per_K": per}, f)


def write_rmab(d, rng):
    J = lambda name: os.path.join(d, name)                                      # noqa: E731
    K5, K3 = [8, 16, 32, 64, 128], [8, 32, 128]
    rmab_file(J("s22_scale_mean.json"), {
        "pf": [.52, .52, .51, .52, .52], "adv": [.27, .20, .17, .155, .15],
        "pi": [1.0, .98, .97, .96, .96], "whittle": [.27, .12, .06, .03, .016],
        "lagrange": [.21, .09, .04, .015, .005], "lagrange_relax": [.205, .088, .039, .015, .005]},
        None, K5, rng)
    hard = [.0075, .016, .016, .016, .016]
    truth = [.0084, -.001, -.003, .0001, -.0019]
    pick3 = lambda g: [g[0], g[2], g[4]]                                        # noqa: E731
    rmab_file(J("s22_gap_ties.json"), {"pf": [.15] * 5, "whittle": [.02, .022, .021, .02, .02],
                                       "lagrange": hard,
                                       "lagrange_relax": [x + .001 for x in hard]}, 30.0, K5, rng)
    rmab_file(J("s22_gap_prices.json"), {"pf": [.15] * 5, "whittle": [.02, .022, .021, .02, .02],
                                         "lagrange": [x + .0005 for x in hard],
                                         "lagrange_relax": [x + .001 for x in hard]}, 30.0, K5, rng)
    rmab_file(J("s22_gap_lpu.json"), {"lagrange": pick3(hard), "lagrange_relax": pick3(hard),
                                      "lagrange_lpu": [.009, .017, .0165]}, 30.0, K3, rng)
    rmab_file(J("s22_gap_grid.json"), {"pf": [.15] * 3, "lagrange": [.007, .0155, .0158],
                                       "lagrange_relax": [.0072, .016, .016]}, 30.0, K3, rng)
    rmab_file(J("s23_gap_truth.json"), {"pf": [.15] * 5, "whittle": [.012, .004, .003, .002, .002],
                                        "lagrange": truth}, 30.0, K5, rng, belief=None)
    rmab_file(J("s23_gap_m25.json"), {"pf": [.15] * 3, "whittle": [.025, .027, .027],
                                      "lagrange": [.02, .025, .026]}, 30.0, K3, rng, margin=0.25)
    rmab_file(J("s23_gap_m50.json"), {"pf": [.15] * 3, "whittle": [.033, .035, .036],
                                      "lagrange": [.03, .035, .036]}, 30.0, K3, rng, margin=0.5)
    rmab_file(J("s24_soft_auto.json"), {
        "pf": [.15] * 5, "whittle": [.02, .022, .021, .02, .02], "lagrange": hard,
        "whittle_soft": [.012, .005, .004, .003, .003], "lagrange_soft": truth}, 30.0, K5, rng)
    for s, fn in ((0.05, "s24_soft_s05.json"), (0.1, "s24_soft_s10.json"),
                  (0.2, "s24_soft_s20.json")):
        rmab_file(J(fn), {"pf": [.15] * 3, "lagrange": pick3(hard),
                          "lagrange_soft": [x + 0.004 * s for x in pick3(truth)]}, 30.0, K3, rng,
                  soft_sigma=s)
    sg, bs = [0.031, 0.035, 0.040, 0.043, 0.045, 0.047], [0.010, 0.008, 0.006, 0.005, 0.004, 0.004]
    with open(J("s24_soft_auto.txt"), "w") as f:                # rmab_scale.py's line, verbatim
        f.write("calibration-aware programs (*_soft): the curves' error per prefix length, sigma "
                + " ".join(f"{v:.3f}" for v in sg) + " dB, bias (truth - belief) "
                + " ".join(f"{v:+.3f}" for v in bs) + " dB (measured)\n")


def latency_file(path, curves, target, Ts, at=(16, 24), ref="lagrange"):
    from tools.latency import needed
    match = {}
    for T0 in at:
        level = curves[ref][Ts.index(T0)]
        rows = {}
        for p, c in curves.items():
            if p == ref:
                continue
            T, flag = needed(Ts, c, level)
            rows[p] = {"needed": T, "flag": flag, "ci": [T - 0.6, T + 0.6],
                       "saved": 1 - T0 / T if not flag else float("nan"),
                       "saved_ci": [1 - T0 / (T - .6), 1 - T0 / (T + .6)]}
        match[str(T0)] = {"level": level, "rows": rows}
    a = dict(utility=NUB["utility"], belief=NUB["belief"], users=8, slots=Ts, frames=300,
             target=target, ref=ref, at=list(at), policies=list(curves))
    with open(path, "w") as f:
        json.dump({"args": a, "slots": Ts, "curves": curves, "match": match, "levels": {}}, f)


def write_latency(d):
    Ts = [12, 16, 20, 24, 28, 32]
    q = lambda off: [float(31.0 + off + 2.0 * np.log2(T / 24)) for T in Ts]     # noqa: E731
    latency_file(os.path.join(d, "s21_latency_mean.json"),
                 {"pf": q(0.0), "slope_adv": q(0.47), "lagrange": q(0.52)}, None, Ts)
    latency_file(os.path.join(d, "s21_latency_tgt.json"),
                 {"padc": [.35, .47, .58, .66, .72, .77], "online": [.40, .55, .67, .75, .81, .85],
                  "whittle": [.45, .58, .69, .75, .80, .84],
                  "lagrange": [.44, .58, .69, .767, .82, .86]}, 30.0, Ts)
    latency_file(os.path.join(d, "s26_latency_tgt.json"),
                 {"padc": [.35, .47, .58, .66, .72, .77], "online": [.40, .55, .67, .75, .81, .85],
                  "whittle_soft": [.46, .59, .70, .77, .81, .85],
                  "lagrange_soft": [.45, .59, .71, .783, .83, .87]}, 30.0, Ts,
                 ref="lagrange_soft")


# ---------------------------------------------------------------- banks, decoded replays -----
def write_banks(d):
    from alloc.utility import Utility, nested_rho
    from tools.codesign import build_bank
    rng = np.random.default_rng(0)
    N = 24
    base = 10 ** (-np.array([28.7, 31.0, 32.6, 33.4, 33.5, 33.5]) / 10)
    dsrc = np.stack([base * 10 ** (-rng.uniform(-3, 3) / 10) for _ in range(N)])
    csens = dsrc * rng.uniform(2.0, 4.0, (N, 1)) * np.linspace(1.0, 0.6, P)
    u = Utility(1.9, [nested_rho(MASTER, L) for L in range(1, P + 1)], dsrc, csens,
                [f"kodim{i:02d}" for i in range(1, N + 1)], meta={"nested": MASTER.tolist()})
    b = u.perturbed(0.04, np.random.default_rng(11))
    for name, kw in (("bank_sa.json", {}), ("bank_sa_cqi95.json", {"cqi_corr": 0.95}),
                     ("bank_sa_cqi80.json", {"cqi_corr": 0.8}),
                     ("bank_sa_cqi60.json", {"cqi_corr": 0.6}),
                     ("bank_sa_cqi30.json", {"cqi_corr": 0.3}),
                     ("bank_sa_cqi0.json", {"cqi_corr": 0.0}),
                     ("bank_lift_k2.json", {"users": 2, "slots": 6, "frames": 200}),
                     ("bank_lift_k4.json", {"users": 4, "slots": 12, "frames": 100}),
                     ("bank_sa_k16.json", {"users": 16, "slots": 48, "frames": 24}),
                     ("bank_lift_k24.json", {"users": 24, "slots": 72, "frames": 16})):
        kw = dict({"users": 8, "slots": 24, "frames": 60}, **kw)
        with open(os.path.join(d, name), "w") as f:
            json.dump(build_bank(u, "slope_adv", belief=b, **kw), f)


def decoded_file(path, effects, rng, target=None, level=31.0, spread=2.4, frames=60, users=8):
    """A decoded replay (tools/mixed_snr.py --design file --replay-metrics msssim lpips)."""
    names = [f"kodim{i:02d}" for i in range(1, 25)]
    q0 = rng.normal(level, spread, (frames, users))
    img = rng.integers(len(names), size=(frames, users))
    mu = rng.uniform(0.0, 15.0, (frames, users))
    entries = []
    for p, eff in effects.items():
        for f in range(frames):
            for k in range(users):
                q = float(q0[f, k] + eff + rng.normal(0.0, 0.35))
                n = int(rng.integers(1, P))
                a = float(q + rng.normal(-0.01, 0.1))
                ms = 0.955 + 0.004 * (a - 31) + rng.normal(0, .002)
                lp = 0.18 - 0.012 * (a - 31) + rng.normal(0, .004)
                entries.append({"frame": f, "policy": p, "user": k, "image": names[img[f, k]],
                                "mean_snr": float(mu[f, k]),
                                "snrs": [float(s) for s in mu[f, k] + 2.5 + rng.normal(0, 2, n)],
                                "predicted_psnr": q, "actual_psnr": a,
                                "actual_msssim": float(ms), "actual_lpips": float(lp)})
    with open(path, "w") as f:
        json.dump({"chunking": "phase", "utility": NUB["utility"], "target": target,
                   "entries": entries, "decoded_by": {"checkpoint": "synthetic",
                                                      "run_name": "synthetic",
                                                      "metrics": ["psnr", "msssim", "lpips"]}}, f)


def write_decoded(d, rng):
    J = lambda name: os.path.join(d, name)                                      # noqa: E731
    mean = {"pf": 0.0, "slope_adv": 0.47, "pi": -0.5, "whittle": 0.5, "lagrange": 0.53,
            "lagrange_relax": 0.53}
    decoded_file(J("s22_dec_mean.json"), mean, rng)
    decoded_file(J("s22_dec_r80_mean.json"), {p: 0.95 * v for p, v in mean.items()}, rng,
                 level=30.6)
    decoded_file(J("s22_clic_dec_mean.json"), {p: v for p, v in mean.items() if p != "whittle"},
                 rng, level=31.6)
    tg = {"padc": 0.0, "online": 0.45, "plan_adv": 0.5, "whittle": 0.4, "lagrange": 0.55,
          "lagrange_relax": 0.55, "whittle_soft": 0.58, "lagrange_soft": 0.65}
    sel = lambda *ps: {p: tg[p] for p in ps}                                    # noqa: E731
    kw = dict(target=30.0, level=30.4, spread=1.6)
    decoded_file(J("s22_dec_tgt.json"), sel("padc", "online", "whittle", "lagrange",
                                            "lagrange_relax"), rng, **kw)
    decoded_file(J("s22_dec_r80_tgt.json"), sel("padc", "plan_adv", "whittle", "lagrange",
                                                "lagrange_relax"), rng, **kw)
    decoded_file(J("s22_clic_dec_tgt.json"), sel("padc", "online", "lagrange", "lagrange_relax"),
                 rng, **kw)
    decoded_file(J("s24_dec_auto.json"), sel("padc", "online", "whittle", "lagrange",
                                             "whittle_soft", "lagrange_soft"), rng, **kw)
    decoded_file(J("s24_dec_r80.json"), sel("padc", "plan_adv", "lagrange", "lagrange_soft"),
                 rng, **kw)
    decoded_file(J("s24_dec_s15.json"), sel("padc", "lagrange", "lagrange_soft"), rng, **kw)
    decoded_file(J("s24_dec_clic.json"), sel("padc", "online", "lagrange", "lagrange_soft"),
                 rng, **kw)


# ------------------------------------------------------------------------------ build --------
def build(d=None):
    """Write every input into d (default: a new temporary directory); return d."""
    d = d or tempfile.mkdtemp(prefix="paper_figs_selftest_")
    os.makedirs(d, exist_ok=True)
    for fn in os.listdir(d):                                   # the text files are appended to
        if fn.endswith(".txt"):
            os.remove(os.path.join(d, fn))
    J = lambda name: os.path.join(d, name)                                      # noqa: E731
    rng = np.random.default_rng(7)
    kodak = [f"kodim{i:02d}" for i in range(1, 25)]
    clic = [f"clic{i:03d}" for i in range(60)]
    allL = list(range(1, P + 1))
    mixed_run(J("p2_full_snrc4.json"), code(24, 1), allL, kodak, 1)
    mixed_run(J("p2_full_s1.json"), code(24, 1, gain_db=[0.3] * P, kind="constant"), allL, kodak,
              2)
    hd = code(24, 1, gain_db=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    pc = code(24, 1, gain_db=[0.15, 0.25, 0.30, 0.35, 0.38, 0.40])
    for L, tag in zip(allL, ("48", "24", "16", "12", "5-48", "8")):
        mixed_run(J(f"p4_full_h{tag}.json"), hd, [L], kodak, 10 + L)
        mixed_run(J(f"p3_full_fix{L}.json"), pc, [L], kodak, 20 + L)
    mixed_run(J("s22_clic_full_snrc4_a.json"), code(60, 3), allL, clic, 3)
    mixed_run(J("s22_clic_full_s1_a.json"), code(60, 3, kind="constant"), allL, clic, 4)
    with open(J("s13_abstractions.txt"), "w") as f:
        f.write(compare([J("p2_full_snrc4.json")]))
        f.write(compare([J("p2_full_s1.json")]))
        f.write(compare([J(f"p4_full_h{t}.json") for t in ("48", "24", "16", "12", "5-48", "8")]))
        f.write(compare([J(f"p3_full_fix{L}.json") for L in allL]))
    with open(J("s22_clic.txt"), "w") as f:
        f.write(compare([J("s22_clic_full_snrc4_a.json")]))
    m, t = write_sims(d, rng)
    clic_args = dict(utility="results/utility_clic_nested.json",
                     belief="results/belief_clic_nested.json")
    sim_file(J("s22_clic.txt"), [
        (args(**clic_args), JSCC, mean_summary(dict(zip(m, (0, .51, -.40, .60, .63, .63))), 31.6,
                                               rng), 0, 1),
        (args(target=30.0, **clic_args), JSCC,
         target_summary(dict(zip(t, (.50, .58, .64, .645, .65, .657, .655))), rng), 0, 1)])
    with open(J("s22_clic.txt"), "a") as f:
        f.write(compare([J("s22_clic_full_s1_a.json")]))
    write_rmab(d, rng)
    write_latency(d)
    write_banks(d)
    write_decoded(d, rng)
    return d


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else None))
