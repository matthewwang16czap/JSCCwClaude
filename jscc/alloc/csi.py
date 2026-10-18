"""Outdated channel reports for the scheduling tools (tools/schedule_sim.py --cqi-corr,
tools/rmab_scale.py, alloc/whittle.Arm(cqi_corr=...)).

A slot's channel is h ~ CN(0, 1) (Rayleigh block fading, |h|^2 ~ Exp(1)). The transmitter
is told a report of it taken earlier, the standard outdated-CSI model

    h_rep = r h + sqrt(1 - r^2) e,   e ~ CN(0, 1) independent of h,

with r = |J0(2 pi f_D tau)| for Doppler f_D and report age tau (Jakes; only |r| matters, the
power correlation is r^2): r = 1 is a fresh report, r = 0 one that says nothing about the slot. Given the report, h ~ CN(r h_rep, 1 - r^2)
(the one-step Kalman predictor of a Gauss-Markov channel), which is all a transmitter that
knows its reports are stale can use:

    outdated(rng, fade_db, r)   report fades (dB) of the true fades (dB); the true channel is
                                not redrawn, so the same frames serve every r
    posterior(rep_db, r)        deterministic equal-weight samples of the true |h|^2 given
                                report fades (dB)
    equivalent_snr(...)         a JSCC transmitter's decision SNR: the SNR whose noise term
                                phi(noise(.)) is the posterior mean of the slot's. The additive
                                model's MSE is linear in phi, so this is the slot's expected
                                distortion, and its history tracks E[Y | reports]
    goodput_snr(...)            a digital transmitter's decision SNR: the rate of largest
                                expected goodput under the posterior (the best any link
                                adaptation can do with the report; no backoff on top)
    kernel(ng, r)               P[i, j] = P(true gain in fading bin j | report in bin i) on
                                alloc/whittle.Arm's equiprobable Exp(1) bins
"""

import numpy as np

from alloc.utility import noise, phi


def outdated(rng, fade_db, r):
    """Report fades (dB) of the true fades `fade_db` (dB), correlation r in [0, 1]. Only the
    magnitude of h matters: e is circular, so |r h + s e| given |h| has the law of
    |r |h| + s e|."""
    fade_db = np.asarray(fade_db, dtype=np.float64)
    if r >= 1.0:
        return fade_db.copy()
    a = np.sqrt(10.0 ** (fade_db / 10.0))
    e = (rng.standard_normal(fade_db.shape) + 1j * rng.standard_normal(fade_db.shape)) / np.sqrt(2.0)
    return 10.0 * np.log10(np.maximum(np.abs(r * a + np.sqrt(1.0 - r * r) * e) ** 2, 1e-12))


def unit_noise(nr=32, na=16):
    """Deterministic equal-weight samples of e ~ CN(0, 1): |e|^2 at the conditional means of
    nr equiprobable cells of Exp(1) (so E|e|^2 = 1 exactly), times na angles."""
    u = np.arange(nr + 1) / nr
    F = (1.0 - u) * (1.0 - np.log(np.maximum(1.0 - u, 1e-300)))   # int_u^1 -ln(1 - v) dv
    F[-1] = 0.0
    rad = np.sqrt((F[:-1] - F[1:]) * nr)
    ang = 2.0 * np.pi * (np.arange(na) + 0.5) / na
    return (rad[:, None] * np.exp(1j * ang)[None, :]).ravel()


def posterior(rep_db, r, nr=32, na=16):
    """Equal-weight samples (..., nr * na) of the true |h|^2 given report fades rep_db (dB)."""
    a = np.sqrt(10.0 ** (np.asarray(rep_db, dtype=np.float64) / 10.0))
    s = np.sqrt(max(1.0 - r * r, 0.0))
    return np.abs(r * a[..., None] + s * unit_noise(nr, na)) ** 2


def _true_snr(rep_db, mean_db, r, clip):
    """Samples (..., S) of the slot's true SNR (dB, clipped) given reports; mean_db
    broadcasts against rep_db."""
    post = posterior(rep_db, r)
    mean = np.asarray(mean_db, dtype=np.float64)[..., None]
    return np.clip(mean + 10.0 * np.log10(np.maximum(post, 1e-12)), clip[0], clip[1])


def equivalent_snr(rep_db, mean_db, kappa, r, clip):
    """rep_db (K, T) report fades, mean_db (K,), kappa (K,) or scalar: the decision SNRs (K, T)
    of a JSCC transmitter that knows its reports have correlation r with the slots."""
    g = _true_snr(rep_db, np.asarray(mean_db, dtype=np.float64)[:, None], r, clip)   # (K, T, S)
    k = np.broadcast_to(np.asarray(kappa, dtype=np.float64), (g.shape[0],))[:, None]
    f = phi(noise(g), k[..., None]).mean(-1)                                           # (K, T)
    n_eq = f / np.maximum(1.0 - f / k, 1e-12)
    return np.clip(-10.0 * np.log10(np.maximum(n_eq, 1e-30)), clip[0], clip[1])


def goodput_snr(rep_db, mean_db, efficiency, r, clip, step=0.25):
    """The decision SNRs (K, T) of a digital transmitter that knows its reports are stale: per
    slot, the SNR (on a step dB grid) whose rate has the largest expected goodput, a block
    being lost when the true SNR does not support its rate."""
    g = _true_snr(rep_db, np.asarray(mean_db, dtype=np.float64)[:, None], r, clip)   # (K, T, S)
    e_true = efficiency(g)
    cand = np.arange(clip[0], clip[1] + 1e-9, step)
    best_v = np.full(g.shape[:2], -1.0)
    best_s = np.full(g.shape[:2], cand[0])
    for c, ec in zip(cand, efficiency(cand)):
        v = ec * np.mean(ec <= e_true + 1e-12, axis=-1)
        up = v > best_v + 1e-12
        best_v = np.where(up, v, best_v)
        best_s = np.where(up, c, best_s)
    return best_s


def kernel(ng, r, sub=8):
    """P[i, j]: the probability that the slot's true gain |h|^2 falls in fading bin j when
    its report falls in bin i, bins = ng equiprobable cells of Exp(1) (alloc/whittle.Arm).
    The identity for r = 1, rows equal to the uniform law for r = 0."""
    if r >= 1.0:
        return np.eye(ng)
    edges = -np.log(1.0 - np.arange(ng) / ng)                            # lower bin edges
    u = (np.arange(ng)[:, None] + (np.arange(sub)[None, :] + 0.5) / sub) / ng
    post = posterior(10.0 * np.log10(-np.log(1.0 - u)), r)              # (ng, sub, S)
    j = np.clip(np.searchsorted(edges, post, side="right") - 1, 0, ng - 1)
    P = np.stack([np.bincount(j[i].ravel(), minlength=ng) for i in range(ng)]).astype(np.float64)
    return P / P.sum(1, keepdims=True)
