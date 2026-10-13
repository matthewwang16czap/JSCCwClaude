"""Additive noise-sensitivity model of a prefix token codec (docs/PROBLEM.md).

An image decoded from the first L phases of its prefix, phase j received at noise
variance n_j (a slot of SNR -10 log10 n_j after equalisation), has MSE

    D(L; n_1..n_L) = Dsrc(L) + C(L) * sum_j rho_L[j] * phi(n_j),   phi(n) = n / (1 + n / kappa)

    Dsrc(L)   the MSE of a noise-free prefix of L phases           (per image)
    C(L)      its noise sensitivity                                (per image)
    rho_L     the share of C(L) each phase carries, sums to 1     (pooled over images)
    kappa     the saturation of the noise response                (pooled)

Measured on the --snr-chunks 4 ViT (two chunks per prefix, Kodak means): <= 0.09 dB PSNR
error at 1/24, 1/12 and 1/8 with one shared kappa ~ 1.8, while decoders trained at one
SNR per image do not obey it (docs/NOTES.md). fit() estimates it from tools/mixed_snr.py
runs (tools/utility_fit.py); tools/schedule_sim.py schedules with it.
"""

import json

import numpy as np


def noise(snr_db):
    """Noise variance per complex symbol at unit signal power."""
    return 10.0 ** (-np.asarray(snr_db, dtype=np.float64) / 10.0)


def phi(n, kappa):
    n = np.asarray(n, dtype=np.float64)
    return n / (1.0 + n / kappa)


class Utility:
    """Per-image MSE / PSNR of a delivery history (one SNR per phase, in prefix order)."""

    def __init__(self, kappa, rho, dsrc, csens, names, d_empty=0.1, meta=None):
        k = np.asarray(kappa, dtype=np.float64)
        self.kappa = float(k) if k.ndim == 0 else k                   # pooled, or one per image
        self.rho = [np.asarray(r, dtype=np.float64) for r in rho]     # rho[L - 1]: (L,) or (N, L)
        self.dsrc = np.asarray(dsrc, dtype=np.float64)                # (N images, P phases)
        self.csens = np.asarray(csens, dtype=np.float64)              # (N, P)
        self.names = list(names)
        self.d_empty = float(d_empty)                                 # nothing received
        self.meta = dict(meta or {})
        self.P = self.dsrc.shape[1]
        if len(self.rho) != self.P or any(r.shape[-1] != L + 1 for L, r in enumerate(self.rho)):
            raise ValueError("rho must hold one share vector per prefix length 1..P")

    def mse(self, i, snrs_db):
        L = len(snrs_db)
        if L == 0:
            return self.d_empty
        if L > self.P:
            raise ValueError(f"{L} phases, the model covers {self.P}")
        kappa = self.kappa if np.ndim(self.kappa) == 0 else self.kappa[i]
        rho = self.rho[L - 1] if self.rho[L - 1].ndim == 1 else self.rho[L - 1][i]
        x = float(np.dot(rho, phi(noise(snrs_db), kappa)))
        return float(self.dsrc[i, L - 1] + self.csens[i, L - 1] * x)

    def psnr(self, i, snrs_db):
        return float(-10.0 * np.log10(max(self.mse(i, snrs_db), 1e-12)))

    def perturbed(self, sigma, rng):
        """The transmitter's belief: each image's Dsrc and C scaled by its own log-normal
        factors (a curve predictor that is off by ~sigma in log MSE)."""
        if sigma <= 0:
            return self
        N = len(self.names)
        fd = np.exp(sigma * rng.standard_normal((N, 1)))
        fc = np.exp(sigma * rng.standard_normal((N, 1)))
        return Utility(self.kappa, self.rho, self.dsrc * fd, self.csens * fc, self.names,
                       self.d_empty, self.meta)

    def shifted(self, bonus_db):
        """The same model with every prefix length L better by bonus_db[L - 1] dB at any
        constant SNR (Dsrc and C scaled together): one model per length (CBR) without the
        nesting penalty, if it decoded mixed-SNR slots like this one."""
        f = 10.0 ** (-np.asarray(bonus_db, dtype=np.float64).reshape(1, -1) / 10.0)
        if f.shape[1] != self.P:
            raise ValueError(f"{f.shape[1]} bonus values for {self.P} prefix lengths")
        return Utility(self.kappa, self.rho, self.dsrc * f, self.csens * f, self.names,
                       self.d_empty, self.meta)

    def average(self):
        """The content-blind transmitter's curve: geometric means over the images (its PSNR
        is about the pool's mean PSNR)."""
        g = lambda a: np.exp(np.log(np.maximum(a, 1e-12)).mean(0, keepdims=True))  # noqa: E731
        return Utility(np.mean(self.kappa), [r if r.ndim == 1 else r.mean(0) for r in self.rho],
                       g(self.dsrc), g(self.csens), ["average"], self.d_empty)

    def to_dict(self):
        return {"kappa": np.asarray(self.kappa).tolist(), "rho": [r.tolist() for r in self.rho],
                "dsrc": self.dsrc.tolist(), "csens": self.csens.tolist(), "images": self.names,
                "d_empty": self.d_empty, "meta": self.meta}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def from_dict(cls, d):
        return cls(d["kappa"], d["rho"], d["dsrc"], d["csens"], d["images"],
                   d.get("d_empty", 0.1), d.get("meta"))

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


def load_utility(path, **digital):
    """A utility file of either kind: this model (tools/utility_fit.py) or a digital
    baseline (tools/digital_rd.py, alloc/digital.py; `digital` overrides its link)."""
    with open(path) as f:
        d = json.load(f)
    if d.get("kind") == "digital":
        from alloc.digital import DigitalUtility
        return DigitalUtility.from_dict(d, **{k: v for k, v in digital.items() if v is not None})
    return Utility.from_dict(d)


# -- fitting --------------------------------------------------------------------------

def _fit_level(D, F, fit, iters=60, rho=None):
    """One prefix length: D (N, P) MSEs, F (P, k) phi of each profile's chunks, `fit` the
    profiles used for fitting. Returns (dsrc (N,), csens (N,), rho (k,)); a given `rho` is
    kept (only dsrc and csens are fitted)."""
    k = F.shape[1]
    fixed = rho is not None
    rho = np.full(k, 1.0 / k) if rho is None else np.asarray(rho, dtype=np.float64)
    Df, Ff = D[:, fit], F[fit]
    w2 = 1.0 / Df ** 2                                         # relative error ~ PSNR error
    for _ in range(iters):
        x = (Ff @ rho)[None, :]                                # (1, Pf)
        s0, s1, s2 = w2.sum(1), (w2 * x).sum(1), (w2 * x * x).sum(1)
        t0, t1 = (w2 * Df).sum(1), (w2 * x * Df).sum(1)
        det = s0 * s2 - s1 * s1                                # per image: D = a + c x
        safe = np.abs(det) > 1e-30
        c = np.where(safe, (s0 * t1 - s1 * t0) / np.where(safe, det, 1.0), 0.0)
        csens = np.maximum(c, 1e-12)
        dsrc = np.maximum((t0 - csens * s1) / s0, 0.0)
        if k == 1 or fixed:
            break
        # pooled rho: (D - dsrc) / c ~ F rho, weights c / D, subject to sum(rho) = 1
        Y = (Df - dsrc[:, None]) / csens[:, None]
        W = csens[:, None] / Df
        A = (Ff[None, :, :] * W[:, :, None]).reshape(-1, k)
        b = (Y * W).reshape(-1)
        M = np.zeros((k + 1, k + 1))
        M[:k, :k] = A.T @ A
        M[:k, k] = M[k, :k] = 1.0
        rhs = np.concatenate([A.T @ b, [1.0]])
        new = np.clip(np.linalg.lstsq(M, rhs, rcond=None)[0][:k], 0.0, None)
        new = new / new.sum() if new.sum() > 0 else np.full(k, 1.0 / k)
        if np.abs(new - rho).max() < 1e-10:
            rho = new
            break
        rho = new
    return dsrc, csens, rho


def nested_rho(master, L):
    """The NESTED sensitivity model: a prefix of L phases weighs its phases like the first L
    entries of one master profile (each phase's share relative to the others does not depend
    on how long the prefix is), so rho_L = master[:L] / sum(master[:L])."""
    m = np.asarray(master, dtype=np.float64)[:L]
    return m / m.sum()


def _fit_nested(Ds, Fs, masks, iters=40):
    """Phase levels (level l has l + 1 chunks) under one master profile: alternately every
    image's dsrc and csens per level, and the master by weighted least squares over all
    levels (the scale of each level is absorbed by its csens). Returns (master, params)."""
    P = max(F.shape[1] for F in Fs)
    longest = int(np.argmax([F.shape[1] for F in Fs]))
    master = _fit_level(Ds[longest], Fs[longest], masks[longest])[2]
    for _ in range(iters):
        params = [_fit_level(D, F, m, rho=nested_rho(master, F.shape[1]))
                  for D, F, m in zip(Ds, Fs, masks)]
        rows, rhs = [], []
        for (d, c, _), D, F, m in zip(params, Ds, Fs, masks):
            k, S = F.shape[1], master[:F.shape[1]].sum()
            Df, Ff = D[:, m], F[m]                             # D ~ d + (c / S) F master[:k]
            W = (c[:, None] / S) / Df                          # relative error weights
            A = np.zeros((Df.size, P))
            A[:, :k] = (W[:, :, None] * Ff[None, :, :]).reshape(-1, k)
            rows.append(A)
            rhs.append(((Df - d[:, None]) / Df).reshape(-1))
        A, b = np.concatenate(rows), np.concatenate(rhs)
        M = np.zeros((P + 1, P + 1))
        M[:P, :P] = A.T @ A
        M[:P, P] = M[P, :P] = 1.0
        new = np.clip(np.linalg.lstsq(M, np.concatenate([A.T @ b, [1.0]]), rcond=None)[0][:P],
                      1e-9, None)
        new = new / new.sum()
        done = np.abs(new - master).max() < 1e-9
        master = new
        if done:
            break
    params = [_fit_level(D, F, m, rho=nested_rho(master, F.shape[1]))
              for D, F, m in zip(Ds, Fs, masks)]
    return master, params


def predict_level(dsrc, csens, rho, F):
    """(N, P) predicted MSE of every profile."""
    return dsrc[:, None] + csens[:, None] * (F @ rho)[None, :]


def fit(levels, kappas=None, nested=False):
    """levels: [{"key": name, "profiles": (P, k) chunk SNRs, "kinds": [...], "psnr": (N, P)}].
    Profiles of kind "random" are held out. Returns (kappa, {key: (dsrc, csens, rho)}, report).
    nested: one master profile for every level (nested_rho; the levels must be chunked by
    phase, level with k chunks = the first k phases); the report then holds "master"."""
    if kappas is None:
        kappas = np.logspace(-1, 2, 31)

    def run(kappa):
        out, sq, cnt = {}, 0.0, 0
        Ds = [10.0 ** (-np.asarray(lv["psnr"], dtype=np.float64) / 10.0) for lv in levels]
        Fs = [phi(noise(np.asarray(lv["profiles"], dtype=np.float64)), kappa) for lv in levels]
        masks = [np.array([kd != "random" for kd in lv["kinds"]]) for lv in levels]
        if nested:
            master, params = _fit_nested(Ds, Fs, masks)
            out["__master__"] = master
        else:
            params = [_fit_level(D, F, m) for D, F, m in zip(Ds, Fs, masks)]
        for lv, D, F, m, (d, c, r) in zip(levels, Ds, Fs, masks, params):
            e = -10 * np.log10(predict_level(d, c, r, F)[:, m]) + 10 * np.log10(D[:, m])
            sq, cnt = sq + float((e ** 2).sum()), cnt + e.size
            out[lv["key"]] = (d, c, r)
        return np.sqrt(sq / max(cnt, 1)), out

    scored = [(run(k)[0], k) for k in kappas]
    best = min(scored)[1]
    lo, hi = best / 1.3, best * 1.3
    for _ in range(20):                                        # golden-section refinement
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if run(a)[0] < run(b)[0]:
            hi = b
        else:
            lo = a
    kappa = 0.5 * (lo + hi)
    rms, params = run(kappa)
    master = params.pop("__master__", None)
    report = {}
    for lv in levels:
        d, c, r = params[lv["key"]]
        D = 10.0 ** (-np.asarray(lv["psnr"], dtype=np.float64) / 10.0)
        F = phi(noise(np.asarray(lv["profiles"], dtype=np.float64)), kappa)
        e = -10 * np.log10(predict_level(d, c, r, F)) + 10 * np.log10(D)
        kinds = np.array(lv["kinds"])
        rep = {}
        for name, m in (("fit", kinds != "random"), ("held_out", kinds == "random")):
            if m.any():
                em = e[:, m]
                rep[name] = {"bias": float(em.mean()), "mae": float(np.abs(em).mean()),
                             "p95": float(np.percentile(np.abs(em), 95)), "n": int(em.size)}
        rep["rho"] = r.tolist()
        report[lv["key"]] = rep
    out = {"fit_rms": rms, "levels": report}
    if master is not None:
        out["master"] = master.tolist()
    return kappa, params, out
