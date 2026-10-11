"""Separation baseline for tools/schedule_sim.py: an image codec's rate-distortion points
(tools/digital_rd.py: BPG, HEIC, JPEG 2000, WebP) sent over the same slots as the token
prefixes, behind the same interface as alloc/utility.Utility.

A slot carries `sym_per_pixel` complex channel uses per pixel (one phase of the prefix
code: CBR 1/48 = 1/16 symbols per pixel) at the slot's SNR, which the transmitter knows
(the same CQI the token schedulers get), so link adaptation is ideal and error-free:

    link "cqi"      the LTE 4-bit CQI table (TS 36.213 Table 7.2.3-1), efficiency of the
                    highest entry whose SNR threshold (AWGN, 10% BLER, the usual link-level
                    values) the slot clears; below the lowest, the slot carries nothing
    link "shannon"  log2(1 + SNR / gap), gap = --gap-db (0: capacity, an upper bound)

At the deadline a user holds B bits per pixel. Mode "step": it decodes the best R-D point
whose file fits in B (as if the transmitter could switch files for free: optimistic for a
non-scalable codec); mode "interp": PSNR interpolated in bpp between the points (an ideal
scalable codec at the codec's efficiency). Below the smallest file: nothing (d_empty).
Every assumption here favours the digital scheme.
"""

import numpy as np

CQI_SNR_DB = np.array([-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7,
                       21.0, 22.7])
CQI_EFF = np.array([0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063,
                    2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547])


def efficiency(snr_db, link="cqi", gap_db=0.0):
    """Bits per complex channel use at SNR snr_db (dB)."""
    snr_db = np.asarray(snr_db, dtype=np.float64)
    if link == "shannon":
        return np.log2(1.0 + 10.0 ** ((snr_db - gap_db) / 10.0))
    if link == "cqi":
        idx = np.searchsorted(CQI_SNR_DB, snr_db, side="right") - 1
        return np.where(idx >= 0, CQI_EFF[np.clip(idx, 0, None)], 0.0)
    raise ValueError(f"unknown link {link!r}")


class DigitalUtility:
    kind = "digital"

    def __init__(self, curves, names, sym_per_pixel=1.0 / 16, link="cqi", gap_db=0.0, P=12,
                 mode="step", d_empty=0.1, meta=None):
        self.curves = []
        for c in curves:                                         # [(bpp, psnr), ...] per image
            c = np.asarray(c, dtype=np.float64).reshape(-1, 2)
            c = c[np.argsort(c[:, 0], kind="stable")]
            self.curves.append(np.stack([c[:, 0], np.maximum.accumulate(c[:, 1])], 1))
        self.names = list(names)
        self.sym_per_pixel, self.link, self.gap_db = float(sym_per_pixel), link, float(gap_db)
        self.P, self.mode, self.d_empty = int(P), mode, float(d_empty)
        self.meta = dict(meta or {})
        if mode not in ("step", "interp"):
            raise ValueError(f"unknown mode {mode!r}")

    def bpp(self, snrs_db):
        return float(self.sym_per_pixel * efficiency(snrs_db, self.link, self.gap_db).sum())

    def psnr(self, i, snrs_db):
        if len(snrs_db) > self.P:
            raise ValueError(f"{len(snrs_db)} slots, at most {self.P}")
        b, c = self.bpp(snrs_db), self.curves[i]
        floor = -10.0 * np.log10(self.d_empty)
        if b < c[0, 0]:
            return floor
        if self.mode == "step":
            return float(c[np.searchsorted(c[:, 0], b, side="right") - 1, 1])
        return float(np.interp(b, c[:, 0], c[:, 1]))

    def mse(self, i, snrs_db):
        return float(10.0 ** (-self.psnr(i, snrs_db) / 10.0))

    def perturbed(self, sigma, rng):
        return self                                              # the encoder has the files

    def average(self):
        """The pool-average curve (mean PSNR over images on a common bpp grid)."""
        grid = np.unique(np.concatenate([c[:, 0] for c in self.curves]))
        tmp = DigitalUtility(self.curves, self.names, mode="interp", d_empty=self.d_empty)
        q = [[tmp._at(i, b) for b in grid] for i in range(len(self.curves))]
        return DigitalUtility([list(zip(grid, np.mean(q, 0)))], ["average"], self.sym_per_pixel,
                              self.link, self.gap_db, self.P, "interp", self.d_empty)

    def _at(self, i, b):
        c = self.curves[i]
        if b < c[0, 0]:
            return -10.0 * np.log10(self.d_empty)
        return float(np.interp(b, c[:, 0], c[:, 1]))

    def to_dict(self):
        return {"kind": "digital", "images": self.names,
                "curves": [c.tolist() for c in self.curves], "sym_per_pixel": self.sym_per_pixel,
                "link": self.link, "gap_db": self.gap_db, "P": self.P, "mode": self.mode,
                "d_empty": self.d_empty, "meta": self.meta}

    @classmethod
    def from_dict(cls, d, **override):
        kw = {k: d[k] for k in ("sym_per_pixel", "link", "gap_db", "P", "mode", "d_empty")
              if k in d}
        kw.update(override)
        return cls(d["curves"], d["images"], meta=d.get("meta"), **kw)
