"""Curve files (.npz) written by tools/alloc_sweep.py and read by tools/alloc_policy.py.

    scores     (N, S, D, K, M)  metric under the channel: image, SNR column,
                                noise draw, candidate budget, metric
    clean      (N, K, M)        the same through a noise-free channel
    snr        (N, S)           SNR (dB) of each column for each image
    units      (K,)             candidate budgets (tokens per tile, or channels)
    cbr        (K,)             their CBRs
    metrics    (M,)             metric names
    feat_img   (N, 14)          image statistics (alloc/sweep.py IMAGE_FEATURES)
    feat_code  (N, 24)          codeword energy profile
    pixels     (N,)             pixels per image (cost weight of its budget)
    names      (N,)             image identifiers
    meta       json             checkpoint, backbone, channel, split, dataset, ...
"""

import json
import os

import numpy as np

KEYS = ("scores", "clean", "snr", "units", "cbr", "metrics", "feat_img", "feat_code",
        "pixels", "names")


def save_curves(path, arrays, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = {k: arrays[k] for k in KEYS}
    out["metrics"] = np.asarray([str(m) for m in arrays["metrics"]])
    out["names"] = np.asarray([str(n) for n in arrays["names"]])
    np.savez_compressed(path, meta=np.asarray(json.dumps(meta, default=str)), **out)


def load_curves(*paths):
    """One curve file, or several concatenated along the image axis (they must
    share budgets and metrics)."""
    parts = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            d = {k: z[k] for k in KEYS}
            d["metrics"] = [str(m) for m in d["metrics"]]
            d["names"] = [str(n) for n in d["names"]]
            d["meta"] = json.loads(str(z["meta"]))
        parts.append(d)
    if len(parts) == 1:
        return parts[0]
    ref = parts[0]
    for d in parts[1:]:
        if list(d["units"]) != list(ref["units"]) or d["metrics"] != ref["metrics"]:
            raise ValueError("curve files with different budgets or metrics cannot be joined")
        if d["scores"].shape[1:] != ref["scores"].shape[1:]:
            raise ValueError("curve files with different SNR columns / draws cannot be joined")
    out = {k: np.concatenate([d[k] for d in parts], 0)
           for k in ("scores", "clean", "snr", "feat_img", "feat_code", "pixels")}
    out.update(units=ref["units"], cbr=ref["cbr"], metrics=ref["metrics"],
               names=[n for d in parts for n in d["names"]],
               meta={"joined": [d["meta"] for d in parts]})
    return out


def subset(d, idx):
    """The curves of the images `idx`."""
    idx = np.asarray(idx)
    out = dict(d)
    for k in ("scores", "clean", "snr", "feat_img", "feat_code", "pixels"):
        out[k] = d[k][idx]
    out["names"] = [d["names"][i] for i in idx]
    return out


def grid_snrs(d):
    """The SNR of each column if every image shares it (a fixed grid), else None."""
    s = d["snr"]
    if np.allclose(s, s[:1], atol=1e-6):
        return [float(v) for v in s[0]]
    return None
