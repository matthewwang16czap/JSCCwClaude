"""The policy: predict each image's quality-vs-budget curve, then allocate.

A small MLP maps what the TRANSMITTER knows -- content statistics of the
image, the energy profile of its codeword, a few noise-free decodes it can run
locally, and the channel SNR (fed back) -- to the image's curve under the
channel for every candidate budget and every metric at once. Allocation is
then the equal-slope rule of alloc/core.py on the predicted curves, so the
target average rate (and the objective's mix of metrics) is a test-time knob:
one predictor serves every CBR without retraining.

The head predicts non-negative increments and sums them, so every predicted
curve is non-decreasing in the budget (signed metrics: larger is better).
Targets are gains over the smallest candidate, divided by the metric's
typical spread on the training curves, so dB and LPIPS weigh alike.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alloc.core import SIGN, metric_scales

FEATURE_SETS = ("img", "code", "clean")


class CurveNet(nn.Module):
    def __init__(self, d_in, n_metrics, n_budgets, hidden=256, depth=3, dropout=0.1):
        super().__init__()
        layers, d = [], d_in
        for _ in range(max(depth - 1, 1)):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, n_metrics * (n_budgets - 1))
        self.m, self.k = n_metrics, n_budgets

    def forward(self, f):
        inc = F.softplus(self.head(self.body(f))).view(-1, self.m, self.k - 1)
        return torch.cumsum(inc, dim=-1)       # (B, metrics, K-1): gain over budget 0


def clean_points(K, n):
    """Indices of the noise-free decodes the transmitter runs (first, last, and
    evenly in between)."""
    return sorted(set(np.linspace(0, K - 1, max(int(n), 2)).round().astype(int).tolist()))


def build_inputs(data, pred_metrics, scales, features, n_clean):
    """(N*S, F) rows in (image, snr-column) order."""
    N, S = data["snr"].shape
    names = list(data["metrics"])
    parts = []
    if "img" in features:
        parts.append(data["feat_img"])
    if "code" in features:
        parts.append(data["feat_code"])
    if "clean" in features:
        K = data["clean"].shape[1]
        idx = clean_points(K, n_clean)
        cols = []
        for m in pred_metrics:
            j = names.index(m)
            c = SIGN[m] * data["clean"][:, idx, j] / scales[m]
            cols.append(np.nan_to_num(c))
        parts.append(np.concatenate(cols, 1))
    per_image = np.concatenate(parts, 1) if parts else np.zeros((N, 0))
    x = np.repeat(per_image, S, axis=0)
    snr = ((data["snr"].reshape(-1, 1) - 10.0) / 10.0)
    return np.concatenate([x, snr], 1).astype(np.float32)


def build_targets(data, pred_metrics, scales):
    """(N*S, metrics, K-1): signed gain over the smallest budget / spread."""
    v = np.nanmean(data["scores"], axis=2)               # mean over draws: (N, S, K, M)
    names = list(data["metrics"])
    ys = []
    for m in pred_metrics:
        j = names.index(m)
        ys.append(SIGN[m] * (v[..., 1:, j] - v[..., :1, j]) / scales[m])
    y = np.stack(ys, axis=2)                              # (N, S, metrics, K-1)
    return np.nan_to_num(y).reshape(-1, len(pred_metrics), y.shape[-1]).astype(np.float32)


def train_scales(data, pred_metrics):
    v = np.nanmean(data["scores"], axis=(1, 2))           # (N, K, M)
    s = metric_scales(v)
    names = list(data["metrics"])
    return {m: float(s[names.index(m)]) for m in pred_metrics}


def fit(train, valid, pred_metrics, features=FEATURE_SETS, n_clean=3, hidden=256, depth=3,
        dropout=0.1, lr=1e-3, weight_decay=1e-4, epochs=400, batch=256, patience=40,
        seed=0, device="cpu", log=print):
    """Train on `train` curves, early-stop on `valid` curves. Returns a checkpoint dict."""
    torch.manual_seed(seed)
    scales = train_scales(train, pred_metrics)
    xtr = build_inputs(train, pred_metrics, scales, features, n_clean)
    ytr = build_targets(train, pred_metrics, scales)
    xva = build_inputs(valid, pred_metrics, scales, features, n_clean)
    yva = build_targets(valid, pred_metrics, scales)
    mu, sd = xtr.mean(0), xtr.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    mu[-1], sd[-1] = 0.0, 1.0                              # the SNR is already centred

    def tens(a):
        return torch.tensor(a, device=device)

    xtr_t, ytr_t = tens((xtr - mu) / sd), tens(ytr)
    xva_t, yva_t = tens((xva - mu) / sd), tens(yva)
    K = ytr.shape[-1] + 1
    net = CurveNet(xtr.shape[1], len(pred_metrics), K, hidden, depth, dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = float("inf"), None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr_t), generator=g).to(device)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            loss = F.mse_loss(net(xtr_t[b]), ytr_t[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tr = float(F.mse_loss(net(xtr_t), ytr_t))
            va = float(F.mse_loss(net(xva_t), yva_t))
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 20 == 0 or bad == 0:
            log(f"  epoch {ep:4d}  train mse {tr:.5f}  valid mse {va:.5f}"
                f"{'  *' if bad == 0 else ''}")
        if bad >= patience:
            log(f"  early stop after epoch {ep} (best valid mse {best:.5f})")
            break
    # what always predicting the mean training curve would score on valid
    base = float(((yva - ytr.mean(0, keepdims=True)) ** 2).mean())
    return {"state": best_state, "metrics": list(pred_metrics), "scales": scales,
            "features": list(features), "n_clean": int(n_clean), "units": train["units"].tolist(),
            "mu": mu.tolist(), "sd": sd.tolist(),
            "arch": {"hidden": hidden, "depth": depth, "dropout": dropout, "d_in": xtr.shape[1],
                     "k": K},
            "valid_mse": best, "valid_mse_mean_curve": base,
            "train_meta": str(train.get("meta", ""))}


def load(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["arch"]
    net = CurveNet(a["d_in"], len(ck["metrics"]), a["k"], a["hidden"], a["depth"], a["dropout"])
    net.load_state_dict(ck["state"])
    return net.to(device).eval(), ck


@torch.no_grad()
def predict(net, ck, data, snr_offset=0.0, device="cpu"):
    """Predicted signed, scaled gain curves (N, S, K, metrics); gain 0 at budget 0.
    `snr_offset` feeds the predictor a wrong SNR (robustness to feedback error)."""
    if [int(u) for u in data["units"]] != [int(u) for u in ck["units"]]:
        raise ValueError(f"these curves use budgets {list(data['units'])}, the predictor was "
                         f"trained on {ck['units']}: sweep with the same --candidates")
    missing = [m for m in ck["metrics"] if m not in list(data["metrics"])]
    if missing:
        raise ValueError(f"the curves lack {missing}, which the predictor needs")
    shifted = dict(data)
    shifted["snr"] = data["snr"] + snr_offset
    x = build_inputs(shifted, ck["metrics"], ck["scales"], ck["features"], ck["n_clean"])
    x = (x - np.asarray(ck["mu"], dtype=np.float32)) / np.asarray(ck["sd"], dtype=np.float32)
    g = net(torch.tensor(x, device=device)).cpu().numpy()   # (N*S, metrics, K-1)
    N, S = data["snr"].shape
    g = np.concatenate([np.zeros_like(g[..., :1]), g], axis=-1)
    return g.reshape(N, S, len(ck["metrics"]), -1).transpose(0, 1, 3, 2)


def predicted_objective(pred, ck, spec):
    """(N, S, K, metrics) -> (N, S, K) for an objective spec."""
    q = np.zeros(pred.shape[:3])
    for name, w in spec:
        if name not in ck["metrics"]:
            raise ValueError(f"the predictor was trained on {ck['metrics']}, not {name!r}")
        q += w * pred[..., ck["metrics"].index(name)]
    return q
