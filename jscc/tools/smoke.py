"""CPU smoke test: all four models end to end, plus the invariants the
experiments rely on. Needs neither data nor a GPU (a few minutes on one core).

    python tools/smoke.py
"""

import importlib.util
import os
import sys
import tempfile

os.environ["JSCC_RANDOM_WEIGHTS"] = "1"   # pretrained networks with random weights: no downloads

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from configs.config import Config  # noqa: E402
from net.network import JSCC  # noqa: E402
from net.tokens import spread_order  # noqa: E402
from utils.common import load_weights, save_weights  # noqa: E402
from utils.metrics import ALL_METRICS, MetricSuite  # noqa: E402
from utils.parser import create_parser  # noqa: E402

SMALL = {  # 128 px images, tiny transformers; token geometry scaled from 256 px
    "swin": ["--model-size", "small"],
    "hybrid": ["--model-size", "small", "--tile", "128", "--token-dim", "64", "--depth", "2",
               "--heads", "2"],
    "vit": ["--tile", "128", "--token-dim", "64", "--depth", "2", "--heads", "2"],
    "adatok": ["--tile", "128", "--token-dim", "64", "--depth", "2", "--heads", "2"],
}


# Sionna when installed; else the pure-PyTorch reference (same statistics)
BACKEND = "sionna" if importlib.util.find_spec("sionna") else "torch"


def parse(argv):
    return create_parser().parse_args(["--out-dir", tempfile.gettempdir(),
                                       "--channel-backend", BACKEND, *argv])


def build(backbone, *extra):
    cfg = Config(parse(["--backbone", backbone, "--img-size", "128", *SMALL[backbone], *extra]))
    cfg.device = torch.device("cpu")
    torch.manual_seed(0)
    return JSCC(cfg).eval(), cfg


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"    ok  {msg}")


def diff(a, b):
    return float((a.float() - b.float()).abs().max())


def raises(argv, msg):
    try:
        Config(parse(argv))
    except ValueError:
        ok(True, msg)
        return
    ok(False, msg)


def check_backbone(backbone, x):
    print(f"[{backbone}]")
    model, cfg = build(backbone, "--rates-per-step", "2")
    print(f"    {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters at smoke "
          f"size; units per CBR {cfg.cbr_units}")

    model.train()
    _, loss, _ = model(x, torch.tensor([0.0, 10.0]))
    loss.backward()
    ok(bool(torch.isfinite(loss)), f"one training step, 2 budgets, loss {float(loss):.4f}")
    g = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
    ok(g > 0, "the gradient reaches the encoder through the channel")
    model.zero_grad(set_to_none=True)
    model.eval()

    # exact budget and unit power at every predefined CBR (noise-free channel)
    model.channel.kind = "none"
    seen = {}
    original = model.channel.forward

    def spy(z, snr, mask=None):
        y = original(z, snr, mask)
        per_row = z.shape[1] * z.shape[2] // 2 if mask is None else int(mask.sum()) // z.shape[0]
        seen.update(rows=z.shape[0], per_row=per_row,
                    power=float(y.pow(2).sum()) / (z.shape[0] * per_row))
        return y

    model.channel.forward = spy
    with torch.no_grad():
        for c in cfg.cbrs:
            out, _, m = model(x, 10.0, c)
            per_image = seen["per_row"] * seen["rows"] // x.shape[0]
            ok(per_image == c * 3 * x.shape[2] * x.shape[3] and abs(seen["power"] - 1) < 1e-4
               and out.shape == x.shape and abs(m["cbr"][0] - float(c)) < 1e-12,
               f"CBR {c}: {per_image} complex symbols per image at unit power")
    del model.channel.forward

    if model.token:
        with torch.no_grad():
            enc = model.encode(x)
            l = cfg.token_list[2] + 5          # mid-phase on purpose
            rx = model.send(enc, l, 10.0)
            ref = model.decode(enc, rx, l)
            junk = rx.clone()
            junk[:, l:] = 100 * torch.randn_like(junk[:, l:])
            ok(diff(model.decode(enc, junk, l), ref) == 0.0,
               "untransmitted slots cannot reach the decoder")
            ok(diff(model.decode(enc, rx, torch.full((x.shape[0],), l)), ref) < 1e-5,
               "per-sample budget path == uniform path")
            la, lb = cfg.token_list[0] + 3, cfg.token_list[3] - 7
            mixed = torch.tensor([la, lb])
            out = model.decode(enc, model.send(enc, mixed, 10.0), mixed)
            ra = model.decode(enc, model.send(enc, la, 10.0), la)
            rb = model.decode(enc, model.send(enc, lb, 10.0), lb)
            ok(diff(out[0], ra[0]) < 1e-5 and diff(out[1], rb[1]) < 1e-5,
               f"mixed per-image budgets ({la}, {lb}) decode like uniform ones")
            if cfg.zero_init:
                t = model.encoder.trunk
                h = torch.randn(2, 10, cfg.token_dim)
                ok(diff(t(h), t.norm(h)) < 1e-6, "zero-init: the encoder trunk starts as LayerNorm")
            if model.decoder.mixer is not None:
                mixer, model.decoder.mixer = model.decoder.mixer, None
                plain = model.decode(enc, rx, l)
                model.decoder.mixer = mixer
                ok(diff(plain, ref) < 1e-6, f"{cfg.rate_mod} budget modulation starts as identity")

    model.channel.kind = "awgn"
    d = model.diagnose(x, 10.0, cfg.cbrs[2])
    ok(all(k in d for k in ("psnr", "zeroed", "swapped", "common")),
       "diagnose: " + " ".join(f"{k} {v:.3f}" for k, v in d.items()))

    big = torch.rand(1, 3, 256, 384)
    out, sent = model.reconstruct(big, 10.0, cfg.cbrs[2])
    vals = MetricSuite(["psnr", "ssim", "msssim"])(out, big, keys=[0])
    ok(out.shape == big.shape and set(vals) == {"psnr", "ssim", "msssim"}
       and abs(sent - float(cfg.cbrs[2])) < 1e-12,
       "256x384 input decodes at its own size, pixel metrics computed")
    if backbone in ("vit", "adatok"):
        model.channel.kind = "none"
        u, worst = cfg.token_list[2], 0.0
        with torch.no_grad():
            enc = model.encode(big)
            whole = model.decode(enc, model.send(enc, u, 10.0), u)
            for i in range(2):
                for j in range(3):
                    box = (..., slice(i * 128, (i + 1) * 128), slice(j * 128, (j + 1) * 128))
                    et = model.encode(big[box])
                    worst = max(worst, diff(model.decode(et, model.send(et, u, 10.0), u), whole[box]))
        ok(worst < 1e-5, "tiles are coded and decoded independently, as in training")
        model.channel.kind = "awgn"

    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01 * torch.randn_like(p))
        for name, b in model.named_buffers():
            if name.endswith("dc.mean"):
                b.normal_()
    model.channel.kind = "none"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "w.pt")
        save_weights(model, path)
        twin, _ = build(backbone)
        load_weights(twin, path)
        twin.channel.kind = "none"
        with torch.no_grad():
            ok(diff(model(x, 10.0, cfg.cbrs[1])[0], twin(x, 10.0, cfg.cbrs[1])[0]) == 0.0,
               "checkpoint round trip reproduces the outputs")


def check_perception(x):
    print("[perception] teacher and metric networks with random weights")
    flags = ["--rates-per-step", "2", "--sem-weight", "0.5", "--align-weight", "0.5",
             "--imp-weight", "0.5", "--lpips-weight", "0.5"]
    for backbone in ("vit", "swin"):
        model, cfg = build(backbone, *flags)
        model.train()
        _, loss, m = model(x, torch.tensor([0.0, 10.0]))
        loss.backward()
        ok(bool(torch.isfinite(loss)) and all(k in m for k in ("sem", "align", "lpips")),
           f"{backbone}: one step with every perceptual term, loss {float(loss):.4f}, sem "
           f"{m['sem']:.3f} align {m['align']:.3f} lpips {m['lpips']:.3f}")
        g = sum(float(p.grad.abs().sum()) for p in model.objective.align.parameters()
                if p.grad is not None)
        teacher = model.objective.teacher.net(x.device)
        ok(g > 0 and all(p.grad is None for p in teacher.parameters()),
           f"{backbone}: the alignment head learns, the teacher stays frozen")
        o = model.objective
        ok(o.factor(cfg.cbr_units[0]) == 1.0 and o.factor(cfg.cbr_units[-1]) == 0.0,
           f"{backbone}: the schedule is 1 at the smallest budget and 0 at the full one")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "w.pt")
            save_weights(model, path)
            keys = torch.load(path, weights_only=True).keys()
            ok(not any(k.startswith("objective.") for k in keys),
               f"{backbone}: the training-only head stays out of the weight file")
    ref = torch.rand(1, 3, 256, 384)
    vals = MetricSuite(["all"])((ref + 0.05 * torch.randn_like(ref)).clamp(0, 1), ref, keys=[0])
    ok(set(vals) == set(ALL_METRICS),
       "every evaluation metric computed: " + " ".join(f"{k} {float(v[0]):.3f}" for k, v in vals.items()))


def check_allocation(x):
    """The per-image budget policy (alloc/): exact allocation, cross-fitting,
    the common-random-number channel, the sweep, and the curve predictor."""
    import itertools

    import numpy as np

    from alloc import predictor as P
    from alloc.core import allocate, compare, isotonic, metric_scales, path, segments
    from alloc.sweep import codeword, draw_noise, send, sweep

    class OneBatch:                     # the sweep reads len(loader.dataset) and iterates
        def __init__(self, x):
            self.x, self.dataset = x, x

        def __iter__(self):
            return iter([self.x])

    print("[allocation]")
    rng = np.random.default_rng(0)
    N, K = 4, 4
    units = np.array([10.0, 20.0, 30.0, 50.0])
    cost = rng.uniform(0.5, 1.5, N)
    q = np.cumsum(rng.uniform(0, 1, (N, K)) * np.linspace(1.5, 0.2, K), 1)
    segs = segments(q, cost, units)
    avg, _ = path(segs, q[:, :, None], cost, units, np.array([1.0]))
    exact = True
    for t in avg[1:-1]:
        w, _ = allocate(segs, cost, units, t)
        best = max(q[np.arange(N), list(c)].sum() for c in itertools.product(range(K), repeat=N)
                   if (cost * units[list(c)]).sum() / cost.sum() <= t + 1e-9)
        exact &= abs((w * q).sum() - best) < 1e-9
        exact &= abs((w @ units * cost).sum() / cost.sum() - t) < 1e-9
    ok(exact, "equal-slope allocation == brute-force optimum, target rate met exactly")
    y = isotonic(np.array([1.0, 3.0, 2.0, 2.0, 5.0, 4.0]))
    ok(bool(np.all(np.diff(y) >= 0)) and abs(y.sum() - 17.0) < 1e-9, "isotonic fit")
    Nn, D, units2 = 300, 4, np.linspace(1024, 6144, 11)
    flat = 25 + 3 * np.log2(units2 / units2[0])
    v = np.stack([np.tile(flat, (Nn, 1)), np.tile(0.4 - flat / 100, (Nn, 1))], -1)
    draws = v[:, None] + rng.normal(0, 1, (Nn, D, 11, 2)) * np.array([0.3, 0.01])
    sc = metric_scales(draws.mean(1))
    res = compare(draws, np.ones(Nn), units2, ["psnr", "lpips"],
                  {"oracle": ("oracle", [("psnr", 1.0)])}, [3072.0], sc, "psnr")
    gain = float((res["oracle"][3072.0]["values"] - res["uniform"][3072.0])[:, 0].mean())
    ok(gain < 0.02, f"cross-fitted oracle harvests no noise on identical images ({gain:+.3f} dB)")

    from tools.alloc_policy import summarise, verdicts

    def gate_passes(p, seed):
        g = np.random.default_rng(seed)
        v2 = np.stack([p, 0.4 - p / 100], -1)
        dr = v2[:, None] + g.normal(0, 1, (len(p), 4, 11, 2)) * np.array([0.3, 0.01])
        res = compare(dr, np.ones(len(p)), units2, ["psnr", "lpips"],
                      {"oracle": ("oracle", [("psnr", 1.0)])}, [3072.0],
                      metric_scales(dr.mean(1)), "psnr")
        tab = summarise([res], ["oracle"], ["psnr", "lpips"], [("t", 3072.0)], 1.0, "psnr", 0)
        return verdicts(tab, "oracle", ["psnr", "lpips"], "psnr", 0.1, 0.05)["t"]["pass"]

    lg2 = np.log2(units2 / units2[0])
    hetero = np.concatenate([np.tile(30 + np.minimum(lg2, 1.0), (150, 1)),
                             np.tile(22 + 3 * lg2, (150, 1))])
    false_pass = float(np.mean([gate_passes(np.full((24, 11), 30.0), s) for s in range(100)]))
    ok(false_pass <= 0.08 and gate_passes(hetero, 0),
       f"gate: structure-free Kodak-sized curves pass at chance level ({100 * false_pass:.0f}% of "
       f"100 trials), easy/hard images pass")
    vv = np.array([[0.0, 2.0, 1.0], [0.0, 2.0, 1.0]])[:, None, :, None]
    bad = compare(vv, np.ones(2), np.array([1.0, 2.0, 3.0]), ["psnr"],
                  {"bad": np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]])}, [2.0], np.ones(1),
                  "psnr")["bad"][2.0]
    ok(bad["reached"] == 0.0 and bad["saving"] <= 0,
       "a method that never reaches uniform's quality is charged the whole range, not skipped")

    for backbone in ("vit", "swin"):
        model, cfg = build(backbone)
        with torch.no_grad():
            enc = model.encode(x)
            u1, u2 = cfg.cbr_units[1], cfg.cbr_units[3]
            model.channel.kind = "none"
            same = all(diff(model.send(enc, u, 10.0), send(model, enc, u)) < 1e-6 for u in (u1, u2))
            ok(same, f"{backbone}: the sweep's channel == model.send (noise-free)")
            model.channel.kind = "awgn"
            noise, _ = draw_noise(model, enc, "awgn", torch.Generator().manual_seed(1))
            snr = torch.full((codeword(model, enc).shape[0],), 7.0)
            n1 = send(model, enc, u1, snr, noise, None, "awgn") - send(model, enc, u1)
            n2 = send(model, enc, u2, snr, noise, None, "awgn") - send(model, enc, u2)
            shared = diff(n1[:, :u1], n2[:, :u1]) if model.token else \
                diff(n1[..., :u1 // 2], n2[..., :u1 // 2])
            ok(shared < 1e-6, f"{backbone}: every budget sees the same noise on its prefix")
            units = [cfg.cbr_units[0], cfg.cbr_units[2], cfg.cbr_units[-1]]
            arr = sweep(model, cfg, OneBatch(x), units, ["psnr"], draws=1, snrs=[5.0],
                        log=lambda s: None)
            model.channel.kind = "none"
            suite = MetricSuite(["psnr"])
            direct = [float(suite(model(x, 10.0, cfg.cbr_for_units(u))[0], x, [0, 1])["psnr"].mean())
                      for u in units]
            model.channel.kind = "awgn"
            ok(arr["scores"].shape == (2, 1, 1, 3, 1) and arr["feat_code"].shape == (2, 24)
               and np.allclose(arr["clean"][:, :, 0].mean(0), direct, atol=1e-4),
               f"{backbone}: sweep shapes; its noise-free curve matches the model's own decode")
    m, c = build("vit", "--channel-type", "rayleigh")
    with torch.no_grad():
        enc = m.encode(x)
        noise, fading = draw_noise(m, enc, "rayleigh", torch.Generator().manual_seed(2))
        y = send(m, enc, c.cbr_units[2], torch.full((codeword(m, enc).shape[0],), 5.0),
                 noise, fading, "rayleigh", "mmse")
    ok(bool(torch.isfinite(y).all()), "Rayleigh + MMSE through the shared-noise channel")

    Ns, Ks = 600, 11
    un = np.linspace(1024, 6144, Ks)

    def synth(n, snr):
        f = rng.normal(0, 1, (n, 14))
        a = 3.0 + 0.8 * f[:, 3]
        lg = np.log2(un / un[0])
        ps = 26 + a[:, None, None] * lg[None, None] * np.ones((1, snr.shape[1], 1))
        v = np.stack([ps, 0.45 - 0.02 * ps], -1)
        noise = rng.normal(0, 1, (n, snr.shape[1], 2, Ks, 2)) * np.array([0.1, 0.002])
        return {"scores": v[:, :, None] + noise,
                "clean": v[:, 0], "snr": snr, "units": un, "metrics": ["psnr", "lpips"],
                "feat_img": f, "feat_code": rng.normal(0, 1, (n, 24)), "pixels": np.ones(n)}

    tr = synth(Ns, rng.uniform(-2, 22, (Ns, 2)))
    va = synth(100, rng.uniform(-2, 22, (100, 2)))
    ck = P.fit(tr, va, ["psnr", "lpips"], epochs=60, patience=60, log=lambda s: None)
    ok(ck["valid_mse"] < 0.5 * ck["valid_mse_mean_curve"],
       f"the curve predictor learns per-image structure (valid mse {ck['valid_mse']:.4f} vs "
       f"{ck['valid_mse_mean_curve']:.4f} for the mean curve)")
    te = synth(50, np.tile([4.0, 10.0], (50, 1)))
    net = P.CurveNet(ck["arch"]["d_in"], 2, Ks, ck["arch"]["hidden"], ck["arch"]["depth"])
    net.load_state_dict(ck["state"])
    q = P.predicted_objective(P.predict(net.eval(), ck, te), ck, [("lpips", 1.0)])
    ok(q.shape == (50, 2, Ks) and bool(np.all(np.diff(q, axis=-1) >= -1e-6)),
       "predicted curves are non-decreasing, one per image and SNR")


def check_frontier():
    """Story B's test (tools/frontier.py): run names, the schedule, the bracket,
    and the verdict on synthetic models on / above / below a concave curve."""
    from types import SimpleNamespace

    import numpy as np

    from alloc.io import save_curves
    from net.loss import Objective
    from tools import frontier as F

    print("[frontier]")
    H = "history/vit_p8d768x10s4_awgn_DIV2K_256"
    cases = {f"{H}_p1/models/last.pt": ("none", {}, 1.0),
             f"{H}_lp0.5-const_p1/models/last.pt": ("constant", {"lp": 0.5}, 1.0),
             f"{H}_lp0.5_p1/models/last.pt": ("budget", {"lp": 0.5}, 1.0),
             f"{H}_lp0.5-g2_p1/models/last.pt": ("budget", {"lp": 0.5}, 2.0),
             f"{H}_sem0.5-al0.5-imp0.5-const_p1/models/last.pt":
                 ("constant", {"sem": 0.5, "al": 0.5, "imp": 0.5}, 1.0)}
    good = all((lambda r: (r["schedule"], r["weights"], r["gamma"]) == want)(
        F.parse_run(F.run_dir(path))) for path, want in cases.items())
    ok(good, "run names -> perceptual weights, schedule, gamma")
    obj = SimpleNamespace(schedule="budget", u_lo=1024.0, u_hi=6144.0, gamma=2.0)
    same = all(abs(Objective.factor(obj, u) - F.schedule_factor(u, 1024, 6144, 2.0)) < 1e-12
               for u in (1024, 1536, 3072, 4096, 6144))
    ok(same and abs(F.schedule_factor(4096, 1024, 6144, 1.0) - 0.4) < 1e-12,
       "the schedule w(u) is the training loss's")

    # bracket of a concave curve: chord below it, bound above it, everywhere
    f = lambda x: -((31.0 - x) / 0.5) ** 2 * 0.01 + 0.3 * np.sqrt(max(31.0 - x, 0.0))
    pts = [(x, f(x)) for x in (30.0, 30.6, 31.0)]
    Hh = F.hull(pts)
    inside = all(F.bracket(Hh, x)[0] - 1e-12 <= f(x) <= F.bracket(Hh, x)[1] + 1e-12
                 for x in np.linspace(30.0, 31.0, 41))
    ok(inside and len(F.hull(pts + [(30.3, -1.0)])) == 3,
       "chord <= concave curve <= bound; a dominated constant point is dropped")

    units = np.linspace(1024, 6144, 11)

    def data(weight_of, shift_db=0.0, seed=0, N=30):
        rng = np.random.default_rng(seed)
        img = np.random.default_rng(99).normal(0, 1, N)
        sc = np.zeros((N, 2, 2, len(units), 3))
        for k, u in enumerate(units):
            w, a, b = weight_of(u), 0.45, 0.035
            g = b * np.sqrt(w) + shift_db * b / (2 * a * np.sqrt(max(w, 0.05)))
            base = np.stack([26 + 3 * np.log2(u / 1024) - a * w + 0.8 * img,
                             0.17 - 0.02 * np.log2(u / 1024) - g + 0.01 * img,
                             0.30 - 0.03 * np.log2(u / 1024) - 1.6 * g + 0.015 * img], -1)
            sc[:, :, :, k] = base[:, None, None] + \
                rng.normal(0, 1, (N, 2, 2, 3)) * np.array([0.02, 0.0005, 0.0008])
        return {"scores": sc, "clean": sc.mean((1, 2)), "units": units, "cbr": units / 49152,
                "metrics": ["psnr", "dists", "lpips_vgg"], "names": [f"i{i}" for i in range(N)],
                "snr": np.tile([4.0, 10.0], (N, 1)), "feat_img": np.zeros((N, 14)),
                "feat_code": np.zeros((N, 24)), "pixels": np.ones(N)}

    sched = lambda u: 0.5 * F.schedule_factor(u, 1024, 6144, 1.0)
    with tempfile.TemporaryDirectory() as tmp:
        paths = {}
        specs = [("mse", lambda u: 0.0, 0.0, 1, f"{H}_p1"),
                 ("c02", lambda u: 0.2, 0.0, 2, f"{H}_lp0.2-const_p1"),
                 ("c05", lambda u: 0.5, 0.0, 3, f"{H}_lp0.5-const_p1"),
                 ("on", sched, 0.0, 4, f"{H}_lp0.5_p1"),
                 ("above", sched, 0.25, 5, f"{H}_lp0.5_p1"),
                 ("below", sched, -0.25, 6, f"{H}_lp0.5_p1")]
        for name, wf, shift, seed, run in specs:
            paths[name] = os.path.join(tmp, f"{name}.npz")
            save_curves(paths[name], data(wf, shift, seed),
                        {"checkpoint": f"{run}/models/last.pt", "seed": 42, "snrs": [4.0, 10.0],
                         "predefined_units": [1024, 2048, 3072, 4096, 6144]})
        consts = [paths["mse"], paths["c02"], paths["c05"]]
        got = {}
        for name in ("on", "above", "below"):
            models = F.load_models(consts, [f"{name}={paths[name]}"], None)
            F.check_same(models)
            res = F.analyse(models, ["dists", "lpips_vgg"], "psnr", 300, 0, 0.05)[name]
            got[name] = F.verdict(res, ["dists", "lpips_vgg"], 0.05, [0.0, 0.2, 0.5])[0]
            if name == "on":
                never_above = all(r["class"] != "ABOVE" for m in res["metrics"].values()
                                  for r in m)
                matched = sorted(r["weight"] for r in res["matched"])
        ok(got == {"on": "DEAD", "above": "LIVES", "below": "DEAD"} and never_above
           and matched == [0.0, 0.2, 0.5],
           f"verdicts: on the curve {got['on']}, above it {got['above']}, below it "
           f"{got['below']}; the schedule meets the constant weights at 1/48, 1/12, 1/8")
        two = F.load_models([paths["mse"], paths["c05"]], [paths["on"]], None)
        res = F.analyse(two, ["dists", "lpips_vgg"], "psnr", 300, 0, 0.05)
        res = next(iter(res.values()))
        v2 = F.verdict(res, ["dists", "lpips_vgg"], 0.05, [0.0, 0.5])[0]
        ok(v2 in ("OPEN", "DEAD") and all(r["class"] != "ABOVE" for m in res["metrics"].values()
                                          for r in m),
           f"with two constant weights a model on the curve is never ABOVE ({v2})")
        other = data(lambda u: 0.0, 0.0, 7)
        other["names"] = [f"x{i}" for i in range(len(other["names"]))]
        save_curves(os.path.join(tmp, "other.npz"), other,
                    {"checkpoint": f"{H}_p1/models/last.pt", "seed": 42, "snrs": [4.0, 10.0]})
        try:
            F.check_same(F.load_models([paths["mse"], os.path.join(tmp, "other.npz")],
                                       [paths["on"]], None))
            refused = False
        except SystemExit:
            refused = True
        ok(refused, "curve files of different images are refused")


def check_freeze(x):
    """--freeze: one half stays bit-identical through training steps and runs in
    eval mode; the other half trains; the flag needs --pretrained and names the run."""
    from utils.common import param_groups

    print("[tail probes]")
    raises(["--backbone", "vit", "--freeze", "encoder"], "--freeze without --pretrained is refused")
    raises(["--backbone", "vit", "--top-prob", "0.5", "--fixed-cbr", "1/8"],
           "--top-prob with --fixed-cbr is refused")
    for tp in (0.5, 1.0):
        model, cfg = build("vit", "--top-prob", str(tp))
        torch.manual_seed(1)
        got = [model.sample_units()[0] for _ in range(2000)]
        share = sum(g == cfg.cbr_units[-1] for g in got) / len(got)
        ok(abs(share - tp) < 0.04 and min(got) >= cfg.cbr_units[0],
           f"--top-prob {tp}: {100 * share:.0f}% of budgets are the full one, the rest uniform")
    cfg = Config(parse(["--backbone", "vit", "--freeze", "decoder", "--pretrained", "x.pt",
                        "--fixed-cbr", "1/8", "--tag", "pr"]))
    ok(cfg.run_name.endswith("_fix1-8_frz-dec_pr"), f"run name marks the probe ({cfg.run_name})")
    for backbone in ("vit", "swin"):
        for part in ("encoder", "decoder"):
            model, cfg = build(backbone)
            other = "decoder" if part == "encoder" else "encoder"
            model.freeze(part)
            model.train()
            snap = {n: p.detach().clone() for n, p in model.named_parameters()}
            opt = torch.optim.AdamW(param_groups(model, 0.0), lr=1e-3)
            for _ in range(2):
                _, loss, _ = model(x, torch.full((x.shape[0],), 10.0), want_metrics=False)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            halves = model.halves()
            names = {h: {n for m in halves[h] for n, _ in m.named_parameters()} for h in halves}
            prefix = {h: [k for k, m in model.named_children() if any(m is mm for mm in halves[h])]
                      for h in halves}

            def changed(h):
                keys = [n for n in snap if any(n.startswith(pf + ".") for pf in prefix[h])]
                now = {n: q.detach() for n, q in model.named_parameters()}
                return keys, [n for n in keys if diff(snap[n], now[n]) > 0]

            fk, fc = changed(part)
            ok_ = fk and not fc and all(not m.training for m in halves[part])
            tk, tc = changed(other)
            ok(ok_ and len(tc) > 0 and all(m.training for m in halves[other]) and names[part],
               f"{backbone}: frozen {part} bit-identical and in eval mode after 2 steps; "
               f"{len(tc)}/{len(tk)} {other} tensors moved")


def main():
    torch.set_num_threads(os.cpu_count() or 1)
    print(f"[shared] channel backend: {BACKEND}")
    order = spread_order(8, 8).tolist()
    ok(sorted(order) == list(range(64)), "spread order is a permutation")
    ok(all((i // 8) % 2 == 0 and (i % 8) % 2 == 0 for i in order[:16]),
       "its first quarter is the stride-2 sub-lattice")
    raises(["--backbone", "adatok", "--zero-init"], "adatok refuses --zero-init")
    raises(["--backbone", "hybrid", "--sym-per-token", "40"], "fractional token counts are refused")
    raises(["--backbone", "hybrid", "--sym-per-token", "64"], "a partial number of phases is refused")
    raises(["--img-size", "200"], "an image size off the Swin grid is refused")
    torch.manual_seed(0)
    x = torch.rand(2, 3, 128, 128)
    for backbone in ("swin", "hybrid", "vit", "adatok"):
        check_backbone(backbone, x)
    check_perception(x)
    check_allocation(x)
    check_frontier()
    check_freeze(x)
    print("all checks passed")


if __name__ == "__main__":
    main()
