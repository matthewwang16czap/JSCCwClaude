"""Measure the channel conventions (net/channel.py) for both backends. CPU,
seconds. Sionna is checked when it is installed.

    python tools/check_channel.py
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from net.channel import Channel, seed_channel  # noqa: E402

BACKENDS = ("torch", "sionna") if importlib.util.find_spec("sionna") else ("torch",)


def ok(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def complex_of(t):
    c = t.shape[-1] // 2
    return torch.complex(t[..., :c], t[..., c:])


def seed(backend, s):
    torch.manual_seed(s)
    if backend == "sionna":
        seed_channel(s)


def measure(kind, backend, snr, eq="mmse", n=400_000):
    seed(backend, 0)
    x = torch.randn(4, n // 4, 2)
    y = Channel(kind, backend, eq)(x, snr)
    return Channel.normalize(complex_of(x)), complex_of(y)


def rayleigh_theory(no, n=2_000_000):
    h2 = torch.distributions.Exponential(1.0).sample((n,))   # |h|^2 of CN(0, 1)
    g = h2 / (h2 + no)
    return float(g.mean()), float(g.var()), float((g * no / (h2 + no)).mean())


def main():
    for backend in BACKENDS:
        print(f"[{backend}]")
        for snr in (-2.0, 5.0, 13.0):
            no = 10 ** (-snr / 10)
            xc, yc = measure("awgn", backend, snr)
            e = yc - xc
            ok(abs(float(e.abs().pow(2).mean()) / no - 1) < 0.02
               and abs(float(e.real.var()) / (no / 2) - 1) < 0.02,
               f"awgn {snr:+.0f} dB: noise variance no per complex symbol, no/2 per real dim")
            ok(abs(float(yc.abs().pow(2).mean()) - (1 + no)) < 0.02 * (1 + no),
               f"awgn {snr:+.0f} dB: received power 1 + no")
            xc, yc = measure("rayleigh", backend, snr)
            mg, vg, vn = rayleigh_theory(no)
            gain = float((yc * xc.conj()).real.mean() / xc.abs().pow(2).mean())
            resid = float((yc - mg * xc).abs().pow(2).mean())
            ok(abs(gain - mg) < 0.01 and abs(resid / (vg + vn) - 1) < 0.03,
               f"rayleigh+MMSE {snr:+.0f} dB: E[g] {gain:.3f} (theory {mg:.3f}), residual "
               f"{resid:.4f} (theory {vg + vn:.4f})")
        for eq in ("mmse", "zf"):
            xc, yc = measure("rayleigh", backend, 80.0, eq=eq, n=40_000)
            ok(float(torch.quantile((yc - xc).abs().flatten(), 0.99)) < 1e-2,
               f"rayleigh+{eq.upper()} at 80 dB returns the input (h is removed)")
        x = torch.randn(3, 50, 8)
        a = [Channel("awgn", backend)(x, 5.0) for _ in range(2) if seed(backend, 7) is None]
        ok(torch.equal(a[0], a[1]), "seeding reproduces the noise exactly")
        if backend == "sionna":
            torch.manual_seed(7)
            b = Channel("awgn", backend)(x, 5.0)
            torch.manual_seed(7)
            ok(not torch.equal(b, Channel("awgn", backend)(x, 5.0)),
               "torch.manual_seed alone does not reach Sionna's generators (hence seed_channel)")
    h2 = torch.logspace(-4, 4, 10001)
    for no in (0.05, 1.0, 3.0):
        ok(float((h2 / (h2 + no) * no / (h2 + no)).max()) <= 0.25 + 1e-9,
           f"post-MMSE noise variance <= 1/4 at no = {no}")
    x = torch.randn(3, 10, 8)
    mask = torch.zeros(3, 10, 4, dtype=torch.bool)
    mask[:, :6] = True
    for backend in BACKENDS:
        seed(backend, 1)
        yc = complex_of(Channel("awgn", backend)(x, 10.0, mask=mask))
        ok(float(yc[:, 6:].abs().max()) == 0.0,
           f"{backend}: masked symbols leave the channel as exact zeros")
    yc = complex_of(Channel("none")(x, 10.0, mask=mask))
    ok(float((yc[:, :6].abs().pow(2).mean(dim=(1, 2)) - 1).abs().max()) < 1e-5,
       "unit power per transmitted symbol, per row")
    print("channel conventions hold")


if __name__ == "__main__":
    main()
