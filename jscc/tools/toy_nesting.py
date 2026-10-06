"""A linear-Gaussian toy: does nesting by itself cost anything, in the one case with
a closed form?

    python tools/toy_nesting.py [--snr 7] [--alpha 1.5]

Source: n independent Gaussian components with variances lambda_j = j^-alpha (an
image-like decaying spectrum). Code: analog linear JSCC over AWGN. A rate k sends
the k strongest components, each scaled to power p_j with sum_j p_j = k (unit mean
power per transmitted real symbol, the convention of net/channel.py); the
receiver is the MMSE estimator, so component j ends with error
lambda_j / (1 + p_j / sigma^2), and an unsent one with lambda_j.

    optimal   p_j(k) re-chosen per rate (reverse water-filling): what a code
              built for that rate alone would use (per-rate specialist, and
              what a cascade's per-level gain / power profile can represent)
    nested    ONE power shape for all rates, as in the prefix scheme: rate k
              takes the head of the same shape and the channel renormalises
              the total. The shape is optimised for the mean dB over the rates.

The gap is the penalty of nesting that a LINEAR code can show, in dB of PSNR. It
is a lower bound on nothing and an upper bound on nothing: it says whether the
power-allocation mismatch alone could explain the 0.3-0.7 dB the nested ViT
loses to a fixed-rate specialist at 1/8 (docs/NOTES.md, tail probes). If it is
tiny, the penalty of a real network lives elsewhere (ordering constraints on
nonlinear features, optimisation, sampling), and so must any theory of why a
cascade wins.
"""

import argparse
import math

import torch

RATES = (32, 64, 96, 128, 192)       # real channels per 16x16 position of the Swin latent


def optimal_distortion(lam, k, sigma2):
    """min sum_{j<=k} lam_j / (1 + p_j / sigma2) + sum_{j>k} lam_j,  sum p = k, p >= 0."""
    lk = lam[:k]
    lo, hi = 1e-12, 1e12              # bisection on the water level mu: p_j = (sigma sqrt(lam_j/mu) - sigma2)+
    for _ in range(200):
        mu = math.sqrt(lo * hi)
        p = (math.sqrt(sigma2) * torch.sqrt(lk / mu) - sigma2).clamp_min(0.0)
        lo, hi = (mu, hi) if p.sum() > k else (lo, mu)
    p = p * (k / p.sum())
    return float((lk / (1.0 + p / sigma2)).sum() + lam[k:].sum())


def nested_distortion(lam, shape, k, sigma2):
    p = shape[:k] * (k / shape[:k].sum())
    return (lam[:k] / (1.0 + p / sigma2)).sum() + lam[k:].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=768, help="source components per position (16x16x3)")
    ap.add_argument("--snr", type=float, nargs="+", default=[1.0, 7.0, 13.0])
    ap.add_argument("--alpha", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    a = ap.parse_args()
    torch.set_num_threads(1)
    print(f"{'alpha':>5s} {'SNR':>4s} | {'optimal PSNR-like dB per rate (32 64 96 128 192)':^52s} | "
          f"{'nested minus optimal, dB':^34s}")
    for alpha in a.alpha:
        lam = torch.arange(1, a.n + 1, dtype=torch.float64) ** -alpha
        for snr in a.snr:
            sigma2 = 10 ** (-snr / 10)
            opt = [optimal_distortion(lam, k, sigma2) for k in RATES]
            logit = torch.zeros(RATES[-1], dtype=torch.float64, requires_grad=True)
            opt_adam = torch.optim.Adam([logit], lr=0.05)
            for _ in range(600):
                shape = torch.exp(logit)
                loss = sum(10 * torch.log10(nested_distortion(lam, shape, k, sigma2) / d)
                           for k, d in zip(RATES, opt)) / len(RATES)
                opt_adam.zero_grad()
                loss.backward()
                opt_adam.step()
            shape = torch.exp(logit).detach()
            gaps = [10 * math.log10(float(nested_distortion(lam, shape, k, sigma2)) / d)
                    for k, d in zip(RATES, opt)]
            psnr = " ".join(f"{-10 * math.log10(d / float(lam.sum())):6.2f}" for d in opt)
            print(f"{alpha:5.1f} {snr:4.0f} | {psnr:^52s} | " + " ".join(f"{-g:+6.3f}" for g in gaps))
    print("\nnested minus optimal is negative or zero: the dB a single shared power shape loses at each rate.")


if __name__ == "__main__":
    main()
