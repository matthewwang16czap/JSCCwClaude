"""Training objective: a pixel loss plus optional perceptual terms, per budget.

For every budget u decoded in the step

    L(u) = pixel(u) + w(u) * sum_t lambda_t * r_t(u) * term_t(u)

  pixel   MSE (or 1 - MS-SSIM). With --imp-weight a the squared error is
          weighted by 1 + a*w(u)*(m - 1), where m is the DINOv2 teacher's CLS
          attention over the image (mean 1, capped at 4): short prefixes are
          scored mostly on what the teacher attends to
  sem     1 - cosine similarity of the teacher's patch features of the
          reconstruction and of the input (--sem-weight)
  align   REPA-style: a small head maps the decoder's hidden state at mid depth
          to the teacher's patch features of the input (--align-weight). This
          shapes the transmitted tokens themselves, not just the output
  lpips   LPIPS (AlexNet) of reconstruction vs input (--lpips-weight)

w(u) = ((u_max - u) / (u_max - u_min))^gamma under --perc-schedule budget: 1 at
the smallest budget, 0 at the full one, so the full budget keeps a pure pixel
objective. `constant` sets w = 1 for every budget (the ablation).
r_t(u) = pixel(u) / term_t(u), detached: every lambda is RELATIVE to the pixel
loss, so one setting transfers across backbones and budgets whose MSE differs
tenfold. The loss is the mean over the step's budgets, computed in fp32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.metrics import ms_ssim, per_image_mse, psnr
from utils.perception import Teacher, lpips_net


def _ratio(pixel, term):
    return pixel.detach() / term.detach().clamp_min(1e-8)


def importance_map(att, size, cap=4.0):
    """Teacher attention (B, h, w) -> pixel weights (B, 1, H, W), mean 1, capped."""
    a = att / att.mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    a = a.clamp(max=cap)
    a = a / a.mean(dim=(1, 2), keepdim=True)
    return F.interpolate(a.unsqueeze(1), size=size, mode="bilinear", align_corners=False)


class Objective(nn.Module):
    def __init__(self, cfg, tap_dim=None):
        super().__init__()
        self.loss = cfg.loss
        self.w_sem, self.w_align = cfg.sem_weight, cfg.align_weight
        self.w_imp, self.w_lpips = cfg.imp_weight, cfg.lpips_weight
        self.schedule, self.gamma = cfg.perc_schedule, cfg.perc_gamma
        self.u_lo, self.u_hi = float(cfg.cbr_units[0]), float(cfg.cbr_units[-1])
        needs_teacher = bool(self.w_sem or self.w_align or self.w_imp)
        self.teacher = Teacher(cfg.teacher, cfg.teacher_weights) if needs_teacher else None
        self.lpips = lpips_net("alex") if self.w_lpips else None
        self.align = None
        if self.w_align:
            d = self.teacher.dim
            self.align = nn.Sequential(nn.Linear(tap_dim, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    @property
    def wants_taps(self):
        return self.align is not None

    def factor(self, u):
        """w(u): the weight of the perceptual terms at budget u."""
        if self.schedule == "constant":
            return 1.0
        t = (self.u_hi - float(u)) / max(self.u_hi - self.u_lo, 1e-9)
        return min(1.0, max(0.0, t)) ** self.gamma

    def forward(self, recons, target, units, taps=None, want_metrics=True):
        with torch.autocast(device_type=target.device.type, enabled=False):
            return self._forward(recons, target, units, taps, want_metrics)

    def _align_loss(self, grid, t_patch, t_hw):
        g = F.adaptive_avg_pool2d(grid.float(), t_hw).flatten(2).transpose(1, 2)
        return (1.0 - F.cosine_similarity(self.align(g), t_patch, dim=-1)).mean()

    def _forward(self, recons, target, units, taps, want_metrics):
        x = target.float()
        t_patch = t_hw = imp = None
        if self.teacher is not None:
            if self.w_imp:
                _, t_patch, t_hw, att = self.teacher.encode_with_attention(x)
                imp = importance_map(att, x.shape[-2:])
            else:
                with torch.no_grad():
                    _, t_patch, t_hw = self.teacher.encode(x)
        total, parts = 0.0, {}
        for i, (r, u) in enumerate(zip(recons, units)):
            r, w = r.float(), self.factor(u)
            if self.loss == "mse":
                err = (r - x).pow(2)
                if imp is not None:
                    err = err * (1.0 + self.w_imp * w * (imp - 1.0))
                pixel = err.mean()
            else:
                pixel = 1.0 - ms_ssim(r, x)
            term = pixel
            if self.w_sem and w > 0:
                _, r_patch, _ = self.teacher.encode(r)
                sem = (1.0 - F.cosine_similarity(r_patch, t_patch, dim=-1)).mean()
                term = term + self.w_sem * w * _ratio(pixel, sem) * sem
                parts.setdefault("sem", []).append(sem.detach())
            if self.align is not None:   # always in the graph (DDP), weight may be 0
                al = self._align_loss(taps[i], t_patch, t_hw)
                term = term + self.w_align * w * _ratio(pixel, al) * al
                parts.setdefault("align", []).append(al.detach())
            if self.lpips is not None and w > 0:
                lp = self.lpips.get(x.device)(r * 2 - 1, x * 2 - 1).mean()
                term = term + self.w_lpips * w * _ratio(pixel, lp) * lp
                parts.setdefault("lpips", []).append(lp.detach())
            total = total + term
        loss = total / len(recons)
        if not want_metrics:
            return loss, {}
        with torch.no_grad():
            mses = [per_image_mse(r.float().clamp(0, 1), x) for r in recons]
            m = {"mse": [float(v.mean()) for v in mses],
                 "psnr": [float(psnr(v).mean()) for v in mses]}
            m.update({k: float(torch.stack(v).mean()) for k, v in parts.items()})
        return loss, m
