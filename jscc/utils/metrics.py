"""Image metrics: the pixel-level functions and the evaluation suite.

MetricSuite scores 8-bit-quantised reconstructions in fp32:

  psnr, ssim, msssim            pixels and structure
  lpips, lpips_vgg, dists       perceptual, full reference (lower is better)
  cls_top1, cls_prob            semantics: does an ImageNet ConvNeXt-T still see
                                the original's class (top-1 agreement; the
                                probability it gives that class)
  det_f1                        objects: COCO Faster R-CNN on the reconstruction
                                against its detections on the original
                                (same class, IoU >= 0.5, score >= 0.5)
  obj_psnr, bg_psnr, obj_lpips  inside / outside the boxes found on the original
  dino_sim                      DINOv2 patch-feature cosine similarity. This is
                                the training teacher: circular for runs trained
                                with --sem-weight or --align-weight

A metric undefined for an image (the detector found nothing in the original)
is NaN there and left out of the averages. A network that cannot load is
dropped with a warning and its metrics are omitted. Results on the originals
are cached per image, so the loader must be deterministic (valid and test are).
"""

import warnings

import torch
import torch.nn.functional as F

from utils.perception import (DISTSNet, Lazy, Teacher, box_mask, classifier, classify,
                              detector, lpips_net, match_f1)

MSSSIM_BETAS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
ALL_METRICS = ("psnr", "ssim", "msssim", "lpips", "lpips_vgg", "dists", "cls_top1", "cls_prob",
               "det_f1", "obj_psnr", "bg_psnr", "obj_lpips", "dino_sim")


def per_image_mse(x, y):
    return (x - y).pow(2).mean(dim=(1, 2, 3))


def psnr(mse, data_range=1.0):
    return 10.0 * torch.log10(data_range ** 2 / mse.clamp_min(1e-10))


def quantize_8bit(x):
    return torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0


def ssim(x, y, reduction="elementwise_mean"):
    from torchmetrics.functional.image import structural_similarity_index_measure as f
    return f(x.float(), y.float(), data_range=1.0, reduction=reduction)


def ms_ssim(x, y, reduction="elementwise_mean"):
    """MS-SSIM with as many scales as the image supports (5 at >= 161 px)."""
    from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as f
    size, n = min(x.shape[-2:]), 1
    while n < len(MSSSIM_BETAS) and 10 * 2 ** n < size:
        n += 1
    betas = MSSSIM_BETAS[:n]
    if n < len(MSSSIM_BETAS):
        total = sum(betas)
        betas = tuple(b / total for b in betas)
    return f(x.float(), y.float(), data_range=1.0, betas=betas, reduction=reduction)


def check_metrics(names):
    names = list(ALL_METRICS) if "all" in names else list(names)
    bad = [n for n in names if n not in ALL_METRICS]
    if bad:
        raise ValueError(f"unknown metric(s) {bad}; choose from {ALL_METRICS} or 'all'")
    return [n for n in ALL_METRICS if n in names]


class MetricSuite:
    def __init__(self, names, teacher="vit_small_patch14_reg4_dinov2.lvd142m", teacher_weights=None):
        self.names = check_metrics(names)
        self.nets = {"lpips_map": lpips_net("alex", spatial=True), "lpips_vgg": lpips_net("vgg"),
                     "dists": Lazy(DISTSNet), "classifier": classifier(), "detector": detector()}
        self.teacher = Teacher(teacher, teacher_weights)
        self.failed, self.cache = set(), {}

    def _net(self, key, device):
        if key in self.failed:
            return None
        try:
            if key == "teacher":
                self.teacher.net(device)
                return self.teacher
            return self.nets[key].get(device)
        except Exception as e:  # missing package or weights: skip, don't crash a run
            warnings.warn(f"{key} unavailable, its metrics are skipped: {e}")
            self.failed.add(key)
            return None

    def _ref(self, name, key, fn):
        if (name, key) not in self.cache:
            self.cache[(name, key)] = fn()
        return self.cache[(name, key)]

    @torch.no_grad()
    def __call__(self, recon, target, keys):
        """Per-image metrics {name: (B,) float tensor on CPU}; `keys` identify
        the images (their index in the loader) for the cache of originals."""
        with torch.autocast(device_type=target.device.type, enabled=False):
            return self._compute(quantize_8bit(recon.float()), target.float(), list(keys))

    def _compute(self, r, x, keys):
        want, dev, B = set(self.names), x.device, x.shape[0]
        out = {}
        sq = (r - x).pow(2)
        if "psnr" in want:
            out["psnr"] = psnr(sq.mean(dim=(1, 2, 3)))
        if "ssim" in want:
            out["ssim"] = ssim(r, x, reduction="none")
        if "msssim" in want:
            out["msssim"] = ms_ssim(r, x, reduction="none")
        lp_map = None
        if want & {"lpips", "obj_lpips"}:
            net = self._net("lpips_map", dev)
            if net is not None:
                lp_map = net(r * 2 - 1, x * 2 - 1)
                if "lpips" in want:
                    out["lpips"] = lp_map.mean(dim=(1, 2, 3))
        if "lpips_vgg" in want and (net := self._net("lpips_vgg", dev)) is not None:
            out["lpips_vgg"] = net(r * 2 - 1, x * 2 - 1).reshape(B)
        if "dists" in want and (net := self._net("dists", dev)) is not None:
            out["dists"] = net(r, x)
        if want & {"cls_top1", "cls_prob"} and (net := self._net("classifier", dev)) is not None:
            pr = classify(net, r)
            top = torch.stack([self._ref("cls", k, lambda i=i: classify(net, x[i:i + 1])[0].argmax())
                               for i, k in enumerate(keys)])
            if "cls_top1" in want:
                out["cls_top1"] = (pr.argmax(-1) == top).float()
            if "cls_prob" in want:
                out["cls_prob"] = pr.gather(1, top[:, None]).squeeze(1)
        obj = {"det_f1", "obj_psnr", "bg_psnr", "obj_lpips"}
        if want & obj and (net := self._net("detector", dev)) is not None:
            refs = [self._ref("det", k, lambda i=i: {n: v.cpu() for n, v in net([x[i]])[0].items()})
                    for i, k in enumerate(keys)]
            if "det_f1" in want:
                outs = net(list(r))
                out["det_f1"] = torch.tensor([match_f1(ref, {n: v.cpu() for n, v in o.items()})
                                              for ref, o in zip(refs, outs)])
            H, W = x.shape[-2:]
            nan = torch.tensor(float("nan"))
            op, bp, ol = [], [], []
            for i, ref in enumerate(refs):
                if len(ref["boxes"]) == 0:
                    op.append(nan), bp.append(nan), ol.append(nan)
                    continue
                m = box_mask(ref["boxes"], H, W, dev)
                e = sq[i].mean(0)
                op.append(psnr(e[m].mean()).cpu())
                bp.append(psnr(e[~m].mean()).cpu() if bool((~m).any()) else nan)
                ol.append(lp_map[i, 0][m].mean().cpu() if lp_map is not None else nan)
            for name, vals in (("obj_psnr", op), ("bg_psnr", bp), ("obj_lpips", ol)):
                if name in want and not (name == "obj_lpips" and lp_map is None):
                    out[name] = torch.stack(vals)
        if "dino_sim" in want and (t := self._net("teacher", dev)) is not None:
            _, rp, _ = t.encode(r)
            ref = torch.stack([self._ref("dino", k, lambda i=i: t.encode(x[i:i + 1])[1][0])
                               for i, k in enumerate(keys)])
            out["dino_sim"] = F.cosine_similarity(rp, ref, dim=-1).mean(-1)
        return {k: v.detach().float().reshape(-1).cpu() for k, v in out.items()}
