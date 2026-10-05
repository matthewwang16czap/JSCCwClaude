"""Frozen pretrained networks for the perceptual objective and the evaluation
suite: a DINOv2 teacher (timm), LPIPS (AlexNet, VGG), DISTS, an ImageNet
classifier (ConvNeXt-T) and a COCO detector (Faster R-CNN v2).

None is trained or saved: every wrapper keeps its network outside the module
registry, builds it on first use and runs it in fp32. Weights come from the
usual caches: torchvision (download.pytorch.org), timm through the Hugging Face
hub (set HF_ENDPOINT for a mirror) or a local file (--teacher-weights).
JSCC_RANDOM_WEIGHTS=1 builds everything with random weights, for CPU tests
without downloads. tools/fetch_models.py downloads and checks them all.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

RANDOM = os.environ.get("JSCC_RANDOM_WEIGHTS") == "1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    return (x - x.new_tensor(mean).view(1, -1, 1, 1)) / x.new_tensor(std).view(1, -1, 1, 1)


def resize_short(x, short, multiple=1):
    """Resize so that the short side is `short`, both sides rounded to `multiple`."""
    H, W = x.shape[-2:]
    s = short / min(H, W)
    h = max(multiple, round(H * s / multiple) * multiple)
    w = max(multiple, round(W * s / multiple) * multiple)
    if (h, w) == (H, W):
        return x
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)


class Lazy:
    """A frozen network, built on first use without shifting the caller's
    random stream, and moved to whichever device it is asked for."""

    def __init__(self, factory):
        self.factory, self.net = factory, None

    def get(self, device):
        device = torch.device(device)
        if self.net is None:
            with torch.random.fork_rng(devices=[]):
                net = self.factory()
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            self.net = net
        if next(self.net.parameters()).device != device:
            self.net.to(device)
        return self.net


class Teacher:
    """DINOv2 through timm. The default, ViT-S/14 with registers, has attention
    maps free of the background artefacts of the original DINOv2."""

    def __init__(self, name, weights=None):
        self.name, self.weights = name, weights
        self.lazy = Lazy(self._factory)
        self.info = None

    def _factory(self):
        import timm
        kw = dict(num_classes=0, dynamic_img_size=True)
        if RANDOM:
            return timm.create_model(self.name, pretrained=False, **kw)
        if self.weights:
            kw["pretrained_cfg_overlay"] = dict(file=self.weights)
        return timm.create_model(self.name, pretrained=True, **kw)

    def net(self, device):
        m = self.lazy.get(device)
        if self.info is None:
            cfg = getattr(m, "pretrained_cfg", None) or {}
            self.info = dict(dim=m.num_features, prefix=m.num_prefix_tokens,
                             patch=m.patch_embed.patch_size[0],
                             mean=tuple(cfg.get("mean", IMAGENET_MEAN)),
                             std=tuple(cfg.get("std", IMAGENET_STD)))
        return m

    @property
    def dim(self):
        if self.info is None:
            self.net("cpu")
        return self.info["dim"]

    def _prep(self, x):
        p = self.info["patch"]
        x = resize_short(x, 224, p)
        return normalize(x, self.info["mean"], self.info["std"]), (x.shape[-2] // p, x.shape[-1] // p)

    def encode(self, x):
        """[0, 1] images -> (cls (B, D), patches (B, h*w, D), (h, w)). Gradients
        flow to x (the weights are frozen)."""
        m = self.net(x.device)
        t, hw = self._prep(x)
        tok = m.forward_features(t)
        return tok[:, 0], tok[:, self.info["prefix"]:], hw

    @torch.no_grad()
    def encode_with_attention(self, x):
        """encode(), plus the last block's CLS attention over the patches (B, h, w)."""
        m = self.net(x.device)
        t, hw = self._prep(x)
        z = m.norm_pre(m.patch_drop(m._pos_embed(m.patch_embed(t))))
        for blk in m.blocks[:-1]:
            z = blk(z)
        last, pre = m.blocks[-1], self.info["prefix"]
        attn = last.attn
        h = last.norm1(z)
        B, N, C = h.shape
        qkv = attn.qkv(h).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k = attn.q_norm(qkv[0]), attn.k_norm(qkv[1])
        a = ((q[:, :, :1] * attn.scale) @ k.transpose(-2, -1)).softmax(-1)
        a = a[:, :, 0, pre:].mean(1).reshape(B, *hw)
        tok = m.norm(last(z))
        return tok[:, 0], tok[:, pre:], hw, a


def lpips_net(kind="alex", spatial=False):
    def factory():
        import lpips
        return lpips.LPIPS(net=kind, spatial=spatial, pnet_rand=RANDOM, verbose=False)
    return Lazy(factory)


class _L2Pool(nn.Module):
    """DISTS' Hanning-window L2 pooling (replaces VGG's max pooling)."""

    def __init__(self, channels):
        super().__init__()
        g = torch.tensor([0.5, 1.0, 0.5])
        g = g[:, None] * g[None, :]
        self.register_buffer("filter", (g / g.sum())[None, None].repeat(channels, 1, 1, 1))

    def forward(self, x):
        out = F.conv2d(x.pow(2), self.filter, stride=2, padding=1, groups=x.shape[1])
        return (out + 1e-12).sqrt()


class DISTSNet(nn.Module):
    """DISTS (Ding et al., 2020), reproducing the reference DISTS_pytorch code
    with torchvision's VGG16 and the alpha/beta weights that package ships."""

    CHANNELS = (3, 64, 128, 256, 512, 512)

    def __init__(self):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(weights=None if RANDOM else models.VGG16_Weights.IMAGENET1K_V1).features

        def stage(pool, lo, hi):
            return nn.Sequential(*([_L2Pool(pool)] if pool else []), *[vgg[i] for i in range(lo, hi)])

        self.stages = nn.ModuleList([stage(0, 0, 4), stage(64, 5, 9), stage(128, 10, 16),
                                     stage(256, 17, 23), stage(512, 24, 30)])
        n = sum(self.CHANNELS)
        alpha = torch.empty(1, n, 1, 1).normal_(0.1, 0.01)
        beta = torch.empty(1, n, 1, 1).normal_(0.1, 0.01)
        if not RANDOM:
            import DISTS_pytorch
            path = os.path.join(os.path.dirname(DISTS_pytorch.__file__), "weights.pt")
            w = torch.load(path, map_location="cpu", weights_only=True)
            alpha, beta = w["alpha"], w["beta"]
        self.register_buffer("alpha", alpha)
        self.register_buffer("beta", beta)

    def forward(self, x, y):
        """Per-image DISTS of [0, 1] images, lower is better."""
        feats = []
        for img in (x, y):
            h, f = normalize(img), [img]
            for s in self.stages:
                h = s(h)
                f.append(h)
            feats.append(f)
        w = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w, self.CHANNELS, dim=1)
        beta = torch.split(self.beta / w, self.CHANNELS, dim=1)
        d1 = d2 = 0.0
        for k in range(len(self.CHANNELS)):
            a, b = feats[0][k], feats[1][k]
            am, bm = a.mean([2, 3], keepdim=True), b.mean([2, 3], keepdim=True)
            d1 = d1 + (alpha[k] * (2 * am * bm + 1e-6) / (am ** 2 + bm ** 2 + 1e-6)).sum(1, keepdim=True)
            av = (a - am).pow(2).mean([2, 3], keepdim=True)
            bv = (b - bm).pow(2).mean([2, 3], keepdim=True)
            cov = (a * b).mean([2, 3], keepdim=True) - am * bm
            d2 = d2 + (beta[k] * (2 * cov + 1e-6) / (av + bv + 1e-6)).sum(1, keepdim=True)
        return 1.0 - (d1 + d2).reshape(-1)


def classifier():
    def factory():
        from torchvision import models
        return models.convnext_tiny(
            weights=None if RANDOM else models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    return Lazy(factory)


def classify(net, x):
    """[0, 1] images -> ImageNet class probabilities (resize 236, centre crop 224)."""
    x = resize_short(x, 236)
    h, w = x.shape[-2:]
    t, l = (h - 224) // 2, (w - 224) // 2
    return net(normalize(x[..., t:t + 224, l:l + 224])).softmax(-1)


def detector():
    def factory():
        from torchvision.models import detection
        weights = None if RANDOM else detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        return detection.fasterrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None,
                                                    box_score_thresh=0.5)
    return Lazy(factory)


def match_f1(ref, out, iou_thr=0.5):
    """F1 of the detections `out` against `ref` (same class, IoU >= 0.5,
    greedy by ref score). NaN when `ref` found nothing."""
    from torchvision.ops import box_iou
    n_ref, n_out = len(ref["boxes"]), len(out["boxes"])
    if n_ref == 0:
        return float("nan")
    if n_out == 0:
        return 0.0
    iou = box_iou(ref["boxes"], out["boxes"]) * (ref["labels"][:, None] == out["labels"][None, :])
    used, tp = set(), 0
    for i in torch.argsort(ref["scores"], descending=True).tolist():
        cand = [(float(iou[i, j]), j) for j in range(n_out) if j not in used and iou[i, j] >= iou_thr]
        if cand:
            used.add(max(cand)[1])
            tp += 1
    if tp == 0:
        return 0.0
    p, r = tp / n_out, tp / n_ref
    return 2 * p * r / (p + r)


def box_mask(boxes, H, W, device):
    m = torch.zeros(H, W, dtype=torch.bool, device=device)
    for x0, y0, x1, y1 in boxes.round().long().tolist():
        m[max(0, y0):min(H, y1), max(0, x0):min(W, x1)] = True
    return m
