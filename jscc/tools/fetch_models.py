"""Download and check every pretrained network that the perceptual objective
and the evaluation suite use. Run it once on the server before training.

    python tools/fetch_models.py [--teacher NAME] [--teacher-weights FILE]

Sources: timm weights come from the Hugging Face hub (export
HF_ENDPOINT=https://hf-mirror.com if huggingface.co is unreachable, or pass a
local file with --teacher-weights); torchvision weights from
download.pytorch.org; the LPIPS and DISTS layer weights ship inside their pip
packages.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.perception import (DISTSNet, Lazy, RANDOM, Teacher, classifier,  # noqa: E402
                              detector, lpips_net)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="vit_small_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--teacher-weights", default=None)
    a = ap.parse_args()
    if RANDOM:
        print("JSCC_RANDOM_WEIGHTS=1 is set: nothing is downloaded")
    checks = [("DINOv2 teacher (timm)", lambda: Teacher(a.teacher, a.teacher_weights).net("cpu")),
              ("LPIPS AlexNet", lambda: lpips_net("alex").get("cpu")),
              ("LPIPS VGG", lambda: lpips_net("vgg").get("cpu")),
              ("DISTS (VGG16 + DISTS_pytorch weights)", lambda: Lazy(DISTSNet).get("cpu")),
              ("ConvNeXt-T classifier", lambda: classifier().get("cpu")),
              ("Faster R-CNN v2 detector", lambda: detector().get("cpu"))]
    failed = 0
    for name, fn in checks:
        try:
            fn()
            print(f"  ok      {name}")
        except Exception as e:
            failed += 1
            print(f"  FAILED  {name}: {type(e).__name__}: {str(e)[:200]}")
    print("all networks ready" if not failed else f"{failed} network(s) missing: their "
          f"metrics are skipped, and a missing teacher stops --sem/--align/--imp runs")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
