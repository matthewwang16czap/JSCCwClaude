"""Rate-distortion points of a standard image codec on a test set, for the separation
baseline of tools/schedule_sim.py (alloc/digital.py). CPU; BPG takes ~1-2 min on Kodak.

    python tools/digital_rd.py --codec bpg --images $JSCC_DATA_ROOT/Kodak --out results/rd_bpg.json

Codecs:
    bpg      bpgenc / bpgdec on PATH (libbpg; HEVC intra; the usual DeepJSCC baseline),
             quantiser --bpg-q (51 = smallest file); --bpg-args passes extra encoder flags
             (e.g. "-f 444")
    heic     pillow-heif (pip install pillow-heif; HEVC intra through libheif/x265)
    jpeg2000 Pillow with OpenJPEG (a scalable codec)
    webp     Pillow
    jpeg     Pillow

PSNR is over the whole RGB image at 8 bits, as tools/mixed_snr.py and the trainer score
the codec (MSE over pixels and channels, peak 1). bpp = 8 x file bytes / pixels. Image
names are the file names, so they match the utility files of tools/utility_fit.py.
"""

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm")


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) / 255.0 - b.astype(np.float64) / 255.0) ** 2)
    return float(-10.0 * np.log10(max(mse, 1e-12)))


def run_bpg(path, q, extra, tmp):
    out, rec = os.path.join(tmp, "x.bpg"), os.path.join(tmp, "x.png")
    subprocess.run(["bpgenc", "-q", str(q), *extra, "-o", out, path], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["bpgdec", "-o", rec, out], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    with Image.open(rec) as im:
        return os.path.getsize(out), np.asarray(im.convert("RGB"))


def run_pillow(img, fmt, q):
    buf = io.BytesIO()
    if fmt == "JPEG2000":
        img.save(buf, "JPEG2000", quality_mode="rates", quality_layers=[q], irreversible=True)
    elif fmt == "HEIF":
        img.save(buf, "HEIF", quality=int(q))
    else:
        img.save(buf, fmt, quality=int(q))
    n = buf.tell()
    buf.seek(0)
    with Image.open(buf) as im:
        return n, np.asarray(im.convert("RGB"))


def qualities(a):
    if a.codec == "bpg":
        return a.bpg_q
    if a.codec == "jpeg2000":                       # compression ratios, large = small file
        return [400, 300, 200, 150, 120, 100, 80, 64, 50, 40, 32, 25, 20, 16, 12, 10, 8, 6]
    return [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--codec", required=True, choices=["bpg", "heic", "jpeg2000", "webp", "jpeg"])
    ap.add_argument("--images", required=True, help="folder of test images (e.g. Kodak)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bpg-q", nargs="+", type=int, default=list(range(51, 9, -2)))
    ap.add_argument("--bpg-args", default="", help="extra bpgenc flags, e.g. '-f 444'")
    ap.add_argument("--sym-per-pixel", type=float, default=1.0 / 16,
                    help="complex channel uses per pixel in one slot (one prefix phase)")
    ap.add_argument("--link", default="cqi", choices=["cqi", "shannon"])
    ap.add_argument("--max-phases", type=int, default=12, help="slots one user may take")
    a = ap.parse_args()
    if a.codec == "bpg" and not (shutil.which("bpgenc") and shutil.which("bpgdec")):
        raise SystemExit("bpgenc / bpgdec not on PATH: build libbpg (https://bellard.org/bpg/) "
                         "or use --codec heic (pip install pillow-heif)")
    if a.codec == "heic":
        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            raise SystemExit("pip install pillow-heif")
        register_heif_opener()
    paths = sorted(os.path.join(a.images, f) for f in os.listdir(a.images)
                   if f.lower().endswith(EXT))
    if not paths:
        raise SystemExit(f"no images in {a.images}")
    fmt = {"heic": "HEIF", "jpeg2000": "JPEG2000", "webp": "WEBP", "jpeg": "JPEG"}.get(a.codec)
    names, curves = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for k, p in enumerate(paths):
            with Image.open(p) as im:
                img = im.convert("RGB")
            ref = np.asarray(img)
            pix = ref.shape[0] * ref.shape[1]
            pts = []
            for q in qualities(a):
                n, rec = (run_bpg(p, q, a.bpg_args.split(), tmp) if a.codec == "bpg"
                          else run_pillow(img, fmt, q))
                pts.append([8.0 * n / pix, psnr(ref, rec)])
            names.append(os.path.basename(p))
            curves.append(sorted(pts))
            if (k + 1) % 6 == 0 or k + 1 == len(paths):
                print(f"  {k + 1}/{len(paths)} images", flush=True)
    c = np.array([[np.interp(b, np.array(cv)[:, 0], np.array(cv)[:, 1]) for b in (0.1, 0.25, 0.5, 1.0)]
                  for cv in curves])
    print(f"{a.codec}: {len(names)} images, {len(curves[0])} points each; mean PSNR at 0.1 / 0.25 / "
          f"0.5 / 1.0 bpp: " + " / ".join(f"{v:.2f}" for v in c.mean(0)))
    out = {"kind": "digital", "codec": a.codec, "images": names, "curves": curves,
           "sym_per_pixel": a.sym_per_pixel, "link": a.link, "gap_db": 0.0, "P": a.max_phases,
           "mode": "step", "d_empty": 0.1,
           "meta": {"images_dir": a.images, "bpg_args": a.bpg_args if a.codec == "bpg" else None}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
