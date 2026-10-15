"""
metric_study.py - check whether the image metrics rank known cases correctly.

For each photo (with <id>_render.png and <id>_grey.png from
render_mapillary_views.py), builds the same masked crop simple_compare.py
scores, then scores the photo against controlled versions whose quality order
is known:

    identity     the photo itself (perfect)
    blur_1..3    the photo increasingly blurred inside the building outline
    shift_1..2   the photo slid sideways by 2% and 6% of the crop width (misalignment)
    scramble     the photo's 64 px blocks shuffled (right colours, wrong layout)

and against two real candidates:

    grey         the untextured grey model
    pipeline     the pipeline's textured render

Run with the metrics environment:
    ~/eval-venv/bin/python scripts/eval/metric_study.py outputs/views ID [ID ...]
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyiqa
import torch
from dreamsim import dreamsim
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor

MAX_SIDE = 1024
BLUR_SIGMAS = [2, 6, 18]  # pixels at MAX_SIDE scale
SHIFTS = [0.02, 0.06]  # fraction of the crop width
BLOCK = 64

# (closer to the photo, further from the photo) pairs every metric should get right
KNOWN_ORDER = [
    ("identity", "blur_1"), ("blur_1", "blur_2"), ("blur_2", "blur_3"),
    ("identity", "shift_1"), ("shift_1", "shift_2"),
    ("identity", "scramble"), ("blur_1", "scramble"), ("shift_1", "scramble"),
]
# A no-reference metric only sees one image, so it can only judge sharpness/structure
KNOWN_ORDER_NR = [
    ("identity", "blur_1"), ("blur_1", "blur_2"), ("blur_2", "blur_3"), ("identity", "scramble"),
]

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def crop_and_scale(img: np.ndarray, mask: np.ndarray, box, size) -> Image.Image:
    out = img.copy()
    out[~mask] = 0
    x0, y0, x1, y1 = box
    return Image.fromarray(out[y0:y1, x0:x1]).resize(size, Image.BICUBIC)


def build_variants(views: Path, img_id: str) -> tuple[Image.Image, dict[str, Image.Image]]:
    photo = np.array(ImageOps.exif_transpose(Image.open(views / f"{img_id}.jpg")).convert("RGB"))
    h, w = photo.shape[:2]
    render = np.array(Image.open(views / f"{img_id}_render.png").convert("RGBA").resize((w, h), Image.BICUBIC))
    grey = np.array(Image.open(views / f"{img_id}_grey.png").convert("RGBA").resize((w, h), Image.BICUBIC))
    mask = render[..., 3] > 127
    ys, xs = np.nonzero(mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    crop_w, crop_h = box[2] - box[0], box[3] - box[1]
    scale = min(1.0, MAX_SIDE / max(crop_w, crop_h))
    size = (max(32, int(crop_w * scale) // 32 * 32), max(32, int(crop_h * scale) // 32 * 32))

    def crop(img):
        return crop_and_scale(img, mask, box, size)

    reference = crop(photo)
    variants = {"identity": reference.copy()}
    for i, sigma in enumerate(BLUR_SIGMAS, 1):
        variants[f"blur_{i}"] = crop(cv2.GaussianBlur(photo, (0, 0), sigma / scale))
    for i, frac in enumerate(SHIFTS, 1):
        shift = np.float32([[1, 0, int(frac * crop_w)], [0, 1, 0]])
        shifted = cv2.warpAffine(photo, shift, (w, h), borderMode=cv2.BORDER_REPLICATE)
        variants[f"shift_{i}"] = crop(shifted)

    ref_np = np.array(reference)
    rng = np.random.default_rng(0)
    gh, gw = size[1] // BLOCK, size[0] // BLOCK
    blocks = [ref_np[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK] for r in range(gh) for c in range(gw)]
    order = rng.permutation(len(blocks))
    scrambled = ref_np.copy()
    for slot, src in enumerate(order):
        r, c = divmod(slot, gw)
        scrambled[r * BLOCK:(r + 1) * BLOCK, c * BLOCK:(c + 1) * BLOCK] = blocks[src]
    small_mask = np.array(Image.fromarray(mask[box[1]:box[3], box[0]:box[2]]).resize(size, Image.NEAREST))
    scrambled[~small_mask] = 0
    variants["scramble"] = Image.fromarray(scrambled)

    variants["grey"] = crop(grey[..., :3])
    variants["pipeline"] = crop(render[..., :3])
    return reference, variants


def main():
    views, img_ids = Path(sys.argv[1]), sys.argv[2:]
    print(f"Running on: {device}")

    ds_model, ds_preprocess = dreamsim(pretrained=True, device=device)
    fr_metrics = {name: pyiqa.create_metric(name, device="cpu" if name == "ssim" else device)
                  for name in ["afine_all", "dists", "lpips", "ssim"]}
    nr_metric = pyiqa.create_metric("qrealign-mini", device=device)
    timings: dict[str, list[float]] = {}

    def timed(name, fn):
        start = time.perf_counter()
        value = float(fn())
        timings.setdefault(name, []).append(time.perf_counter() - start)
        return value

    rows = []
    for img_id in img_ids:
        reference, variants = build_variants(views, img_id)
        ref_t = to_tensor(reference).unsqueeze(0)
        ref_quality = timed("qrealign", lambda: nr_metric(ref_t.clone().to(device)))
        print(f"\n{img_id}: crop {reference.size[0]}x{reference.size[1]}, photo Q-ReAlign {ref_quality:.3f}")
        for variant, img in variants.items():
            img_t = to_tensor(img).unsqueeze(0)
            with torch.no_grad():
                row = {"photo": img_id, "variant": variant}
                row["dreamsim"] = timed("dreamsim", lambda: ds_model(
                    ds_preprocess(img).to(device), ds_preprocess(reference).to(device)))
                for name, metric in fr_metrics.items():
                    dev = "cpu" if name == "ssim" else device
                    row[name] = timed(name, lambda: metric(img_t.clone().to(dev), ref_t.clone().to(dev)))
                row["qrealign"] = timed("qrealign", lambda: nr_metric(img_t.clone().to(device)))
            rows.append(row)
            print(f"  {variant:<9} " + "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k not in ("photo", "variant")))

    out_csv = views / "metric_study.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lower_better = {"dreamsim": True, "afine_all": True, "dists": True, "lpips": True, "ssim": False, "qrealign": False}
    print("\n=== Ranking accuracy on cases with a known answer ===")
    for metric, lower in lower_better.items():
        pairs = KNOWN_ORDER_NR if metric == "qrealign" else KNOWN_ORDER
        correct = total = 0
        misses: dict[tuple, int] = {}
        for img_id in img_ids:
            score = {r["variant"]: r[metric] for r in rows if r["photo"] == img_id}
            for better, worse in pairs:
                ok = score[better] < score[worse] if lower else score[better] > score[worse]
                correct += ok
                total += 1
                if not ok:
                    misses[(better, worse)] = misses.get((better, worse), 0) + 1
        miss_txt = ", ".join(f"{b}<{w} x{n}" for (b, w), n in misses.items()) or "none"
        print(f"{metric:<10} {correct}/{total} correct ({100 * correct / total:.0f}%)  "
              f"{np.mean(timings[metric]):.2f}s per call  misses: {miss_txt}")

    print("\n=== Mean score per variant (all photos) ===")
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    print(f"{'variant':<10}" + "".join(f"{m:>11}" for m in lower_better))
    for v in variants:
        means = [np.mean([r[m] for r in rows if r["variant"] == v]) for m in lower_better]
        print(f"{v:<10}" + "".join(f"{x:>11.3f}" for x in means))
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
