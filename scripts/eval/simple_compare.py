"""
simple_compare.py - give it two images, get back similarity/quality scores.

Run:
    python simple_compare.py real_photo.jpg render.png

First run downloads the models (a few GB total), later runs are fast.
"""
import sys

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

if len(sys.argv) != 3:
    sys.exit("usage: python simple_compare.py <real_photo> <render_or_second_image>")

photo_path, render_path = sys.argv[1], sys.argv[2]
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running on: {device}\n")

import numpy as np
from PIL import ImageOps

# exif_transpose = apply the rotation flag phones store, so photos aren't sideways
photo = ImageOps.exif_transpose(Image.open(photo_path)).convert("RGB")
render_rgba = ImageOps.exif_transpose(Image.open(render_path)).convert("RGBA")
render_rgba = render_rgba.resize(photo.size, Image.BICUBIC)

# If the render has a transparent background, only score the building:
# black out non-building pixels in BOTH images and crop to the building.
alpha = np.array(render_rgba)[..., 3]
mask = alpha > 127
if mask.all():
    print("Render has no transparent background -> comparing the WHOLE images.")
else:
    if not mask.any():
        sys.exit("render has no visible model pixels")
    print(f"Building covers {100 * mask.mean():.0f}% of the frame -> scoring only those pixels.")
    ys, xs = np.nonzero(mask)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    photo_np = np.array(photo)
    render_np = np.array(render_rgba.convert("RGB"))
    photo_np[~mask] = 0
    render_np[~mask] = 0
    photo = Image.fromarray(photo_np).crop(box)
    render_rgba = Image.fromarray(render_np).crop(box).convert("RGBA")
render = render_rgba.convert("RGB")

# A-FINE needs width and height divisible by 32, so round both images down to that
w, h = max(32, photo.width // 32 * 32), max(32, photo.height // 32 * 32)
photo = photo.resize((w, h), Image.BICUBIC)
render = render.resize((w, h), Image.BICUBIC)
photo_t = to_tensor(photo).unsqueeze(0).to(device)
render_t = to_tensor(render).unsqueeze(0).to(device)

results = []

# ---- DreamSim: how different do the two images look to a person? ----------
try:
    from dreamsim import dreamsim
    model, preprocess = dreamsim(pretrained=True, device=device)
    with torch.no_grad():
        d = model(preprocess(photo).to(device), preprocess(render).to(device))
    results.append(("DreamSim", float(d), "lower is better (0 = identical)"))
except Exception as e:
    results.append(("DreamSim", None, f"failed: {e}"))

# ---- pyiqa metrics that compare the two images ------------------------------
import pyiqa

available = pyiqa.list_models()
afine_name = next((m for m in available if "afine" in m.lower().replace("-", "")), None)

for name in [afine_name or "afine", "dists", "lpips", "ssim"]:
    if name == "afine" and afine_name is None:
        results.append(("AFINE", None, "not in your pyiqa version - see note in chat"))
        continue
    # SSIM uses float64 math, which Apple's GPU (mps) can't do, so run it on the CPU
    dev = "cpu" if name == "ssim" else device
    try:
        metric = pyiqa.create_metric(name, device=dev)
        with torch.no_grad():
            # .clone() = give each metric its own copy; some metrics modify the image in place
            score = float(metric(render_t.clone().to(dev), photo_t.clone().to(dev)))   # (judged, reference)
        direction = "lower is better" if metric.lower_better else "higher is better"
        results.append((name.upper(), score, direction))
    except Exception as e:
        results.append((name.upper(), None, f"failed: {e}"))

# ---- Q-ReAlign: judges ONLY the render, no reference photo ------------------
try:
    metric = pyiqa.create_metric("qrealign-mini", device=device)
    with torch.no_grad():
        r = float(metric(render_t.clone()))
        p = float(metric(photo_t.clone()))
    results.append(("Q-ReAlign (render)", r, "higher is better (0-1)"))
    results.append(("Q-ReAlign (photo)", p, "same scale, for comparison"))
except Exception as e:
    results.append(("Q-ReAlign", None, f"failed: {e}"))

# ---- print --------------------------------------------------------------------
print(f"{'metric':<20}{'score':>10}   meaning")
print("-" * 60)
for name, score, note in results:
    s = f"{score:.4f}" if score is not None else "-"
    print(f"{name:<20}{s:>10}   {note}")
