"""
pose_outline_score.py - how well does each pose source put the building where it is in the photo?

For every photo that has a pose in all the given runs (sidecars from export_single_building.py
--poses ...), renders the model's outline from each run's pose for that photo and segments the
photo (Mask2Former, Cityscapes classes). The score is the share of outline pixels the
segmentation calls building or wall; road, sidewalk, terrain and sky count against it, and
occluders (cars, people, poles, vegetation, ...) are ignored. Use a low model (e.g.
export_single_building.py --height 6) so every part of the outline should be building
whatever the real height. A pose that slides the building along a row of facades still lands
on building pixels: this catches sideways and heading errors, not along-street ones.

Only the outline is compared, not textures, so photos need not be held out.

Writes <out>/outline_scores.csv, <out>/outline_examples.png and prints a summary.

Usage (pipeline .venv, from the repo root):
    .venv/bin/python scripts/eval/pose_outline_score.py outputs/ab/outline/way_467844847.glb \
        raw=outputs/ab/raw/way_467844847.json computed=outputs/ab/computed/way_467844847.json \
        colmap=outputs/ab/colmap/way_467844847.json --only-source colmap --out outputs/ab/outline
"""
import argparse
import csv
import json
import tempfile
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from PIL import Image, ImageOps
from pyproj import Transformer
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

import render_mapillary_views as rv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEG_MODEL_ID = "facebook/mask2former-swin-large-cityscapes-semantic"
BUILDING = {"building", "wall"}
NOT_BUILDING = {"road", "sidewalk", "terrain", "sky"}
WIDTH = 1024
MIN_COVERAGE = 0.005
COLORS = {"raw": "red", "computed": "orange", "colmap": "lime"}


def segment(model, proc, photo: Image.Image) -> np.ndarray:
    inputs = proc(images=photo, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    return proc.post_process_semantic_segmentation(out, target_sizes=[photo.size[::-1]])[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("glb", type=Path)
    parser.add_argument("runs", nargs="+", help="name=sidecar.json")
    parser.add_argument("--only-source", default=None,
                        help="Only photos whose pose in run NAME came from that source (e.g. colmap)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(rv.REPO_ROOT / ".env")
    args.out.mkdir(parents=True, exist_ok=True)

    runs = dict(r.split("=", 1) for r in args.runs)
    sidecars = {name: json.loads(Path(p).read_text()) for name, p in runs.items()}
    ids = set.intersection(*(set(s["camera_poses"]) for s in sidecars.values()))
    if args.only_source:
        ids = {i for i in ids if sidecars[args.only_source]["camera_poses"][i]["pose_source"] == args.only_source}
    ids = sorted(ids)
    print(f"{len(ids)} photos with a pose in every run")

    eval_sidecar = json.loads(args.glb.with_suffix(".json").read_text())
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{eval_sidecar['utm_epsg']}", always_xy=True)
    meta = rv.fetch_metadata(ids)

    with tempfile.TemporaryDirectory() as tmp:
        jobs = []
        for i in ids:
            for name, side in sidecars.items():
                job = rv.camera_job(meta[i], eval_sidecar, tr, scale=WIDTH / meta[i]["width"],
                                    pose=side["camera_poses"][i])
                if job:
                    jobs.append({**job, "out": f"{tmp}/{i}_{name}.png"})
        rv.run_blender(args.glb, jobs)
        alpha = {}
        for i in ids:
            for name in sidecars:
                p = Path(f"{tmp}/{i}_{name}.png")
                alpha[i, name] = np.array(Image.open(p))[..., 3] > 127 if p.exists() else None

    visible = [i for i in ids if any(a is not None and a.mean() >= MIN_COVERAGE
                                     for a in (alpha[i, n] for n in sidecars))]
    print(f"{len(visible)} photos where the building is in view for at least one run")

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    proc = Mask2FormerImageProcessor.from_pretrained(SEG_MODEL_ID)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL_ID).to(device).eval()
    labels = model.config.id2label
    building_ids = [k for k, v in labels.items() if v in BUILDING]
    not_building_ids = [k for k, v in labels.items() if v in NOT_BUILDING]

    rows, examples = [], {}
    for n, i in enumerate(visible, 1):
        photo = ImageOps.exif_transpose(Image.open(BytesIO(
            requests.get(meta[i]["thumb_original_url"], timeout=60).content))).convert("RGB")
        some = next(a for a in (alpha[i, r] for r in sidecars) if a is not None)
        photo = photo.resize(some.shape[::-1], Image.BICUBIC)
        seg = segment(model, proc, photo)
        is_b, is_nb = np.isin(seg, building_ids), np.isin(seg, not_building_ids)
        row = {"photo": i}
        for name in sidecars:
            a = alpha[i, name]
            cov = 0.0 if a is None else float(a.mean())
            hit = 0 if a is None else int((a & is_b).sum())
            miss = 0 if a is None else int((a & is_nb).sum())
            row[f"{name}_coverage"] = round(cov, 4)
            row[f"{name}_score"] = round(hit / (hit + miss), 4) if hit + miss and cov >= MIN_COVERAGE else None
        rows.append(row)
        examples[i] = (photo, seg, {name: alpha[i, name] for name in sidecars})
        print(f"[{n}/{len(visible)}] {i}  " + "  ".join(
            f"{name} {row[f'{name}_score'] if row[f'{name}_score'] is not None else 'off-screen':>6}" for name in sidecars))

    with open(args.out / "outline_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nShare of the model's outline on building pixels (1.0 = all on the building), {len(rows)} photos")
    print(f"{'run':<10}{'mean':>7}{'median':>8}{'< 0.5':>7}{'off-screen':>12}{'best on':>9}")
    for name in sidecars:
        s = [r[f"{name}_score"] for r in rows if r[f"{name}_score"] is not None]
        off = sum(r[f"{name}_score"] is None for r in rows)
        best = sum(
            max(sidecars, key=lambda k: -1 if r[f"{k}_score"] is None else r[f"{k}_score"]) == name for r in rows
        )
        print(f"{name:<10}{np.mean(s):>7.3f}{np.median(s):>8.3f}{sum(x < 0.5 for x in s):>7}{off:>12}{best:>9}")

    by_spread = sorted(rows, key=lambda r: -np.ptp([r[f"{k}_score"] or 0 for k in sidecars]))
    shown = by_spread[:2] + by_spread[len(by_spread) // 2:len(by_spread) // 2 + 2]
    fig, axes = plt.subplots(len(shown), 1, figsize=(10, 3.2 * len(shown)), squeeze=False)
    for ax, r in zip(axes[:, 0], shown):
        photo, seg, alphas = examples[r["photo"]]
        ax.imshow(photo)
        ax.imshow(np.ma.masked_where(~np.isin(seg, building_ids), np.ones(seg.shape)), cmap="winter", alpha=0.3)
        for name, a in alphas.items():
            if a is not None and a.any():
                ax.contour(a, levels=[0.5], colors=COLORS.get(name, "white"), linewidths=1.5)
        ax.set_title(f"{r['photo']}: " + ", ".join(
            f"{k} {r[f'{k}_score'] if r[f'{k}_score'] is not None else 'off-screen'}" for k in sidecars), fontsize=9)
        ax.axis("off")
    fig.suptitle("Model outline from each run's pose (red raw, orange computed, green colmap); blue = segmented "
                 "building.\nTop two: biggest disagreement between runs; bottom two: typical photos", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "outline_examples.png", dpi=110)
    print(f"\nWrote {args.out / 'outline_scores.csv'} and {args.out / 'outline_examples.png'}")


if __name__ == "__main__":
    main()
