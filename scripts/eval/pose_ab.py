"""
pose_ab.py - score models textured with different camera poses on the same held-out photos.

Each run folder holds views/ from render_mapillary_views.py --own-poses: per photo
<id>.jpg, <id>_render.png (rendered from that run's pose for the photo) and <id>_grey.png.
Scores use the masked crop of metric_study.py: lower DreamSim / LPIPS is closer to the
photo. The grey (untextured) model is scored too, as a no-texture baseline.

Writes <root>/pose_ab.csv and <root>/pose_ab.png.

Run with the metrics environment:
    ~/eval-venv/bin/python scripts/eval/pose_ab.py outputs/ab raw computed colmap --ids ID [ID ...]
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyiqa
import torch
from dreamsim import dreamsim
from torchvision.transforms.functional import to_tensor

from metric_study import build_variants, device


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("root", type=Path)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()

    ds_model, ds_preprocess = dreamsim(pretrained=True, device=device)
    lpips = pyiqa.create_metric("lpips", device=device)

    rows, crops = [], {}
    for img_id in args.ids:
        for run in args.runs:
            views = args.root / run / "views"
            if not (views / f"{img_id}_render.png").exists():
                print(f"{run}/{img_id}: no render, skipped")
                continue
            reference, variants = build_variants(views, img_id)
            crops[img_id, run] = (reference, variants["pipeline"])
            ref_t = to_tensor(reference).unsqueeze(0).to(device)
            for kind in ("pipeline", "grey"):
                img = variants[kind]
                with torch.no_grad():
                    ds = float(ds_model(ds_preprocess(img).to(device), ds_preprocess(reference).to(device)))
                    lp = float(lpips(to_tensor(img).unsqueeze(0).to(device), ref_t))
                rows.append({"photo": img_id, "run": run, "model": kind, "dreamsim": ds, "lpips": lp})
            r = rows[-2]
            print(f"{img_id} {run:<20} textured DreamSim {r['dreamsim']:.3f}  LPIPS {r['lpips']:.3f}")

    with open(args.root / "pose_ab.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'run':<20}{'DreamSim':>10}{'LPIPS':>8}{'grey DreamSim':>15}{'best on':>9}")
    for run in args.runs:
        tex = [r for r in rows if r["run"] == run and r["model"] == "pipeline"]
        grey = [r for r in rows if r["run"] == run and r["model"] == "grey"]
        wins = sum(
            min((r for r in rows if r["photo"] == t["photo"] and r["model"] == "pipeline"),
                key=lambda r: r["dreamsim"])["run"] == run
            for t in tex
        )
        print(f"{run:<20}{np.mean([r['dreamsim'] for r in tex]):>10.3f}{np.mean([r['lpips'] for r in tex]):>8.3f}"
              f"{np.mean([r['dreamsim'] for r in grey]):>15.3f}{f'{wins}/{len(tex)}':>9}")

    score = {(r["photo"], r["run"]): r for r in rows if r["model"] == "pipeline"}
    fig, axes = plt.subplots(len(args.ids), len(args.runs) + 1, figsize=(3.2 * (len(args.runs) + 1), 2.6 * len(args.ids)),
                             squeeze=False)
    for i, img_id in enumerate(args.ids):
        best = min((r for r in args.runs if (img_id, r) in score), key=lambda r: score[img_id, r]["dreamsim"],
                   default=None)
        for j, run in enumerate([None, *args.runs]):
            ax = axes[i, j]
            ax.axis("off")
            if run is None:
                ref = next((crops[img_id, r][0] for r in args.runs if (img_id, r) in crops), None)
                if ref is not None:
                    ax.imshow(ref)
                ax.set_title(f"photo {img_id}", fontsize=8)
            elif (img_id, run) in crops:
                ax.imshow(crops[img_id, run][1])
                s = score[img_id, run]
                ax.set_title(f"{run}\nDreamSim {s['dreamsim']:.3f}  LPIPS {s['lpips']:.3f}", fontsize=8,
                             fontweight="bold" if run == best else "normal", color="green" if run == best else "black")
    fig.suptitle("Held-out photos vs. the model rendered from each run's own pose (lower is closer; best in green)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.root / "pose_ab.png", dpi=130)
    print(f"\nWrote {args.root / 'pose_ab.csv'} and {args.root / 'pose_ab.png'}")


if __name__ == "__main__":
    main()
