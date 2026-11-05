"""Pose alignment, stage 3 (segmentation): building and road masks for every ROI image.

Runs Mask2Former (segmentation.model: Swin-Large trained on Mapillary Vistas, 65
street-scene classes) on every frame of every sequence of an ROI and reduces its classes
to one mask per frame,
saved as a PNG in outputs/alignment/mapillary_roi_N/data/masks/<sequence folder>/<frame>.png:

    0  other (sidewalk, vegetation, poles, signs, ...)
  100  road surface (road, lane markings, crosswalks, bike lane, manholes, ...)
  200  building
  255  sky, vehicles and people (excluded from COLMAP features with --mask-moving)

Later stages need these masks: the COLMAP layer labels each 3D point by majority vote over
the pixels it was seen at (building points become facades, road points give the ground
plane and camera height) and can mask features with them. --previews also saves colour
overlays (data/masks/<sequence folder>/previews/) for checking by eye.

Each frame's class map is also kept in data/labels/<image key>.png (segmentation.model.
LabelCache): texturing builds its obstruction masks from these instead of segmenting the
same photos again. Frames that already have a mask are skipped unless --force.

The checkpoint's image processor resizes every frame to 384 x 384 before the model (aspect not
kept); the mask is scaled back to the frame. --batch N segments N frames per model call (default 4:
12% faster than one at a time, the same masks to 1 pixel in 10^7 on 120 roi_13 frames); --half runs
the model in 16-bit floats (no faster on an M-series GPU, 0.05% of pixels different; off).

Usage:
  python -m segmentation.street_masks outputs/alignment/mapillary_roi_1
  python -m segmentation.street_masks outputs/alignment/mapillary_roi_1 --previews
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pose_alignment.geo_common import (
    MASK_BUILDING,
    MASK_NAMES,
    MASK_OTHER,
    MASK_ROAD,
    MASK_SKY_OR_MOVING,
    RoiPaths,
)
from segmentation.model import (
    SEG_MODEL_ID,
    LabelCache,
    class_lut,
    id2label,
    load_model,
    segment,
    select_device,
)

DEFAULT_MODEL = SEG_MODEL_ID
# Mapillary Vistas class names per mask value.
CLASS_GROUPS = {
    MASK_BUILDING: ["Building"],
    MASK_ROAD: [
        "Road",
        "Lane Marking - General",
        "Lane Marking - Crosswalk",
        "Crosswalk - Plain",
        "Bike Lane",
        "Manhole",
        "Pothole",
        "Catch Basin",
    ],
    MASK_SKY_OR_MOVING: [
        "Sky",
        "Person",
        "Bicyclist",
        "Motorcyclist",
        "Other Rider",
        "Bicycle",
        "Boat",
        "Bus",
        "Car",
        "Caravan",
        "Motorcycle",
        "On Rails",
        "Other Vehicle",
        "Trailer",
        "Truck",
        "Wheeled Slow",
        "Car Mount",
        "Ego Vehicle",
        "Bird",
        "Ground Animal",
    ],
}
PREVIEW_COLOURS = {
    MASK_OTHER: (0, 0, 0),
    MASK_ROAD: (128, 64, 128),
    MASK_BUILDING: (0, 140, 255),
    MASK_SKY_OR_MOVING: (230, 200, 120),
}  # BGR


def missing_masks(paths):
    """{sequence folder: [frame files without a mask]}."""
    out = {}
    for seq in paths.sequences():
        frames = sorted(
            p.name for p in (paths.images / seq["folder"]).glob("frame_*.jpg")
        )
        todo = [
            f
            for f in frames
            if not (paths.masks / seq["folder"] / f"{Path(f).stem}.png").exists()
        ]
        if todo:
            out[seq["folder"]] = todo
    return out


def segment_roi(
    roi_dir,
    model_name=DEFAULT_MODEL,
    device=None,
    previews=False,
    force=False,
    batch=4,
    half=False,
    out_root=None,
    limit=None,
):
    """Write a mask for every frame of the ROI (into out_root instead of data/masks, and only the first
    `limit` frames per sequence, for tests); returns the masks folder."""
    paths = RoiPaths(roi_dir)
    if out_root:
        paths.masks = Path(out_root)
    sequences = paths.sequences()
    todo = {}
    for seq in sequences:
        frames = sorted((paths.images / seq["folder"]).glob("frame_*.jpg"))[:limit]
        todo[seq["folder"]] = [
            p
            for p in frames
            if force or not (paths.masks / seq["folder"] / f"{p.stem}.png").exists()
        ]
    total = sum(len(v) for v in todo.values())
    print(f"{len(sequences)} sequence(s), {total} frame(s) to segment")
    paths.masks.mkdir(parents=True, exist_ok=True)
    if total:
        device = select_device(device, allow_mps=True)
        print(f"Model {model_name} on {device}")
        _, model = load_model(model_name, device, half=half)
        names = id2label(model)
        lut = np.full(256, MASK_OTHER, np.uint8)
        for value, classes in CLASS_GROUPS.items():
            lut = np.where(class_lut(names, classes) > 0, np.uint8(value), lut)
        labels = LabelCache(paths.labels)
        labels.start(model_name, names)
        (paths.masks / "classes.json").write_text(
            json.dumps(
                {
                    "model": model_name,
                    "values": {str(v): n for v, n in MASK_NAMES.items()},
                    "vistas_classes": {
                        MASK_NAMES[v]: c for v, c in CLASS_GROUPS.items()
                    },
                },
                indent=1,
            )
        )

        t0, done = time.time(), 0
        for folder, frames in todo.items():
            meta = paths.frames(folder)
            out = paths.masks / folder
            out.mkdir(parents=True, exist_ok=True)
            if previews:
                (out / "previews").mkdir(exist_ok=True)
            for k0 in range(0, len(frames), batch):
                chunk = frames[k0 : k0 + batch]
                imgs = [cv2.imread(str(p)) for p in chunk]
                segs = segment(imgs, model_name, device, half=half)
                for p, img, seg in zip(chunk, imgs, segs):
                    mask = lut[seg]
                    cv2.imwrite(str(out / f"{p.stem}.png"), mask)
                    if p.name in meta:
                        labels.save(RoiPaths.image_key(folder, meta[p.name]), seg)
                    if previews:
                        colour = np.zeros_like(img)
                        for value, bgr in PREVIEW_COLOURS.items():
                            colour[mask == value] = bgr
                        overlay = np.where(
                            mask[..., None] == MASK_OTHER,
                            img,
                            (0.45 * img + 0.55 * colour).astype(np.uint8),
                        )
                        w = min(1280, img.shape[1])
                        cv2.imwrite(
                            str(out / "previews" / f"{p.stem}.jpg"),
                            cv2.resize(
                                overlay, (w, round(w * img.shape[0] / img.shape[1]))
                            ),
                            [cv2.IMWRITE_JPEG_QUALITY, 85],
                        )
                    done += 1
                    if done % 25 == 0 or done == total:
                        print(
                            f"  {done}/{total} frames ({(time.time() - t0) / done:.2f} s/frame)"
                        )

    # Pixel shares, as a quick sanity check (few building pixels = little to align to).
    for seq in sequences:
        counts = np.zeros(256, np.int64)
        for m in sorted((paths.masks / seq["folder"]).glob("frame_*.png"))[::5]:
            counts += np.bincount(
                cv2.imread(str(m), cv2.IMREAD_GRAYSCALE).ravel(), minlength=256
            )
        if counts.sum():
            print(
                f"  {seq['folder']}: "
                + ", ".join(
                    f"{n} {100 * counts[v] / counts.sum():.0f}%"
                    for v, n in MASK_NAMES.items()
                )
            )
    print(f"Masks in {paths.masks}")
    return paths.masks


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "roi_dir",
        type=Path,
        help="ROI folder from data_ingest/mapillary_sequences.py (outputs/alignment/mapillary_roi_N)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face Mask2Former semantic checkpoint (Vistas classes)",
    )
    parser.add_argument("--device", help="mps, cuda or cpu (default: best available)")
    parser.add_argument(
        "--previews",
        action="store_true",
        help="Also save colour overlays for checking by eye",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-segment frames that already have masks"
    )
    parser.add_argument("--batch", type=int, default=4, help="Frames per model call")
    parser.add_argument(
        "--half", action="store_true", help="Run the model in 16-bit floats"
    )
    parser.add_argument(
        "--out", type=Path, help="Write the masks here instead of data/masks (tests)"
    )
    parser.add_argument(
        "--limit", type=int, help="Only the first N frames of each sequence (tests)"
    )
    args = parser.parse_args()
    segment_roi(
        args.roi_dir,
        args.model,
        args.device,
        args.previews,
        args.force,
        args.batch,
        args.half,
        args.out,
        args.limit,
    )


if __name__ == "__main__":
    main()
