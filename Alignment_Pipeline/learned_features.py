"""Feature matching for the reconstruct stage: ALIKED keypoints matched with LightGlue
(nn_features.py swaps in nearest-neighbour matching for frames close together).

Writes them in the formats of COLMAP's documented route for custom features, so COLMAP's own
geometric verification, mapper and everything after stay unchanged (reconstruct.py runs
feature_importer and matches_importer --match_type raw on them):
  <out>/features/<image name>.txt   "N 128", then one "x y scale orientation" line per keypoint
                                    plus 128 zero descriptor values (COLMAP needs the field;
                                    matching is already done)
  <out>/matches.txt                 per pair "name1 name2", one "idx1 idx2" line per match, blank line

Pairs are sequential like COLMAP's sequential matcher: each frame with its next --overlap frames
(capture order = file name order). Keypoints on black pixels of the COLMAP feature masks
(<mask dir>/<image name>.png: burned-in overlays, sky, vehicles, people) are dropped.

Runs as its own process (reconstruct.py calls it): PyTorch and pycolmap each ship an OpenMP
runtime and crash when loaded together, and LightGlue would import pycolmap for its SIFT backend,
so pycolmap is kept out of this process.

Usage:
  python learned_features.py --images <dir> --out <dir> [--overlap 10] [--masks <dir>]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # ops MPS lacks run on the CPU
sys.modules.setdefault("pycolmap", None)  # LightGlue's optional SIFT backend; see the docstring

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lightglue import ALIKED, LightGlue  # noqa: E402
from lightglue.utils import load_image, rbd  # noqa: E402

MAX_KEYPOINTS = 2048
RESIZE = 1024  # long image side the features are extracted at


@torch.inference_mode()
def extract_and_match(images, out, overlap=10, mask_dir=None):
    """Keypoints for every frame in `images`, LightGlue matches between sequential pairs."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    names = sorted(str(p.relative_to(images)) for p in Path(images).rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    extractor = ALIKED(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
    # LightGlue's adaptive pruning saves time on the CPU but costs it on MPS (0.29 vs 0.11 s per
    # pair on an M1 Pro, same matches), so it is switched off there
    pruning = {} if device.type == "cpu" else {"depth_confidence": -1, "width_confidence": -1}
    matcher = LightGlue(features="aliked", **pruning).eval().to(device)
    feat_dir = Path(out) / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    feats, counts = {}, []
    for name in names:
        f = extractor.extract(load_image(Path(images) / name).to(device), resize=RESIZE)
        kp = f["keypoints"][0].cpu().numpy()  # original image pixels, pixel centres at integers
        keep = np.ones(len(kp), bool)
        mask = cv2.imread(str(Path(mask_dir) / f"{name}.png"), cv2.IMREAD_GRAYSCALE) if mask_dir else None
        if mask is not None:
            h, w = mask.shape
            x = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
            y = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)
            keep = mask[y, x] > 0
        idx = torch.from_numpy(np.flatnonzero(keep)).to(device)
        feats[name] = {k: v[:, idx] if v.ndim >= 2 and v.shape[1] == len(kp) else v for k, v in f.items()}
        kp = kp[keep]
        counts.append(len(kp))
        # COLMAP keypoints have pixel centres at +0.5; scale 1, orientation 0; zero descriptors
        rows = np.c_[kp + 0.5, np.ones(len(kp)), np.zeros(len(kp))]
        (feat_dir / name).parent.mkdir(parents=True, exist_ok=True)
        with open(feat_dir / f"{name}.txt", "w") as fh:
            fh.write(f"{len(kp)} 128\n")
            zeros = " ".join(["0"] * 128)
            fh.writelines(f"{x:.3f} {y:.3f} {s:g} {o:g} {zeros}\n" for x, y, s, o in rows)
    t_extract = time.time() - t0

    t0 = time.time()
    todo = [(a, b) for i, a in enumerate(names) for b in names[i + 1:i + 1 + overlap]]
    pairs, n_matches = 0, []
    with open(Path(out) / "matches.txt", "w") as fh:
        for a, b in todo:
            m = rbd(matcher({"image0": feats[a], "image1": feats[b]}))["matches"].cpu().numpy()
            pairs += 1
            n_matches.append(len(m))
            fh.write(f"{a} {b}\n")
            fh.writelines(f"{p} {q}\n" for p, q in m)
            fh.write("\n")
    stats = {"features": "aliked", "device": str(device), "images": len(names), "pairs": pairs,
             "keypoints_median": float(np.median(counts)), "matches_median": float(np.median(n_matches)),
             "extract_s": round(t_extract, 1), "match_s": round(time.time() - t0, 1)}
    (Path(out) / "learned_features.json").write_text(json.dumps(stats, indent=1))
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overlap", type=int, default=10, help="Match each frame with its next N frames")
    parser.add_argument("--masks", type=Path, help="COLMAP feature masks (<image name>.png, black = ignore)")
    args = parser.parse_args()
    print(json.dumps(extract_and_match(args.images, args.out, args.overlap, args.masks)))


if __name__ == "__main__":
    main()
