"""Feature matching for the reconstruct stage: ALIKED keypoints matched with LightGlue, or with
plain mutual nearest-neighbour matching of the descriptors for frames close together (--matcher nn;
nn_features.py is the same with nn as the default).

Writes them in the formats of COLMAP's documented route for custom features, so COLMAP's own
geometric verification, mapper and everything after stay unchanged (reconstruct.py runs
feature_importer and matches_importer --match_type raw on them):
  <out>/features/<image name>.txt   "N 128", then one "x y scale orientation" line per keypoint
                                    plus 128 zero descriptor values (COLMAP needs the field;
                                    matching is already done)
  <out>/matches.txt                 per pair "name1 name2", one "idx1 idx2" line per match, blank line

Pairs are sequential like COLMAP's sequential matcher: each frame with its next --overlap frames
(capture order = file name order), or those listed in --pairs ("name1 name2" per line; a rig's
images, reconstruct.py). Keypoints on black pixels of the COLMAP feature masks
(<mask dir>/<image name>.png: burned-in overlays, sky, vehicles, people) are dropped.

--jobs jobs.json runs many such jobs (a list of {images, out, masks, pairs, overlap, matcher}) in one
process: the networks are loaded once, and each image file is read and its keypoints extracted
once however many jobs use it (reconstruct.py's leg work folders hard-link the same frames, and
legs overlap), instead of one process per leg camera.

Nearest-neighbour matching: a pair matches when each keypoint is the other's closest descriptor and
the closest is clearly closer than the second closest (ratio test). Against LightGlue (roi_2, 5, 7,
11): ~15x faster, equal results on ordinary streets; worse where neighbouring frames look very
different (frames ~9 m apart on roi_5's long leg, a parked truck on roi_11).

Runs as its own process (reconstruct.py calls it): PyTorch and pycolmap each ship an OpenMP
runtime and crash when loaded together, and LightGlue would import pycolmap for its SIFT backend,
so pycolmap is kept out of this process.

Usage:
  python learned_features.py --images <dir> --out <dir> [--overlap 10] [--masks <dir>] [--matcher nn]
  python learned_features.py --jobs jobs.json
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
RATIO = 0.9    # nearest-neighbour matching: closest / second-closest descriptor distance


class NearestNeighbour(torch.nn.Module):
    """Stands in for LightGlue: same call and output layout ({"matches": [K x 2]})."""

    def forward(self, data):
        d0, d1 = data["image0"]["descriptors"][0], data["image1"]["descriptors"][0]
        if len(d0) < 2 or len(d1) < 2:
            return {"matches": [torch.zeros((0, 2), dtype=torch.long, device=d0.device)]}
        dist = torch.cdist(d0, d1)
        best2, idx2 = dist.topk(2, dim=1, largest=False)
        nn01 = idx2[:, 0]
        nn10 = dist.argmin(dim=0)
        i = torch.arange(len(d0), device=d0.device)
        keep = (nn10[nn01] == i) & (best2[:, 0] < RATIO * best2[:, 1])
        return {"matches": [torch.stack([i[keep], nn01[keep]], dim=1)]}


class Matcher:
    """ALIKED extraction, cached per image file, and the matchers, loaded once."""

    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.extractor = ALIKED(max_num_keypoints=MAX_KEYPOINTS).eval().to(self.device)
        self.matchers = {"nn": NearestNeighbour()}
        self.cache, self.extracted, self.reused, self.extract_s = {}, 0, 0, 0.0

    def matcher(self, name):
        if name not in self.matchers:
            # LightGlue's adaptive pruning saves time on the CPU but costs it on MPS (0.29 vs 0.11 s per
            # pair on an M1 Pro, same matches), so it is switched off there
            pruning = {} if self.device.type == "cpu" else {"depth_confidence": -1, "width_confidence": -1}
            self.matchers[name] = LightGlue(features="aliked", **pruning).eval().to(self.device)
        return self.matchers[name]

    def features(self, path):
        """ALIKED features of an image file (CPU tensors), extracted once per file: hard links of one
        frame share its inode."""
        st = os.stat(path)
        key = (st.st_dev, st.st_ino)
        if key in self.cache:
            self.reused += 1
            return self.cache[key]
        t0 = time.time()
        f = self.extractor.extract(load_image(path).to(self.device), resize=RESIZE)
        f = {k: v.cpu() for k, v in f.items()}
        self.cache[key] = f
        self.extracted += 1
        self.extract_s += time.time() - t0
        return f

    @torch.inference_mode()
    def run(self, images, out, overlap=10, mask_dir=None, pairs_file=None, matcher="lightglue"):
        """Keypoints for every frame in `images` (masked), matches between the pairs."""
        names = sorted(str(p.relative_to(images)) for p in Path(images).rglob("*")
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        feat_dir = Path(out) / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        t0, n0 = time.time(), self.extract_s
        feats, counts = {}, []
        for name in names:
            f = self.features(Path(images) / name)
            kp = f["keypoints"][0].numpy()  # original image pixels, pixel centres at integers
            keep = np.ones(len(kp), bool)
            mask = cv2.imread(str(Path(mask_dir) / f"{name}.png"), cv2.IMREAD_GRAYSCALE) if mask_dir else None
            if mask is not None:
                h, w = mask.shape
                x = np.clip(np.round(kp[:, 0]).astype(int), 0, w - 1)
                y = np.clip(np.round(kp[:, 1]).astype(int), 0, h - 1)
                keep = mask[y, x] > 0
            idx = torch.from_numpy(np.flatnonzero(keep))
            feats[name] = {k: (v[:, idx] if v.ndim >= 2 and v.shape[1] == len(kp) else v).to(self.device)
                           for k, v in f.items()}
            kp = kp[keep]
            counts.append(len(kp))
            # COLMAP keypoints have pixel centres at +0.5; scale 1, orientation 0; zero descriptors
            rows = np.c_[kp + 0.5, np.ones(len(kp)), np.zeros(len(kp))]
            (feat_dir / name).parent.mkdir(parents=True, exist_ok=True)
            with open(feat_dir / f"{name}.txt", "w") as fh:
                fh.write(f"{len(kp)} 128\n")
                zeros = " ".join(["0"] * 128)
                fh.writelines(f"{x:.3f} {y:.3f} {s:g} {o:g} {zeros}\n" for x, y, s, o in rows)
        t_prepare = time.time() - t0

        t0 = time.time()
        todo = ([tuple(l.split()) for l in Path(pairs_file).read_text().splitlines() if l.strip()] if pairs_file else
                [(a, b) for i, a in enumerate(names) for b in names[i + 1:i + 1 + overlap]])
        m_net, n_matches = self.matcher(matcher), []
        with open(Path(out) / "matches.txt", "w") as fh:
            for a, b in todo:
                m = rbd(m_net({"image0": feats[a], "image1": feats[b]}))["matches"].cpu().numpy()
                n_matches.append(len(m))
                fh.write(f"{a} {b}\n")
                fh.writelines(f"{p} {q}\n" for p, q in m)
                fh.write("\n")
        stats = {"features": "aliked", "matcher": matcher, "device": str(self.device), "images": len(names),
                 "pairs": len(todo), "keypoints_median": float(np.median(counts)) if counts else 0.0,
                 "matches_median": float(np.median(n_matches)) if n_matches else 0.0,
                 "extract_s": round(self.extract_s - n0, 1), "prepare_s": round(t_prepare, 1),
                 "match_s": round(time.time() - t0, 1)}
        (Path(out) / "learned_features.json").write_text(json.dumps(stats, indent=1))
        return stats


def main(default_matcher="lightglue"):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--overlap", type=int, default=10, help="Match each frame with its next N frames")
    parser.add_argument("--masks", type=Path, help="COLMAP feature masks (<image name>.png, black = ignore)")
    parser.add_argument("--pairs", type=Path, help="Match these pairs ('name1 name2' per line) instead of sequential ones")
    parser.add_argument("--matcher", choices=["lightglue", "nn"], default=default_matcher)
    parser.add_argument("--jobs", type=Path, help="Run a list of jobs ({images, out, masks, pairs, overlap, matcher}) in one process")
    args = parser.parse_args()
    t0 = time.time()
    m = Matcher()
    jobs = (json.loads(args.jobs.read_text()) if args.jobs else
            [{"images": args.images, "out": args.out, "masks": args.masks, "pairs": args.pairs,
              "overlap": args.overlap, "matcher": args.matcher}])
    for job in jobs:
        stats = m.run(Path(job["images"]), Path(job["out"]), job.get("overlap", 10),
                      Path(job["masks"]) if job.get("masks") else None,
                      Path(job["pairs"]) if job.get("pairs") else None, job.get("matcher", default_matcher))
        print(json.dumps({"out": str(job["out"]), **stats}), flush=True)
    print(json.dumps({"jobs": len(jobs), "images_extracted": m.extracted, "images_reused": m.reused,
                      "extract_s": round(m.extract_s, 1), "total_s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
