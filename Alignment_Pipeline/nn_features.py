"""Fast feature matching for frames close together: learned_features.py with LightGlue swapped
for plain mutual nearest-neighbour matching of the ALIKED descriptors (no network): a pair
matches when each keypoint is the other's closest descriptor and the closest is clearly closer than the second
closest (ratio test). Same arguments and outputs as learned_features.py. ALIKED (shipped in the
lightglue package) still finds the keypoints; LightGlue's matching network is not run.

Against LightGlue (roi_2, 5, 7, 11): matching ~15x faster (2-6 s per leg instead of 35-70 s),
equal results on ordinary streets; worse where neighbouring frames look very different (frames
~9 m apart on roi_5's long leg, a parked truck on roi_11), where it finds a third of the matches.

Usage: python nn_features.py --images <dir> --out <dir> [--overlap 10] [--masks <dir>]
"""

import sys

import torch

import learned_features

RATIO = 0.9  # closest / second-closest descriptor distance


class NearestNeighbour(torch.nn.Module):
    """Stands in for LightGlue: same call and output layout ({"matches": [K x 2]})."""

    def __init__(self, **_):
        super().__init__()

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


learned_features.LightGlue = NearestNeighbour

if __name__ == "__main__":
    sys.argv[0] = learned_features.__file__
    learned_features.main()
