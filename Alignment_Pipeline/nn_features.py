"""Fast feature matching for frames close together: learned_features.py with mutual nearest-neighbour
matching of the ALIKED descriptors instead of LightGlue (no network). Same arguments and outputs
as learned_features.py (which now takes --matcher nn itself; this keeps the old command working).

Usage: python nn_features.py --images <dir> --out <dir> [--overlap 10] [--masks <dir>]
"""

import learned_features

if __name__ == "__main__":
    learned_features.main(default_matcher="nn")
