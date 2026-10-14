"""Streets inside a bounding box, as an ROI for the alignment layers.

The standalone alignment pipeline starts from streets drawn in route_gui.py; the main
pipeline starts from a bounding box. This builds the same ROI document (route_gui.make_roi)
from the OSM streets in the box:
  1. Fetch the drivable OSM ways (STREET_TYPES) in the box, grown by --margin so a street
     just outside it, from which its buildings were photographed, is included.
  2. Chain ways of the same street into one line: OSM splits a street into a way per block,
     and each line is searched for its own best Mapillary run. Ways are only joined where
     the street carries on through the joint, so a divided road's two carriageways are not
     joined into a U-turn.
  3. Clip the lines to the grown box and drop those shorter than --min-length (too few
     frames for COLMAP), and drop a line running within two corridor widths of a longer one
     of the same street (the other carriageway: the same captures would be found twice).

Usage:
  python -m pose_alignment.streets_from_bbox MIN_LON MIN_LAT MAX_LON MAX_LAT [--out FILE]
  python -m pose_alignment.run_pipeline --roi FILE      # then align the cameras along them
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from .geo_common import ROUTES_DIR, LocalFrame, line_length_m, overpass
from .route_gui import make_roi

# Service roads (parking aisles, driveways) and footpaths are left out: their captures are
# sparse and they rarely face street-front walls.
STREET_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
                "living_street"]


def fetch_streets(bbox):
    """OSM street ways in bbox (min_lon, min_lat, max_lon, max_lat): [(name, [(lon, lat)])]."""
    w, s, e, n = bbox
    data = overpass(f'[out:json][timeout:90];way["highway"~"^({"|".join(STREET_TYPES)})$"]({s},{w},{n},{e});'
                    "out geom;")
    return [(el.get("tags", {}).get("name") or f"unnamed way {el['id']}",
             [(p["lon"], p["lat"]) for p in el["geometry"]])
            for el in data["elements"] if len(el.get("geometry", [])) >= 2]


def turn_deg(p, q, r):
    """How sharply the path p -> q -> r turns at q (deg, 0 = straight on)."""
    a = math.atan2(q[1] - p[1], q[0] - p[0])
    b = math.atan2(r[1] - q[1], r[0] - q[0])
    return abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi))


def join(a, b, max_turn):
    """a and b as one polyline if they share an end point and the street carries on through it
    without turning more than max_turn (a divided road's two carriageways meet at a U-turn)."""
    for x, y in ((a, b), (a, b[::-1]), (b, a), (b[::-1], a)):
        if x[-1] == y[0] and turn_deg(x[-2], x[-1], y[1]) <= max_turn:
            return x + y[1:]
    return None


def chain(lines, max_turn=100.0):
    """Join polylines that share an end point (OSM ways of one street share their end nodes)."""
    lines = [list(line) for line in lines]
    joined = True
    while joined:
        joined = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                merged = join(lines[i], lines[j], max_turn)
                if merged:
                    lines[i] = merged
                    del lines[j]
                    joined = True
                    break
            if joined:
                break
    return lines


def distance_to_polyline(p, pts):
    """Distance (m) from point p to a polyline, all (x, y)."""
    best = math.inf
    for a, b in zip(pts, pts[1:]):
        d = (b[0] - a[0], b[1] - a[1])
        t = max(0.0, min(1.0, ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / max(d[0] ** 2 + d[1] ** 2, 1e-12)))
        best = min(best, math.dist(p, (a[0] + t * d[0], a[1] + t * d[1])))
    return best


def densify(pts, step=5.0):
    """Points every `step` metres along a polyline."""
    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        n = max(1, math.ceil(math.dist(a, b) / step))
        out += [(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) for k in range(1, n + 1)]
    return out


def parallel_to(pts, others, max_dist, share=0.8):
    """Whether most of a polyline lies within max_dist of one of the others (the second
    carriageway of a divided road): its Mapillary search would find the same captures."""
    samples = densify(pts)
    return any(sum(distance_to_polyline(p, o) <= max_dist for p in samples) >= share * len(samples) for o in others)


def clip_segment(p, q, lo, hi):
    """The part of segment p-q inside the box lo-hi (Liang-Barsky), or None."""
    t0, t1 = 0.0, 1.0
    d = (q[0] - p[0], q[1] - p[1])
    for k in range(2):
        for num, den in ((p[k] - lo[k], -d[k]), (hi[k] - p[k], d[k])):
            if den == 0:
                if num < 0:
                    return None
                continue
            t = num / den
            if den < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return (p[0] + t0 * d[0], p[1] + t0 * d[1]), (p[0] + t1 * d[0], p[1] + t1 * d[1])


def clip_polyline(pts, lo, hi):
    """Pieces of a polyline (x, y points) inside the box lo-hi."""
    pieces, current = [], []
    for p, q in zip(pts, pts[1:]):
        seg = clip_segment(p, q, lo, hi)
        if seg is None:
            if current:
                pieces.append(current)
            current = []
            continue
        a, b = seg
        if current and math.dist(current[-1], a) < 1e-6:
            current.append(b)
        else:
            if current:
                pieces.append(current)
            current = [a, b]
    if current:
        pieces.append(current)
    return pieces


def streets_roi(bbox, corridor_m=15.0, margin_m=25.0, min_length_m=60.0, name="bbox"):
    """ROI document (route_gui.make_roi, plus line_names) of the streets in bbox
    (min_lon, min_lat, max_lon, max_lat). Raises ValueError when no street is long enough."""
    w, s, e, n = bbox
    frame = LocalFrame((w + e) / 2, (s + n) / 2)
    (x0, y0), (x1, y1) = frame.xy(w, s), frame.xy(e, n)
    lo, hi = (x0 - margin_m, y0 - margin_m), (x1 + margin_m, y1 + margin_m)
    grown = (*frame.lonlat(*lo), *frame.lonlat(*hi))

    by_name = {}
    for street, line in fetch_streets(grown):
        by_name.setdefault(street, []).append([frame.xy(*p) for p in line])
    lines, names = [], []
    for street, ways in sorted(by_name.items()):
        pieces = [piece for line in chain(ways) for piece in clip_polyline(line, lo, hi)]
        kept = []
        for piece in sorted(pieces, key=lambda pts: -sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))):
            lonlat = [frame.lonlat(*p) for p in piece]
            if line_length_m(lonlat) >= min_length_m and not parallel_to(piece, kept, 2 * corridor_m):
                kept.append(piece)
                lines.append(lonlat)
                names.append(street)
    if not lines:
        raise ValueError(f"no street of {min_length_m:.0f} m or more within {margin_m:.0f} m of the box")
    return {**make_roi(name, lines, corridor_m), "line_names": names}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--corridor", type=float, default=15.0, help="Corridor half-width around each street (m)")
    parser.add_argument("--margin", type=float, default=25.0, help="Also take streets this far outside the box (m)")
    parser.add_argument("--min-length", type=float, default=60.0, help="Drop clipped streets shorter than this (m)")
    parser.add_argument("--out", type=Path, help="Where to save the ROI (default outputs/alignment/routes/bbox_<time>.json)")
    args = parser.parse_args()

    roi = streets_roi(args.bbox, args.corridor, args.margin, args.min_length)
    for street, line in zip(roi["line_names"], roi["lines"]):
        print(f"  {street}: {line_length_m(line):.0f} m")
    out = args.out or ROUTES_DIR / f"bbox_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(roi, indent=1))
    print(f"{len(roi['lines'])} street line(s) saved to {out}")


if __name__ == "__main__":
    main()
