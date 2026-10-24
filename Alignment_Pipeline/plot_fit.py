"""Figure of how each leg's sparse point cloud sits on the OSM footprints after alignment.

  left   plan view: every sparse point, coloured by its segmentation label (building, road,
         other), over the OSM footprints; the aligned cameras with their viewing directions
  right  3D view: the same points at their heights above the ground, with the OSM footprints
         extruded to their tagged (or default) heights, so facade points can be seen standing on
         the walls
Reads data/reconstruction models and data/alignment/<leg>_<folder>/transform.json; writes
results/fit_<leg>_<folder>.png.

Usage: python plot_fit.py ../Alignment_Pipeline/rois/mapillary_roi_2
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from common import MASK_BUILDING, MASK_ROAD, LocalFrame, RoiPaths, label_points, load_cameras, load_points  # noqa: E402
from align import point_pieces  # noqa: E402
from osm import building_heights  # noqa: E402

COL = {"building": "#d62728", "road": "#8c8c8c", "other": "#4c9be8"}


def plot_leg(p, folder_dir, buildings):
    tf = json.loads((folder_dir / "transform.json").read_text())
    stats = json.loads((folder_dir / "stats.json").read_text())
    model = Path(tf["model"])
    folder = stats["folder"]
    origin, basis = np.array(tf["origin"]), np.array(tf["basis"])
    pieces = tf.get("pieces") or [{"images": None, "angle_rad": tf["angle_rad"], "scale_m_per_unit": tf["scale_m_per_unit"],
                                   "translation_m": tf["translation_m"]}]

    def to_enu(X, idx):
        """Levelled, then each piece's similarity (idx: piece per row)."""
        L = (X - origin) @ basis.T
        out = np.empty_like(L)
        for k, pc in enumerate(pieces):
            s, a, t = pc["scale_m_per_unit"], pc["angle_rad"], np.array(pc["translation_m"])
            Rz = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            m = idx == k
            out[m] = np.c_[s * L[m, :2] @ Rz.T + t, s * (L[m, 2] - tf["ground_z"])]
        return out

    names, C, R = load_cameras(model)
    xyz, rgb, track = load_points(model)
    piece_of = {n: k for k, pc in enumerate(pieces) for n in (pc["images"] or names)}
    cam_piece = np.array([piece_of[n] for n in names])
    pt_piece = point_pieces(model, names, piece_of) if len(pieces) > 1 else np.zeros(len(xyz), int)
    P, Cw = to_enu(xyz, pt_piece), to_enu(C, cam_piece)
    bld, _ = label_points(model, p.masks / folder, MASK_BUILDING, 0.5)
    road, _ = label_points(model, p.masks / folder, MASK_ROAD, 0.5)
    label = np.where(bld, "building", np.where(road, "road", "other"))
    view = np.array([(basis @ R[i][:, 2]) for i in range(len(names))])  # camera z axis, levelled
    for k, pc in enumerate(pieces):
        a = pc["angle_rad"]
        m = cam_piece == k
        view[m, :2] = view[m, :2] @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]).T

    enu = LocalFrame(*tf["enu_origin_lon_lat"])
    # extent: the cameras and most of the building points (setbacks can be 50 m+), plus a margin
    bxy = P[bld & (P[:, 2] > 0) & (P[:, 2] < 40), :2]
    ext = np.vstack([Cw[:, :2], np.percentile(bxy, [2, 98], axis=0)]) if len(bxy) else Cw[:, :2]
    lo, hi = ext.min(axis=0) - 25, ext.max(axis=0) + 25
    keep = (P[:, 0] > lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] > lo[1]) & (P[:, 1] < hi[1]) & (P[:, 2] > -3) & (P[:, 2] < 40)
    fps = []
    for tags, ring in buildings:
        xy = np.array([enu.xy(lon, lat) for lon, lat in ring])
        if np.all(xy.max(axis=0) >= lo) and np.all(xy.min(axis=0) <= hi):
            fps.append((xy, *building_heights(tags, 3.5, 3)[:2]))

    fig = plt.figure(figsize=(19, 8.5))
    ax = fig.add_subplot(1, 2, 1)
    for xy, _, _ in fps:
        ax.fill(*xy.T, color="#ececec", ec="#555", lw=0.8, zorder=0)
    for name in ("other", "road", "building"):
        m = keep & (label == name)
        ax.scatter(P[m, 0], P[m, 1], s=0.9 if name == "building" else 0.5, c=COL[name], alpha=0.8, zorder=1,
                   label=f"{name} points ({m.sum()})")
    for k in range(len(pieces)):  # one line per piece; a dotted link shows each cut
        m = cam_piece == k
        ax.plot(*Cw[m, :2].T, "-", color="black", lw=1.2, zorder=2,
                label=(f"cameras ({len(names)}, {len(pieces)} pieces)" if len(pieces) > 1 else f"cameras ({len(names)})")
                if k == 0 else None)
        if k:
            j0, j1 = np.flatnonzero(cam_piece == k - 1)[-1], np.flatnonzero(m)[0]
            ax.plot(*Cw[[j0, j1], :2].T, ":", color="black", lw=1.2, zorder=2)
    for c, v in zip(Cw[::2], view[::2]):
        d = v[:2] / (np.linalg.norm(v[:2]) + 1e-9) * 6
        ax.annotate("", xy=c[:2] + d, xytext=c[:2], arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8})
    ptf = stats["points_to_footprint_median_m"]
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.legend(fontsize=8, loc="lower right", markerscale=8)
    ax.set_title(f"Plan view: building points to footprints {ptf['aligned']:.2f} m median "
                 f"({100 * stats['points_to_footprint_within_1m']:.0f}% within 1 m)", fontsize=10, loc="left")

    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    for xy, base, top in fps:
        top = min(top, 30)
        walls = [[(*xy[i], base), (*xy[i + 1], base), (*xy[i + 1], top), (*xy[i], top)] for i in range(len(xy) - 1)]
        ax3.add_collection3d(Poly3DCollection(walls, facecolor="#d9d9d9", edgecolor="#777", linewidths=0.3, alpha=0.25))
    for name in ("road", "other", "building"):
        m = keep & (label == name)
        sub = np.flatnonzero(m)
        sub = sub[np.random.default_rng(0).choice(len(sub), min(len(sub), 6000), replace=False)] if len(sub) else sub
        ax3.scatter(P[sub, 0], P[sub, 1], P[sub, 2], s=0.35, c=COL[name], alpha=0.8, depthshade=False)
    ax3.plot(*Cw.T, color="black", lw=1.5)
    ax3.set_xlim(lo[0], hi[0])
    ax3.set_ylim(lo[1], hi[1])
    ax3.set_zlim(0, 30)
    span = hi - lo
    ax3.set_box_aspect((span[0], span[1], 30))
    ax3.view_init(elev=28, azim=-60)
    ax3.set_xlabel("east (m)")
    ax3.set_ylabel("north (m)")
    ax3.set_zlabel("height (m)")
    ax3.set_title("3D: points and OSM footprints extruded to their heights", fontsize=10, loc="left")
    mapper = json.loads(p.legs_json.read_text()).get("mapper", "global") if p.legs_json.exists() else "global"
    fig.suptitle(f"{p.name} {stats['leg']}/{folder}: GeoCalib + ALIKED + "
                 f"{'GLOMAP' if mapper == 'global' else 'COLMAP incremental'}, "
                 f"scale x{stats['scale_vs_mapillary']:.2f} vs GPS placement, camera height {stats['camera_height_m']:.2f} m, "
                 f"{'reliable' if stats['reliable'] else 'UNRELIABLE'}", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    out = p.results / f"fit_{stats['leg']}_{folder}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roi_dir", type=Path)
    parser.add_argument("--tag", help="Variant run (see reconstruct.py --tag)")
    args = parser.parse_args()
    p = RoiPaths(args.roi_dir, args.tag)
    buildings = json.loads((p.alignment / "osm_buildings.json").read_text())
    for d in sorted(x for x in p.alignment.iterdir() if (x / "transform.json").exists()):
        print(plot_leg(p, d, buildings))


if __name__ == "__main__":
    main()
