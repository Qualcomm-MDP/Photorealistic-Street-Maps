"""Result images of the OSM alignment layer: before/after maps and raw GPS vs pipeline."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLOURS = ["crimson", "purple", "teal", "darkgoldenrod", "navy", "darkgreen", "chocolate", "slategray"]


def _xy(frame, lonlat):
    a = np.asarray(lonlat, float).reshape(-1, 2)
    return np.array([frame.xy(lon, lat) for lon, lat in a]).reshape(-1, 2)


def _buildings(ax, frame, buildings, lo, hi, **style):
    for _, ring in buildings:
        r = _xy(frame, ring)
        if np.all(r.max(axis=0) >= lo) and np.all(r.min(axis=0) <= hi):
            ax.fill(r[:, 0], r[:, 1], **style)


def _finish(ax, lo, hi, title):
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)


def plot_before_after(path, legs, frame, buildings, title_note):
    """Two panels for one or more legs: placed with Mapillary's positions only vs aligned to
    the walls. legs: [{"id", "street", "reliable", "plan"}], plan from OSMAlignmentLayer."""
    allpts = np.vstack([_xy(frame, c) for leg in legs for c in leg["plan"]["cameras_aligned"]]
                       + [_xy(frame, leg["plan"]["raw_gps"]) for leg in legs])
    lo, hi = allpts.min(axis=0) - 45, allpts.max(axis=0) + 45
    d_before = np.concatenate([leg["plan"]["facade_dist_mapillary"] for leg in legs])
    d_after = np.concatenate([leg["plan"]["facade_dist_aligned"] for leg in legs])
    aspect = (hi[1] - lo[1]) / (hi[0] - lo[0])
    fig, axes = plt.subplots(1, 2, figsize=(18, min(18, max(4.5, 8.5 * aspect + 1.5))))
    panels = [("mapillary", f"Before (Mapillary-only): median facade-to-wall {np.median(d_before):.2f} m"),
              ("aligned", f"After (facades matched to OSM walls): median {np.median(d_after):.2f} m")]
    for ax, (key, title) in zip(axes, panels):
        _buildings(ax, frame, buildings, lo, hi, facecolor="0.88", edgecolor="0.6", lw=0.6)
        walls = [_xy(frame, w) for leg in legs for w in leg["plan"]["walls"]]
        for j, w in enumerate(walls):
            ax.plot(w[:, 0], w[:, 1], color="tab:blue", lw=2.2, alpha=0.6, label="street-facing OSM walls" if j == 0 else None)
        for k, leg in enumerate(legs):
            plan = leg["plan"]
            pts = _xy(frame, plan[f"points_{key}"])
            ax.scatter(pts[:, 0], pts[:, 1], s=1.2, c=np.asarray(plan["points_rgb"]).reshape(-1, 3) / 255,
                       linewidths=0, zorder=3)
            for j, seg in enumerate(plan[f"segments_{key}"]):
                seg = _xy(frame, seg)
                ax.plot(seg[:, 0], seg[:, 1], color="darkorange", lw=2.4, zorder=4,
                        label="facade segments (point cloud)" if k == 0 and j == 0 else None)
            g = _xy(frame, plan["raw_gps"])
            ax.plot(g[:, 0], g[:, 1], "+", color="0.35", ms=5, zorder=5, label="Mapillary raw GPS" if k == 0 else None)
            if key == "mapillary":
                m = _xy(frame, plan["cameras_mapillary"])
                ax.plot(m[:, 0], m[:, 1], "x", color="tab:green", ms=4, zorder=5,
                        label="Mapillary computed (OpenSfM)" if k == 0 else None)
            else:
                colour = COLOURS[k % len(COLOURS)]
                for c in plan["cameras_aligned"]:
                    c = _xy(frame, c)
                    ax.plot(c[:, 0], c[:, 1], "o-", color=colour, ms=2.5, lw=1, zorder=6)
                ax.plot([], [], "o-", color=colour, ms=3, label=f"{leg['id']} cameras ({leg['street']})"
                        + ("" if leg.get("reliable", True) else ", UNRELIABLE"))
        _finish(ax, lo, hi, title)
    fig.suptitle(title_note, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_gps_vs_pipeline(path, records, frame, buildings):
    """Raw GPS (with compass heading) vs pipeline camera positions (with heading), linked."""
    pairs = [r for r in records if r.get("pipeline") and r["raw_gps"].get("lon") is not None]
    if not pairs:
        return
    A = np.array([frame.xy(r["pipeline"]["lon"], r["pipeline"]["lat"]) for r in pairs])
    G = np.array([frame.xy(r["raw_gps"]["lon"], r["raw_gps"]["lat"]) for r in pairs])
    M = np.array([frame.xy(r["mapillary_computed"]["lon"], r["mapillary_computed"]["lat"]) for r in pairs])
    lo, hi = np.vstack([A, G]).min(axis=0) - 30, np.vstack([A, G]).max(axis=0) + 30
    aspect = (hi[1] - lo[1]) / (hi[0] - lo[0])
    fig, ax = plt.subplots(figsize=(11, min(18, max(4, 11 * aspect + 1))))
    _buildings(ax, frame, buildings, lo, hi, facecolor="0.9", edgecolor="0.45", lw=0.8)
    for a, g in zip(A, G):
        ax.plot([g[0], a[0]], [g[1], a[1]], color="0.45", lw=0.7)
    ax.plot(M[:, 0], M[:, 1], ".", color="tab:green", ms=3, alpha=0.4, label="Mapillary computed (OpenSfM), for reference")
    ax.plot(G[:, 0], G[:, 1], "o", color="tab:blue", ms=3.5, label="Mapillary raw GPS (+ compass heading)")
    ax.plot(A[:, 0], A[:, 1], "o", color="crimson", ms=3.5, label="pipeline (aligned to OSM)")
    for points, headings, colour in ((A, [r["pipeline"]["heading_deg"] for r in pairs], "crimson"),
                                     (G, [r["raw_gps"]["compass_deg"] for r in pairs], "tab:blue")):
        for p, hd in list(zip(points, headings))[::4]:
            if hd is not None:
                hd = math.radians(hd)
                ax.arrow(p[0], p[1], 4 * math.sin(hd), 4 * math.cos(hd), color=colour, width=0.15, head_width=1.0)
    off = np.linalg.norm(A - G, axis=1)
    _finish(ax, lo, hi, f"Raw GPS vs pipeline camera positions: median {np.median(off):.1f} m apart (max {off.max():.1f} m)")
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
