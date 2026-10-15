"""
metric_study_figure.py - slide figure of one photo's metric_study.csv results.

Shows every variant next to its six metric scores. Cells are coloured by how
each metric ranks the variants (green = closest to the photo, red = furthest),
and red boxes mark pairs whose order is known but the metric got backwards.
The bottom rows count correct orderings across every photo in the CSV.

Run with the metrics environment, after metric_study.py:
    ~/eval-venv/bin/python scripts/eval/metric_study_figure.py outputs/views ID
"""
import csv
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from metric_study import KNOWN_ORDER, KNOWN_ORDER_NR, build_variants

ROWS = [
    ("identity", "Photo itself"),
    ("shift_1", "Shifted 2%"),
    ("shift_2", "Shifted 6%"),
    ("blur_1", "Light blur"),
    ("blur_2", "Medium blur"),
    ("blur_3", "Heavy blur"),
    ("scramble", "Shuffled blocks"),
    ("pipeline", "Pipeline render"),
    ("grey", "Untextured grey model"),
]
LABEL = dict(ROWS)
# (key, name, lower is better, subtitle, verdict)
METRICS = [
    ("dreamsim", "DreamSim", True, "lower = closer", "Keep (headline)"),
    ("lpips", "LPIPS", True, "lower = closer", "Keep"),
    ("dists", "DISTS", True, "lower = closer", "Optional"),
    ("ssim", "SSIM", False, "higher = closer", "Drop as headline"),
    ("qrealign", "Q-ReAlign", False, "higher = better quality\n(no reference)", "Keep (realism only)"),
    ("afine_all", "A-FINE", True, "lower = closer", "Drop"),
]
VERDICT_COLOUR = {"Keep": "#2e7d32", "Opti": "#b26a00", "Drop": "#c62828"}
# A 2% slide is a smaller error than heavy blur, and real texture should beat none
EXTRA_PAIRS = [("shift_1", "blur_3"), ("pipeline", "grey")]
PAIRS_FR = KNOWN_ORDER + EXTRA_PAIRS
PAIRS_NR = KNOWN_ORDER_NR + EXTRA_PAIRS

LABEL_W, MET_W, ROW_H, HEAD_H, TITLE_H, FOOT_H = 2.1, 1.5, 0.95, 0.95, 1.1, 0.6


def misses(scores: dict[str, float], lower: bool, pairs) -> list[tuple[str, str]]:
    return [(b, w) for b, w in pairs
            if not (scores[b] < scores[w] if lower else scores[b] > scores[w])]


def main():
    views, img_id = Path(sys.argv[1]), sys.argv[2]
    with open(views / "metric_study.csv") as f:
        rows = [{k: (v if k in ("photo", "variant") else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    photos = list(dict.fromkeys(r["photo"] for r in rows))
    table = {p: {r["variant"]: r for r in rows if r["photo"] == p} for p in photos}
    this = table[img_id]

    _, variants = build_variants(views, img_id)
    aspect = variants["identity"].width / variants["identity"].height
    img_h = ROW_H - 0.12
    img_w = img_h * aspect
    x_img = LABEL_W
    x_met = x_img + img_w + 0.25

    photo_misses = {}
    for key, name, lower, _, _ in METRICS:
        pairs = PAIRS_NR if key == "qrealign" else PAIRS_FR
        photo_misses[key] = misses({v: this[v][key] for v in this}, lower, pairs)
    miss_lines = []
    for key, name, *_ in METRICS:
        if photo_misses[key]:
            verb = "rated higher quality than" if key == "qrealign" else "scored closer than"
            text = "; ".join(f"{LABEL[w]} {verb} {LABEL[b]}" for b, w in photo_misses[key])
            miss_lines += textwrap.wrap(f"{name}:  {text}", 150, subsequent_indent="        ")

    total_w = x_met + MET_W * len(METRICS) + 0.15
    y_rows = TITLE_H + HEAD_H
    y_foot = y_rows + ROW_H * len(ROWS) + 0.1
    y_notes = y_foot + 2 * FOOT_H + 0.25
    total_h = y_notes + 0.35 + 0.3 * len(miss_lines) + 0.2

    plt.rcParams["font.family"] = ["Helvetica", "Arial", "DejaVu Sans"]
    fig = plt.figure(figsize=(total_w, total_h), dpi=200)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")

    ax.text(0.2, 0.45, f"How each metric scores known edits of one photo (Mapillary {img_id})",
            fontsize=17, weight="bold", va="center")
    ax.text(0.2, 0.85, "Each version is compared with the original photo inside the building outline.  "
            "Cell colour = that metric's ranking (green = judged closest, red = judged furthest).  "
            "Red box = the metric got a known order backwards.",
            fontsize=10.5, color="#444", va="center")

    cmap = plt.get_cmap("RdYlGn")
    for j, (key, name, lower, sub, _) in enumerate(METRICS):
        cx = x_met + MET_W * (j + 0.5)
        ax.text(cx, TITLE_H + 0.3, name, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(cx, TITLE_H + 0.68, sub, ha="center", va="center", fontsize=8.5, color="#555")

        values = np.array([this[v][key] for v, _ in ROWS])
        order = np.argsort(values if lower else -values)
        rank = np.empty(len(ROWS), int)
        rank[order] = np.arange(len(ROWS))
        missed = {v for pair in photo_misses[key] for v in pair}
        for i, (v, _) in enumerate(ROWS):
            x0, y0 = x_met + MET_W * j, y_rows + ROW_H * i
            colour = np.array(cmap(1 - rank[i] / (len(ROWS) - 1)))
            colour[:3] = 0.45 * colour[:3] + 0.55
            ax.add_patch(Rectangle((x0 + 0.04, y0 + 0.04), MET_W - 0.08, ROW_H - 0.08,
                                   facecolor=colour, edgecolor="none"))
            if v in missed:
                ax.add_patch(Rectangle((x0 + 0.06, y0 + 0.06), MET_W - 0.12, ROW_H - 0.12,
                                       facecolor="none", edgecolor="#d50000", linewidth=2.6))
            digits = 2 if key == "afine_all" else 3
            ax.text(x0 + MET_W / 2, y0 + ROW_H / 2, f"{round(values[i], digits) + 0.0:.{digits}f}", ha="center",
                    va="center", fontsize=12.5, family="monospace")

        pairs = PAIRS_NR if key == "qrealign" else PAIRS_FR
        right = sum(len(pairs) - len(misses({v: table[p][v][key] for v in table[p]}, lower, pairs))
                    for p in photos)
        total = len(pairs) * len(photos)
        ax.text(cx, y_foot + FOOT_H / 2, f"{right}/{total}  ({100 * right / total:.0f}%)",
                ha="center", va="center", fontsize=12, weight="bold")
        verdict = METRICS[j][4]
        ax.text(cx, y_foot + FOOT_H * 1.5, verdict, ha="center", va="center", fontsize=11.5,
                weight="bold", color=VERDICT_COLOUR[verdict[:4]])

    for i, (v, label) in enumerate(ROWS):
        y0 = y_rows + ROW_H * i
        ax.text(LABEL_W - 0.15, y0 + ROW_H / 2, label, ha="right", va="center", fontsize=12,
                weight="bold" if v in ("pipeline", "grey") else "normal")
        ax.imshow(np.array(variants[v]), extent=(x_img, x_img + img_w, y0 + 0.06 + img_h, y0 + 0.06))
        if i:
            ax.plot([0.2, total_w - 0.15], [y0, y0], color="#ddd", linewidth=0.8)
    ax.plot([0.2, total_w - 0.15], [y_rows + ROW_H * 7, y_rows + ROW_H * 7], color="#888", linewidth=1.4)

    ax.plot([0.2, total_w - 0.15], [y_foot, y_foot], color="#888", linewidth=1.4)
    ax.text(x_met - 0.25, y_foot + FOOT_H / 2, f"Known orderings right, all {len(photos)} photos",
            ha="right", va="center", fontsize=11, weight="bold")
    ax.text(x_met - 0.25, y_foot + FOOT_H * 1.5, "Verdict", ha="right", va="center",
            fontsize=11, weight="bold")

    ax.text(0.2, y_notes, "Backwards orderings on this photo:", fontsize=11, weight="bold", va="center")
    for k, line in enumerate(miss_lines):
        ax.text(0.35, y_notes + 0.35 + 0.3 * k, line, fontsize=10, va="center", color="#333")

    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    out = views / f"metric_study_{img_id}.png"
    fig.savefig(out, facecolor="white")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
