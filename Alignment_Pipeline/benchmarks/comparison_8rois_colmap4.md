# SIFT vs ALIKED + LightGlue (incremental / GLOMAP / GPS priors), 8 ROIs

Each cell shows frames aligned / reliable · median facade-to-wall after OSM alignment. A fit
over 5 m is marked "failed": the leg was flagged unreliable. Legs are planned the same way
everywhere: split at turns over 35°, curves trimmed.

- **SIFT:** the original COLMAP 3.13 runs.
- **LG incr 3.13:** ALIKED + LightGlue with the incremental mapper on COLMAP 3.13.
- **4.2 columns:** ALIKED + LightGlue on COLMAP 4.2.1. GLOMAP is followed by
  re-triangulation; GPS-prior uses raw GPS with 5 m standard deviation and a robust loss.

Every LightGlue run was recomputed from scratch. Times are wall clock on an M1 Pro (16 GB),
including the OSM stage.

| ROI | frames | SIFT 3.13 | LG incr 3.13 | LG incr 4.2 | LG GLOMAP 4.2 | LG GPS-prior 4.2 |
|---|---|---|---|---|---|---|
| roi_2 | 36 | 29/29 · 0.35 m | 30/30 · 0.18 m · 1.1 min | 30/30 · 0.17 m · 1.6 min | 30/30 · 0.22 m · 1.2 min | 30/30 · 0.21 m · 1.7 min |
| roi_4 | 36 | 30/30 · 0.23 m | 30/30 · 0.45 m · 1.7 min | 30/30 · 0.11 m · 1.5 min | 30/30 · 0.09 m · 1.6 min | 30/30 · 0.38 m · 2.1 min |
| roi_5 | 51 | 47/47 · 0.12 m | 47/47 · 0.51 m · 2.6 min | 48/48 · 0.64 m · 2.6 min | 48/48 · 0.39 m · 2.1 min | 48/48 · 0.37 m · 2.1 min |
| roi_7 | 60 | 21/21 · 0.09 m | 37/37 · 0.18 m · 2.6 min | 37/37 · 0.28 m · 1.8 min | 21/21 · 0.22 m · 2.1 min | 37/37 · 0.17 m · 2.6 min |
| roi_8 | 32 | 23/23 · 0.42 m | 23/23 · 0.42 m · 2.1 min | 23/23 · 0.49 m · 1.6 min | 23/23 · 0.39 m · 1.4 min | 23/23 · 0.45 m · 2.2 min |
| roi_9 | 38 | 27/27 · 0.08 m | 33/0 · failed · 1.5 min | 33/33 · 0.96 m · 2.5 min | 33/33 · 0.07 m · 1.4 min | 33/0 · failed · 1.7 min |
| roi_10 | 27 | not aligned | 21/21 · 0.04 m · 1.1 min | 21/21 · 0.04 m · 2.2 min | 21/21 · 0.11 m · 2.2 min | 21/21 · 0.04 m · 1.3 min |
| roi_11 | 141 | 111/111 · 0.60 m | 82/82 · 1.14 m · 7.3 min | 82/50 · 1.32 m · 8.5 min | 82/50 · 1.18 m · 8.6 min | 82/50 · 1.59 m · 8.8 min |
| **total** | 421 | **288/288** | **303/270** · 20.1 min | **304/272** · 22.4 min | **288/256** · 20.6 min | **304/239** · 22.5 min |
