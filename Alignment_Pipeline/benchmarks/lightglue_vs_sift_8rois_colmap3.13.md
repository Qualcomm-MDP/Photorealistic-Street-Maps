# ALIKED + LightGlue vs COLMAP SIFT, 8 ROIs (2026-10-14/15)

Same frames, legs planned the same way (split at turns >35°, curves trimmed), same OSM buildings.
Every ROI was recomputed from scratch, with no reused models. Times are on an M1 Pro (16 GB) with
LightGlue on the Apple GPU (MPS). The SIFT time is COLMAP's own command time from its logs; the
original SIFT runs' wall-clock time was not recorded. Facade-to-wall is the median distance from
the reconstructed facades to the OSM walls after alignment.

| ROI | Frames | Registered SIFT / LG | Aligned SIFT / LG | Reliable SIFT / LG | Facade-to-wall SIFT / LG (m) | Features + matching (min) | COLMAP SIFT / LG (min) | LG reconstruction stage (min) | LG OSM stage (min) | LG wall clock (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| roi_2 | 36 | 29 / 30 | 29 / 30 | 29 / 30 | 0.354 / 0.184 | 0.73 | 1.31 / 0.2 | 1.1 | 0.1 | 1.14 |
| roi_4 | 36 | 30 / 30 | 30 / 30 | 30 / 30 | 0.227 / 0.447 | 0.78 | 1.62 / 0.28 | 1.2 | 0.5 | 1.69 |
| roi_5 | 51 | 47 / 47 | 47 / 47 | 47 / 47 | 0.122 / 0.511 | 1.30 | 2.61 / 0.6 | 2.3 | 0.3 | 2.59 |
| roi_7 | 60 | 37 / 37 | 21 / 37 | 21 / 37 | 0.086 / 0.184 | 1.31 | 1.27 / 0.53 | 2.5 | 0.1 | 2.58 |
| roi_8 | 32 | 23 / 23 | 23 / 23 | 23 / 23 | 0.417 / 0.422 | 0.57 | 1.01 / 0.3 | 1.1 | 1.0 | 2.1 |
| roi_9 | 38 | 27 / 33 | 27 / 33 | 27 / 0 | 0.076 / 39.909 | 0.86 | 1.84 / 0.3 | 1.4 | 0.1 | 1.52 |
| roi_10 | 27 | 12 / 21 | - / 21 | - / 21 | - / 0.036 | 0.50 | 1.91 / 0.27 | 1.1 | 0.0 | 1.14 |
| roi_11 | 141 | 113 / 131 | 111 / 82 | 111 / 82 | 0.598 / 1.143 | 4.52 | 4.29 / 1.87 | 7.1 | 0.2 | 7.31 |

**Totals (8 ROIs, 421 frames).**

- **Aligned:** SIFT 288, LightGlue 303.
- **Reliable:** SIFT 288, LightGlue 270.
- **LightGlue wall clock:** 20.1 min in total, 2.9 s per frame.
- **COLMAP command time:** SIFT 15.9 min, LightGlue 4.3 min.
