# lightglue_prior vs COLMAP (SIFT), 2026-10-15 03:16

Variant: aliked + LightGlue, prior mapper, extra options: --force. Times in minutes on this machine. SIFT time is COLMAP's own command time from its logs (the original runs' wall clock was not recorded); for the variant, both wall clock (whole run incl. OSM stage) and COLMAP command time are given.

| ROI | frames | legs (SIFT/var) | registered (SIFT/var) | aligned (SIFT/var) | reliable (SIFT/var) | facade-to-wall m (SIFT/var) | GPS to pipeline m (SIFT/var) | features+match min | COLMAP min (SIFT/var) | var recon stage min | var OSM min | var wall min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mapillary_roi_2 | 36 | 1/1 | 29/30 | 29/30 | 29/30 | 0.354/0.209 | 4.86/8.85 | 0.71 | 1.31/0.15 | 1.5 | 0.2 | 1.69 |
| mapillary_roi_4 | 36 | 1/1 | 30/30 | 30/30 | 30/30 | 0.227/0.383 | 7.45/9.03 | 0.76 | 1.62/0.24 | 1.4 | 0.7 | 2.11 |
| mapillary_roi_5 | 51 | 1/1 | 47/48 | 47/48 | 47/48 | 0.122/0.373 | 10.79/11.09 | 1.30 | 2.61/0.52 | 1.9 | 0.2 | 2.14 |
| mapillary_roi_7 | 60 | 2/2 | 37/37 | 21/37 | 21/37 | 0.086/0.167 | 4.89/7.84 | 1.30 | 1.27/0.48 | 2.5 | 0.1 | 2.56 |
| mapillary_roi_8 | 32 | 1/1 | 23/23 | 23/23 | 23/23 | 0.417/0.454 | 9.26/8.82 | 0.56 | 1.01/0.26 | 1.6 | 0.6 | 2.23 |
| mapillary_roi_9 | 38 | 1/1 | 27/33 | 27/33 | 27/0 | 0.076/38.713 | 10.7/10.17 | 0.84 | 1.84/0.25 | 1.6 | 0.1 | 1.73 |
| mapillary_roi_10 | 27 | 1/1 | 12/21 | -/21 | -/21 | -/0.036 | -/3.1 | 0.49 | 1.91/0.24 | 1.3 | 0.0 | 1.31 |
| mapillary_roi_11 | 141 | 2/2 | 113/131 | 111/82 | 111/50 | 0.598/1.589 | 17.36/24.07 | 4.50 | 4.29/1.46 | 8.5 | 0.3 | 8.77 |

Totals over 8 ROIs: 421 frames; aligned SIFT 288 vs lightglue_prior 304; reliable SIFT 288 vs 239; wall clock 22.5 min (3.2 s per frame).
