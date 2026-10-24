# COLMAP SIFT vs ALIKED + LightGlue, 4 ROIs (2026-10-15)

Both methods ran on COLMAP 4.2.1 with the incremental mapper and the same leg plan, each
recomputed from scratch, then aligned to the same OSM buildings.

- **Machine:** M1 Pro, 16 GB.
- **SIFT:** runs on the CPU (Homebrew COLMAP has no GPU support).
- **ALIKED + LightGlue:** runs on the Apple GPU (MPS).
- **CPU time:** user + sys summed over all processes; GPU time is not included.
- **Facade-to-wall:** median distance after alignment. There is no ground truth, so it is a proxy.

| ROI | Method | Wall s | CPU s | Peak MB | Feature s | Matching s | Mapping s | Points | Building 3+ | Mean track | Reproj px | Aligned | Reliable | Facade-to-wall m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| roi_2 | SIFT | 86.1 | 232.8 | 7041 | 39.4 | 8.1 | 4.0 | 9325 | 2694 | 3.19 | 0.609 | 29/30 | 29 | 0.193 |
| roi_2 | ALIKED+LightGlue | 101.9 | 33.0 | 776 | 6.0 | 39.1 | 5.3 | 7398 | 2614 | 4.2 | 1.022 | 30/30 | 30 | 0.204 |
| roi_7 | SIFT | 126.8 | 305.2 | 5392 | 57.7 | 12.5 | 4.4 | 8197 | 1419 | 2.48 | 0.788 | 37/37 | 21 | 0.165 |
| roi_7 | ALIKED+LightGlue | 133.2 | 34.0 | 702 | 8.2 | 57.3 | 7.3 | 11515 | 1667 | 2.85 | 1.16 | 37/37 | 37 | 0.067 |
| roi_9 | SIFT | 92.6 | 285.0 | 2936 | 68.9 | 13.6 | 3.5 | 6843 | 1376 | 2.73 | 0.691 | 27/33 | 27 | 0.065 |
| roi_9 | ALIKED+LightGlue | 101.8 | 36.6 | 711 | 6.1 | 50.6 | 8.9 | 7956 | 2303 | 3.4 | 1.023 | 33/33 | 0 | 31.536 |
| roi_17 | SIFT | 262.7 | 773.6 | 2835 | 206.9 | 22.6 | 22.0 | 15149 | 2193 | 2.7 | 0.56 | 64/64 | 64 | 1.844 |
| roi_17 | ALIKED+LightGlue | 270.2 | 227.5 | 582 | 11.2 | 153.6 | 45.4 | 21337 | 2264 | 3.44 | 1.165 | 64/64 | 64 | 0.585 |
