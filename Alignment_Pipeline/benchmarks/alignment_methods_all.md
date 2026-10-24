# OSM alignment: facade segments (RANSAC) vs building points to footprints (2026-10-20)

Both methods ran on the same reconstructions with the same levelling and priors.

- **"Points to footprint"** is the median distance of the thinned building points to the nearest OSM
  footprint edge after alignment. The points method minimises roughly this quantity, so it
  favours that method.
- **"Cameras differ"** is the median horizontal distance between the two methods' camera positions.

| ROI | Reconstruction | Reliable (facades) | Reliable (points) | Points to footprint (facades) | Points to footprint (points) | Cameras differ |
|---|---|---|---|---|---|---|
| roi_2 | lightglue | 30/36 | 30/36 | 0.973 | 0.893 | 0.42 m |
| roi_4 | lightglue | 30/36 | 30/36 | 0.892 | 0.788 | 0.3 m |
| roi_5 | lightglue | 48/51 | 48/51 | 1.112 | 1.085 | 1.33 m |
| roi_7 | lightglue | 37/60 | 37/60 | 1.286 | 0.913 | 0.93 m |
| roi_8 | lightglue | 23/32 | 23/32 | 0.673 | 0.639 | 0.77 m |
| roi_9 | lightglue | 33/38 | 33/38 | 2.523 | 1.949 | 2.35 m |
| roi_10 | lightglue | 0/- | 21/27 | - | 1.667 | - m |
| roi_11 | lightglue | 50/141 | 50/141 | 1.944 | 1.916 | 9.42 m |
| roi_15 | lightglue | 21/21 | 21/21 | 2.742 | 1.209 | 8.64 m |
| roi_16 | lightglue | 21/21 | 21/21 | 1.363 | 0.777 | 7.64 m |
| roi_17 | lightglue | 64/64 | 64/64 | 0.743 | 0.806 | 0.44 m |
| roi_2 | colmap | 29/36 | 29/36 | 3.023 | 1.845 | 2.63 m |
| roi_4 | colmap | 30/36 | 30/36 | 0.834 | 0.616 | 1.24 m |
| roi_5 | colmap | 47/51 | 47/51 | 0.531 | 0.624 | 4.4 m |
| roi_7 | colmap | 21/60 | 37/60 | 0.468 | 0.685 | 0.47 m |
| roi_8 | colmap | 23/32 | 23/32 | 0.579 | 0.562 | 0.18 m |
| roi_9 | colmap | 27/38 | 27/38 | 0.915 | 0.908 | 0.84 m |
| roi_10 | colmap | 0/- | 12/27 | - | 1.645 | - m |
| roi_11 | colmap | 111/141 | 50/141 | 2.109 | 1.964 | 2.35 m |
| roi_14 | colmap | 22/22 | 22/22 | 2.659 | 0.772 | 9.84 m |

## Independent check against a multi-pass reference

roi_17 aligns the same photos as roi_14, roi_15 and roi_16 with three passes' evidence. There the
two methods agree within 0.44 m. The reference is the average of the two methods' roi_17
positions, and each single-pass result is compared with it:

| Single pass | Facades: median distance to reference | Points: median distance to reference |
|---|---|---|
| roi_14 (SIFT) | 13.9 m | **4.3 m** |
| roi_15 (LightGlue) | 13.1 m | **7.7 m** |
| roi_16 (LightGlue) | 9.2 m | **1.9 m** |

All six results were marked reliable. roi_15's Mapillary focal is wrong (1.34 instead of 0.80),
which hurts both methods there.

## GLOMAP reconstructions (lightglue_global, 2026-10-20)

Source: alignment_methods_20261020_2323.csv. "Before" is Mapillary's GPS track alone.

| ROI | Reliable (facades) | Reliable (points) | Points to footprint: before | Points to footprint (facades) | Points to footprint (points) | Cameras differ |
|---|---|---|---|---|---|---|
| roi_2 | 30/36 | 30/36 | 3.65 | 2.13 | **1.06** | 7.01 m |
| roi_4 | 30/36 | 30/36 | 3.28 | 1.64 | **1.08** | 2.71 m |
| roi_5 | 48/51 | 48/51 | 2.77 | 1.06 | 1.11 | 0.57 m |
| roi_7 | 21/60 | **37/60** | 2.11 | 0.68 | 0.81 | 0.68 m |
| roi_8 | 23/32 | 23/32 | 5.09 | 0.60 | 0.59 | 0.43 m |
| roi_9 | 33/38 | 33/38 | 9.65 | 1.85 | 1.85 | 0.11 m |
| roi_10 | 21/27 (false pass) | 0/27 | 10.90 | 7.92 | 2.17 | 7.14 m |
| roi_11 | 50/141 | **82/141** | 3.84 | 1.83 | **1.56** | 4.89 m |

roi_10 shows how the facade method can fail without saying so. It found facades on only one side
of the street and fit them to a wall at 0.11 m, so it marked the leg reliable. The building cloud as
a whole, however, ends up 7.9 m from the footprints. The points method fits the cloud at 2.17 m, just
over its 2 m reliability threshold, so it marks the leg unreliable.
