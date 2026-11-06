"""Wall textures blended from every photo that sees them (the pose alignment's facade bake).

Each wall triangle is baked on a grid of TEXEL_M texels laid out in world coordinates
(along the wall and up), so the triangles of one wall sample the same points and meet
without a seam. Every texel takes a weighted average over the photos that see it:

    weight = (cos(view angle) x unobstructed / (1 + distance / 15 m)) ** SHARPEN

so the most square-on, closest, unobstructed view dominates and the others soften the
seams between photos. A photo counts for a wall only within REACH_M, in front of it, and
where no other building's wall crosses the line of sight (a plan-view test per texel
column). Obstructions (segmentation masks, when they are kept rather than painted over)
get no weight. Texels no photo sees stay untextured.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from common.per_patch_confidence import confidence_score, clip_confidence_score

TEXEL_M = 0.1  # metres per texel
MAX_TEXELS = 1024  # longest patch side; larger faces get coarser texels
REACH_M = 60.0  # a photo paints walls within this distance
SHARPEN = 8  # weight exponent: the best view dominates, its neighbours soften seams
MAX_IMAGE_SIDE = 2048  # photos are read at most this large for baking
VERTICAL_NZ = 0.2  # |normal z| below this: a wall


def _seg_hits(P0, P1, A, B):
    """For segments P0[i]->P1[i] (N, 2) and edges A[j]->B[j] (M, 2): (N,) any proper crossing."""
    if len(A) == 0:
        return np.zeros(len(P0), bool)

    def orient(p, q, r):
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (
            q[..., 1] - p[..., 1]
        ) * (r[..., 0] - p[..., 0])

    p0, p1 = P0[:, None], P1[:, None]
    a, b = A[None], B[None]
    d1, d2 = orient(a, b, p0), orient(a, b, p1)
    d3, d4 = orient(p0, p1, a), orient(p0, p1, b)
    return ((d1 * d2 < 0) & (d3 * d4 < 0)).any(axis=1)


def _wall_frames(verts_proj, faces, normals, skip):
    """The wall triangles' texel grids: {face index: frame} and the walls' plan-view
    segments (A, B: (M, 2)) for the occlusion test."""
    frames, seg_a, seg_b = {}, [], []
    for fi in np.flatnonzero(np.abs(normals[:, 2]) < VERTICAL_NZ):
        n = normals[fi, :2]
        n_len = float(np.linalg.norm(n))
        if n_len < 0.5:
            continue
        n = n / n_len  # outward, horizontal
        V = verts_proj[faces[fi]]
        e1 = np.array([-n[1], n[0]])  # along the wall, left to right seen from outside
        s, t = V[:, :2] @ e1, V[:, 2]
        # its plan-view segment: the triangle's extent along the wall
        d = float(V[0, :2] @ n)
        seg = (d * n + s.min() * e1, d * n + s.max() * e1)
        seg_a.append(seg[0])
        seg_b.append(seg[1])
        if fi in skip:
            continue
        s0, s1, t0, t1 = s.min(), s.max(), t.min(), t.max()
        if s1 - s0 < 0.05 or t1 - t0 < 0.05:
            continue
        texel = max(TEXEL_M, (s1 - s0) / MAX_TEXELS, (t1 - t0) / MAX_TEXELS)
        W = max(2, int(math.ceil((s1 - s0) / texel)))
        H = max(2, int(math.ceil((t1 - t0) / texel)))
        cols = s0 + (np.arange(W) + 0.5) * texel
        frames[int(fi)] = {
            "n": n,
            "e1": e1,
            "s0": s0,
            "t1": t1,
            "texel": texel,
            "W": W,
            "H": H,
            "colxy": d * n[None] + cols[:, None] * e1[None],  # (W, 2)
            "z": t1 - (np.arange(H) + 0.5) * texel,  # (H,), row 0 at the top
            "centre": V.mean(axis=0),
            "seg": seg,
            "vs": s,
            "vt": t,
        }
    return frames, np.array(seg_a).reshape(-1, 2), np.array(seg_b).reshape(-1, 2)


def bake_walls(
    verts_proj, faces, normals, cam_data, skip, reporter, grey=225
) -> tuple[list[dict], dict]:
    """Patches (texturing's patch records: fi, patch RGB, pw, ph, v0_px..v2_px) of every
    wall triangle a photo sees, and stats. cam_data: texturing's cameras (img_path, w, h,
    K, R world-to-camera, Cw, mask: a small obstruction mask or None)."""
    frames, A, B = _wall_frames(verts_proj, faces, normals, skip)
    if not frames:
        return [], {"walls": 0}
    fis = np.array(list(frames))
    centres = np.array([frames[f]["centre"] for f in fis])
    outward = np.array([frames[f]["n"] for f in fis])
    acc: dict[int, np.ndarray] = {}
    wsum: dict[int, np.ndarray] = {}
    confidence_map: dict[int, np.ndarray] = {}
    lo_ab, hi_ab = np.minimum(A, B), np.maximum(A, B)

    for cd in reporter.progress(cam_data, desc="Blending photos onto walls"):
        C = np.asarray(cd["Cw"], float)
        to_cam = C[None, :2] - centres[:, :2]
        dist = np.linalg.norm(to_cam, axis=1)
        near = np.flatnonzero(
            (dist < REACH_M) & (np.einsum("ij,ij->i", to_cam, outward) > 0.5)
        )
        if not len(near):
            continue
        img = cv2.imread(str(cd["img_path"]))
        if img is None:
            continue
        k = min(1.0, MAX_IMAGE_SIDE / max(img.shape[:2]))
        if k < 1.0:
            img = cv2.resize(
                img,
                (round(img.shape[1] * k), round(img.shape[0] * k)),
                interpolation=cv2.INTER_AREA,
            )
        sx, sy = img.shape[1] / cd["w"], img.shape[0] / cd["h"]
        K = cd["K"]
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
        R = np.asarray(cd["R"], float)
        mask = cd.get("mask")
        for fi in fis[near]:
            fr = frames[int(fi)]
            colxy, z = fr["colxy"], fr["z"]
            # other buildings' walls between the camera and each texel column
            lo = np.minimum(colxy.min(0), C[:2]) - 1
            hi = np.maximum(colxy.max(0), C[:2]) + 1
            sel = np.all((lo_ab <= hi) & (hi_ab >= lo), axis=1)
            a_sel, b_sel = A[sel], B[sel]
            own = np.all(np.isclose(a_sel, fr["seg"][0]), axis=1) & np.all(
                np.isclose(b_sel, fr["seg"][1]), axis=1
            )
            to_c = C[:2] - colxy
            ends = (
                colxy
                + to_c * (0.3 / np.maximum(np.linalg.norm(to_c, axis=1), 1e-6))[:, None]
            )
            blocked = _seg_hits(
                np.broadcast_to(C[:2], colxy.shape), ends, a_sel[~own], b_sel[~own]
            )
            if blocked.all():
                continue
            H, W = fr["H"], fr["W"]
            P = np.empty((H, W, 3))
            P[..., :2] = colxy[None]
            P[..., 2] = z[:, None]
            Xc = (P - C) @ R.T
            zc = Xc[..., 2]
            ok = zc > 0.5
            zs = np.where(ok, zc, 1.0)
            u = fx * Xc[..., 0] / zs + cx
            v = fy * Xc[..., 1] / zs + cy
            ok &= (u >= 0) & (u < img.shape[1] - 1) & (v >= 0) & (v < img.shape[0] - 1)
            ok &= ~blocked[None, :]
            if not ok.any():
                continue
            ray = C - P
            rn = np.linalg.norm(ray, axis=2)
            cos = np.clip((ray[..., :2] @ fr["n"]) / rn, 0, 1)
            clear = np.ones_like(cos)
            if mask is not None:
                mu = np.clip(
                    (u * mask.shape[1] / img.shape[1]).astype(int), 0, mask.shape[1] - 1
                )
                mv = np.clip(
                    (v * mask.shape[0] / img.shape[0]).astype(int), 0, mask.shape[0] - 1
                )
                clear = 1.0 - mask[mv, mu] / 255.0
            w = ((cos * clear / (1.0 + rn / 15.0)) ** SHARPEN * ok).astype(np.float32)
            if not w.any():
                continue
            col = cv2.remap(
                img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR
            ).astype(np.float32)
            if int(fi) not in acc:
                acc[int(fi)] = np.zeros((H, W, 3), np.float32)
                wsum[int(fi)] = np.zeros((H, W), np.float32)
                confidence_map[int(fi)] = np.zeros((H, W), np.float32)
            acc[int(fi)] += col * w[..., None]
            wsum[int(fi)] += w
            confidence = confidence_score(rn, cos, clear, average=False)

            # Keep the highest confidence observation for each texel
            confidence_map[int(fi)] = np.maximum(
                confidence_map[int(fi)],
                confidence
            )


    patches, seen, total = [], 0, 0
    for fi, a in acc.items():
        fr, ws = frames[fi], wsum[fi]
        has = ws > 1e-30
        tex = np.full(a.shape, grey, np.uint8)
        tex[has] = np.clip(a[has] / ws[has][:, None], 0, 255).astype(np.uint8)
        tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
        face_confidence = confidence_map[int(fi)].astype(np.uint8)
        tex_rgba = np.dstack((tex, face_confidence))
        seen += int(has.sum())
        total += has.size

        def to_px(s, t, _fr=fr):
            return (s - _fr["s0"]) / _fr["texel"], (_fr["t1"] - t) / _fr["texel"]

        patches.append(
            {
                "fi": fi,
                "patch": tex_rgba,
                "pw": fr["W"],
                "ph": fr["H"],
                **{f"v{j}_px": to_px(fr["vs"][j], fr["vt"][j]) for j in range(3)},
            }
        )
    return patches, {
        "walls": len(frames),
        "textured_walls": len(patches),
        "coverage": round(seen / max(total, 1), 3),
    }
