from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

SEG_MODEL_ID = "facebook/mask2former-swin-large-cityscapes-semantic"

OBSTRUCTION_CLASSES = {
    5,
    6,
    7,
    8,
    11,
    12,
}

LAMA_MAX_DIM = 1024

_seg_proc: Mask2FormerImageProcessor | None = None
_seg_model: Mask2FormerForUniversalSegmentation | None = None
_lama: SimpleLama | None = None


def _load_seg_model() -> None:
    global _seg_proc, _seg_model
    if _seg_model is not None:
        return
    print(f"Loading segmentation model ({SEG_MODEL_ID}) …")
    _seg_proc = Mask2FormerImageProcessor.from_pretrained(SEG_MODEL_ID)
    _seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL_ID)
    _seg_model.eval()
    if torch.cuda.is_available():
        _seg_model = _seg_model.cuda()  # type: ignore[assignment]
        print("  -> running on GPU")
    else:
        print("  -> running on CPU (this will be slow)")


def _load_lama() -> SimpleLama:
    global _lama
    if _lama is not None:
        return _lama
    print("Loading LaMa inpainting model …")
    _orig = torch.jit.load
    torch.jit.load = lambda *a, **kw: _orig(*a, **{**kw, "map_location": "cpu"})
    try:
        _lama = SimpleLama()
    finally:
        torch.jit.load = _orig
    return _lama


def build_obstruction_mask(img_bgr: np.ndarray) -> np.ndarray:
    _load_seg_model()
    assert _seg_model is not None
    assert _seg_proc is not None
    h, w = img_bgr.shape[:2]
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    device = next(_seg_model.parameters()).device
    inputs = _seg_proc(images=pil, return_tensors="pt").to(device)  # type: ignore[operator]
    with torch.no_grad():
        outputs = _seg_model(**inputs)
    seg = _seg_proc.post_process_semantic_segmentation(outputs, target_sizes=[(h, w)])[
        0
    ]  # type: ignore[arg-type]
    seg_np = seg.cpu().numpy()

    mask = np.zeros((h, w), dtype=np.uint8)
    for cls_id in OBSTRUCTION_CLASSES:
        mask[seg_np == cls_id] = 255

    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    return mask


def synthesize_texture(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lama = _load_lama()
    h, w = img_bgr.shape[:2]

    scale = min(1.0, LAMA_MAX_DIM / max(h, w))
    ih, iw = int(h * scale), int(w * scale)
    img_s = cv2.resize(img_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
    msk_s = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

    pil_img = Image.fromarray(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(msk_s)
    result_s = cv2.cvtColor(np.array(lama(pil_img, pil_mask)), cv2.COLOR_RGB2BGR)

    inpainted = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img_bgr.copy()
    out[mask > 0] = inpainted[mask > 0]
    return out


def remove_obstructions(img_bgr: np.ndarray) -> np.ndarray:
    mask = build_obstruction_mask(img_bgr)
    #n_obs = int((mask > 0).sum())
    return synthesize_texture(img_bgr, mask)
