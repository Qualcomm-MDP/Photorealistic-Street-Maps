from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from simple_lama_inpainting import SimpleLama

from common.reporter import Reporter, get_reporter
from segmentation.model import (
    class_lut,
    id2label,
    load_model,
    segment,
    select_device,
)

# Classes painted over before texturing (Mapillary Vistas and Cityscapes names, matched in
# any case): things in front of the facades. Vistas also marks the capture vehicle itself
# (Ego Vehicle, Car Mount), which Cityscapes' 19 classes cannot.
OBSTRUCTION_NAMES = {
    # Cityscapes
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    # Mapillary Vistas (Pole, Traffic Light, Vegetation, Person, Car, ... as above)
    "utility pole",
    "street light",
    "traffic sign (front)",
    "traffic sign (back)",
    "traffic sign frame",
    "bicyclist",
    "motorcyclist",
    "other rider",
    "boat",
    "caravan",
    "on rails",
    "other vehicle",
    "trailer",
    "wheeled slow",
    "car mount",
    "ego vehicle",
    "bird",
    "ground animal",
}

LAMA_MAX_DIM = 1024

_lama: SimpleLama | None = None
_lama_cpu_only = False  # set once LaMa failed on the GPU: the CPU from then on


def _load_lama(
    device: str | None = None, reporter: Reporter | None = None
) -> SimpleLama:
    """LaMa on the best device (select_device; on Apple's GPU about 14x faster than the
    CPU, the same result). The TorchScript file is loaded onto the CPU first (it was saved
    on CUDA), then moved."""
    global _lama, _lama_cpu_only
    target = torch.device("cpu") if _lama_cpu_only else select_device(device)
    if _lama is not None and _lama.device == target:
        return _lama
    (reporter or get_reporter()).info(f"Loading LaMa inpainting model on {target} …")
    _orig = torch.jit.load
    torch.jit.load = lambda *a, **kw: _orig(*a, **{**kw, "map_location": "cpu"})
    try:
        lama = SimpleLama(device=torch.device("cpu"))
    finally:
        torch.jit.load = _orig
    if target.type != "cpu":
        try:
            lama.model.to(target)
            lama.device = target
        except RuntimeError as exc:
            (reporter or get_reporter()).warning(
                f"LaMa on {target} failed ({exc}); using the CPU"
            )
            _lama_cpu_only = True
    _lama = lama
    return _lama


def obstruction_mask(labels: np.ndarray, names: dict[int, str]) -> np.ndarray:
    """255 where the class map (with its class names) shows an obstruction, grown by 7 px."""
    mask = class_lut(names, OBSTRUCTION_NAMES, value=255)[labels]
    return cv2.dilate(mask, np.ones((15, 15), np.uint8))


def build_obstruction_mask(
    img_bgr: np.ndarray,
    model_id: str | None = None,
    device: str | None = None,
    reporter: Reporter | None = None,
    labels: tuple[np.ndarray, dict[int, str]] | None = None,
) -> np.ndarray:
    """labels: the image's class map and class names when it was segmented already
    (segmentation.model.LabelCache); otherwise the model runs on the image."""
    if labels is None:
        seg = segment([img_bgr], model_id, device, reporter)[0]
        _, model = load_model(model_id, device, reporter)
        labels = (seg, id2label(model))
    return obstruction_mask(*labels)


def synthesize_texture(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    reporter: Reporter | None = None,
    device: str | None = None,
) -> np.ndarray:
    lama = _load_lama(device, reporter)
    h, w = img_bgr.shape[:2]

    scale = min(1.0, LAMA_MAX_DIM / max(h, w))
    ih, iw = int(h * scale), int(w * scale)
    img_s = cv2.resize(img_bgr, (iw, ih), interpolation=cv2.INTER_AREA)
    msk_s = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

    pil_img = Image.fromarray(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(msk_s)
    try:
        result = lama(pil_img, pil_mask)
    except RuntimeError as exc:
        if lama.device.type == "cpu":
            raise
        (reporter or get_reporter()).warning(
            f"LaMa on {lama.device} failed ({exc}); using the CPU"
        )
        global _lama_cpu_only
        _lama_cpu_only = True
        lama.model.to(torch.device("cpu"))
        lama.device = torch.device("cpu")
        result = lama(pil_img, pil_mask)
    result_s = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    inpainted = cv2.resize(result_s, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img_bgr.copy()
    out[mask > 0] = inpainted[mask > 0]
    return out


def remove_obstructions(
    img_bgr: np.ndarray,
    model_id: str | None = None,
    device: str | None = None,
    reporter: Reporter | None = None,
    labels: tuple[np.ndarray, dict[int, str]] | None = None,
) -> np.ndarray:
    mask = build_obstruction_mask(img_bgr, model_id, device, reporter, labels)
    return synthesize_texture(img_bgr, mask, reporter, device)
