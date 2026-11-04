"""Mask2Former semantic segmentation, shared by obstruction removal (texturing) and the
pose alignment's building/road masks (street_masks.py).

The default checkpoint is Swin-Large trained on Mapillary Vistas (65 street-scene classes,
crowdsourced Mapillary photos from many cameras and countries). Cityscapes checkpoints
(19 classes) work too: callers pick classes by name. Both checkpoints' image processors
resize every image to 384 x 384 before the model, and the class map is scaled back to the
image.

LabelCache keeps class maps on disk by image key, so an image segmented by one stage (the
alignment) is not segmented again by another (texturing).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from common.reporter import Reporter, get_reporter

# Override with the SEG_MODEL env var (any Mask2Former semantic checkpoint).
SEG_MODEL_ID = "facebook/mask2former-swin-large-mapillary-vistas-semantic"

_seg_proc: Mask2FormerImageProcessor | None = None
_seg_model: Mask2FormerForUniversalSegmentation | None = None
_seg_key: tuple[str, str, bool] | None = None


def select_device(
    requested: str | None = None, allow_mps: bool = False
) -> torch.device:
    """The requested device, else SEG_DEVICE (cuda/mps/cpu), else CUDA when available,
    else (with allow_mps) Apple's GPU, else the CPU."""
    requested = (requested or os.environ.get("SEG_DEVICE", "")).strip().lower()
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_id_or_default(model_id: str | None = None) -> str:
    return model_id or os.environ.get("SEG_MODEL", "").strip() or SEG_MODEL_ID


def load_model(
    model_id: str | None = None,
    device: str | torch.device | None = None,
    reporter: Reporter | None = None,
    half: bool = False,
) -> tuple[Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation]:
    """The segmentation model and its processor, reloaded only if the model, device or
    precision changed."""
    global _seg_proc, _seg_model, _seg_key
    resolved_device = (
        device if isinstance(device, torch.device) else select_device(device)
    )
    model_id = model_id_or_default(model_id)
    key = (model_id, str(resolved_device), half)
    if _seg_model is not None and _seg_proc is not None and _seg_key == key:
        return _seg_proc, _seg_model
    # Forget the cached key first, so a load that fails halfway (e.g. out of
    # GPU memory in .to()) can't leave a half-replaced model that looks cached.
    _seg_key = None
    (reporter or get_reporter()).info(
        f"Loading segmentation model ({model_id}) on {resolved_device} …"
    )
    _seg_proc = Mask2FormerImageProcessor.from_pretrained(model_id)
    _seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
    _seg_model.eval()
    _seg_model = _seg_model.to(resolved_device)  # type: ignore[assignment]
    if half:
        _seg_model = _seg_model.half()  # type: ignore[union-attr]
    _seg_key = key
    return _seg_proc, _seg_model  # type: ignore[return-value]


def id2label(model: Mask2FormerForUniversalSegmentation) -> dict[int, str]:
    return {int(k): v for k, v in model.config.id2label.items()}


def class_lut(labels: dict[int, str], names, value=1, default=0) -> np.ndarray:
    """A 256-entry lookup table: `value` for the classes whose name (any case) is in names."""
    wanted = {n.lower() for n in names}
    lut = np.full(256, default, np.uint8)
    for i, n in labels.items():
        if n.lower() in wanted:
            lut[i] = value
    return lut


def segment(
    images_bgr: list[np.ndarray],
    model_id: str | None = None,
    device: str | torch.device | None = None,
    reporter: Reporter | None = None,
    half: bool = False,
) -> list[np.ndarray]:
    """Class maps (uint8 class ids, each image's size) of a batch of BGR images."""
    proc, model = load_model(model_id, device, reporter, half)
    dev = next(model.parameters()).device
    # BGR -> RGB as a contiguous copy: with torchvision installed, transformers converts
    # images with torch.from_numpy, which rejects the reversed (negative-stride) view
    rgbs = [np.ascontiguousarray(img[:, :, ::-1]) for img in images_bgr]
    inputs = proc(images=rgbs, return_tensors="pt").to(dev)  # type: ignore[operator]
    if half:
        inputs["pixel_values"] = inputs["pixel_values"].half()
    with torch.no_grad():
        outputs = model(**inputs)
    segs = proc.post_process_semantic_segmentation(
        outputs,
        target_sizes=[img.shape[:2] for img in images_bgr],  # type: ignore[arg-type]
    )
    return [s.cpu().numpy().astype(np.uint8) for s in segs]


class LabelCache:
    """Class maps on disk, <root>/<key>.png, with the model's class names in model.json."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def model(self) -> tuple[str | None, dict[int, str]]:
        """(model id, {class id: name}) of the cached maps; (None, {}) when there are none."""
        f = self.root / "model.json"
        if not f.exists():
            return None, {}
        doc = json.loads(f.read_text())
        return doc["model"], {int(k): v for k, v in doc["id2label"].items()}

    def start(self, model_id: str, labels: dict[int, str]) -> None:
        """Use the cache for this model's maps (dropping maps of another model)."""
        self.root.mkdir(parents=True, exist_ok=True)
        if self.model()[0] not in (None, model_id):
            for p in self.root.glob("*.png"):
                p.unlink()
        (self.root / "model.json").write_text(
            json.dumps({"model": model_id, "id2label": labels}, indent=1)
        )

    def path(self, key: str) -> Path:
        return self.root / f"{key}.png"

    def save(self, key: str, labels: np.ndarray) -> None:
        cv2.imwrite(str(self.path(key)), labels)

    def get(self, key: str, shape: tuple[int, int] | None = None) -> np.ndarray | None:
        """The class map of key (scaled to shape, nearest class), or None."""
        p = self.path(key)
        if not p.exists():
            return None
        labels = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if labels is not None and shape is not None and labels.shape != tuple(shape):
            labels = cv2.resize(
                labels, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
            )
        return labels
