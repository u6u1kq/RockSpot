"""Thin YOLOv8 wrapper. Returns instance masks; does not train or save runs."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

# Public COCO checkpoint. Any other missing path is a user file, not a download.
AUTO_DOWNLOAD_WEIGHTS = {"yolov8n.pt"}

# Printed only when the resolved checkpoint sits under weights/legacy/ or the path says legacy.
LEGACY_WEIGHT_WARNING = (
    "legacy / non-commercial: detector weights are a development leftover "
    "(MineralImage5k prototype). Not cleared for commercial release."
)


@dataclass
class Detection:
    xyxy: list[float]
    conf: float
    class_id: int
    class_name: str
    mask_area_px: int | None = None
    centroid_xy: list[float] | None = None
    mask: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "xyxy": self.xyxy,
            "conf": self.conf,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


def resolve_segment_weights(weights: str, repo_root: Path) -> str:
    """Return a path YOLO can open, or the bare ``yolov8n.pt`` name."""
    raw = str(weights).strip()
    if not raw:
        raise ValueError("segment_weights is empty")
    path = Path(raw)
    if path.is_file():
        return str(path.resolve())
    rooted = repo_root / path
    if rooted.is_file():
        return str(rooted.resolve())
    if path.name in AUTO_DOWNLOAD_WEIGHTS and path.parent == Path("."):
        return path.name
    raise FileNotFoundError(
        f"segment weights not found: {raw}. Expected weights/rock_yolov8s_seg_boulder_ft3.pt."
    )


def resolve_detector_weights(weights: str, repo_root: Path) -> str:
    """Alias kept so older imports still resolve a weight path."""
    return resolve_segment_weights(weights, repo_root)


def legacy_weight_warning(weights: str) -> str | None:
    """Warn when a checkpoint is under weights/legacy or the path says legacy."""
    path = Path(weights)
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    posix = path.as_posix().lower()
    if "legacy" in parts or "legacy" in name or "weights/legacy" in posix:
        return LEGACY_WEIGHT_WARNING
    return None


def _class_name(names: object, class_id: int) -> str:
    if isinstance(names, dict):
        if class_id in names:
            return str(names[class_id])
        key = str(class_id)
        if key in names:
            return str(names[key])
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _binary_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.shape != (height, width):
        image = Image.fromarray((array > 0.5).astype(np.uint8) * 255, mode="L")
        image = image.resize((width, height), Image.Resampling.NEAREST)
        array = np.asarray(image)
    return array > 0.5


class YoloDetector:
    def __init__(
        self,
        weights: str,
        device: str,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
    ) -> None:
        self.weights = weights
        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.legacy_warning = legacy_weight_warning(weights)
        if self.legacy_warning:
            print(f"warning: {self.legacy_warning}", file=sys.stderr)
        self.model = YOLO(weights)
        inner = getattr(self.model, "model", None)
        if inner is not None:
            inner.eval()

    def predict(self, image: Image.Image) -> list[Detection]:
        """Return one instance per mask. No masks means no instances."""
        results = self.model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
            save=False,
            retina_masks=True,
        )
        result = results[0]
        boxes = result.boxes
        masks = result.masks
        if boxes is None or len(boxes) == 0 or masks is None:
            return []
        mask_data = masks.data.detach().cpu().numpy()
        if mask_data.ndim != 3 or mask_data.shape[0] == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu()
        confs = boxes.conf.detach().cpu()
        classes = boxes.cls.detach().cpu()
        names = result.names or {}
        width, height = image.size
        count = min(int(xyxy.shape[0]), int(mask_data.shape[0]))
        found: list[Detection] = []
        for i in range(count):
            binary = _binary_mask(mask_data[i], height, width)
            area = int(binary.sum())
            if area <= 0:
                continue
            ys, xs = np.nonzero(binary)
            class_id = int(classes[i].item())
            found.append(
                Detection(
                    xyxy=[float(v) for v in xyxy[i].tolist()],
                    conf=float(confs[i].item()),
                    class_id=class_id,
                    class_name=_class_name(names, class_id),
                    mask_area_px=area,
                    centroid_xy=[float(xs.mean()), float(ys.mean())],
                    mask=binary,
                )
            )
        return found

    def resolved_weights(self) -> str:
        for attr in ("ckpt_path", "pt_path"):
            value = getattr(self.model, attr, None)
            if value:
                return str(value)
        trainer = getattr(self.model, "trainer", None)
        if trainer is not None and getattr(trainer, "last", None):
            return str(trainer.last)
        return self.weights
