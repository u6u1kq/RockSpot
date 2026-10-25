"""Image in, instance masks, mask crops, then one class and feature per crop."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

from rockspot.classify import ClassifierInfo, RockClassifier
from rockspot.detect import (
    Detection,
    YoloDetector,
    legacy_weight_warning,
    resolve_segment_weights,
)

CROP_SEGMENT = "segment"
CROP_FULL_IMAGE = "full_image"
DEFAULT_SEGMENT_WEIGHTS = "weights/rock_yolov8s_seg_boulder_ft3.pt"
# YOLO letterbox gray. Pixels outside the instance mask are filled with this.
NEUTRAL_RGB = (114, 114, 114)


@dataclass
class InferConfig:
    repo_root: Path
    device: str
    segment_weights: str
    conf: float
    iou: float
    imgsz: int
    max_det: int
    classifier_weights: str | None
    class_names: str | None
    classify_imgsz: int
    feature_dim_note: int | None
    output_dir: Path


@dataclass
class CropItem:
    crop_source: str
    box: Detection | None
    image: Image.Image
    mask_image: Image.Image | None
    class_id: int
    class_name: str
    score: float
    feature: object
    feature_dim: int
    confidence: float | None
    mask_area_px: int | None
    mask_area_ratio: float | None
    bbox_xyxy: list[float] | None
    bbox_area_px: float | None
    centroid_xy: list[float] | None
    equivalent_diameter_px: float | None


@dataclass
class PipelineResult:
    image: str
    device: str
    fallback_full_image: bool
    note: str | None
    segment: dict
    classifier: ClassifierInfo
    items: list[CropItem]


def load_config(path: Path) -> InferConfig:
    path = path.resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    repo_root = path.parent.parent
    output = Path(str(data.get("output_dir") or "outputs"))
    if not output.is_absolute():
        output = repo_root / output
    segment_weights = _optional_text(data.get("segment_weights"))
    if segment_weights is None:
        # Explicit older key only. The default below is the segmentation checkpoint.
        segment_weights = _optional_text(data.get("detector_weights"))
    if segment_weights is None:
        segment_weights = DEFAULT_SEGMENT_WEIGHTS
    cfg = InferConfig(
        repo_root=repo_root,
        device=str(data.get("device") or "auto"),
        segment_weights=segment_weights,
        conf=_as_float(data.get("conf", 0.25), "conf"),
        iou=_as_float(data.get("iou", 0.45), "iou"),
        imgsz=_as_int(data.get("imgsz", 640), "imgsz"),
        max_det=_as_int(data.get("max_det", 50), "max_det"),
        classifier_weights=_optional_text(data.get("classifier_weights")),
        class_names=_optional_text(data.get("class_names")),
        classify_imgsz=_as_int(data.get("classify_imgsz", 224), "classify_imgsz"),
        feature_dim_note=_optional_int(data.get("feature_dim")),
        output_dir=output,
    )
    if not 0.0 < cfg.conf <= 1.0:
        raise ValueError("conf must be in (0, 1]")
    if not 0.0 < cfg.iou <= 1.0:
        raise ValueError("iou must be in (0, 1]")
    if cfg.imgsz < 32 or cfg.classify_imgsz < 32:
        raise ValueError("imgsz and classify_imgsz must be >= 32")
    if cfg.max_det < 1:
        raise ValueError("max_det must be >= 1")
    return cfg


def resolve_device(name: str) -> str:
    text = (name or "auto").strip().lower()
    if text == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device is {name} but CUDA is not available")
    return text


def run_inference(image_path: Path, cfg: InferConfig) -> PipelineResult:
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    device = resolve_device(cfg.device)
    with Image.open(image_path) as handle:
        transposed = ImageOps.exif_transpose(handle)
        if transposed is None:
            raise ValueError(f"could not read image: {image_path}")
        image = transposed.convert("RGB").copy()
    weights = resolve_segment_weights(cfg.segment_weights, cfg.repo_root)
    detector = YoloDetector(
        weights=weights,
        device=device,
        conf=cfg.conf,
        iou=cfg.iou,
        imgsz=cfg.imgsz,
        max_det=cfg.max_det,
    )
    resolved_weights = detector.resolved_weights()
    legacy_warning = detector.legacy_warning or legacy_weight_warning(resolved_weights)
    if legacy_warning and not detector.legacy_warning:
        print(f"warning: {legacy_warning}", file=sys.stderr)
    instances = detector.predict(image)
    classifier = RockClassifier(
        device=device,
        imgsz=cfg.classify_imgsz,
        repo_root=cfg.repo_root,
        requested_weights=cfg.classifier_weights,
        requested_class_names=cfg.class_names,
    )
    _note_feature_dim(cfg, classifier.info)

    width, height = image.size
    crops = []
    for inst in instances:
        if inst.mask is None:
            continue
        cropped, mask_image = mask_crop(image, inst.mask, inst.xyxy)
        crops.append((CROP_SEGMENT, inst, cropped, mask_image))
    if crops:
        fallback = False
        note = None
    else:
        # No masks. Do not load another detector. Stage 2 still runs once on the full image.
        crops = [(CROP_FULL_IMAGE, None, image, None)]
        fallback = True
        note = (
            "The segmenter found nothing. Stage 2 ran once on the full image "
            "(crop_source=full_image)."
        )

    predictions = classifier.predict_batch([crop for _, _, crop, _ in crops])
    if len(predictions) != len(crops):
        raise RuntimeError("classifier returned a different number of results than crops")
    items = []
    for (source, box, crop, mask_image), pred in zip(crops, predictions):
        geometry = _geometry(box, width, height)
        items.append(
            CropItem(
                crop_source=source,
                box=box,
                image=crop,
                mask_image=mask_image,
                class_id=pred.class_id,
                class_name=pred.class_name,
                score=pred.score,
                feature=pred.feature,
                feature_dim=pred.feature_dim,
                confidence=geometry["confidence"],
                mask_area_px=geometry["mask_area_px"],
                mask_area_ratio=geometry["mask_area_ratio"],
                bbox_xyxy=geometry["bbox_xyxy"],
                bbox_area_px=geometry["bbox_area_px"],
                centroid_xy=geometry["centroid_xy"],
                equivalent_diameter_px=geometry["equivalent_diameter_px"],
            )
        )
    segment = {
        "weights": cfg.segment_weights,
        "weights_resolved": resolved_weights,
        "conf": cfg.conf,
        "iou": cfg.iou,
        "imgsz": cfg.imgsz,
        "max_det": cfg.max_det,
        "num_instances": len(instances),
        "num_detections": len(instances),
        "warning": legacy_warning,
    }
    return PipelineResult(
        image=image_path.as_posix(),
        device=device,
        fallback_full_image=fallback,
        note=note,
        segment=segment,
        classifier=classifier.info,
        items=items,
    )


def mask_crop(
    image: Image.Image,
    mask: np.ndarray,
    xyxy: list[float],
) -> tuple[Image.Image, Image.Image]:
    """Crop to the box. Pixels outside the mask are filled with neutral gray."""
    width, height = image.size
    x1, y1, x2, y2 = xyxy
    left = max(0, min(math.floor(x1), width - 1))
    top = max(0, min(math.floor(y1), height - 1))
    right = max(left + 1, min(math.ceil(x2), width))
    bottom = max(top + 1, min(math.ceil(y2), height))
    if mask.shape != (height, width):
        raise RuntimeError(f"mask shape {mask.shape} does not match image {(height, width)}")
    arr = np.asarray(image, dtype=np.uint8)
    crop = np.array(arr[top:bottom, left:right], copy=True)
    sub = np.asarray(mask[top:bottom, left:right], dtype=bool)
    crop[~sub] = NEUTRAL_RGB
    mask_u8 = np.where(sub, 255, 0).astype(np.uint8)
    return Image.fromarray(crop), Image.fromarray(mask_u8, mode="L")


def _geometry(det: Detection | None, width: int, height: int) -> dict:
    empty = {
        "confidence": None,
        "mask_area_px": None,
        "mask_area_ratio": None,
        "bbox_xyxy": None,
        "bbox_area_px": None,
        "centroid_xy": None,
        "equivalent_diameter_px": None,
    }
    if det is None or det.mask_area_px is None:
        return empty
    area = int(det.mask_area_px)
    x1, y1, x2, y2 = (float(v) for v in det.xyxy)
    pixels = int(width) * int(height)
    ratio = float(area) / float(pixels) if pixels else None
    diameter = 2.0 * math.sqrt(area / math.pi)
    centroid = None if det.centroid_xy is None else [float(v) for v in det.centroid_xy]
    return {
        "confidence": float(det.conf),
        "mask_area_px": area,
        "mask_area_ratio": ratio,
        "bbox_xyxy": [x1, y1, x2, y2],
        "bbox_area_px": max(0.0, (x2 - x1) * (y2 - y1)),
        "centroid_xy": centroid,
        "equivalent_diameter_px": float(diameter),
    }


def _note_feature_dim(cfg: InferConfig, info: ClassifierInfo) -> None:
    if cfg.feature_dim_note is None or cfg.feature_dim_note == info.feature_dim:
        return
    message = f"config feature_dim is {cfg.feature_dim_note}; model feature_dim is {info.feature_dim}"
    info.warning = f"{info.warning}; {message}" if info.warning else message


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return text


def _optional_int(value: object) -> int | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _as_int(value, "feature_dim")


def _as_float(value: object, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _as_int(value: object, name: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)
