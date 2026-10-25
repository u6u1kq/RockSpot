"""MobileNetV3-Small: pooled feature plus top-1 class.

classifier_weights empty, or the file missing on disk, uses ImageNet.
A configured path that is not on disk is ``imagenet_fallback`` — that file was
not loaded. ``weights_source`` records which weights actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# torchvision MobileNetV3-Small: features -> avgpool -> flatten is 576-d,
# then the classifier head maps 576 -> 1024 -> num_classes.
FEATURE_DIM = 576
ARCH = "mobilenet_v3_small"
_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")


@dataclass
class Prediction:
    class_id: int
    class_name: str
    score: float
    feature: np.ndarray
    feature_dim: int


@dataclass
class ClassifierInfo:
    arch: str
    weights_source: str
    weights_path: str | None
    requested_weights: str | None
    class_names_source: str
    class_names_path: str | None
    feature_dim: int
    num_classes: int
    warning: str | None

    def to_dict(self) -> dict:
        return {
            "arch": self.arch,
            "weights_source": self.weights_source,
            "weights_path": self.weights_path,
            "requested_weights": self.requested_weights,
            "class_names_source": self.class_names_source,
            "class_names_path": self.class_names_path,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "warning": self.warning,
        }


def find_existing_file(value: str | None, repo_root: Path) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"null", "none"}:
        return None
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    rooted = repo_root / path
    if rooted.is_file():
        return rooted.resolve()
    return None


def load_class_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return _names_from_mapping(yaml.safe_load(text), path)
    if suffix == ".json":
        import json

        return _names_from_mapping(json.loads(text), path)
    names = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not names:
        raise ValueError(f"class_names file has no names: {path}")
    return names


def _names_from_mapping(data: object, path: Path) -> list[str]:
    if isinstance(data, list):
        return [str(item) for item in data]
    if isinstance(data, dict):
        names = data.get("names", data)
        if isinstance(names, list):
            return [str(item) for item in names]
        if isinstance(names, dict):
            ordered = sorted(names, key=lambda key: int(key))
            return [str(names[key]) for key in ordered]
    raise ValueError(f"class_names file must be a list or a names map: {path}")


class RockClassifier:
    def __init__(
        self,
        device: str,
        imgsz: int,
        repo_root: Path,
        requested_weights: str | None,
        requested_class_names: str | None,
    ) -> None:
        self.device = device
        self.preprocess = _build_preprocess(imgsz)
        weights_file = find_existing_file(requested_weights, repo_root)
        names_file = find_existing_file(requested_class_names, repo_root)
        self.model, self.info, self.class_names = _load_model(
            weights_file=weights_file,
            requested_weights=_blank_to_none(requested_weights),
            names_file=names_file,
            requested_class_names=_blank_to_none(requested_class_names),
        )
        self.model.to(device)
        self.model.eval()

    def predict_batch(self, images: list[Image.Image]) -> list[Prediction]:
        if not images:
            return []
        batch = torch.stack([_as_tensor(self.preprocess(image)) for image in images])
        batch = batch.to(self.device)
        with torch.inference_mode():
            features = _pooled_features(self.model, batch)
            logits = self.model.classifier(features)
            scores, indices = torch.softmax(logits, dim=1).max(dim=1)
        feat_cpu = features.detach().cpu().numpy()
        predictions: list[Prediction] = []
        for i in range(feat_cpu.shape[0]):
            class_id = int(indices[i].item())
            vector = np.asarray(feat_cpu[i], dtype=np.float32).reshape(-1)
            predictions.append(
                Prediction(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    score=float(scores[i].item()),
                    feature=vector,
                    feature_dim=int(vector.shape[0]),
                )
            )
        return predictions


def _blank_to_none(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none"}:
        return None
    return text


def _as_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    raise TypeError("classifier preprocess did not return a tensor")


def _build_preprocess(imgsz: int):
    if imgsz == 224:
        return MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    resize = max(imgsz, int(round(imgsz * 256 / 224)))
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _imagenet_names() -> list[str]:
    categories = MobileNet_V3_Small_Weights.IMAGENET1K_V1.meta["categories"]
    return [str(name) for name in categories]


def _load_model(
    weights_file: Path | None,
    requested_weights: str | None,
    names_file: Path | None,
    requested_class_names: str | None,
) -> tuple[nn.Module, ClassifierInfo, list[str]]:
    warnings: list[str] = []
    if requested_weights and weights_file is None:
        # Path was set, file is absent. Say so; do not treat ImageNet as that checkpoint.
        warnings.append(
            f"classifier_weights not found ({requested_weights}); using ImageNet MobileNetV3-Small"
        )
        if requested_class_names:
            warnings.append("class_names not applied because the checkpoint is missing")
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        names = _imagenet_names()
        info = _info(
            model,
            weights_source="imagenet_fallback",
            weights_path=None,
            requested_weights=requested_weights,
            class_names_source="imagenet",
            class_names_path=None,
            num_classes=len(names),
            warnings=warnings,
        )
        return model, info, names

    if weights_file is None:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        names, names_source, names_path = _resolve_names(
            num_classes=1000,
            names_file=names_file,
            requested_class_names=requested_class_names,
            warnings=warnings,
        )
        info = _info(
            model,
            weights_source="imagenet",
            weights_path=None,
            requested_weights=None,
            class_names_source=names_source,
            class_names_path=names_path,
            num_classes=len(names),
            warnings=warnings,
        )
        return model, info, names

    state = _read_state_dict(weights_file)
    model = mobilenet_v3_small(weights=None)
    _adapt_classifier_head(model, state)
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    critical = [key for key in missing if key.startswith("features.")]
    if critical:
        sample = ", ".join(critical[:3])
        raise RuntimeError(
            "classifier_weights is missing MobileNetV3-Small backbone tensors "
            f"({sample}). Expected a torchvision mobilenet_v3_small state_dict."
        )
    if missing or unexpected:
        warnings.append(
            f"checkpoint load skipped {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
    num_classes = _num_classes(model)
    names, names_source, names_path = _resolve_names(
        num_classes=num_classes,
        names_file=names_file,
        requested_class_names=requested_class_names,
        warnings=warnings,
    )
    info = _info(
        model,
        weights_source="checkpoint",
        weights_path=str(weights_file),
        requested_weights=requested_weights,
        class_names_source=names_source,
        class_names_path=names_path,
        num_classes=len(names),
        warnings=warnings,
    )
    return model, info, names


def _info(
    model: nn.Module,
    weights_source: str,
    weights_path: str | None,
    requested_weights: str | None,
    class_names_source: str,
    class_names_path: str | None,
    num_classes: int,
    warnings: list[str],
) -> ClassifierInfo:
    feature_dim = _feature_dim(model)
    if feature_dim != FEATURE_DIM:
        warnings.append(f"expected pooled feature dim {FEATURE_DIM}, model reports {feature_dim}")
    return ClassifierInfo(
        arch=ARCH,
        weights_source=weights_source,
        weights_path=weights_path,
        requested_weights=requested_weights,
        class_names_source=class_names_source,
        class_names_path=class_names_path,
        feature_dim=feature_dim,
        num_classes=num_classes,
        warning="; ".join(warnings) if warnings else None,
    )


def _resolve_names(
    num_classes: int,
    names_file: Path | None,
    requested_class_names: str | None,
    warnings: list[str],
) -> tuple[list[str], str, str | None]:
    if names_file is not None:
        names = load_class_names(names_file)
        if len(names) != num_classes:
            warnings.append(
                f"class_names has {len(names)} entries but the head has {num_classes} outputs"
            )
        else:
            return names, "file", str(names_file)
    elif requested_class_names:
        warnings.append(f"class_names file not found ({requested_class_names})")

    if num_classes == 1000:
        return _imagenet_names(), "imagenet", None
    warnings.append("using class_0.. labels")
    return [f"class_{index}" for index in range(num_classes)], "index", None


def _read_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    except Exception:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in _unwrap_state_dict(blob).items():
        if not torch.is_tensor(value):
            continue
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    if not cleaned:
        raise ValueError(f"no tensors in classifier checkpoint: {path}")
    return cleaned


def _unwrap_state_dict(blob: object) -> dict:
    if isinstance(blob, nn.Module):
        return blob.state_dict()
    if not isinstance(blob, dict):
        raise ValueError("classifier checkpoint must be a state_dict or a dict that contains one")
    for key in _STATE_DICT_KEYS:
        inner = blob.get(key)
        if isinstance(inner, nn.Module):
            return inner.state_dict()
        if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
            return inner
    if any(torch.is_tensor(v) for v in blob.values()):
        return blob
    raise ValueError("could not find a state_dict in classifier checkpoint")


def _adapt_classifier_head(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Match the final Linear width to the checkpoint before load_state_dict."""
    if "classifier.3.weight" in state:
        out_features, in_features = (int(v) for v in state["classifier.3.weight"].shape)
        current = model.classifier[3]
        if (
            not isinstance(current, nn.Linear)
            or current.out_features != out_features
            or current.in_features != in_features
        ):
            model.classifier[3] = nn.Linear(in_features, out_features)
        return
    weight = state.get("classifier.weight")
    if weight is not None and weight.ndim == 2:
        out_features, in_features = (int(v) for v in weight.shape)
        model.classifier = nn.Linear(in_features, out_features)


def _num_classes(model: nn.Module) -> int:
    head = model.classifier
    if isinstance(head, nn.Linear):
        return int(head.out_features)
    if isinstance(head, nn.Sequential) and isinstance(head[-1], nn.Linear):
        return int(head[-1].out_features)
    raise RuntimeError("MobileNetV3 classifier head has no final Linear")


def _feature_dim(model: nn.Module) -> int:
    head = model.classifier
    if isinstance(head, nn.Linear):
        return int(head.in_features)
    if isinstance(head, nn.Sequential) and isinstance(head[0], nn.Linear):
        return int(head[0].in_features)
    raise RuntimeError("cannot read MobileNetV3-Small feature width")


def _pooled_features(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    hidden = model.features(batch)
    pooled = model.avgpool(hidden)
    return torch.flatten(pooled, 1)
