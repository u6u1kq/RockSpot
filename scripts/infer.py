"""Run RockSpot inference on one image and write crops, features, and JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="RockSpot inference on one image")
    parser.add_argument("--image", required=True, type=Path, help="Path to one validation image")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "infer.yaml",
        help="Inference YAML (default: configs/infer.yaml)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <output_dir>/<image stem>)",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"image not found: {args.image}")

    try:
        import numpy as np

        from rockspot.pipeline import load_config, run_inference
    except ImportError as exc:
        _print_install_help(exc)
        raise SystemExit(1) from exc

    cfg = load_config(args.config)
    result = run_inference(args.image, cfg)

    out = args.out if args.out is not None else cfg.output_dir / args.image.stem
    crops_dir = out / "crops"
    features_dir = out / "features"
    crops_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    masks_dir = out / "masks"
    items: list[dict] = []
    for index, item in enumerate(result.items):
        crop_name = f"crop_{index:03d}.png"
        feature_name = f"feat_{index:03d}.npy"
        item.image.save(crops_dir / crop_name)
        np.save(features_dir / feature_name, item.feature)
        mask_path = None
        if item.mask_image is not None:
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_name = f"mask_{index:03d}.png"
            item.mask_image.save(masks_dir / mask_name)
            mask_path = f"masks/{mask_name}"
        items.append(
            {
                "index": index,
                "crop_source": item.crop_source,
                "confidence": item.confidence,
                "mask_area_px": item.mask_area_px,
                "mask_area_ratio": item.mask_area_ratio,
                "bbox_xyxy": item.bbox_xyxy,
                "bbox_area_px": item.bbox_area_px,
                "centroid_xy": item.centroid_xy,
                "equivalent_diameter_px": item.equivalent_diameter_px,
                "box": None if item.box is None else item.box.to_dict(),
                "crop_path": f"crops/{crop_name}",
                "mask_path": mask_path,
                "class_id": item.class_id,
                "class_name": item.class_name,
                "score": item.score,
                "feature_path": f"features/{feature_name}",
                "feature_dim": item.feature_dim,
                "weights_source": result.classifier.weights_source,
            }
        )

    payload = {
        "image": result.image,
        "device": result.device,
        "fallback_full_image": result.fallback_full_image,
        "note": result.note,
        "segment": result.segment,
        "detector": result.segment,
        "classifier": result.classifier.to_dict(),
        "items": items,
    }
    results_path = out / "results.json"
    results_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"device: {result.device}")
    print(f"weights_resolved: {result.segment.get('weights_resolved')}")
    if result.segment.get("warning"):
        print(f"warning: {result.segment['warning']}")
    print(f"instances: {result.segment['num_instances']}")
    print(f"fallback_full_image: {result.fallback_full_image}")
    print(f"weights_source: {result.classifier.weights_source}")
    if result.note:
        print(result.note)
    if result.classifier.warning:
        print(f"warning: {result.classifier.warning}")
    if items:
        print(f"feature_dim: {items[0]['feature_dim']}")
    print(f"wrote: {results_path}")


def _print_install_help(exc: ImportError) -> None:
    print(f"error: missing dependency ({exc})", file=sys.stderr)
    print("CPU install:", file=sys.stderr)
    print(
        "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        file=sys.stderr,
    )
    print("  pip install ultralytics pyyaml pillow numpy", file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
