"""Run deterministic offline YOLO prediction on explicitly supplied local images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .train import pick_device

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def discover_images(source: str | Path, *, recursive: bool = False) -> list[Path]:
    """Return a stable list of supported local image files."""
    root = Path(source)
    if root.is_file():
        if root.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image suffix: {root.suffix}")
        return [root.resolve()]
    if not root.is_dir():
        raise FileNotFoundError(f"image source does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    images = sorted(
        (path.resolve() for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path).lower(),
    )
    if not images:
        raise ValueError(f"no supported images found under: {root}")
    return images


def source_relative_path(image: Path, source: Path) -> Path:
    """Preserve source hierarchy for directory inputs and one filename for file input."""
    resolved_image = image.resolve()
    resolved_source = source.resolve()
    return resolved_image.relative_to(resolved_source) if resolved_source.is_dir() else Path(image.name)


def artifact_relative_path(image_relative: Path, suffix: str) -> Path:
    """Append an artifact suffix without discarding the source image extension."""
    return image_relative.parent / f"{image_relative.name}{suffix}"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one model artifact without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_detections(
    xywhn: Sequence[Sequence[float]],
    class_ids: Sequence[float],
    confidences: Sequence[float],
    names: dict[int, str] | list[str] | tuple[str, ...],
) -> list[dict[str, Any]]:
    """Convert model arrays into finite, JSON-safe single-image detections."""
    if not (len(xywhn) == len(class_ids) == len(confidences)):
        raise ValueError("prediction arrays have different lengths")
    detections = []
    for box, raw_class_id, raw_confidence in zip(xywhn, class_ids, confidences):
        if len(box) != 4:
            raise ValueError("each normalized box must contain xc, yc, width, height")
        class_id = int(raw_class_id)
        confidence = float(raw_confidence)
        normalized_box = [float(value) for value in box]
        if not all(math.isfinite(value) for value in [*normalized_box, confidence]):
            raise ValueError("prediction values must be finite")
        xc, yc, width, height = normalized_box
        if class_id < 0:
            raise ValueError("class ids must be non-negative")
        if not (0 <= confidence <= 1):
            raise ValueError("prediction confidence must be between zero and one")
        if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError("normalized prediction box is outside YOLO bounds")
        if isinstance(names, dict):
            class_name = names.get(class_id, str(class_id))
        elif 0 <= class_id < len(names):
            class_name = names[class_id]
        else:
            class_name = str(class_id)
        detections.append(
            {
                "class_id": class_id,
                "class_name": str(class_name),
                "confidence": round(confidence, 8),
                "xywhn": [round(value, 8) for value in normalized_box],
            }
        )
    return detections


def to_prediction_lines(detections: Iterable[dict[str, Any]]) -> str:
    """Serialize detections as YOLO labels with confidence in the sixth field."""
    lines = []
    for detection in detections:
        xc, yc, width, height = detection["xywhn"]
        lines.append(
            f"{detection['class_id']} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f} "
            f"{detection['confidence']:.8f}\n"
        )
    return "".join(lines)


def ensure_clean_output(output: str | Path) -> Path:
    """Reject any prior result files so two prediction runs cannot be mixed."""
    root = Path(output)
    if root.is_file():
        raise FileExistsError(f"prediction output is a file, not a directory: {root}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to mix prediction runs; output is not empty: {root}")
    return root


def build_predict_plan(
    *,
    weights: str | Path,
    source: str | Path,
    output: str | Path,
    conf: float,
    iou: float,
    imgsz: int,
    batch: int,
    device: str,
    recursive: bool,
    save_overlays: bool,
) -> dict[str, Any]:
    """Build the complete machine-readable prediction plan without reading inputs."""
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("conf and iou must be between zero and one")
    if imgsz <= 0 or batch <= 0:
        raise ValueError("imgsz and batch must be positive")
    return {
        "schema_version": 1,
        "weights": str(Path(weights).resolve()),
        "source": str(Path(source).resolve()),
        "output": str(Path(output).resolve()),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "recursive": recursive,
        "save_overlays": save_overlays,
    }


def run_prediction(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute one offline run and write labels, overlays, and a JSON manifest."""
    weights = Path(plan["weights"])
    source = Path(plan["source"])
    output = ensure_clean_output(plan["output"])
    if not weights.is_file():
        raise FileNotFoundError(f"weights do not exist: {weights}")
    images = discover_images(source, recursive=bool(plan["recursive"]))
    output.mkdir(parents=True, exist_ok=True)

    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(
        source=[str(image) for image in images],
        conf=plan["conf"],
        iou=plan["iou"],
        imgsz=plan["imgsz"],
        batch=plan["batch"],
        device=plan["device"],
        save=False,
        save_txt=False,
        verbose=False,
        stream=True,
    )

    items = []
    total_detections = 0
    processed = 0
    for index, result in enumerate(results):
        if index >= len(images):
            raise RuntimeError("model returned more results than input images")
        image = images[index]
        relative = source_relative_path(image, source)
        label_path = output / "labels" / artifact_relative_path(relative, ".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        boxes = result.boxes
        xywhn = boxes.xywhn.cpu().numpy().tolist() if boxes is not None else []
        class_ids = boxes.cls.cpu().numpy().tolist() if boxes is not None else []
        confidences = boxes.conf.cpu().numpy().tolist() if boxes is not None else []
        detections = normalize_detections(xywhn, class_ids, confidences, result.names)
        label_path.write_text(to_prediction_lines(detections), encoding="utf-8")

        overlay_path = None
        if plan["save_overlays"]:
            import cv2

            overlay_path = output / "overlays" / artifact_relative_path(relative, ".png")
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(overlay_path), result.plot()):
                raise OSError(f"failed to write prediction overlay: {overlay_path}")

        total_detections += len(detections)
        processed += 1
        items.append(
            {
                "image": str(image),
                "relative_image": relative.as_posix(),
                "label": str(label_path.resolve()),
                "overlay": str(overlay_path.resolve()) if overlay_path is not None else None,
                "detections": detections,
            }
        )
    if processed != len(images):
        raise RuntimeError(f"model returned {processed} results for {len(images)} input images")

    manifest = {
        **plan,
        "weights_sha256": sha256_file(weights),
        "ultralytics_version": ultralytics.__version__,
        "image_count": len(images),
        "detection_count": total_detections,
        "items": items,
    }
    manifest_path = output / "predictions.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = args.device or ("auto" if args.dry_run else pick_device())
    plan = build_predict_plan(
        weights=args.weights,
        source=args.source,
        output=args.out,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        recursive=args.recursive,
        save_overlays=args.save_overlays,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return
    manifest = run_prediction(plan)
    print(
        json.dumps(
            {
                "image_count": manifest["image_count"],
                "detection_count": manifest["detection_count"],
                "output": manifest["output"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
