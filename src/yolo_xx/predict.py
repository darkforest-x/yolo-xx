"""Run deterministic offline YOLO prediction on explicitly supplied local images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from .audit import audit_dataset
from .source_manifest import sha256_file
from .train import pick_device

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
DATASET_CONTEXT_FIELDS = (
    "symbol",
    "source_file",
    "source_sha256",
    "window_start_time",
    "window_end_close_time",
    "available_at",
    "image_sha256",
)


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


def load_dataset_contexts(
    manifest_path: str | Path,
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Load schema-v2 sample timing keyed by its exact relative image path."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("dataset manifest must use schema_version 2")
    detection_spec = payload.get("detection_spec")
    if not isinstance(detection_spec, dict) or not detection_spec.get("timeframe"):
        raise ValueError("dataset manifest is missing detection_spec.timeframe")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("dataset manifest samples must be a list")
    contexts: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("dataset manifest sample must be an object")
        relative_image = sample.get("image")
        if not isinstance(relative_image, str) or not relative_image:
            raise ValueError("dataset manifest sample is missing image")
        normalized = Path(relative_image).as_posix()
        if Path(normalized).is_absolute() or ".." in Path(normalized).parts:
            raise ValueError(f"dataset manifest image must be relative: {relative_image}")
        if normalized in contexts:
            raise ValueError(f"duplicate dataset manifest image: {normalized}")
        missing = [field for field in DATASET_CONTEXT_FIELDS if not sample.get(field)]
        if not sample.get("id"):
            missing.append("id")
        if missing:
            raise ValueError(
                f"dataset manifest sample {normalized} is missing: {', '.join(missing)}"
            )
        if sample["available_at"] != sample["window_end_close_time"]:
            raise ValueError(
                f"dataset manifest sample {normalized} violates availability contract"
            )
        contexts[normalized] = {
            "sample_id": sample["id"],
            "symbol": sample["symbol"],
            "source_file": sample["source_file"],
            "source_sha256": sample["source_sha256"],
            "detector_timeframe": detection_spec["timeframe"],
            "window_start_time": sample["window_start_time"],
            "window_end_close_time": sample["window_end_close_time"],
            "available_at": sample["available_at"],
            "image_sha256": sample["image_sha256"],
        }
    return str(detection_spec["timeframe"]), contexts


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
    dataset_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Build the complete machine-readable prediction plan without reading inputs."""
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("conf and iou must be between zero and one")
    if imgsz <= 0 or batch <= 0:
        raise ValueError("imgsz and batch must be positive")
    plan = {
        "schema_version": 2 if dataset_manifest is not None else 1,
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
    if dataset_manifest is not None:
        plan["dataset_manifest"] = str(Path(dataset_manifest).resolve())
    return plan


def run_prediction(plan: dict[str, Any]) -> dict[str, Any]:
    """Execute one offline run and write labels, overlays, and a JSON manifest."""
    weights = Path(plan["weights"])
    source = Path(plan["source"])
    output = ensure_clean_output(plan["output"])
    if not weights.is_file():
        raise FileNotFoundError(f"weights do not exist: {weights}")
    weights_sha256 = sha256_file(weights)
    images = discover_images(source, recursive=bool(plan["recursive"]))
    image_inputs = [(image, source_relative_path(image, source)) for image in images]
    dataset_contexts = None
    detector_timeframe = None
    dataset_manifest_path = plan.get("dataset_manifest")
    dataset_manifest_sha256 = None
    if dataset_manifest_path is not None:
        dataset_manifest_path = Path(dataset_manifest_path).resolve()
        dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
        dataset_root = dataset_manifest_path.parent
        if source.resolve() != dataset_root:
            raise ValueError(
                "prediction source must equal the dataset manifest parent"
            )
        audit = audit_dataset(dataset_root)
        if not audit["valid"]:
            preview = "; ".join(str(item) for item in audit["errors"][:5])
            raise ValueError(f"dataset manifest failed strong audit: {preview}")
        detector_timeframe, dataset_contexts = load_dataset_contexts(dataset_manifest_path)
        missing = [
            relative.as_posix()
            for _, relative in image_inputs
            if relative.as_posix() not in dataset_contexts
        ]
        if missing:
            raise ValueError(
                "prediction images are missing from dataset manifest: "
                + ", ".join(missing[:5])
            )
        for image, relative in image_inputs:
            expected_hash = dataset_contexts[relative.as_posix()]["image_sha256"]
            if sha256_file(image) != expected_hash:
                raise ValueError(f"prediction image SHA-256 mismatch: {relative.as_posix()}")
    output.mkdir(parents=True, exist_ok=True)

    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    items = []
    total_detections = 0
    for image, relative in image_inputs:
        # Ultralytics 8.4.89 rewrites ``result.path`` to ``image0.jpg`` when a
        # Python list of path strings is supplied.  Isolate each input instead:
        # one call must yield exactly one result whose path is the requested
        # file.  This keeps path identity auditable without trusting batch order.
        results = iter(
            model.predict(
                source=str(image),
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
        )
        try:
            result = next(results)
        except StopIteration as error:
            raise RuntimeError(f"model omitted input result: {image}") from error
        try:
            extra_result = next(results)
        except StopIteration:
            extra_result = None
        if extra_result is not None:
            raise RuntimeError(f"model returned multiple results for one input: {image}")
        raw_result_path = getattr(result, "path", None)
        if raw_result_path is None:
            raise RuntimeError("model result is missing its input path")
        result_path = Path(raw_result_path).resolve()
        if result_path != image.resolve():
            raise RuntimeError(
                f"model result path does not match isolated input: "
                f"expected {image.resolve()}, got {result_path}"
            )
        label_path = output / "labels" / artifact_relative_path(relative, ".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        boxes = result.boxes
        xywhn = boxes.xywhn.cpu().numpy().tolist() if boxes is not None else []
        class_ids = boxes.cls.cpu().numpy().tolist() if boxes is not None else []
        confidences = boxes.conf.cpu().numpy().tolist() if boxes is not None else []
        detections = normalize_detections(xywhn, class_ids, confidences, result.names)
        context = dataset_contexts[relative.as_posix()] if dataset_contexts is not None else None
        if context is not None:
            detections = [
                {**detection, "available_at": context["available_at"]}
                for detection in detections
            ]
        label_path.write_text(to_prediction_lines(detections), encoding="utf-8")

        overlay_path = None
        if plan["save_overlays"]:
            import cv2

            overlay_path = output / "overlays" / artifact_relative_path(relative, ".png")
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(overlay_path), result.plot()):
                raise OSError(f"failed to write prediction overlay: {overlay_path}")

        total_detections += len(detections)
        item = {
            "image": str(image),
            "relative_image": relative.as_posix(),
            "weights_sha256": weights_sha256,
            "label": str(label_path.resolve()),
            "overlay": str(overlay_path.resolve()) if overlay_path is not None else None,
            "detections": detections,
        }
        if context is not None:
            item.update(context)
            item["dataset_manifest_sha256"] = dataset_manifest_sha256
        items.append(item)
    items.sort(key=lambda item: str(item["relative_image"]))
    manifest: dict[str, Any] = {
        **plan,
        "weights_sha256": weights_sha256,
        "ultralytics_version": ultralytics.__version__,
        "image_count": len(images),
        "detection_count": total_detections,
        "items": items,
    }
    if dataset_manifest_path is not None:
        manifest["dataset_manifest_sha256"] = dataset_manifest_sha256
        manifest["detector_timeframe"] = detector_timeframe
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
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        help="optional schema-v2 dataset manifest for non-backfilled sample timing",
    )
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
        dataset_manifest=args.dataset_manifest,
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
