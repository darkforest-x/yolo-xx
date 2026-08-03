"""Batch-predict one audited unlabeled scan arm at a fixed offline threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .predict import normalize_detections, to_prediction_lines
from .scan_set import SCAN_MANIFEST, audit_scan_arm, verify_scan_receipt
from .source_manifest import sha256_file
from .train import pick_device


def build_plan(
    *,
    weights: str | Path,
    arm_dir: str | Path,
    out_dir: str | Path,
    conf: float,
    iou: float,
    imgsz: int,
    batch: int,
    device: str,
    overlay_limit: int,
    receipt: str | Path | None = None,
    receipt_sha256: str | None = None,
) -> dict[str, object]:
    """Return an inspectable plan without reading weights, images, or manifests."""
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("conf and iou must be between zero and one")
    if imgsz <= 0 or batch <= 0 or overlay_limit < 0:
        raise ValueError("imgsz/batch must be positive and overlay_limit non-negative")
    if (receipt is None) != (receipt_sha256 is None):
        raise ValueError("receipt and receipt SHA-256 must be supplied together")
    return {
        "schema_version": 1,
        "plan_type": "yolo_xx_batch_scan_prediction",
        "weights": str(Path(weights).resolve()),
        "scan_arm": str(Path(arm_dir).resolve()),
        "output": str(Path(out_dir).resolve()),
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "overlay_limit": overlay_limit,
        "portable_receipt": str(Path(receipt).resolve()) if receipt is not None else None,
        "portable_receipt_sha256": receipt_sha256,
        "holdout_read": False,
    }


def _clean_output(path: str | Path) -> Path:
    output = Path(path)
    if output.exists() and (output.is_file() or any(output.iterdir())):
        raise FileExistsError(f"refusing to mix scan predictions: {output}")
    return output


def run(plan: dict[str, object]) -> dict[str, object]:
    """Run directory-batched prediction and map results by their exact source path."""
    weights = Path(str(plan["weights"]))
    arm = Path(str(plan["scan_arm"]))
    output = _clean_output(str(plan["output"]))
    if not weights.is_file():
        raise FileNotFoundError(f"weights do not exist: {weights}")
    receipt = plan.get("portable_receipt")
    if receipt is None:
        audit = audit_scan_arm(arm)
        if not audit["valid"]:
            raise ValueError("scan arm failed full audit: " + "; ".join(audit["errors"][:5]))
        audit_mode = "full_source_snapshot"
    else:
        verify_scan_receipt(
            arm_dir=arm,
            receipt=str(receipt),
            expected_receipt_sha256=str(plan["portable_receipt_sha256"]),
        )
        audit_mode = "portable_payload_with_mac_full_audit_receipt"
    manifest_path = arm / SCAN_MANIFEST
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[Path, dict[str, Any]] = {}
    for sample in manifest["samples"]:
        image = (arm / sample["image"]).resolve()
        if image in expected:
            raise ValueError(f"duplicate scan image path: {image}")
        expected[image] = sample
    output.mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir()
    if int(plan["overlay_limit"]) > 0:
        (output / "overlays").mkdir()

    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = model.predict(
        source=str((arm / "images").resolve()),
        conf=float(plan["conf"]),
        iou=float(plan["iou"]),
        imgsz=int(plan["imgsz"]),
        batch=int(plan["batch"]),
        device=str(plan["device"]),
        save=False,
        save_txt=False,
        verbose=False,
        stream=True,
    )
    items: list[dict[str, object]] = []
    seen: set[Path] = set()
    overlay_count = 0
    detection_count = 0
    for result in results:
        raw_path = getattr(result, "path", None)
        if raw_path is None:
            raise RuntimeError("scan result is missing its source path")
        image = Path(raw_path).resolve()
        if image not in expected or image in seen:
            raise RuntimeError(f"scan result path is unexpected or duplicated: {image}")
        seen.add(image)
        sample = expected[image]
        boxes = result.boxes
        xywhn = boxes.xywhn.cpu().numpy().tolist() if boxes is not None else []
        class_ids = boxes.cls.cpu().numpy().tolist() if boxes is not None else []
        confidences = boxes.conf.cpu().numpy().tolist() if boxes is not None else []
        detections = normalize_detections(xywhn, class_ids, confidences, result.names)
        detections = [
            {
                **detection,
                "available_at": sample["available_at"],
                "box_right_fraction": round(
                    detection["xywhn"][0] + detection["xywhn"][2] / 2, 8
                ),
            }
            for detection in detections
        ]
        label_path = output / "labels" / f"{sample['id']}.txt"
        label_path.write_text(to_prediction_lines(detections), encoding="utf-8")
        overlay_path = None
        if detections and overlay_count < int(plan["overlay_limit"]):
            import cv2

            overlay_path = output / "overlays" / f"{sample['id']}.jpg"
            if not cv2.imwrite(
                str(overlay_path), result.plot(), [cv2.IMWRITE_JPEG_QUALITY, 90]
            ):
                raise OSError(f"failed to write scan overlay: {overlay_path}")
            overlay_count += 1
        detection_count += len(detections)
        items.append(
            {
                "sample_id": sample["id"],
                "symbol": sample["symbol"],
                "image": str(image),
                "image_sha256": sample["image_sha256"],
                "window_start_time": sample["window_start_time"],
                "window_end_close_time": sample["window_end_close_time"],
                "available_at": sample["available_at"],
                "label": str(label_path.resolve()),
                "overlay": str(overlay_path.resolve()) if overlay_path is not None else None,
                "detections": detections,
            }
        )
    missing = sorted(str(path) for path in set(expected) - seen)
    if missing:
        raise RuntimeError("model omitted scan images: " + ", ".join(missing[:5]))
    items.sort(key=lambda item: str(item["sample_id"]))
    payload: dict[str, object] = {
        **plan,
        "audit_mode": audit_mode,
        "weights_sha256": sha256_file(weights),
        "scan_manifest_sha256": sha256_file(manifest_path),
        "scan_contract_sha256": manifest["scan_contract_sha256"],
        "timeframe": manifest["timeframe"],
        "window_bars": manifest["window_bars"],
        "ma_periods_bars": manifest["ma_periods_bars"],
        "ultralytics_version": ultralytics.__version__,
        "image_count": len(items),
        "images_with_detections": sum(bool(item["detections"]) for item in items),
        "detection_count": detection_count,
        "overlay_count": overlay_count,
        "items": items,
    }
    (output / "predictions.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--scan-arm", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--overlay-limit", type=int, default=48)
    parser.add_argument("--portable-receipt", type=Path)
    parser.add_argument("--portable-receipt-sha256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    device = args.device or ("auto" if args.dry_run else pick_device())
    plan = build_plan(
        weights=args.weights,
        arm_dir=args.scan_arm,
        out_dir=args.out,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        overlay_limit=args.overlay_limit,
        receipt=args.portable_receipt,
        receipt_sha256=args.portable_receipt_sha256,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    payload = run(plan)
    print(
        json.dumps(
            {
                "timeframe": payload["timeframe"],
                "window_bars": payload["window_bars"],
                "image_count": payload["image_count"],
                "images_with_detections": payload["images_with_detections"],
                "detection_count": payload["detection_count"],
                "output": payload["output"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
