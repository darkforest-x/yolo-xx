"""Strong schema-v2 audit for immutable YOLO dataset artifacts and timing."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .source_manifest import (
    HOLDOUT_START,
    SnapshotFile,
    load_source_manifest,
    sha256_file,
    utc_timestamp,
    verify_snapshot_identity,
)
from .specs import timeframe_minutes


def _safe_dataset_path(root: Path, raw: object, *, field: str, errors: list[str]) -> Path | None:
    if not isinstance(raw, str) or not raw:
        errors.append(f"{field} must be a non-empty relative path")
        return None
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        errors.append(f"{field} must stay below the dataset root: {raw}")
        return None
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        errors.append(f"{field} escapes the dataset root: {raw}")
        return None
    return resolved


def _timestamp(raw: object, *, field: str, errors: list[str]) -> pd.Timestamp | None:
    try:
        return utc_timestamp(raw, field=field)
    except ValueError as error:
        errors.append(str(error))
        return None


def _label_rows(path: Path, *, errors: list[str]) -> list[tuple[int, list[float]]]:
    rows: list[tuple[int, list[float]]] = []
    seen: set[tuple[int, tuple[float, ...]]] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        errors.append(f"cannot read label {path}: {error}")
        return rows
    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric label")
            continue
        xc, yc, width, height = values
        if class_id != 0:
            errors.append(f"{path}:{line_number}: class id must be 0")
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{path}:{line_number}: values must be finite")
        elif not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{path}:{line_number}: normalized box is outside [0, 1]")
        identity = (class_id, tuple(values))
        if identity in seen:
            errors.append(f"{path}:{line_number}: duplicate YOLO label row")
        seen.add(identity)
        rows.append((class_id, values))
    return rows


def _exact_digest(path: Path, expected: object, *, field: str, errors: list[str]) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        errors.append(f"{field} is missing a SHA-256 digest")
        return
    try:
        actual = sha256_file(path)
    except OSError as error:
        errors.append(f"cannot hash {path}: {error}")
        return
    if actual != expected:
        errors.append(f"{field} SHA-256 mismatch: {path}")


def _load_manifest(root: Path, errors: list[str]) -> dict[str, Any] | None:
    path = root / "dataset_manifest.json"
    if not path.is_file():
        errors.append("missing dataset_manifest.json")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid dataset_manifest.json: {error}")
        return None
    if not isinstance(payload, dict):
        errors.append("dataset manifest root must be an object")
        return None
    if payload.get("schema_version") != 2:
        errors.append("dataset manifest schema_version must be 2")
    if payload.get("manifest_type") != "yolo_xx_dataset":
        errors.append("dataset manifest_type must be yolo_xx_dataset")
    return payload


def audit_dataset(dataset: str | Path) -> dict[str, object]:
    """Return a fail-closed schema-v2 audit suitable as a train/eval gate."""
    root = Path(dataset).resolve()
    errors: list[str] = []
    splits: dict[str, dict[str, int]] = {}
    if not (root / "data.yaml").is_file():
        errors.append("missing data.yaml")
    manifest = _load_manifest(root, errors)
    if manifest is None:
        return {
            "schema_version": 2,
            "audit_type": "yolo_xx_dataset_audit",
            "dataset": str(root),
            "valid": False,
            "splits": splits,
            "errors": errors,
        }

    detection_spec = manifest.get("detection_spec")
    timeframe = detection_spec.get("timeframe") if isinstance(detection_spec, dict) else None
    if not isinstance(timeframe, str) or not timeframe:
        errors.append("dataset manifest is missing detection_spec.timeframe")
    raw_window_bars = manifest.get("window_bars")
    if (
        isinstance(raw_window_bars, bool)
        or not isinstance(raw_window_bars, int)
        or raw_window_bars <= 0
    ):
        errors.append("dataset manifest window_bars must be a positive integer")
        window_bars = 0
    else:
        window_bars = raw_window_bars
    end_before = _timestamp(manifest.get("end_before"), field="end_before", errors=errors)
    split_at = _timestamp(manifest.get("split_at"), field="split_at", errors=errors)
    if end_before is not None and end_before > HOLDOUT_START:
        errors.append("end_before is later than holdout start")
    if split_at is not None and split_at > HOLDOUT_START:
        errors.append("split_at is later than holdout start")

    source_records: dict[str, SnapshotFile] = {}
    snapshot_info = manifest.get("source_snapshot")
    if not isinstance(snapshot_info, dict):
        errors.append("dataset manifest is missing source_snapshot")
    else:
        snapshot_path = _safe_dataset_path(
            root,
            snapshot_info.get("manifest"),
            field="source_snapshot.manifest",
            errors=errors,
        )
        if snapshot_path is not None:
            if not snapshot_path.is_file():
                errors.append(f"missing source snapshot manifest: {snapshot_path}")
            else:
                _exact_digest(
                    snapshot_path,
                    snapshot_info.get("sha256"),
                    field="source_snapshot.manifest",
                    errors=errors,
                )
                try:
                    snapshot = load_source_manifest(
                        snapshot_path,
                        expected_source_dir=manifest.get("source_dir"),
                        expected_timeframe=timeframe if isinstance(timeframe, str) else None,
                        end_before=manifest.get("end_before"),
                    )
                    verify_snapshot_identity(snapshot)
                    source_records = {str(record.path): record for record in snapshot.files}
                    if snapshot.manifest_sha256 != snapshot_info.get("sha256"):
                        errors.append("source snapshot parsed hash differs from dataset manifest")
                except (FileNotFoundError, ValueError, OSError) as error:
                    errors.append(f"source snapshot invalid: {error}")

    declared_sources = manifest.get("source_files")
    if not isinstance(declared_sources, list) or not declared_sources:
        errors.append("dataset manifest source_files must be a non-empty list")
    else:
        declared_pairs: dict[str, str] = {}
        for index, item in enumerate(declared_sources):
            if not isinstance(item, dict):
                errors.append(f"source_files[{index}] must be an object")
                continue
            path = item.get("path")
            digest = item.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                errors.append(f"source_files[{index}] is missing path/sha256")
                continue
            if path in declared_pairs:
                errors.append(f"duplicate source_files path: {path}")
            declared_pairs[path] = digest
            if source_records and (
                path not in source_records or source_records[path].sha256 != digest
            ):
                errors.append(f"source_files[{index}] differs from source snapshot: {path}")

    if manifest.get("strict_cadence") is not True:
        errors.append("strict_cadence must be true for train/eval eligibility")

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("dataset manifest samples must be a non-empty list")
        samples = []
    ids: set[str] = set()
    images: set[str] = set()
    labels: set[str] = set()
    train_available: list[pd.Timestamp] = []
    val_starts: list[pd.Timestamp] = []
    split_counts: CounterLike = {"train": [0, 0, 0], "val": [0, 0, 0]}

    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{prefix} must be an object")
            continue
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            errors.append(f"{prefix}.id must be non-empty")
        elif sample_id in ids:
            errors.append(f"duplicate sample id: {sample_id}")
        else:
            ids.add(sample_id)
        split = sample.get("split")
        if split not in ("train", "val"):
            errors.append(f"{prefix}.split must be train or val")
            split = "invalid"

        image_raw = sample.get("image")
        label_raw = sample.get("label")
        image_path = _safe_dataset_path(root, image_raw, field=f"{prefix}.image", errors=errors)
        label_path = _safe_dataset_path(root, label_raw, field=f"{prefix}.label", errors=errors)
        if isinstance(image_raw, str):
            if image_raw in images:
                errors.append(f"duplicate sample image: {image_raw}")
            images.add(image_raw)
            if split in ("train", "val") and not image_raw.startswith(f"images/{split}/"):
                errors.append(f"{prefix}.image path disagrees with split")
        if isinstance(label_raw, str):
            if label_raw in labels:
                errors.append(f"duplicate sample label: {label_raw}")
            labels.add(label_raw)
            if split in ("train", "val") and not label_raw.startswith(f"labels/{split}/"):
                errors.append(f"{prefix}.label path disagrees with split")
        if (
            isinstance(sample_id, str)
            and isinstance(image_raw, str)
            and isinstance(label_raw, str)
            and (
                Path(image_raw).stem != sample_id
                or Path(label_raw).stem != sample_id
                or Path(image_raw).stem != Path(label_raw).stem
            )
        ):
            errors.append(f"{prefix}: id/image/label stems do not correspond")
        label_rows: list[tuple[int, list[float]]] = []
        if image_path is None or not image_path.is_file():
            errors.append(f"{prefix}: image does not exist")
        else:
            _exact_digest(
                image_path, sample.get("image_sha256"), field=f"{prefix}.image", errors=errors
            )
        if label_path is None or not label_path.is_file():
            errors.append(f"{prefix}: label does not exist")
        else:
            _exact_digest(
                label_path, sample.get("label_sha256"), field=f"{prefix}.label", errors=errors
            )
            label_rows = _label_rows(label_path, errors=errors)

        source_file = sample.get("source_file")
        source_sha = sample.get("source_sha256")
        source_record = source_records.get(source_file) if isinstance(source_file, str) else None
        if source_record is None or source_record.sha256 != source_sha:
            errors.append(f"{prefix}: source_file/source_sha256 is not authenticated")

        window_start = _timestamp(
            sample.get("window_start_time"), field=f"{prefix}.window_start_time", errors=errors
        )
        window_end_open = _timestamp(
            sample.get("window_end_open_time"),
            field=f"{prefix}.window_end_open_time",
            errors=errors,
        )
        window_end_close = _timestamp(
            sample.get("window_end_close_time"),
            field=f"{prefix}.window_end_close_time",
            errors=errors,
        )
        available = _timestamp(
            sample.get("available_at"), field=f"{prefix}.available_at", errors=errors
        )
        start_index = sample.get("source_start_index")
        end_index = sample.get("source_end_index")
        if all(value is not None for value in (window_start, window_end_open, window_end_close, available)):
            assert window_start is not None
            assert window_end_open is not None
            assert window_end_close is not None
            assert available is not None
            if not window_start <= window_end_open < window_end_close:
                errors.append(f"{prefix}: window UTC times are out of order")
            if available != window_end_close:
                errors.append(f"{prefix}: available_at must equal window_end_close_time")
            if end_before is not None and available > end_before:
                errors.append(f"{prefix}: available_at exceeds end_before")
            if (
                source_record is None
                or isinstance(start_index, bool)
                or not isinstance(start_index, int)
                or isinstance(end_index, bool)
                or not isinstance(end_index, int)
                or not 0 <= start_index <= end_index < source_record.row_count
            ):
                errors.append(f"{prefix}: source indices are invalid")
            elif isinstance(timeframe, str):
                cadence = pd.Timedelta(minutes=timeframe_minutes(timeframe))
                expected_start = source_record.first_open_time + start_index * cadence
                expected_end_open = source_record.first_open_time + end_index * cadence
                expected_available = expected_end_open + cadence
                if window_start != expected_start:
                    errors.append(f"{prefix}: window_start_time disagrees with source index")
                if window_end_open != expected_end_open:
                    errors.append(f"{prefix}: window_end_open_time disagrees with source index")
                if window_end_close != expected_available:
                    errors.append(f"{prefix}: window_end_close_time disagrees with source index")
                if end_index - start_index + 1 != window_bars:
                    errors.append(f"{prefix}: source index span disagrees with window_bars")
            if split_at is not None:
                if split == "train":
                    train_available.append(available)
                    if not available < split_at:
                        errors.append(f"{prefix}: train available_at is not before split_at")
                elif split == "val":
                    val_starts.append(window_start)
                    if not window_start >= split_at:
                        errors.append(f"{prefix}: val window_start_time is before split_at")

        boxes = sample.get("boxes")
        if not isinstance(boxes, list):
            errors.append(f"{prefix}.boxes must be a list")
            boxes = []
        if sample.get("n_boxes") != len(boxes):
            errors.append(f"{prefix}.n_boxes does not match boxes")
        if len(label_rows) != len(boxes):
            errors.append(f"{prefix}: label row count does not match manifest boxes")
        for box_index, box in enumerate(boxes):
            box_prefix = f"{prefix}.boxes[{box_index}]"
            if not isinstance(box, dict):
                errors.append(f"{box_prefix} must be an object")
                continue
            if box.get("class_id") != 0 or box.get("class_name") != "dense_cluster":
                errors.append(f"{box_prefix}: class contract is invalid")
            box_start = _timestamp(
                box.get("box_start_time"), field=f"{box_prefix}.box_start_time", errors=errors
            )
            box_end = _timestamp(
                box.get("box_end_time"), field=f"{box_prefix}.box_end_time", errors=errors
            )
            box_end_close = _timestamp(
                box.get("box_end_close_time"),
                field=f"{box_prefix}.box_end_close_time",
                errors=errors,
            )
            box_available = _timestamp(
                box.get("available_at"), field=f"{box_prefix}.available_at", errors=errors
            )
            segment = box.get("segment")
            segment_indices: tuple[int, int, int, int] | None = None
            if not isinstance(segment, dict):
                errors.append(f"{box_prefix}.segment must be an object")
            else:
                raw_indices = (
                    segment.get("start_index_in_window"),
                    segment.get("end_index_in_window"),
                    segment.get("start_index_in_source"),
                    segment.get("end_index_in_source"),
                )
                if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_indices):
                    errors.append(f"{box_prefix}.segment indices must be integers")
                else:
                    segment_indices = raw_indices  # type: ignore[assignment]
            if all(value is not None for value in (box_start, box_end, box_end_close, box_available, available)):
                assert box_start is not None
                assert box_end is not None
                assert box_end_close is not None
                assert box_available is not None
                assert available is not None
                if not box_start <= box_end < box_end_close <= available:
                    errors.append(f"{box_prefix}: UTC times are out of causal order")
                if box_available != available:
                    errors.append(f"{box_prefix}: availability differs from sample")
                if (
                    segment_indices is not None
                    and isinstance(start_index, int)
                    and not isinstance(start_index, bool)
                    and isinstance(timeframe, str)
                    and window_start is not None
                ):
                    in_start, in_end, source_start, source_end = segment_indices
                    if not 0 <= in_start <= in_end < window_bars:
                        errors.append(f"{box_prefix}: window segment indices are invalid")
                    if source_start != start_index + in_start or source_end != start_index + in_end:
                        errors.append(f"{box_prefix}: source segment indices are inconsistent")
                    cadence = pd.Timedelta(minutes=timeframe_minutes(timeframe))
                    if box_start != window_start + in_start * cadence:
                        errors.append(f"{box_prefix}: box_start_time disagrees with segment")
                    if box_end != window_start + in_end * cadence:
                        errors.append(f"{box_prefix}: box_end_time disagrees with segment")
                    if box_end_close != box_end + cadence:
                        errors.append(f"{box_prefix}: box_end_close_time disagrees with timeframe")
            xywhn = box.get("xywhn")
            if not isinstance(xywhn, list) or len(xywhn) != 4:
                errors.append(f"{box_prefix}.xywhn must contain four values")
            elif box_index < len(label_rows):
                class_id, label_box = label_rows[box_index]
                try:
                    manifest_box = [float(value) for value in xywhn]
                except (TypeError, ValueError):
                    errors.append(f"{box_prefix}.xywhn must be numeric")
                else:
                    if class_id != box.get("class_id") or any(
                        not math.isclose(left, right, rel_tol=0.0, abs_tol=5e-7)
                        for left, right in zip(manifest_box, label_box)
                    ):
                        errors.append(f"{box_prefix}: xywhn differs from label row")

        if split in ("train", "val"):
            split_counts[split][0] += 1
            split_counts[split][1] += len(boxes)
            split_counts[split][2] += int(not boxes)

    actual_images = {
        path.relative_to(root).as_posix()
        for split in ("train", "val")
        for path in (root / "images" / split).glob("*.png")
        if path.is_file()
    }
    actual_labels = {
        path.relative_to(root).as_posix()
        for split in ("train", "val")
        for path in (root / "labels" / split).glob("*.txt")
        if path.is_file()
    }
    for missing in sorted(images - actual_images):
        errors.append(f"manifest image missing on disk: {missing}")
    for orphan in sorted(actual_images - images):
        errors.append(f"unmanifested image on disk: {orphan}")
    for missing in sorted(labels - actual_labels):
        errors.append(f"manifest label missing on disk: {missing}")
    for orphan in sorted(actual_labels - labels):
        errors.append(f"unmanifested label on disk: {orphan}")

    if train_available and val_starts:
        max_train = max(train_available)
        min_val = min(val_starts)
        if not max_train < min_val:
            errors.append("global split invariant failed: train availability overlaps val window")
        declared_invariant = manifest.get("global_split_invariant")
        if not isinstance(declared_invariant, dict) or declared_invariant.get("holds") is not True:
            errors.append("global_split_invariant declaration is missing or false")
        else:
            declared_train = _timestamp(
                declared_invariant.get("max_train_available_at"),
                field="global_split_invariant.max_train_available_at",
                errors=errors,
            )
            declared_val = _timestamp(
                declared_invariant.get("min_val_window_start_time"),
                field="global_split_invariant.min_val_window_start_time",
                errors=errors,
            )
            if declared_train != max_train or declared_val != min_val:
                errors.append("global_split_invariant extrema do not match samples")
    else:
        errors.append("both train and val samples are required for global split audit")

    for split in ("train", "val"):
        counts = split_counts[split]
        splits[split] = {
            "images": counts[0],
            "labels": counts[0],
            "boxes": counts[1],
            "background_images": counts[2],
        }
    return {
        "schema_version": 2,
        "audit_type": "yolo_xx_dataset_audit",
        "dataset": str(root),
        "valid": not errors,
        "splits": splits,
        "errors": errors,
    }


# Simple type alias kept local so the audit module remains dependency-light.
CounterLike = dict[str, list[int]]


def require_valid_dataset(data_yaml: str | Path) -> dict[str, object]:
    """Raise before model import when a generated dataset fails strong audit."""
    data = Path(data_yaml).resolve()
    if not data.is_file():
        raise FileNotFoundError(f"dataset YAML does not exist: {data}")
    summary = audit_dataset(data.parent)
    if not summary["valid"]:
        preview = "; ".join(str(item) for item in summary["errors"][:5])
        raise ValueError(f"dataset failed schema-v2/pre-holdout audit: {preview}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summary = audit_dataset(args.dataset)
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
