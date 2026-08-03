"""Audit YOLO image/label pairing and normalized single-class label syntax."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _validate_label(path: Path) -> list[str]:
    errors = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields")
            continue
        try:
            class_id = int(parts[0])
            xc, yc, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"{path}:{line_number}: non-numeric label")
            continue
        values = (xc, yc, width, height)
        if class_id != 0:
            errors.append(f"{path}:{line_number}: class id must be 0")
        if not all(math.isfinite(value) for value in values):
            errors.append(f"{path}:{line_number}: values must be finite")
        elif not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"{path}:{line_number}: normalized box is outside [0, 1]")
    return errors


def audit_dataset(dataset: str | Path) -> dict[str, object]:
    """Return a stable, machine-readable dataset audit for train and val."""
    root = Path(dataset)
    errors = []
    splits: dict[str, dict[str, int]] = {}
    if not (root / "data.yaml").is_file():
        errors.append("missing data.yaml")
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"missing split directories: {split}")
            continue
        images = {path.stem for path in image_dir.glob("*.png")}
        labels = {path.stem for path in label_dir.glob("*.txt")}
        missing_labels = sorted(images - labels)
        orphan_labels = sorted(labels - images)
        if not images:
            errors.append(f"{split}: split has no images")
        errors.extend(f"{split}: image without label: {stem}" for stem in missing_labels)
        errors.extend(f"{split}: label without image: {stem}" for stem in orphan_labels)
        box_count = 0
        background_count = 0
        for stem in sorted(images & labels):
            label_path = label_dir / f"{stem}.txt"
            lines = label_path.read_text(encoding="utf-8").splitlines()
            box_count += len(lines)
            background_count += int(not lines)
            errors.extend(_validate_label(label_path))
        splits[split] = {
            "images": len(images),
            "labels": len(labels),
            "boxes": box_count,
            "background_images": background_count,
        }
    return {
        "schema_version": 1,
        "dataset": str(root.resolve()),
        "valid": not errors,
        "splits": splits,
        "errors": errors,
    }


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
