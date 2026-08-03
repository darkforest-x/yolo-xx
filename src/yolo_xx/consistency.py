"""Compare YOLO prediction labels with ground truth using greedy IoU matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

Box = tuple[float, float, float, float]


def load_boxes(path: Path) -> list[Box]:
    """Load normalized `(xc, yc, width, height)` values, ignoring confidence."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            boxes.append(tuple(map(float, parts[1:5])))
    return boxes


def _xyxy(box: Box) -> Box:
    xc, yc, width, height = box
    return xc - width / 2, yc - height / 2, xc + width / 2, yc + height / 2


def iou(left: Box, right: Box) -> float:
    """Calculate intersection-over-union for two normalized YOLO boxes."""
    ax1, ay1, ax2, ay2 = _xyxy(left)
    bx1, by1, bx2, by2 = _xyxy(right)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    area_left = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_right = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def match_greedy(gt: list[Box], pred: list[Box], *, iou_threshold: float = 0.5) -> int:
    """Return the number of one-to-one prediction matches."""
    used: set[int] = set()
    matched = 0
    for gt_box in gt:
        candidates = [
            (iou(gt_box, pred_box), index)
            for index, pred_box in enumerate(pred)
            if index not in used
        ]
        best_iou, best_index = max(candidates, default=(0.0, -1))
        if best_index >= 0 and best_iou >= iou_threshold:
            used.add(best_index)
            matched += 1
    return matched


def compare(
    dataset: str | Path,
    predictions: str | Path,
    *,
    split: str = "val",
    iou_threshold: float = 0.5,
) -> dict[str, object]:
    """Compare flat prediction text files with one dataset split."""
    root = Path(dataset)
    prediction_dir = Path(predictions)
    image_stems = sorted(path.stem for path in (root / "images" / split).glob("*.png"))
    matched = gt_count = prediction_count = 0
    for stem in image_stems:
        gt = load_boxes(root / "labels" / split / f"{stem}.txt")
        pred = load_boxes(prediction_dir / f"{stem}.txt")
        matched += match_greedy(gt, pred, iou_threshold=iou_threshold)
        gt_count += len(gt)
        prediction_count += len(pred)
    return {
        "schema_version": 1,
        "dataset": str(root.resolve()),
        "predictions": str(prediction_dir.resolve()),
        "split": split,
        "iou_threshold": iou_threshold,
        "images": len(image_stems),
        "gt_boxes": gt_count,
        "prediction_boxes": prediction_count,
        "matched": matched,
        "recall": round(matched / gt_count, 6) if gt_count else None,
        "precision": round(matched / prediction_count, 6) if prediction_count else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = compare(
        args.dataset,
        args.predictions,
        split=args.split,
        iou_threshold=args.iou,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
