"""Run one offline Ultralytics validation pass and write exact JSON metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .train import pick_device


def _metrics_summary(metrics: object) -> dict[str, float]:
    box = getattr(metrics, "box")
    return {
        "map50": round(float(box.map50), 6),
        "map50_95": round(float(box.map), 6),
        "precision": round(float(box.mp), 6),
        "recall": round(float(box.mr), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "out": str(args.out.resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device or ("auto" if args.dry_run else pick_device()),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return
    if not args.weights.is_file() or not args.data.is_file():
        raise FileNotFoundError("weights and dataset YAML must both exist")

    from ultralytics import YOLO

    metrics = YOLO(str(args.weights)).val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=plan["device"],
        plots=False,
    )
    summary = {**plan, "metrics": _metrics_summary(metrics)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
