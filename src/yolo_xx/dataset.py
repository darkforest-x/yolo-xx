"""Build a temporally split YOLO dataset from explicitly supplied local CSVs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .data import add_mas, cache_symbol, list_cache_files, load_ohlcv_csv
from .labels import CLASS_NAME, find_dense_segments, label_window, to_yolo_lines
from .render import render_chart

WINDOW_START_MIN = 360
DEFAULT_END_BEFORE = "2026-05-04T00:00:00Z"


@dataclass(frozen=True)
class WindowSpec:
    symbol: str
    cache_path: Path
    start: int
    n_boxes: int
    split: str


def _scan_symbol(
    path: Path,
    *,
    window: int,
    stride: int,
    train_frac: float,
    end_before: str,
) -> list[WindowSpec]:
    frame = add_mas(load_ohlcv_csv(path, end_before=end_before))
    count = len(frame)
    if count < WINDOW_START_MIN + window:
        return []
    cutoff = int(count * train_frac)
    specs = []
    for start in range(WINDOW_START_MIN, count - window + 1, stride):
        end = start + window
        if end <= cutoff:
            split = "train"
        elif start >= cutoff:
            split = "val"
        else:
            continue
        subframe = frame.iloc[start:end].reset_index(drop=True)
        specs.append(
            WindowSpec(
                symbol=cache_symbol(path),
                cache_path=path,
                start=start,
                n_boxes=len(find_dense_segments(subframe)),
                split=split,
            )
        )
    return specs


def _sample_split(
    pool: list[WindowSpec],
    *,
    cap: int,
    target_bg_frac: float,
    rng: random.Random,
) -> list[WindowSpec]:
    background = [item for item in pool if item.n_boxes == 0]
    positive = [item for item in pool if item.n_boxes > 0]
    rng.shuffle(background)
    rng.shuffle(positive)
    chosen = background[: min(len(background), round(cap * target_bg_frac))]
    chosen.extend(positive[: min(len(positive), cap - len(chosen))])
    if len(chosen) < cap:
        selected = set(chosen)
        remainder = [item for item in background + positive if item not in selected]
        rng.shuffle(remainder)
        chosen.extend(remainder[: cap - len(chosen)])
    return chosen


def _balance(
    specs: list[WindowSpec],
    *,
    target_bg_frac: float,
    max_images: int,
    train_frac: float,
    rng: random.Random,
) -> list[WindowSpec]:
    train_cap = round(max_images * train_frac)
    caps = {"train": train_cap, "val": max_images - train_cap}
    chosen = []
    for split in ("train", "val"):
        chosen.extend(
            _sample_split(
                [item for item in specs if item.split == split],
                cap=caps[split],
                target_bg_frac=target_bg_frac,
                rng=rng,
            )
        )
    return chosen


def _ensure_clean_output(output: Path) -> None:
    for relative in ("images/train", "images/val", "labels/train", "labels/val"):
        target = output / relative
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"refusing to mix datasets; output is not empty: {target}")


def build(
    out_dir: Path,
    *,
    cache_dir: Path,
    window: int,
    stride: int,
    train_frac: float,
    target_bg_frac: float,
    max_images: int,
    seed: int,
    end_before: str,
    symbol_contains: str = "",
    min_rows: int = 10_000,
) -> dict[str, object]:
    """Build the dataset and return the exact machine-readable summary."""
    if window <= 0 or stride <= 0 or max_images <= 0:
        raise ValueError("window, stride, and max_images must be positive")
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between zero and one")
    if not 0 <= target_bg_frac <= 1:
        raise ValueError("target_bg_frac must be between zero and one")
    _ensure_clean_output(out_dir)

    rng = random.Random(seed)
    cache_files = list_cache_files(cache_dir, min_rows=min_rows)
    specs = []
    for path in cache_files:
        if symbol_contains and symbol_contains not in path.name:
            continue
        specs.extend(
            _scan_symbol(
                path,
                window=window,
                stride=stride,
                train_frac=train_frac,
                end_before=end_before,
            )
        )
    chosen = _balance(
        specs,
        target_bg_frac=target_bg_frac,
        max_images=max_images,
        train_frac=train_frac,
        rng=rng,
    )
    chosen_splits = Counter(item.split for item in chosen)
    missing_splits = [split for split in ("train", "val") if chosen_splits[split] == 0]
    if missing_splits:
        raise ValueError(
            "dataset build produced no windows for split(s): "
            f"{', '.join(missing_splits)}; check input length, window, stride, and train_frac"
        )

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    box_counts: list[int] = []
    box_dimensions: list[tuple[float, float]] = []
    grouped: dict[Path, list[WindowSpec]] = {}
    for spec in chosen:
        grouped.setdefault(spec.cache_path, []).append(spec)
    for path, file_specs in grouped.items():
        frame = add_mas(load_ohlcv_csv(path, end_before=end_before))
        for spec in file_specs:
            subframe = frame.iloc[spec.start : spec.start + window].reset_index(drop=True)
            name = f"{spec.symbol}_{spec.start:06d}"
            image_path = out_dir / "images" / spec.split / f"{name}.png"
            _, transform = render_chart(subframe, out_path=image_path)
            boxes = label_window(subframe, transform)
            label_path = out_dir / "labels" / spec.split / f"{name}.txt"
            label_path.write_text(to_yolo_lines(boxes), encoding="utf-8")
            stats[f"{spec.split}_images"] += 1
            stats[f"{spec.split}_boxes"] += len(boxes)
            if not boxes:
                stats[f"{spec.split}_background"] += 1
            box_counts.append(len(boxes))
            box_dimensions.extend((width, height) for _, _, width, height in boxes)

    data_yaml = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    summary: dict[str, object] = {
        "schema_version": 1,
        "source_dir": str(cache_dir.resolve()),
        "source_files": [str(path.resolve()) for path in sorted(grouped)],
        "symbols": len(grouped),
        "end_before": end_before,
        "window": window,
        "stride": stride,
        "train_frac": train_frac,
        "seed": seed,
        **{key: int(value) for key, value in sorted(stats.items())},
        "boxes_per_image_mean": round(sum(box_counts) / max(len(box_counts), 1), 3),
        "box_w_mean": round(
            sum(width for width, _ in box_dimensions) / max(len(box_dimensions), 1), 4
        ),
        "box_h_mean": round(
            sum(height for _, height in box_dimensions) / max(len(box_dimensions), 1), 4
        ),
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--target-bg-frac", type=float, default=0.35)
    parser.add_argument("--max-images", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--end-before", default=DEFAULT_END_BEFORE)
    parser.add_argument("--min-rows", type=int, default=10_000)
    parser.add_argument("--symbol-contains", default="")
    args = parser.parse_args()
    summary = build(
        args.out,
        cache_dir=args.cache_dir,
        window=args.window,
        stride=args.stride,
        train_frac=args.train_frac,
        target_bg_frac=args.target_bg_frac,
        max_images=args.max_images,
        seed=args.seed,
        end_before=args.end_before,
        symbol_contains=args.symbol_contains,
        min_rows=args.min_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
