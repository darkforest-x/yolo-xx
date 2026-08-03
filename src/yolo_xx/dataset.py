"""Build a globally time-split YOLO dataset from an authenticated local snapshot."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import pandas as pd

from .data import add_mas, cache_symbol, load_ohlcv_csv
from .labels import CLASS_ID, CLASS_NAME, find_dense_segments, label_segments, to_yolo_lines
from .render import IMG_WIDTH, MARGIN, render_chart
from .source_manifest import (
    HOLDOUT_START,
    SnapshotFile,
    enforce_preholdout,
    load_source_manifest,
    sha256_file,
    utc_iso,
    utc_timestamp,
    verify_loaded_frame,
    verify_snapshot_file,
    verify_snapshot_identity,
)
from .specs import (
    DEFAULT_DENSE_MAX_MINUTES,
    DEFAULT_DENSE_MIN_MINUTES,
    DEFAULT_MA_MINUTES,
    DEFAULT_MERGE_GAP_MINUTES,
    DEFAULT_PHYSICAL_WINDOW_MINUTES,
    DEFAULT_TIMEFRAME,
    DetectionSpec,
    parse_minute_list,
    resolve_window_bars,
)

DEFAULT_END_BEFORE = "2026-05-04T00:00:00Z"


@dataclass(frozen=True)
class WindowSpec:
    """One causal chart window before or after global split assignment."""

    symbol: str
    source: SnapshotFile
    start: int
    n_boxes: int
    window_start_time: pd.Timestamp
    available_at: pd.Timestamp
    split: str = ""


def _read_authenticated_source(
    record: SnapshotFile,
    *,
    detection_spec: DetectionSpec,
    strict_cadence: bool,
) -> pd.DataFrame:
    """Parse one already-authenticated CSV and recheck identity after parsing."""
    verify_snapshot_file(record)
    frame = load_ohlcv_csv(
        record.path,
        end_before=None,
        timeframe=detection_spec.timeframe,
        strict_cadence=strict_cadence,
    )
    verify_loaded_frame(frame, record, timeframe=detection_spec.timeframe)
    verify_snapshot_file(record)
    return add_mas(frame, periods=detection_spec.ma_periods)


def _scan_symbol(
    record: SnapshotFile,
    *,
    window: int,
    stride: int,
    detection_spec: DetectionSpec,
    strict_cadence: bool,
) -> list[WindowSpec]:
    frame = _read_authenticated_source(
        record,
        detection_spec=detection_spec,
        strict_cadence=strict_cadence,
    )
    count = len(frame)
    if count < detection_spec.warmup_bars + window:
        return []
    bar_delta = pd.Timedelta(minutes=detection_spec.bar_minutes)
    specs: list[WindowSpec] = []
    for start in range(detection_spec.warmup_bars, count - window + 1, stride):
        end = start + window
        subframe = frame.iloc[start:end].reset_index(drop=True)
        specs.append(
            WindowSpec(
                symbol=cache_symbol(record.path, timeframe=detection_spec.timeframe),
                source=record,
                start=start,
                n_boxes=len(
                    find_dense_segments(
                        subframe,
                        min_bars=detection_spec.dense_min_bars,
                        merge_gap=detection_spec.merge_gap_bars,
                        max_bars=detection_spec.dense_max_bars,
                    )
                ),
                window_start_time=pd.Timestamp(subframe.iloc[0]["open_time"]),
                available_at=pd.Timestamp(subframe.iloc[-1]["open_time"]) + bar_delta,
            )
        )
    return specs


def _resolve_global_split_at(
    specs: Sequence[WindowSpec],
    *,
    train_frac: float,
    explicit_split_at: object | None,
) -> pd.Timestamp:
    if explicit_split_at is not None:
        return enforce_preholdout(explicit_split_at, field="split_at")
    if not specs:
        raise ValueError("cannot calculate split_at without candidate windows")
    earliest = min(spec.window_start_time for spec in specs)
    latest = max(spec.available_at for spec in specs)
    if earliest >= latest:
        raise ValueError("candidate window time extent is empty")
    target_ns = int(earliest.value + (latest.value - earliest.value) * train_frac)
    target = pd.Timestamp(target_ns, tz="UTC")
    # Snap the computed target to one globally feasible window boundary.  A raw
    # fractional timestamp can leave only cross-boundary windows when stride is
    # equal to the window width.  This is still one global UTC split, never a
    # per-symbol/count split.
    feasible = [
        boundary
        for boundary in sorted({spec.window_start_time for spec in specs})
        if any(spec.available_at < boundary for spec in specs)
        and any(spec.window_start_time >= boundary for spec in specs)
    ]
    if not feasible:
        raise ValueError("no global split_at can produce causally separated train and val windows")
    calculated = min(feasible, key=lambda boundary: abs(boundary.value - target.value))
    if calculated > HOLDOUT_START:
        raise ValueError("calculated split_at is later than holdout start")
    return calculated


def _assign_global_splits(
    specs: Sequence[WindowSpec],
    *,
    split_at: pd.Timestamp,
) -> tuple[list[WindowSpec], int]:
    assigned: list[WindowSpec] = []
    dropped = 0
    for spec in specs:
        if spec.available_at < split_at:
            assigned.append(replace(spec, split="train"))
        elif spec.window_start_time >= split_at:
            assigned.append(replace(spec, split="val"))
        else:
            dropped += 1
    return assigned, dropped


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
    chosen: list[WindowSpec] = []
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
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to mix datasets; output is not empty: {output}")


def _physical_window_plan(
    detection_spec: DetectionSpec,
    *,
    window: int | None,
    stride: int | None,
) -> dict[str, object]:
    resolved_window = resolve_window_bars(detection_spec.timeframe, window)
    resolved_stride = resolved_window if stride is None else int(stride)
    if resolved_stride <= 0:
        raise ValueError("stride bars must be positive")
    pixels_per_bar = (IMG_WIDTH - 2 * MARGIN) / max(resolved_window - 1, 1)
    risks: list[str] = []
    if pixels_per_bar < 1:
        risks.append(
            "subpixel_horizontal_resolution: multiple bars map to one x pixel; "
            "treat 1m/2m as later-stage experiments"
        )
    return {
        "physical_window_minutes": DEFAULT_PHYSICAL_WINDOW_MINUTES,
        "window_bars": resolved_window,
        "window_was_explicit": window is not None,
        "stride_bars": resolved_stride,
        "stride_was_explicit": stride is not None,
        "render_plot_width_pixels": IMG_WIDTH - 2 * MARGIN,
        "pixels_per_bar": round(pixels_per_bar, 6),
        "resolution_risks": risks,
    }


def make_build_plan(
    out_dir: Path,
    *,
    cache_dir: Path,
    window: int | None,
    stride: int | None,
    train_frac: float,
    target_bg_frac: float,
    max_images: int,
    seed: int,
    end_before: str,
    source_manifest: Path | None = None,
    split_at: str | None = None,
    symbol_contains: str = "",
    min_rows: int = 10_000,
    timeframe: str = DEFAULT_TIMEFRAME,
    ma_minutes: Sequence[int] = DEFAULT_MA_MINUTES,
    dense_min_minutes: int = DEFAULT_DENSE_MIN_MINUTES,
    dense_max_minutes: int = DEFAULT_DENSE_MAX_MINUTES,
    merge_gap_minutes: int = DEFAULT_MERGE_GAP_MINUTES,
    strict_cadence: bool = True,
) -> dict[str, object]:
    """Return a validated plan without reading a source manifest or any CSV."""
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between zero and one")
    if not 0 <= target_bg_frac <= 1:
        raise ValueError("target_bg_frac must be between zero and one")
    enforce_preholdout(end_before, field="end_before")
    if split_at is not None:
        enforce_preholdout(split_at, field="split_at")
    detection_spec = DetectionSpec(
        timeframe=timeframe,
        ma_minutes=tuple(ma_minutes),
        dense_min_minutes=dense_min_minutes,
        dense_max_minutes=dense_max_minutes,
        merge_gap_minutes=merge_gap_minutes,
    )
    physical = _physical_window_plan(detection_spec, window=window, stride=stride)
    return {
        "schema_version": 2,
        "action": "build_dataset",
        "cache_dir": str(cache_dir.resolve()),
        "source_manifest": (
            str(source_manifest.resolve()) if source_manifest is not None else None
        ),
        "out_dir": str(out_dir.resolve()),
        "end_before": utc_iso(end_before, field="end_before"),
        "split_at_requested": (
            utc_iso(split_at, field="split_at") if split_at is not None else None
        ),
        **physical,
        "train_frac": train_frac,
        "target_bg_frac": target_bg_frac,
        "max_images": max_images,
        "seed": seed,
        "symbol_contains": symbol_contains,
        "min_rows": min_rows,
        "strict_cadence": strict_cadence,
        "detection_spec": detection_spec.as_dict(),
    }


def build(
    out_dir: Path,
    *,
    cache_dir: Path,
    window: int | None,
    stride: int | None,
    train_frac: float,
    target_bg_frac: float,
    max_images: int,
    seed: int,
    end_before: str,
    source_manifest: Path | None = None,
    split_at: str | None = None,
    symbol_contains: str = "",
    min_rows: int = 10_000,
    timeframe: str = DEFAULT_TIMEFRAME,
    ma_minutes: Sequence[int] = DEFAULT_MA_MINUTES,
    dense_min_minutes: int = DEFAULT_DENSE_MIN_MINUTES,
    dense_max_minutes: int = DEFAULT_DENSE_MAX_MINUTES,
    merge_gap_minutes: int = DEFAULT_MERGE_GAP_MINUTES,
    strict_cadence: bool = True,
) -> dict[str, object]:
    """Build a dataset after authenticating a physical pre-holdout snapshot."""
    plan = make_build_plan(
        out_dir,
        cache_dir=cache_dir,
        window=window,
        stride=stride,
        train_frac=train_frac,
        target_bg_frac=target_bg_frac,
        max_images=max_images,
        seed=seed,
        end_before=end_before,
        source_manifest=source_manifest,
        split_at=split_at,
        symbol_contains=symbol_contains,
        min_rows=min_rows,
        timeframe=timeframe,
        ma_minutes=ma_minutes,
        dense_min_minutes=dense_min_minutes,
        dense_max_minutes=dense_max_minutes,
        merge_gap_minutes=merge_gap_minutes,
        strict_cadence=strict_cadence,
    )
    if source_manifest is None:
        raise ValueError("non-dry-run build requires --source-manifest")
    detection_spec = DetectionSpec(
        timeframe=timeframe,
        ma_minutes=tuple(ma_minutes),
        dense_min_minutes=dense_min_minutes,
        dense_max_minutes=dense_max_minutes,
        merge_gap_minutes=merge_gap_minutes,
    )
    resolved_window = int(plan["window_bars"])
    resolved_stride = int(plan["stride_bars"])
    _ensure_clean_output(out_dir)

    snapshot = load_source_manifest(
        source_manifest,
        expected_source_dir=cache_dir,
        expected_timeframe=detection_spec.timeframe,
        end_before=end_before,
    )
    # This full identity gate happens before the first pd.read_csv call.
    verify_snapshot_identity(snapshot)
    selected_sources = [
        record
        for record in snapshot.files
        if record.row_count >= min_rows
        and (not symbol_contains or symbol_contains in record.path.name)
    ]
    if not selected_sources:
        raise ValueError("source manifest produced no files after explicit filters")

    specs: list[WindowSpec] = []
    for record in selected_sources:
        specs.extend(
            _scan_symbol(
                record,
                window=resolved_window,
                stride=resolved_stride,
                detection_spec=detection_spec,
                strict_cadence=strict_cadence,
            )
        )
    resolved_split_at = _resolve_global_split_at(
        specs,
        train_frac=train_frac,
        explicit_split_at=split_at,
    )
    assigned, dropped_cross_split = _assign_global_splits(
        specs,
        split_at=resolved_split_at,
    )
    chosen = _balance(
        assigned,
        target_bg_frac=target_bg_frac,
        max_images=max_images,
        train_frac=train_frac,
        rng=random.Random(seed),
    )
    chosen_splits = Counter(item.split for item in chosen)
    missing_splits = [name for name in ("train", "val") if chosen_splits[name] == 0]
    if missing_splits:
        raise ValueError(
            "dataset build produced no windows for split(s): "
            f"{', '.join(missing_splits)}; check coverage, split_at, window, and stride"
        )
    max_train_available = max(item.available_at for item in chosen if item.split == "train")
    min_val_start = min(item.window_start_time for item in chosen if item.split == "val")
    if not max_train_available < min_val_start:
        raise AssertionError("global train/val temporal separation invariant failed")

    for name in ("train", "val"):
        (out_dir / "images" / name).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / name).mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    box_counts: list[int] = []
    box_dimensions: list[tuple[float, float]] = []
    manifest_samples: list[dict[str, object]] = []
    grouped: dict[Path, list[WindowSpec]] = {}
    records_by_path = {record.path: record for record in selected_sources}
    for spec in chosen:
        grouped.setdefault(spec.source.path, []).append(spec)
    for path, file_specs in grouped.items():
        record = records_by_path[path]
        frame = _read_authenticated_source(
            record,
            detection_spec=detection_spec,
            strict_cadence=strict_cadence,
        )
        for spec in file_specs:
            subframe = frame.iloc[
                spec.start : spec.start + resolved_window
            ].reset_index(drop=True)
            name = f"{spec.symbol}_{spec.start:06d}"
            image_path = out_dir / "images" / spec.split / f"{name}.png"
            _, transform = render_chart(
                subframe,
                out_path=image_path,
                ma_periods=detection_spec.ma_periods,
            )
            labeled = label_segments(
                subframe,
                transform,
                ma_periods=detection_spec.ma_periods,
                min_bars=detection_spec.dense_min_bars,
                merge_gap=detection_spec.merge_gap_bars,
                max_bars=detection_spec.dense_max_bars,
            )
            boxes = [box for _, box in labeled]
            label_path = out_dir / "labels" / spec.split / f"{name}.txt"
            label_path.write_text(to_yolo_lines(boxes), encoding="utf-8")

            bar_delta = pd.Timedelta(minutes=detection_spec.bar_minutes)
            window_start_time = pd.Timestamp(subframe.iloc[0]["open_time"])
            window_end_open_time = pd.Timestamp(subframe.iloc[-1]["open_time"])
            window_end_close_time = window_end_open_time + bar_delta
            available_at = utc_iso(window_end_close_time)
            if window_start_time != spec.window_start_time or window_end_close_time != spec.available_at:
                raise AssertionError("rescanned source window timing changed")
            manifest_boxes: list[dict[str, object]] = []
            for segment, box in labeled:
                box_start_time = pd.Timestamp(subframe.iloc[segment.start]["open_time"])
                box_end_time = pd.Timestamp(subframe.iloc[segment.end]["open_time"])
                box_end_close_time = box_end_time + bar_delta
                if box_end_close_time > window_end_close_time:
                    raise AssertionError("box segment extends beyond its source window")
                manifest_boxes.append(
                    {
                        "class_id": CLASS_ID,
                        "class_name": CLASS_NAME,
                        "segment": {
                            "start_index_in_window": segment.start,
                            "end_index_in_window": segment.end,
                            "start_index_in_source": spec.start + segment.start,
                            "end_index_in_source": spec.start + segment.end,
                        },
                        "box_start_time": utc_iso(box_start_time),
                        "box_end_time": utc_iso(box_end_time),
                        "box_end_close_time": utc_iso(box_end_close_time),
                        "available_at": available_at,
                        "xywhn": [round(float(value), 6) for value in box],
                    }
                )
            image_relative = image_path.relative_to(out_dir).as_posix()
            label_relative = label_path.relative_to(out_dir).as_posix()
            manifest_samples.append(
                {
                    "id": name,
                    "symbol": spec.symbol,
                    "split": spec.split,
                    "source_file": str(path),
                    "source_sha256": record.sha256,
                    "source_start_index": spec.start,
                    "source_end_index": spec.start + resolved_window - 1,
                    "image": image_relative,
                    "image_sha256": sha256_file(image_path),
                    "label": label_relative,
                    "label_sha256": sha256_file(label_path),
                    "window_start_time": utc_iso(window_start_time),
                    "window_end_open_time": utc_iso(window_end_open_time),
                    "window_end_close_time": available_at,
                    "available_at": available_at,
                    "n_boxes": len(manifest_boxes),
                    "boxes": manifest_boxes,
                }
            )
            stats[f"{spec.split}_images"] += 1
            stats[f"{spec.split}_boxes"] += len(boxes)
            if not boxes:
                stats[f"{spec.split}_background"] += 1
            box_counts.append(len(boxes))
            box_dimensions.extend((width, height) for _, _, width, height in boxes)
        # Detect drift while the authenticated in-memory frame was rendered.
        verify_snapshot_file(record)

    data_yaml = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    (out_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
    snapshot_copy = out_dir / "source_snapshot_manifest.json"
    snapshot_copy.write_bytes(snapshot.manifest_path.read_bytes())
    if sha256_file(snapshot_copy) != snapshot.manifest_sha256:
        raise AssertionError("copied source snapshot manifest hash changed")
    if sha256_file(snapshot.manifest_path) != snapshot.manifest_sha256:
        raise ValueError("source manifest drifted during dataset build")

    split_at_iso = utc_iso(resolved_split_at)
    manifest: dict[str, object] = {
        "schema_version": 2,
        "manifest_type": "yolo_xx_dataset",
        "created_from": "authenticated_local_ohlcv_snapshot_only",
        "source_dir": str(snapshot.source_dir),
        "source_snapshot": {
            "manifest": snapshot_copy.relative_to(out_dir).as_posix(),
            "sha256": snapshot.manifest_sha256,
            "cutoff_exclusive": utc_iso(snapshot.cutoff_exclusive),
            "timeframe": snapshot.timeframe,
        },
        "source_files": [records_by_path[path].as_manifest_dict() for path in sorted(grouped)],
        "end_before": utc_iso(end_before, field="end_before"),
        "split_at": split_at_iso,
        "dropped_cross_split": dropped_cross_split,
        "global_split_invariant": {
            "max_train_available_at": utc_iso(max_train_available),
            "min_val_window_start_time": utc_iso(min_val_start),
            "holds": True,
        },
        "physical_window_minutes": plan["physical_window_minutes"],
        "window_bars": resolved_window,
        "stride_bars": resolved_stride,
        "pixels_per_bar": plan["pixels_per_bar"],
        "resolution_risks": plan["resolution_risks"],
        "strict_cadence": strict_cadence,
        "availability_contract": (
            "Every box is available only at its source window_end_close_time; "
            "box_*_time is descriptive and must not be used as detection availability."
        ),
        "detection_spec": detection_spec.as_dict(),
        "samples": sorted(manifest_samples, key=lambda item: str(item["image"])),
    }
    manifest_path = out_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "schema_version": 2,
        "source_dir": str(snapshot.source_dir),
        "source_manifest": str(snapshot.manifest_path),
        "source_manifest_sha256": snapshot.manifest_sha256,
        "source_files": [str(path) for path in sorted(grouped)],
        "symbols": len(grouped),
        "end_before": utc_iso(end_before, field="end_before"),
        "split_at": split_at_iso,
        "dropped_cross_split": dropped_cross_split,
        "window": resolved_window,
        "physical_window_minutes": plan["physical_window_minutes"],
        "pixels_per_bar": plan["pixels_per_bar"],
        "resolution_risks": plan["resolution_risks"],
        "stride": resolved_stride,
        "train_frac": train_frac,
        "seed": seed,
        "strict_cadence": strict_cadence,
        "detection_spec": detection_spec.as_dict(),
        "dataset_manifest": str(manifest_path.resolve()),
        "build_plan": plan,
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--window", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--split-at")
    parser.add_argument("--target-bg-frac", type=float, default=0.35)
    parser.add_argument("--max-images", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--end-before", default=DEFAULT_END_BEFORE)
    parser.add_argument("--min-rows", type=int, default=10_000)
    parser.add_argument("--symbol-contains", default="")
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument(
        "--ma-minutes",
        type=parse_minute_list,
        default=DEFAULT_MA_MINUTES,
        help="comma-separated physical MA horizons (default: 300,900,1800)",
    )
    parser.add_argument("--dense-min-minutes", type=int, default=DEFAULT_DENSE_MIN_MINUTES)
    parser.add_argument("--dense-max-minutes", type=int, default=DEFAULT_DENSE_MAX_MINUTES)
    parser.add_argument("--merge-gap-minutes", type=int, default=DEFAULT_MERGE_GAP_MINUTES)
    parser.add_argument(
        "--strict-cadence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reject duplicate, out-of-order, or missing candles (default: true)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and print the plan without reading manifests/CSVs or writing output",
    )
    args = parser.parse_args(argv)
    build_kwargs = {
        "cache_dir": args.cache_dir,
        "source_manifest": args.source_manifest,
        "window": args.window,
        "stride": args.stride,
        "train_frac": args.train_frac,
        "split_at": args.split_at,
        "target_bg_frac": args.target_bg_frac,
        "max_images": args.max_images,
        "seed": args.seed,
        "end_before": args.end_before,
        "symbol_contains": args.symbol_contains,
        "min_rows": args.min_rows,
        "timeframe": args.timeframe,
        "ma_minutes": args.ma_minutes,
        "dense_min_minutes": args.dense_min_minutes,
        "dense_max_minutes": args.dense_max_minutes,
        "merge_gap_minutes": args.merge_gap_minutes,
        "strict_cadence": args.strict_cadence,
    }
    if args.dry_run:
        plan = make_build_plan(args.out, **build_kwargs)
        print(json.dumps({"dry_run": True, **plan}, indent=2, sort_keys=True))
        return
    summary = build(args.out, **build_kwargs)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
