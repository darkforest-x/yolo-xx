"""Build an audited, sample-paired 200-bar versus 96-bar owner-short A/B dataset.

The builder uses only the immutable pre-holdout OHLCV snapshot and owner-reviewed
short boxes.  It pairs every retained positive anchor with one chart-only,
symbol/time/split-matched background window.  A background is accepted only when
both views are free of owner boxes and the frozen dense-MA rule, which avoids
silently turning an unreviewed candidate pattern into a negative label.

No model, outcome, return, threshold, exchange, or network code is used here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from .audit import audit_dataset
from .data import add_mas, load_ohlcv_csv
from .labels import find_dense_segments
from .manual_short import (
    BAR_DELTA,
    BAR_MINUTES,
    CLASS_ID,
    CLASS_NAME,
    DEFAULT_RIGHT_CONTEXTS,
    DEFAULT_SPLIT_AT,
    ManualBox,
    _record_symbol,
    _remap_short_box,
    _right_context,
    load_short_annotations,
)
from .render import IMG_HEIGHT, IMG_WIDTH, MARGIN, render_chart
from .source_manifest import (
    HOLDOUT_START,
    SnapshotFile,
    load_source_manifest,
    sha256_file,
    utc_iso,
    utc_timestamp,
    verify_loaded_frame,
    verify_snapshot_file,
    verify_snapshot_identity,
)
from .specs import DetectionSpec

TIMEFRAME = "15m"
WINDOWS = (200, 96)
WIDEST_WINDOW = max(WINDOWS)
DEFAULT_SEED = 20260804
PAIR_MANIFEST = "pair_manifest.json"


@dataclass(frozen=True)
class BoxInterval:
    """One owner box's inclusive source-time interval."""

    box: ManualBox
    start: pd.Timestamp
    end: pd.Timestamp


@dataclass(frozen=True)
class PositiveAnchor:
    """One deduplicated positive window endpoint shared by both A/B arms."""

    anchor_id: str
    symbol: str
    split: str
    window_end_open: pd.Timestamp
    right_context_bars: int
    core_box_ids: tuple[str, ...]

    @property
    def available_at(self) -> pd.Timestamp:
        return self.window_end_open + BAR_DELTA


@dataclass(frozen=True)
class MatchedPair:
    """One positive anchor and its shared negative endpoint."""

    pair_id: str
    positive: PositiveAnchor
    negative_end_open: pd.Timestamp
    negative_search_day_offset: int
    # Extra endpoints that look dense but carry no owner box.  The frozen 1:1
    # recipe deliberately excluded these, because an unlabelled dense window may
    # be a setup the owner simply never reviewed.  They are only admitted here
    # when an outcome check says the short would have lost, which removes that
    # ambiguity without ever forcing an unreviewed pattern to be a negative.
    hard_negative_ends: tuple[pd.Timestamp, ...] = ()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _window_start(end_open: pd.Timestamp, window: int) -> pd.Timestamp:
    return end_open - (window - 1) * BAR_DELTA


def _box_interval(box: ManualBox) -> BoxInterval:
    return BoxInterval(
        box=box,
        start=box.cut_time - (box.bar_b1 - box.bar_b0) * BAR_DELTA,
        end=box.cut_time,
    )


def _intersects(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> bool:
    return start_a <= end_b and start_b <= end_a


def _visible_intervals(
    intervals: Sequence[BoxInterval],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[BoxInterval, ...]:
    return tuple(item for item in intervals if start <= item.start and item.end <= end)


def _has_partial_box(
    intervals: Sequence[BoxInterval],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> bool:
    return any(
        _intersects(start, end, item.start, item.end)
        and not (start <= item.start and item.end <= end)
        for item in intervals
    )


def _assign_split(end_open: pd.Timestamp, split_at: pd.Timestamp) -> str | None:
    available = end_open + BAR_DELTA
    widest_start = _window_start(end_open, WIDEST_WINDOW)
    if available < split_at:
        return "train"
    if widest_start >= split_at:
        return "val"
    return None


def _make_positive_anchors(
    boxes: Sequence[ManualBox],
    *,
    split_at: pd.Timestamp,
    right_contexts: tuple[int, ...],
) -> tuple[list[PositiveAnchor], dict[str, list[BoxInterval]], Counter[str]]:
    """Create shared endpoints, collapse exact duplicates, and reject cropped boxes."""
    by_symbol: dict[str, list[BoxInterval]] = defaultdict(list)
    for box in boxes:
        by_symbol[box.symbol].append(_box_interval(box))
    for symbol in by_symbol:
        by_symbol[symbol].sort(key=lambda item: (item.start, item.end, item.box.box_id))

    endpoint_groups: dict[tuple[str, int], list[ManualBox]] = defaultdict(list)
    stats: Counter[str] = Counter()
    for box in boxes:
        context = _right_context(box.box_id, right_contexts)
        if box.width_bars + context > min(WINDOWS):
            stats["dropped_core_does_not_fit_w96"] += 1
            continue
        end = box.cut_time + context * BAR_DELTA
        endpoint_groups[(box.symbol, int(end.value))].append(box)

    anchors: list[PositiveAnchor] = []
    for (symbol, end_ns), core_boxes in sorted(endpoint_groups.items()):
        end = pd.Timestamp(end_ns, tz="UTC")
        split = _assign_split(end, split_at)
        if split is None:
            stats["dropped_cross_split"] += len(core_boxes)
            continue
        intervals = by_symbol[symbol]
        invalid_partial = any(
            _has_partial_box(intervals, start=_window_start(end, window), end=end)
            for window in WINDOWS
        )
        if invalid_partial:
            stats["dropped_partial_owner_box"] += len(core_boxes)
            continue
        visible_w96 = {
            item.box.box_id
            for item in _visible_intervals(
                intervals, start=_window_start(end, min(WINDOWS)), end=end
            )
        }
        core_ids = tuple(sorted(box.box_id for box in core_boxes))
        if not set(core_ids).issubset(visible_w96):
            stats["dropped_core_visibility_invariant"] += len(core_boxes)
            continue
        anchor_id = core_ids[0]
        anchors.append(
            PositiveAnchor(
                anchor_id=anchor_id,
                symbol=symbol,
                split=split,
                window_end_open=end,
                right_context_bars=_right_context(anchor_id, right_contexts),
                core_box_ids=core_ids,
            )
        )
        stats[f"candidate_{split}_anchors"] += 1
        stats["candidate_core_boxes"] += len(core_boxes)
        stats["collapsed_duplicate_endpoints"] += len(core_boxes) - 1
    return anchors, by_symbol, stats


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _limit_anchors(
    anchors: Sequence[PositiveAnchor],
    *,
    max_positive_anchors: int | None,
    seed: int,
) -> list[PositiveAnchor]:
    if max_positive_anchors is None or len(anchors) <= max_positive_anchors:
        return list(anchors)
    if max_positive_anchors < 2:
        raise ValueError("max_positive_anchors must be at least 2")
    train_target = max_positive_anchors // 2
    val_target = max_positive_anchors - train_target
    selected: list[PositiveAnchor] = []
    for split, target in (("train", train_target), ("val", val_target)):
        pool = sorted(
            (item for item in anchors if item.split == split),
            key=lambda item: _stable_rank(seed, item.anchor_id),
        )
        selected.extend(pool[:target])
    if len(selected) < max_positive_anchors:
        chosen = {item.anchor_id for item in selected}
        remainder = sorted(
            (item for item in anchors if item.anchor_id not in chosen),
            key=lambda item: _stable_rank(seed, item.anchor_id),
        )
        selected.extend(remainder[: max_positive_anchors - len(selected)])
    return sorted(selected, key=lambda item: (item.symbol, item.window_end_open, item.anchor_id))


def _candidate_day_offsets(anchor_id: str, seed: int) -> list[int]:
    """Return deterministic nearby-day offsets, nearest days preferred in shuffled bands."""
    rng_seed = int(_stable_rank(seed, anchor_id)[:16], 16)
    rng = random.Random(rng_seed)
    offsets: list[int] = []
    for lower, upper in ((1, 14), (15, 45), (46, 90), (91, 180), (181, 365)):
        band = [sign * day for day in range(lower, upper + 1) for sign in (-1, 1)]
        rng.shuffle(band)
        offsets.extend(band)
    return offsets


def _is_rule_clear(frame: pd.DataFrame, end_index: int) -> bool:
    for window in WINDOWS:
        start = end_index - window + 1
        if start < 0:
            return False
        view = frame.iloc[start : end_index + 1]
        if find_dense_segments(view):
            return False
    return True


def _match_symbol_negatives(
    anchors: Sequence[PositiveAnchor],
    *,
    intervals: Sequence[BoxInterval],
    frame: pd.DataFrame,
    split_at: pd.Timestamp,
    seed: int,
    hard_negatives_per_anchor: int = 0,
    hard_negative_filter: Callable[[str, pd.Timestamp], bool] | None = None,
) -> tuple[list[MatchedPair], int]:
    """Match one unique rule-clear background to every possible positive anchor."""
    time_to_index = {
        int(pd.Timestamp(value).value): index for index, value in enumerate(frame["open_time"])
    }
    used_endpoints: set[int] = set()
    pairs: list[MatchedPair] = []
    unmatched = 0
    for anchor in sorted(anchors, key=lambda item: _stable_rank(seed, item.anchor_id)):
        matched: tuple[pd.Timestamp, int] | None = None
        for day_offset in _candidate_day_offsets(anchor.anchor_id, seed):
            end = anchor.window_end_open + pd.Timedelta(days=day_offset)
            end_ns = int(end.value)
            if end_ns in used_endpoints or end_ns not in time_to_index:
                continue
            if _assign_split(end, split_at) != anchor.split:
                continue
            wide_start = _window_start(end, WIDEST_WINDOW)
            if any(_intersects(wide_start, end, item.start, item.end) for item in intervals):
                continue
            end_index = time_to_index[end_ns]
            if not _is_rule_clear(frame, end_index):
                continue
            matched = (end, day_offset)
            break
        if matched is None:
            unmatched += 1
            continue
        negative_end, day_offset = matched
        used_endpoints.add(int(negative_end.value))
        hard: list[pd.Timestamp] = []
        if hard_negatives_per_anchor and hard_negative_filter is not None:
            for extra_offset in _candidate_day_offsets(anchor.anchor_id + "#hard", seed):
                if len(hard) >= hard_negatives_per_anchor:
                    break
                end = anchor.window_end_open + pd.Timedelta(days=extra_offset)
                end_ns = int(end.value)
                if end_ns in used_endpoints or end_ns not in time_to_index:
                    continue
                if _assign_split(end, split_at) != anchor.split:
                    continue
                wide_start = _window_start(end, WIDEST_WINDOW)
                if any(_intersects(wide_start, end, item.start, item.end) for item in intervals):
                    continue
                end_index = time_to_index[end_ns]
                # Inverting the rule-clear test also drops the left-history guard
                # it implied, so the widest window still has to fit the snapshot.
                if end_index < WIDEST_WINDOW - 1:
                    continue
                # The opposite of the matched-background test: we want windows the
                # frozen rule *does* call dense, so the detector learns to reject
                # look-alikes instead of only clean backgrounds.
                if _is_rule_clear(frame, end_index):
                    continue
                if not hard_negative_filter(anchor.symbol, end):
                    continue
                used_endpoints.add(end_ns)
                hard.append(end)
        pairs.append(
            MatchedPair(
                pair_id=anchor.anchor_id,
                positive=anchor,
                negative_end_open=negative_end,
                negative_search_day_offset=day_offset,
                hard_negative_ends=tuple(hard),
            )
        )
    return pairs, unmatched


def _load_authenticated_frame(record: SnapshotFile) -> pd.DataFrame:
    verify_snapshot_file(record)
    frame = load_ohlcv_csv(record.path, timeframe=TIMEFRAME, strict_cadence=True)
    verify_loaded_frame(frame, record, timeframe=TIMEFRAME)
    frame = add_mas(frame, periods=(20, 60, 120))
    verify_snapshot_file(record)
    return frame


def _render_one(
    *,
    dataset: Path,
    record: SnapshotFile,
    frame: pd.DataFrame,
    time_to_index: dict[int, int],
    intervals: Sequence[BoxInterval],
    sample_id: str,
    match_id: str,
    sample_kind: str,
    split: str,
    window: int,
    end_open: pd.Timestamp,
    right_context_bars: int,
    core_box_ids: Sequence[str],
) -> tuple[dict[str, object], int]:
    end_index = time_to_index[int(end_open.value)]
    start_index = end_index - window + 1
    if start_index < 0:
        raise ValueError(f"{sample_id}: source window starts before snapshot")
    start_time = pd.Timestamp(frame.iloc[start_index]["open_time"])
    available_at = end_open + BAR_DELTA
    subframe = frame.iloc[start_index : end_index + 1].reset_index(drop=True)
    image_path = dataset / "images" / split / f"{sample_id}.png"
    _, transform = render_chart(subframe, out_path=image_path, ma_periods=(20, 60, 120))

    visible = (
        _visible_intervals(intervals, start=start_time, end=end_open)
        if sample_kind == "positive"
        else ()
    )
    if sample_kind == "positive" and not set(core_box_ids).issubset(
        {item.box.box_id for item in visible}
    ):
        raise AssertionError(f"{sample_id}: core owner box disappeared during render")
    if sample_kind == "negative" and any(
        _intersects(start_time, end_open, item.start, item.end) for item in intervals
    ):
        raise AssertionError(f"{sample_id}: negative overlaps an owner box")

    manifest_boxes: list[dict[str, object]] = []
    label_lines: list[str] = []
    y_fallbacks = 0
    boxes_by_coordinates: dict[tuple[float, float, float, float], dict[str, object]] = {}
    collapsed_duplicate_annotations = 0
    for item in visible:
        normalized, source_box_start, source_box_end, fallback = _remap_short_box(
            item.box,
            frame=frame,
            source_start_index=start_index,
            transform=transform,
            time_to_index=time_to_index,
        )
        rounded = tuple(round(float(value), 6) for value in normalized)
        existing = boxes_by_coordinates.get(rounded)
        if existing is not None:
            annotation_ids = existing["annotation_box_ids"]
            assert isinstance(annotation_ids, list)
            annotation_ids.append(item.box.box_id)
            existing["is_core_anchor"] = bool(existing["is_core_anchor"]) or (
                item.box.box_id in core_box_ids
            )
            collapsed_duplicate_annotations += 1
            continue
        label_lines.append(f"{CLASS_ID} " + " ".join(f"{value:.6f}" for value in rounded))
        box_start_time = pd.Timestamp(frame.iloc[source_box_start]["open_time"])
        box_end_time = pd.Timestamp(frame.iloc[source_box_end]["open_time"])
        manifest_box: dict[str, object] = {
                "class_id": CLASS_ID,
                "class_name": CLASS_NAME,
                "annotation_box_id": item.box.box_id,
                "annotation_box_ids": [item.box.box_id],
                "annotation_stem": item.box.stem,
                "is_core_anchor": item.box.box_id in core_box_ids,
                "segment": {
                    "start_index_in_window": source_box_start - start_index,
                    "end_index_in_window": source_box_end - start_index,
                    "start_index_in_source": source_box_start,
                    "end_index_in_source": source_box_end,
                },
                "box_start_time": utc_iso(box_start_time),
                "box_end_time": utc_iso(box_end_time),
                "box_end_close_time": utc_iso(box_end_time + BAR_DELTA),
                "available_at": utc_iso(available_at),
                "right_context_bars": right_context_bars,
                "y_price_fallback": fallback,
                "xywhn": list(rounded),
            }
        manifest_boxes.append(manifest_box)
        boxes_by_coordinates[rounded] = manifest_box
        y_fallbacks += int(fallback)

    label_path = dataset / "labels" / split / f"{sample_id}.txt"
    label_path.write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
    )
    return (
        {
            "id": sample_id,
            "match_id": match_id,
            "sample_kind": sample_kind,
            "symbol": _record_symbol(record),
            "split": split,
            "source_file": str(record.path),
            "source_sha256": record.sha256,
            "source_start_index": start_index,
            "source_end_index": end_index,
            "image": image_path.relative_to(dataset).as_posix(),
            "image_sha256": sha256_file(image_path),
            "label": label_path.relative_to(dataset).as_posix(),
            "label_sha256": sha256_file(label_path),
            "window_start_time": utc_iso(start_time),
            "window_end_open_time": utc_iso(end_open),
            "window_end_close_time": utc_iso(available_at),
            "available_at": utc_iso(available_at),
            "right_context_bars": right_context_bars,
            "core_box_ids": list(core_box_ids),
            "collapsed_duplicate_annotations": collapsed_duplicate_annotations,
            "n_boxes": len(manifest_boxes),
            "boxes": manifest_boxes,
        },
        y_fallbacks,
    )


def _contract_rows(samples: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "id",
        "match_id",
        "sample_kind",
        "symbol",
        "split",
        "window_end_open_time",
        "available_at",
        "right_context_bars",
        "core_box_ids",
    )
    return [
        {field: sample[field] for field in fields}
        for sample in sorted(samples, key=lambda item: str(item["id"]))
    ]


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finalize_dataset(
    *,
    dataset: Path,
    final_output: Path,
    snapshot_root: Path,
    snapshot_manifest: Path,
    snapshot_sha256: str,
    records: dict[str, SnapshotFile],
    samples: list[dict[str, object]],
    split_at: pd.Timestamp,
    window: int,
    pair_contract_sha256: str,
    input_box_count: int,
    dropped_stats: Counter[str],
    y_fallbacks: int,
) -> dict[str, object]:
    data_yaml = (
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )
    (dataset / "data.yaml").write_text(data_yaml, encoding="utf-8")
    snapshot_copy = dataset / "source_snapshot_manifest.json"
    snapshot_copy.write_bytes(snapshot_manifest.read_bytes())

    ordered = sorted(samples, key=lambda item: str(item["image"]))
    train_available = [
        pd.Timestamp(item["available_at"]) for item in ordered if item["split"] == "train"
    ]
    val_starts = [
        pd.Timestamp(item["window_start_time"]) for item in ordered if item["split"] == "val"
    ]
    if not train_available or not val_starts or not max(train_available) < min(val_starts):
        raise AssertionError("paired dataset global split invariant failed")
    used_symbols = sorted({str(item["symbol"]) for item in ordered})
    positives = [item for item in ordered if item["sample_kind"] == "positive"]
    negatives = [item for item in ordered if item["sample_kind"] == "negative"]
    # The matched background per positive is the position-shortcut control and
    # stays strictly 1:1.  Outcome-verified hard negatives sit on top of it and
    # are counted separately, so the control is never diluted by accident.
    matched = [item for item in negatives if str(item["id"]).startswith("neg__")]
    hard = [item for item in negatives if str(item["id"]).startswith("hard")]
    if len(matched) + len(hard) != len(negatives):
        raise AssertionError("negative sample id does not declare its role")
    if len(positives) != len(matched):
        raise AssertionError("every positive must keep exactly one matched background")
    if any(int(item["n_boxes"]) != 0 for item in negatives):
        raise AssertionError("background sample unexpectedly contains a label")

    manifest = {
        "schema_version": 2,
        "manifest_type": "yolo_xx_dataset",
        "created_from": "owner_short_paired_window_ab_with_rule_clear_matched_backgrounds",
        "matched_background_count": len(matched),
        "hard_negative_count": len(hard),
        "source_dir": str(snapshot_root),
        "source_snapshot": {
            "manifest": snapshot_copy.relative_to(dataset).as_posix(),
            "sha256": snapshot_sha256,
            "cutoff_exclusive": utc_iso(HOLDOUT_START),
            "timeframe": TIMEFRAME,
        },
        "source_files": [records[symbol].as_manifest_dict() for symbol in used_symbols],
        "annotation_source": {
            "path": str(snapshot_root / "owner_short_annotations.csv"),
            "sha256": sha256_file(snapshot_root / "owner_short_annotations.csv"),
            "filter": "owner_side == short",
            "input_boxes": input_box_count,
        },
        "end_before": utc_iso(HOLDOUT_START),
        "split_at": utc_iso(split_at),
        "dropped_cross_split": int(dropped_stats["dropped_cross_split"]),
        "global_split_invariant": {
            "max_train_available_at": utc_iso(max(train_available)),
            "min_val_window_start_time": utc_iso(min(val_starts)),
            "holds": True,
        },
        "layout": "paired_staggered_causal",
        "position_policy": "shared_real_endpoint_deterministic_0_8_16_24_context",
        "negative_policy": (
            "same_symbol_split_context_bucket_and_time_of_day; unique nearby-day endpoint; "
            "no owner-box overlap; no frozen dense-rule segment in either 200/96 view"
        ),
        "pair_contract_sha256": pair_contract_sha256,
        "physical_window_minutes": window * BAR_MINUTES,
        "window_bars": window,
        "stride_bars": window,
        "pixels_per_bar": round((IMG_WIDTH - 2 * MARGIN) / max(window - 1, 1), 6),
        "resolution_risks": [],
        "strict_cadence": True,
        "availability_contract": (
            "Every sample and box is available only at the rendered window_end_close_time."
        ),
        "detection_spec": DetectionSpec().as_dict(),
        "samples": ordered,
    }
    _write_json(dataset / "dataset_manifest.json", manifest)
    summary = {
        "schema_version": 2,
        "dataset": str(final_output),
        "window_bars": window,
        "pair_contract_sha256": pair_contract_sha256,
        "source_snapshot_sha256": snapshot_sha256,
        "input_annotation_boxes": input_box_count,
        "positive_images": len(positives),
        "background_images": len(negatives),
        "train_images": sum(item["split"] == "train" for item in ordered),
        "val_images": sum(item["split"] == "val" for item in ordered),
        "train_boxes": sum(
            int(item["n_boxes"]) for item in ordered if item["split"] == "train"
        ),
        "val_boxes": sum(
            int(item["n_boxes"]) for item in ordered if item["split"] == "val"
        ),
        "y_price_fallbacks": y_fallbacks,
        "collapsed_duplicate_annotations": sum(
            int(item.get("collapsed_duplicate_annotations", 0)) for item in ordered
        ),
        "portable_data_yaml": True,
        "holdout_read": False,
    }
    _write_json(dataset / "dataset_summary.json", summary)
    return summary


def audit_pair(pair_root: str | Path) -> dict[str, object]:
    """Verify both datasets and the machine-readable sample pairing contract."""
    root = Path(pair_root).resolve()
    errors: list[str] = []
    audits = {name: audit_dataset(root / name) for name in ("w200", "w96")}
    for name, audit in audits.items():
        if not audit["valid"]:
            errors.append(f"{name} strong dataset audit failed")
    manifests: dict[str, dict[str, object]] = {}
    for name in ("w200", "w96"):
        try:
            manifests[name] = json.loads(
                (root / name / "dataset_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{name} manifest unreadable: {error}")
    contract_hashes: dict[str, str] = {}
    if len(manifests) == 2:
        rows = {
            name: _contract_rows(manifest["samples"])  # type: ignore[arg-type,index]
            for name, manifest in manifests.items()
        }
        contract_hashes = {name: _payload_hash(value) for name, value in rows.items()}
        if rows["w200"] != rows["w96"]:
            errors.append("A/B sample ledgers differ outside the window-specific content")
        for name, manifest in manifests.items():
            if manifest.get("pair_contract_sha256") != contract_hashes[name]:
                errors.append(f"{name} pair_contract_sha256 mismatch")
        ids = {
            name: {str(item["id"]) for item in manifest["samples"]}  # type: ignore[index]
            for name, manifest in manifests.items()
        }
        if ids["w200"] != ids["w96"]:
            errors.append("A/B sample ids differ")
        for name, manifest in manifests.items():
            for sample in manifest["samples"]:  # type: ignore[index]
                if sample.get("sample_kind") == "negative" and sample.get("n_boxes") != 0:
                    errors.append(f"{name}:{sample.get('id')} negative has boxes")

    pair_manifest_path = root / PAIR_MANIFEST
    if not pair_manifest_path.is_file():
        errors.append(f"missing {PAIR_MANIFEST}")
    result = {
        "schema_version": 1,
        "audit_type": "yolo_xx_paired_ab_audit",
        "pair_root": str(root),
        "valid": not errors,
        "contract_sha256": contract_hashes.get("w200"),
        "datasets": {
            name: {
                "valid": bool(audit["valid"]),
                "splits": audit.get("splits", {}),
                "error_count": len(audit.get("errors", [])),
            }
            for name, audit in audits.items()
        },
        "errors": errors,
    }
    return result


def build_pair(
    *,
    snapshot_dir: str | Path,
    out_dir: str | Path,
    split_at: str = DEFAULT_SPLIT_AT,
    right_contexts: Sequence[int] = DEFAULT_RIGHT_CONTEXTS,
    seed: int = DEFAULT_SEED,
    max_positive_anchors: int | None = None,
    hard_negatives_per_anchor: int = 0,
    hard_negative_filter: Callable[[str, pd.Timestamp], bool] | None = None,
) -> dict[str, object]:
    """Build both immutable datasets from one shared positive/background ledger.

    `hard_negatives_per_anchor` admits extra negatives that the frozen rule calls
    dense but that carry no owner box.  They are the windows a continuous scan
    actually confuses the detector with; the 1:1 recipe never showed it any.
    `hard_negative_filter` must disambiguate them (an outcome check), so an
    unreviewed pattern is never forced into the negative class on looks alone.
    """
    snapshot_root = Path(snapshot_dir).resolve()
    output = Path(out_dir).resolve()
    _ensure_new_output(output)
    split_timestamp = utc_timestamp(split_at, field="split_at")
    if split_timestamp >= HOLDOUT_START:
        raise ValueError("split_at must be before holdout")
    contexts = tuple(sorted(set(int(value) for value in right_contexts)))
    if not contexts or any(value < 0 for value in contexts):
        raise ValueError("right_contexts must contain non-negative integers")

    snapshot_manifest = snapshot_root / "source_snapshot.json"
    annotation_path = snapshot_root / "owner_short_annotations.csv"
    boxes, _ = load_short_annotations(annotation_path)
    snapshot = load_source_manifest(
        snapshot_manifest,
        expected_source_dir=snapshot_root,
        expected_timeframe=TIMEFRAME,
        end_before=HOLDOUT_START,
    )
    verify_snapshot_identity(snapshot)
    records = {_record_symbol(record): record for record in snapshot.files}
    anchors, intervals_by_symbol, build_stats = _make_positive_anchors(
        boxes, split_at=split_timestamp, right_contexts=contexts
    )
    anchors = _limit_anchors(
        anchors,
        max_positive_anchors=max_positive_anchors,
        seed=seed,
    )
    missing = sorted({item.symbol for item in anchors} - set(records))
    if missing:
        raise ValueError("snapshot is missing anchor symbols: " + ", ".join(missing))

    pairs: list[MatchedPair] = []
    unmatched = 0
    by_symbol: dict[str, list[PositiveAnchor]] = defaultdict(list)
    for anchor in anchors:
        by_symbol[anchor.symbol].append(anchor)
    for symbol in sorted(by_symbol):
        frame = _load_authenticated_frame(records[symbol])
        matched, missing_count = _match_symbol_negatives(
            by_symbol[symbol],
            intervals=intervals_by_symbol[symbol],
            frame=frame,
            split_at=split_timestamp,
            seed=seed,
            hard_negatives_per_anchor=hard_negatives_per_anchor,
            hard_negative_filter=hard_negative_filter,
        )
        pairs.extend(matched)
        unmatched += missing_count
    if not pairs:
        raise ValueError("no positive/background pairs survived matching")
    if not any(item.positive.split == "train" for item in pairs) or not any(
        item.positive.split == "val" for item in pairs
    ):
        raise ValueError("matched pairs must contain both train and val")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        datasets = {window: staging / f"w{window}" for window in WINDOWS}
        for dataset in datasets.values():
            for split in ("train", "val"):
                (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
                (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)

        samples_by_window: dict[int, list[dict[str, object]]] = defaultdict(list)
        fallbacks_by_window: Counter[int] = Counter()
        pairs_by_symbol: dict[str, list[MatchedPair]] = defaultdict(list)
        for pair in pairs:
            pairs_by_symbol[pair.positive.symbol].append(pair)
        for symbol in sorted(pairs_by_symbol):
            record = records[symbol]
            frame = _load_authenticated_frame(record)
            time_to_index = {
                int(pd.Timestamp(value).value): index
                for index, value in enumerate(frame["open_time"])
            }
            intervals = intervals_by_symbol[symbol]
            for pair in sorted(pairs_by_symbol[symbol], key=lambda item: item.pair_id):
                anchor = pair.positive
                for window in WINDOWS:
                    dataset = datasets[window]
                    positive, positive_fallbacks = _render_one(
                        dataset=dataset,
                        record=record,
                        frame=frame,
                        time_to_index=time_to_index,
                        intervals=intervals,
                        sample_id=f"pos__{pair.pair_id}",
                        match_id=pair.pair_id,
                        sample_kind="positive",
                        split=anchor.split,
                        window=window,
                        end_open=anchor.window_end_open,
                        right_context_bars=anchor.right_context_bars,
                        core_box_ids=anchor.core_box_ids,
                    )
                    negative, negative_fallbacks = _render_one(
                        dataset=dataset,
                        record=record,
                        frame=frame,
                        time_to_index=time_to_index,
                        intervals=intervals,
                        sample_id=f"neg__{pair.pair_id}",
                        match_id=pair.pair_id,
                        sample_kind="negative",
                        split=anchor.split,
                        window=window,
                        end_open=pair.negative_end_open,
                        right_context_bars=anchor.right_context_bars,
                        core_box_ids=(),
                    )
                    samples_by_window[window].extend((positive, negative))
                    fallbacks_by_window[window] += positive_fallbacks + negative_fallbacks
                    for index, hard_end in enumerate(pair.hard_negative_ends):
                        hard_sample, hard_fallbacks = _render_one(
                            dataset=dataset,
                            record=record,
                            frame=frame,
                            time_to_index=time_to_index,
                            intervals=intervals,
                            sample_id=f"hard{index}__{pair.pair_id}",
                            match_id=pair.pair_id,
                            sample_kind="negative",
                            split=anchor.split,
                            window=window,
                            end_open=hard_end,
                            right_context_bars=anchor.right_context_bars,
                            core_box_ids=(),
                        )
                        samples_by_window[window].append(hard_sample)
                        fallbacks_by_window[window] += hard_fallbacks
            verify_snapshot_file(record)

        contract_rows = _contract_rows(samples_by_window[WIDEST_WINDOW])
        contract_hash = _payload_hash(contract_rows)
        if contract_rows != _contract_rows(samples_by_window[min(WINDOWS)]):
            raise AssertionError("paired render changed the shared sample ledger")

        summaries: dict[str, dict[str, object]] = {}
        for window in WINDOWS:
            summaries[f"w{window}"] = _finalize_dataset(
                dataset=datasets[window],
                final_output=output / f"w{window}",
                snapshot_root=snapshot_root,
                snapshot_manifest=snapshot_manifest,
                snapshot_sha256=snapshot.manifest_sha256,
                records=records,
                samples=samples_by_window[window],
                split_at=split_timestamp,
                window=window,
                pair_contract_sha256=contract_hash,
                input_box_count=len(boxes),
                dropped_stats=build_stats,
                y_fallbacks=int(fallbacks_by_window[window]),
            )
            audit = audit_dataset(datasets[window])
            if not audit["valid"]:
                raise ValueError(
                    f"w{window} staging audit failed: " + "; ".join(audit["errors"][:5])
                )

        ledger = [
            {
                "pair_id": pair.pair_id,
                "symbol": pair.positive.symbol,
                "split": pair.positive.split,
                "right_context_bars": pair.positive.right_context_bars,
                "core_box_ids": list(pair.positive.core_box_ids),
                "positive_window_end_open_time": utc_iso(pair.positive.window_end_open),
                "positive_available_at": utc_iso(pair.positive.available_at),
                "negative_window_end_open_time": utc_iso(pair.negative_end_open),
                "negative_available_at": utc_iso(pair.negative_end_open + BAR_DELTA),
                "negative_search_day_offset": pair.negative_search_day_offset,
            }
            for pair in sorted(pairs, key=lambda item: item.pair_id)
        ]
        pair_manifest = {
            "schema_version": 1,
            "manifest_type": "yolo_xx_paired_ab",
            "created_before_training": True,
            "holdout_read": False,
            "source_snapshot_sha256": snapshot.manifest_sha256,
            "annotation_sha256": sha256_file(annotation_path),
            "split_at": utc_iso(split_timestamp),
            "seed": seed,
            "right_context_choices_bars": list(contexts),
            "windows": list(WINDOWS),
            "pair_contract_sha256": contract_hash,
            "candidate_positive_anchors": len(anchors),
            "matched_pairs": len(pairs),
            "unmatched_positive_anchors": unmatched,
            "positive_to_background_ratio": "1:1",
            "build_stats": {key: int(value) for key, value in sorted(build_stats.items())},
            "datasets": {
                name: {
                    "path": name,
                    "dataset_manifest_sha256": sha256_file(
                        staging / name / "dataset_manifest.json"
                    ),
                    "summary": summary,
                }
                for name, summary in summaries.items()
            },
            "ledger": ledger,
        }
        _write_json(staging / PAIR_MANIFEST, pair_manifest)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    final_audit = audit_pair(output)
    _write_json(output / "pair_audit.json", final_audit)
    if not final_audit["valid"]:
        raise ValueError("final pair audit failed: " + "; ".join(final_audit["errors"][:5]))
    summary = {
        "schema_version": 1,
        "pair_root": str(output),
        "pair_contract_sha256": final_audit["contract_sha256"],
        "matched_pairs": len(pairs),
        "positive_images_per_arm": len(pairs),
        "background_images_per_arm": len(pairs),
        "unmatched_positive_anchors": unmatched,
        "candidate_positive_anchors": len(anchors),
        "dataset_audits_valid": True,
        "holdout_read": False,
        "pair_manifest_sha256": sha256_file(output / PAIR_MANIFEST),
    }
    _write_json(output / "pair_summary.json", summary)
    return summary


def make_plan(
    *,
    snapshot_dir: Path,
    out_dir: Path,
    split_at: str,
    right_contexts: Sequence[int],
    seed: int,
    max_positive_anchors: int | None,
) -> dict[str, object]:
    """Return a no-read/no-write build plan for preregistration review."""
    timestamp = utc_timestamp(split_at, field="split_at")
    if timestamp >= HOLDOUT_START:
        raise ValueError("split_at must be before holdout")
    return {
        "dry_run": True,
        "action": "build-paired-ab",
        "snapshot_dir": str(snapshot_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "windows": list(WINDOWS),
        "split_at": utc_iso(timestamp),
        "right_contexts": sorted(set(int(value) for value in right_contexts)),
        "negative_ratio": 1.0,
        "negative_rule": "no owner-box overlap and no dense-rule segment in both arms",
        "seed": seed,
        "max_positive_anchors": max_positive_anchors,
        "training": False,
        "network": False,
        "holdout_read": False,
    }


def _parse_contexts(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(item.strip()) for item in raw.split(",") if item.strip())))
    except ValueError as error:
        raise argparse.ArgumentTypeError("right contexts must be comma-separated integers") from error
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("right contexts must be non-negative")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--snapshot-dir", required=True, type=Path)
    build_parser.add_argument("--out", required=True, type=Path)
    build_parser.add_argument("--split-at", default=DEFAULT_SPLIT_AT)
    build_parser.add_argument(
        "--right-contexts", type=_parse_contexts, default=DEFAULT_RIGHT_CONTEXTS
    )
    build_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build_parser.add_argument("--max-positive-anchors", type=int)
    build_parser.add_argument("--dry-run", action="store_true")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--pair-root", required=True, type=Path)
    audit_parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.action == "audit":
        payload = audit_pair(args.pair_root)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if payload["valid"] else 1
    if args.dry_run:
        payload = make_plan(
            snapshot_dir=args.snapshot_dir,
            out_dir=args.out,
            split_at=args.split_at,
            right_contexts=args.right_contexts,
            seed=args.seed,
            max_positive_anchors=args.max_positive_anchors,
        )
    else:
        payload = build_pair(
            snapshot_dir=args.snapshot_dir,
            out_dir=args.out,
            split_at=args.split_at,
            right_contexts=args.right_contexts,
            seed=args.seed,
            max_positive_anchors=args.max_positive_anchors,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
