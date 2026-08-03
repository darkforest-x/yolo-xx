"""Recover owner-reviewed short-only boxes into immutable YOLO datasets.

This module is deliberately chart-only.  It reads a local owner review sheet,
copies only the authenticated OHLCV prefix before the frozen holdout boundary,
and renders either the original 200-bar windows or a shorter position-balanced
96-bar view.  It never reads outcome columns, trains a model, or contacts a
network service.

The short-window layout is not claimed to make the annotated pattern finish
earlier.  Its purpose is to give narrow clusters more horizontal pixels while
preventing the fixed-right-edge shortcut: each box receives a deterministic
0/8/16/24-bar right context.  Availability remains the close of the complete
rendered window and is recorded explicitly in the dataset manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

from .audit import audit_dataset
from .data import add_mas, load_ohlcv_csv
from .render import (
    IMG_HEIGHT,
    IMG_WIDTH,
    MARGIN,
    ChartTransform,
    make_chart_transform,
    render_chart,
)
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
BAR_MINUTES = 15
BAR_DELTA = pd.Timedelta(minutes=BAR_MINUTES)
BAR_MILLISECONDS = BAR_MINUTES * 60 * 1000
ORIGINAL_WINDOW = 200
SHORT_WINDOW = 96
DEFAULT_SPLIT_AT = "2026-02-15T00:00:00Z"
DEFAULT_RIGHT_CONTEXTS = (0, 8, 16, 24)
CLASS_ID = 0
CLASS_NAME = "dense_cluster"
_SOURCE_RE = re.compile(
    r"^okx_(?P<symbol>.+)_15m_(?P<rows>[0-9]+)(?:_latest)?\.csv$"
)
_REQUIRED_ANNOTATION_COLUMNS = {
    "box_id",
    "symbol",
    "stem",
    "cut_time",
    "bar_b0",
    "bar_b1",
    "yolo_xc",
    "yolo_yc",
    "yolo_w",
    "yolo_h",
    "owner_side",
}
_SNAPSHOT_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class ManualBox:
    """One owner-reviewed short box with no outcome or trading fields."""

    box_id: str
    symbol: str
    stem: str
    cut_time: pd.Timestamp
    bar_b0: int
    bar_b1: int
    xywhn: tuple[float, float, float, float]

    @property
    def width_bars(self) -> int:
        return self.bar_b1 - self.bar_b0 + 1

    @property
    def original_window_start(self) -> pd.Timestamp:
        return self.cut_time - self.bar_b1 * BAR_DELTA


@dataclass(frozen=True)
class RenderRequest:
    """One immutable chart request after layout and temporal split assignment."""

    sample_id: str
    symbol: str
    window_start: pd.Timestamp
    window_end_open: pd.Timestamp
    split: str
    boxes: tuple[ManualBox, ...]
    right_context_bars: int | None

    @property
    def available_at(self) -> pd.Timestamp:
        return self.window_end_open + BAR_DELTA


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _safe_float(raw: object, *, field: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def load_short_annotations(path: str | Path) -> tuple[list[ManualBox], list[str]]:
    """Load only owner_side=short rows and reject late or malformed labels."""
    source = Path(path)
    rows: list[ManualBox] = []
    fieldnames: list[str] = []
    seen_ids: set[str] = set()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = sorted(_REQUIRED_ANNOTATION_COLUMNS - set(fieldnames))
        if missing:
            raise ValueError(f"annotation sheet is missing columns: {', '.join(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            if str(raw.get("owner_side", "")).strip().lower() != "short":
                continue
            box_id = str(raw["box_id"]).strip()
            if not box_id or box_id in seen_ids:
                raise ValueError(f"annotation line {line_number}: duplicate/empty box_id")
            seen_ids.add(box_id)
            cut_time = utc_timestamp(raw["cut_time"], field=f"line {line_number} cut_time")
            if cut_time >= HOLDOUT_START:
                raise ValueError(f"annotation line {line_number}: short box is in holdout")
            try:
                bar_b0 = int(raw["bar_b0"])
                bar_b1 = int(raw["bar_b1"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"annotation line {line_number}: invalid bar span") from error
            if not 0 <= bar_b0 <= bar_b1 < ORIGINAL_WINDOW:
                raise ValueError(f"annotation line {line_number}: box span is outside 200 bars")
            xywhn = tuple(
                _safe_float(raw[name], field=f"line {line_number} {name}")
                for name in ("yolo_xc", "yolo_yc", "yolo_w", "yolo_h")
            )
            xc, yc, width, height = xywhn
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"annotation line {line_number}: invalid normalized box")
            rows.append(
                ManualBox(
                    box_id=box_id,
                    symbol=str(raw["symbol"]).strip(),
                    stem=str(raw["stem"]).strip(),
                    cut_time=cut_time,
                    bar_b0=bar_b0,
                    bar_b1=bar_b1,
                    xywhn=xywhn,  # type: ignore[arg-type]
                )
            )
    if not rows:
        raise ValueError("annotation sheet contains no owner_side=short rows")
    return rows, fieldnames


def _select_source_files(source_dir: Path, symbols: Iterable[str]) -> dict[str, Path]:
    required = set(symbols)
    choices: dict[str, tuple[int, Path]] = {}
    for path in sorted(source_dir.glob("okx_*_15m_*.csv")):
        matched = _SOURCE_RE.fullmatch(path.name)
        if matched is None:
            continue
        symbol = matched.group("symbol")
        if symbol not in required:
            continue
        declared_rows = int(matched.group("rows"))
        current = choices.get(symbol)
        if current is None or (declared_rows, path.name) > (current[0], current[1].name):
            choices[symbol] = (declared_rows, path)
    missing = sorted(required - set(choices))
    if missing:
        raise FileNotFoundError(
            "no 15m OHLCV source found for: " + ", ".join(missing[:20])
        )
    return {symbol: item[1] for symbol, item in choices.items()}


def _parse_source_prefix(
    source: Path,
    destination: Path,
    *,
    cutoff: pd.Timestamp,
) -> dict[str, object]:
    """Copy only rows before cutoff; boundary OHLC is never parsed or written."""
    cutoff_ms = int(cutoff.value // 1_000_000)
    origin_digest = hashlib.sha256()
    first_ms: int | None = None
    last_ms: int | None = None
    count = 0
    boundary_checked = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open(
        "w", newline="", encoding="utf-8"
    ) as destination_handle:
        header_raw = source_handle.readline()
        if not header_raw:
            raise ValueError(f"empty OHLCV file: {source}")
        origin_digest.update(header_raw)
        header = next(csv.reader([header_raw.decode("utf-8").rstrip("\r\n")]))
        required = set(_SNAPSHOT_COLUMNS) | {"open_time"}
        if not required.issubset(header):
            raise ValueError(f"{source}: missing OHLCV/open_time columns")
        indices = {name: header.index(name) for name in required}
        writer = csv.DictWriter(destination_handle, fieldnames=_SNAPSHOT_COLUMNS)
        writer.writeheader()
        previous_ms: int | None = None
        for raw_bytes in source_handle:
            text = raw_bytes.decode("utf-8")
            fields = next(csv.reader([text.rstrip("\r\n")]))
            if len(fields) < len(header):
                raise ValueError(f"{source}: short CSV row before boundary")
            try:
                timestamp_ms = int(fields[indices["ts"]])
            except ValueError as error:
                raise ValueError(f"{source}: invalid ts before boundary") from error
            if timestamp_ms >= cutoff_ms:
                boundary_checked = True
                break
            if previous_ms is not None and timestamp_ms - previous_ms != BAR_MILLISECONDS:
                raise ValueError(f"{source}: pre-holdout rows are not continuous 15m candles")
            values: dict[str, str] = {"ts": str(timestamp_ms)}
            for name in ("open", "high", "low", "close", "volume"):
                raw_value = fields[indices[name]]
                value = _safe_float(raw_value, field=f"{source.name} {name}")
                if name != "volume" and value <= 0:
                    raise ValueError(f"{source}: non-positive {name} before boundary")
                values[name] = raw_value
            writer.writerow(values)
            origin_digest.update(raw_bytes)
            first_ms = timestamp_ms if first_ms is None else first_ms
            last_ms = timestamp_ms
            previous_ms = timestamp_ms
            count += 1
    if not boundary_checked:
        raise ValueError(f"{source}: source does not reach the frozen holdout boundary")
    if count == 0 or first_ms is None or last_ms is None:
        raise ValueError(f"{source}: no pre-holdout rows")
    stat = destination.stat()
    first = pd.Timestamp(first_ms, unit="ms", tz="UTC")
    last = pd.Timestamp(last_ms, unit="ms", tz="UTC")
    return {
        "path": destination.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(destination),
        "row_count": count,
        "first_open_time": utc_iso(first),
        "last_open_time": utc_iso(last),
        "last_closed_at": utc_iso(last + BAR_DELTA),
        "origin_path": str(source.resolve()),
        "origin_size_bytes": source.stat().st_size,
        "origin_mtime_ns": source.stat().st_mtime_ns,
        "origin_preholdout_prefix_sha256": origin_digest.hexdigest(),
        "boundary_timestamp_checked": True,
        "post_cutoff_ohlcv_rows_materialized": 0,
    }


def create_preholdout_snapshot(
    *,
    review_sheet: str | Path,
    source_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, object]:
    """Recover annotations and immutable local candle prefixes into yolo-xx."""
    sheet = Path(review_sheet).resolve()
    source_root = Path(source_dir).resolve()
    output = Path(out_dir).resolve()
    _ensure_new_output(output)
    boxes, fieldnames = load_short_annotations(sheet)
    selected = _select_source_files(source_root, (box.symbol for box in boxes))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        annotation_copy = staging / "owner_short_annotations.csv"
        wanted = {box.box_id for box in boxes}
        with sheet.open(newline="", encoding="utf-8") as source_handle, annotation_copy.open(
            "w", newline="", encoding="utf-8"
        ) as destination_handle:
            reader = csv.DictReader(source_handle)
            writer = csv.DictWriter(destination_handle, fieldnames=fieldnames)
            writer.writeheader()
            copied = 0
            for raw in reader:
                if raw.get("box_id") in wanted:
                    writer.writerow(raw)
                    copied += 1
        if copied != len(boxes):
            raise AssertionError("short annotation copy count changed")

        entries: list[dict[str, object]] = []
        for symbol in sorted(selected):
            source = selected[symbol]
            declared = int(_SOURCE_RE.fullmatch(source.name).group("rows"))  # type: ignore[union-attr]
            destination = staging / f"okx_{symbol}_15m_{declared}.csv"
            entry = _parse_source_prefix(source, destination, cutoff=HOLDOUT_START)
            entry["symbol"] = symbol
            entries.append(entry)

        manifest = {
            "schema_version": 1,
            "manifest_type": "yolo_xx_source_snapshot",
            "immutable": True,
            "source_dir": str(output),
            "timeframe": TIMEFRAME,
            "cutoff_exclusive": utc_iso(HOLDOUT_START),
            "annotation_source": {
                "original_path": str(sheet),
                "original_sha256": sha256_file(sheet),
                "copy": annotation_copy.name,
                "copy_sha256": sha256_file(annotation_copy),
                "filter": "owner_side == short",
                "rows": len(boxes),
                "symbols": len({box.symbol for box in boxes}),
                "min_cut_time": utc_iso(min(box.cut_time for box in boxes)),
                "max_cut_time": utc_iso(max(box.cut_time for box in boxes)),
            },
            "files": entries,
            "safety": {
                "holdout_read": False,
                "boundary_timestamp_only_checked": True,
                "post_cutoff_ohlcv_rows_materialized": 0,
            },
        }
        _write_json(staging / "source_snapshot.json", manifest)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    summary = {
        "snapshot_dir": str(output),
        "manifest": str(output / "source_snapshot.json"),
        "manifest_sha256": sha256_file(output / "source_snapshot.json"),
        "annotation_rows": len(boxes),
        "annotation_sha256": sha256_file(output / "owner_short_annotations.csv"),
        "symbols": len(selected),
        "preholdout_rows": sum(int(item["row_count"]) for item in entries),
        "post_cutoff_ohlcv_rows_materialized": 0,
    }
    _write_json(output / "snapshot_summary.json", summary)
    return summary


def _record_symbol(record: SnapshotFile) -> str:
    matched = _SOURCE_RE.fullmatch(record.path.name)
    if matched is None:
        raise ValueError(f"snapshot filename does not encode a 15m symbol: {record.path.name}")
    return matched.group("symbol")


def _right_context(box_id: str, choices: tuple[int, ...]) -> int:
    digest = hashlib.sha256(box_id.encode("utf-8")).digest()
    return choices[int.from_bytes(digest[:8], "big") % len(choices)]


def _make_requests(
    boxes: Sequence[ManualBox],
    *,
    layout: str,
    window: int,
    split_at: pd.Timestamp,
    right_contexts: tuple[int, ...],
) -> tuple[list[RenderRequest], int]:
    candidates: list[RenderRequest] = []
    if layout == "original":
        if window != ORIGINAL_WINDOW:
            raise ValueError("original layout requires --window 200")
        grouped: dict[str, list[ManualBox]] = defaultdict(list)
        for box in boxes:
            grouped[box.stem].append(box)
        for stem, group in sorted(grouped.items()):
            symbols = {box.symbol for box in group}
            starts = {box.original_window_start for box in group}
            if len(symbols) != 1 or len(starts) != 1:
                raise ValueError(f"{stem}: boxes disagree on source window")
            start = next(iter(starts))
            candidates.append(
                RenderRequest(
                    sample_id=stem,
                    symbol=next(iter(symbols)),
                    window_start=start,
                    window_end_open=start + (window - 1) * BAR_DELTA,
                    split="",
                    boxes=tuple(sorted(group, key=lambda item: item.box_id)),
                    right_context_bars=None,
                )
            )
    elif layout == "staggered_causal":
        if window <= 0:
            raise ValueError("window must be positive")
        if not right_contexts or any(value < 0 for value in right_contexts):
            raise ValueError("right contexts must be non-negative")
        for box in boxes:
            context = _right_context(box.box_id, right_contexts)
            if box.width_bars + context > window:
                raise ValueError(f"{box.box_id}: box and right context do not fit window")
            end = box.cut_time + context * BAR_DELTA
            candidates.append(
                RenderRequest(
                    sample_id=box.box_id,
                    symbol=box.symbol,
                    window_start=end - (window - 1) * BAR_DELTA,
                    window_end_open=end,
                    split="",
                    boxes=(box,),
                    right_context_bars=context,
                )
            )
    else:
        raise ValueError(f"unsupported layout: {layout}")

    assigned: list[RenderRequest] = []
    dropped = 0
    for request in candidates:
        if request.available_at < split_at:
            split = "train"
        elif request.window_start >= split_at:
            split = "val"
        else:
            dropped += 1
            continue
        assigned.append(
            RenderRequest(
                sample_id=request.sample_id,
                symbol=request.symbol,
                window_start=request.window_start,
                window_end_open=request.window_end_open,
                split=split,
                boxes=request.boxes,
                right_context_bars=request.right_context_bars,
            )
        )
    if not any(item.split == "train" for item in assigned) or not any(
        item.split == "val" for item in assigned
    ):
        raise ValueError("global split must produce both train and val samples")
    return assigned, dropped


def _price_at(transform: ChartTransform, y_pixel: float) -> float:
    span = max(transform.price_max - transform.price_min, 1e-12)
    return transform.price_max - (y_pixel - transform.top) / transform.plot_h * span


def _remap_short_box(
    box: ManualBox,
    *,
    frame: pd.DataFrame,
    source_start_index: int,
    transform: ChartTransform,
    time_to_index: dict[int, int],
) -> tuple[tuple[float, float, float, float], int, int, bool]:
    cut_index = time_to_index[box.cut_time.value]
    original_start = cut_index - box.bar_b1
    original = frame.iloc[original_start : original_start + ORIGINAL_WINDOW].reset_index(drop=True)
    if len(original) != ORIGINAL_WINDOW:
        raise ValueError(f"{box.box_id}: original 200-bar source window is incomplete")
    # The remap needs only geometry.  Building a full throwaway raster here used
    # to double chart drawing work during large paired builds.
    original_transform = make_chart_transform(original, ma_periods=(20, 60, 120))
    _, yc, _, height = box.xywhn
    y1 = (yc - height / 2) * original_transform.height
    y2 = (yc + height / 2) * original_transform.height
    price_hi = _price_at(original_transform, min(y1, y2))
    price_lo = _price_at(original_transform, max(y1, y2))

    source_box_start = cut_index - (box.bar_b1 - box.bar_b0)
    source_box_end = cut_index
    local_start = source_box_start - source_start_index
    local_end = source_box_end - source_start_index
    if not 0 <= local_start <= local_end < transform.n_bars:
        raise ValueError(f"{box.box_id}: box does not fit remapped short window")

    fallback = False
    if price_hi < transform.price_min or price_lo > transform.price_max:
        region = frame.iloc[source_box_start : source_box_end + 1]
        price_hi = float(region["high"].max())
        price_lo = float(region["low"].min())
        fallback = True
    price_hi = min(price_hi, transform.price_max)
    price_lo = max(price_lo, transform.price_min)
    if price_hi <= price_lo:
        region = frame.iloc[source_box_start : source_box_end + 1]
        price_hi = min(float(region["high"].max()), transform.price_max)
        price_lo = max(float(region["low"].min()), transform.price_min)
        fallback = True

    x1 = max(0.0, transform.x_at(local_start) - transform.candle_half_w)
    x2 = min(float(transform.width), transform.x_at(local_end) + transform.candle_half_w)
    py1 = float(np.clip(transform.y_at(price_hi), 0, transform.height - 1))
    py2 = float(np.clip(transform.y_at(price_lo), 1, transform.height))
    if x2 - x1 < 4 or py2 - py1 < 4:
        raise ValueError(f"{box.box_id}: remapped box is too small")
    return (
        (x1 + x2) / 2 / transform.width,
        (py1 + py2) / 2 / transform.height,
        (x2 - x1) / transform.width,
        (py2 - py1) / transform.height,
    ), source_box_start, source_box_end, fallback


def _draw_gallery(dataset: Path, samples: Sequence[dict[str, object]], *, count: int = 24) -> str:
    gallery = dataset / "preview"
    gallery.mkdir(parents=True, exist_ok=True)
    selected = sorted(samples, key=lambda item: str(item["id"]))[:count]
    cards: list[str] = []
    for sample in selected:
        image_path = dataset / str(sample["image"])
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for box in sample["boxes"]:  # type: ignore[union-attr]
            xc, yc, bw, bh = box["xywhn"]
            x1, x2 = int((xc - bw / 2) * width), int((xc + bw / 2) * width)
            y1, y2 = int((yc - bh / 2) * height), int((yc + bh / 2) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 170, 0), 3, cv2.LINE_AA)
        name = f"{sample['id']}.jpg"
        cv2.imwrite(str(gallery / name), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cards.append(
            f"<figure><img src='{name}'><figcaption>{sample['id']} | "
            f"{sample['split']} | available {sample['available_at']}</figcaption></figure>"
        )
    html = """<!doctype html><meta charset='utf-8'><title>manual short preview</title>
<style>body{font:14px system-ui;background:#eee}main{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
figure{margin:0;background:white;padding:10px}img{width:100%;height:auto}figcaption{margin-top:6px}</style><main>"""
    (gallery / "index.html").write_text(html + "".join(cards) + "</main>\n", encoding="utf-8")
    return str((gallery / "index.html").resolve())


def build_manual_short_dataset(
    *,
    snapshot_dir: str | Path,
    out_dir: str | Path,
    layout: str,
    window: int,
    split_at: str = DEFAULT_SPLIT_AT,
    right_contexts: Sequence[int] = DEFAULT_RIGHT_CONTEXTS,
) -> dict[str, object]:
    """Render one strong-audit dataset from the recovered immutable snapshot."""
    snapshot_root = Path(snapshot_dir).resolve()
    output = Path(out_dir).resolve()
    _ensure_new_output(output)
    manifest_path = snapshot_root / "source_snapshot.json"
    annotation_path = snapshot_root / "owner_short_annotations.csv"
    boxes, _ = load_short_annotations(annotation_path)
    split_timestamp = utc_timestamp(split_at, field="split_at")
    if split_timestamp >= HOLDOUT_START:
        raise ValueError("split_at must be before holdout")
    contexts = tuple(sorted(set(int(value) for value in right_contexts)))
    requests, dropped_cross_split = _make_requests(
        boxes,
        layout=layout,
        window=window,
        split_at=split_timestamp,
        right_contexts=contexts,
    )

    snapshot = load_source_manifest(
        manifest_path,
        expected_source_dir=snapshot_root,
        expected_timeframe=TIMEFRAME,
        end_before=HOLDOUT_START,
    )
    verify_snapshot_identity(snapshot)
    records = {_record_symbol(record): record for record in snapshot.files}
    missing = sorted({request.symbol for request in requests} - set(records))
    if missing:
        raise ValueError("snapshot is missing requested symbols: " + ", ".join(missing))

    by_symbol: dict[str, list[RenderRequest]] = defaultdict(list)
    for request in requests:
        by_symbol[request.symbol].append(request)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for split in ("train", "val"):
            (staging / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging / "labels" / split).mkdir(parents=True, exist_ok=True)

        samples: list[dict[str, object]] = []
        stats: Counter[str] = Counter()
        box_rights: list[float] = []
        right_context_counts: Counter[int] = Counter()
        y_fallbacks = 0
        used_records: set[str] = set()
        for symbol in sorted(by_symbol):
            record = records[symbol]
            verify_snapshot_file(record)
            frame = load_ohlcv_csv(record.path, timeframe=TIMEFRAME, strict_cadence=True)
            verify_loaded_frame(frame, record, timeframe=TIMEFRAME)
            frame = add_mas(frame, periods=(20, 60, 120))
            time_to_index = {
                int(pd.Timestamp(value).value): index
                for index, value in enumerate(frame["open_time"])
            }
            for request in sorted(by_symbol[symbol], key=lambda item: item.sample_id):
                start_ns = request.window_start.value
                end_ns = request.window_end_open.value
                if start_ns not in time_to_index or end_ns not in time_to_index:
                    raise ValueError(f"{request.sample_id}: requested window is missing")
                source_start = time_to_index[start_ns]
                source_end = time_to_index[end_ns]
                if source_end - source_start + 1 != window:
                    raise ValueError(f"{request.sample_id}: requested window is not contiguous")
                subframe = frame.iloc[source_start : source_end + 1].reset_index(drop=True)
                image_path = staging / "images" / request.split / f"{request.sample_id}.png"
                _, transform = render_chart(
                    subframe,
                    out_path=image_path,
                    ma_periods=(20, 60, 120),
                )
                manifest_boxes: list[dict[str, object]] = []
                label_lines: list[str] = []
                for box in request.boxes:
                    if layout == "original":
                        xywhn = box.xywhn
                        source_box_end = time_to_index[box.cut_time.value]
                        source_box_start = source_box_end - (box.bar_b1 - box.bar_b0)
                        fallback = False
                    else:
                        xywhn, source_box_start, source_box_end, fallback = _remap_short_box(
                            box,
                            frame=frame,
                            source_start_index=source_start,
                            transform=transform,
                            time_to_index=time_to_index,
                        )
                    y_fallbacks += int(fallback)
                    normalized = tuple(round(float(value), 6) for value in xywhn)
                    label_lines.append(
                        f"{CLASS_ID} " + " ".join(f"{value:.6f}" for value in normalized)
                    )
                    box_rights.append(normalized[0] + normalized[2] / 2)
                    box_start_time = pd.Timestamp(frame.iloc[source_box_start]["open_time"])
                    box_end_time = pd.Timestamp(frame.iloc[source_box_end]["open_time"])
                    manifest_boxes.append(
                        {
                            "class_id": CLASS_ID,
                            "class_name": CLASS_NAME,
                            "annotation_box_id": box.box_id,
                            "annotation_stem": box.stem,
                            "segment": {
                                "start_index_in_window": source_box_start - source_start,
                                "end_index_in_window": source_box_end - source_start,
                                "start_index_in_source": source_box_start,
                                "end_index_in_source": source_box_end,
                            },
                            "box_start_time": utc_iso(box_start_time),
                            "box_end_time": utc_iso(box_end_time),
                            "box_end_close_time": utc_iso(box_end_time + BAR_DELTA),
                            "available_at": utc_iso(request.available_at),
                            "right_context_bars": request.right_context_bars,
                            "y_price_fallback": fallback,
                            "xywhn": list(normalized),
                        }
                    )
                label_path = staging / "labels" / request.split / f"{request.sample_id}.txt"
                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                image_relative = image_path.relative_to(staging).as_posix()
                label_relative = label_path.relative_to(staging).as_posix()
                samples.append(
                    {
                        "id": request.sample_id,
                        "symbol": symbol,
                        "split": request.split,
                        "source_file": str(record.path),
                        "source_sha256": record.sha256,
                        "source_start_index": source_start,
                        "source_end_index": source_end,
                        "image": image_relative,
                        "image_sha256": sha256_file(image_path),
                        "label": label_relative,
                        "label_sha256": sha256_file(label_path),
                        "window_start_time": utc_iso(request.window_start),
                        "window_end_open_time": utc_iso(request.window_end_open),
                        "window_end_close_time": utc_iso(request.available_at),
                        "available_at": utc_iso(request.available_at),
                        "right_context_bars": request.right_context_bars,
                        "n_boxes": len(manifest_boxes),
                        "boxes": manifest_boxes,
                    }
                )
                stats[f"{request.split}_images"] += 1
                stats[f"{request.split}_boxes"] += len(manifest_boxes)
                if request.right_context_bars is not None:
                    right_context_counts[request.right_context_bars] += 1
                used_records.add(symbol)
            verify_snapshot_file(record)

        max_train = max(
            pd.Timestamp(item["available_at"])
            for item in samples
            if item["split"] == "train"
        )
        min_val = min(
            pd.Timestamp(item["window_start_time"])
            for item in samples
            if item["split"] == "val"
        )
        if not max_train < min_val:
            raise AssertionError("global train/val split invariant failed")

        data_yaml = (
            f"path: {output}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            f"  0: {CLASS_NAME}\n"
        )
        (staging / "data.yaml").write_text(data_yaml, encoding="utf-8")
        snapshot_copy = staging / "source_snapshot_manifest.json"
        snapshot_copy.write_bytes(manifest_path.read_bytes())
        detection_spec = DetectionSpec().as_dict()
        right_series = pd.Series(box_rights, dtype=float)
        dataset_manifest = {
            "schema_version": 2,
            "manifest_type": "yolo_xx_dataset",
            "created_from": "owner_reviewed_short_boxes_and_authenticated_preholdout_ohlcv",
            "source_dir": str(snapshot_root),
            "source_snapshot": {
                "manifest": snapshot_copy.relative_to(staging).as_posix(),
                "sha256": snapshot.manifest_sha256,
                "cutoff_exclusive": utc_iso(snapshot.cutoff_exclusive),
                "timeframe": snapshot.timeframe,
            },
            "source_files": [records[symbol].as_manifest_dict() for symbol in sorted(used_records)],
            "annotation_source": {
                "path": str(annotation_path),
                "sha256": sha256_file(annotation_path),
                "filter": "owner_side == short",
                "input_boxes": len(boxes),
            },
            "end_before": utc_iso(HOLDOUT_START),
            "split_at": utc_iso(split_timestamp),
            "dropped_cross_split": dropped_cross_split,
            "global_split_invariant": {
                "max_train_available_at": utc_iso(max_train),
                "min_val_window_start_time": utc_iso(min_val),
                "holds": True,
            },
            "layout": layout,
            "position_policy": (
                "preserve_original_owner_box_coordinates"
                if layout == "original"
                else "deterministic_balanced_right_context_no_geometric_augmentation"
            ),
            "right_context_choices_bars": list(contexts) if layout != "original" else [],
            "right_context_counts": {str(key): value for key, value in sorted(right_context_counts.items())},
            "box_right_fraction": {
                "min": round(float(right_series.min()), 6),
                "p25": round(float(right_series.quantile(0.25)), 6),
                "median": round(float(right_series.median()), 6),
                "p75": round(float(right_series.quantile(0.75)), 6),
                "max": round(float(right_series.max()), 6),
                "at_or_above_0_95": int((right_series >= 0.95).sum()),
            },
            "physical_window_minutes": window * BAR_MINUTES,
            "window_bars": window,
            "stride_bars": window,
            "pixels_per_bar": round((IMG_WIDTH - 2 * MARGIN) / max(window - 1, 1), 6),
            "resolution_risks": [],
            "strict_cadence": True,
            "availability_contract": (
                "Every box is available only at its rendered window_end_close_time; "
                "box_end_time is descriptive and never substitutes for availability."
            ),
            "detection_spec": detection_spec,
            "samples": sorted(samples, key=lambda item: str(item["image"])),
        }
        _write_json(staging / "dataset_manifest.json", dataset_manifest)
        summary: dict[str, object] = {
            "schema_version": 2,
            "dataset": str(output),
            "layout": layout,
            "window_bars": window,
            "physical_window_minutes": window * BAR_MINUTES,
            "pixels_per_bar": dataset_manifest["pixels_per_bar"],
            "split_at": utc_iso(split_timestamp),
            "dropped_cross_split": dropped_cross_split,
            "input_annotation_boxes": len(boxes),
            "symbols": len(used_records),
            "source_snapshot_sha256": snapshot.manifest_sha256,
            "annotation_sha256": sha256_file(annotation_path),
            "right_context_counts": dataset_manifest["right_context_counts"],
            "box_right_fraction": dataset_manifest["box_right_fraction"],
            "y_price_fallbacks": y_fallbacks,
            "background_images": 0,
            "training_readiness_warning": (
                "Positive-only recovery artifact; add an audited position-matched negative pool "
                "before claiming detector precision."
            ),
            **{key: int(value) for key, value in sorted(stats.items())},
        }
        _write_json(staging / "dataset_summary.json", summary)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    audit = audit_dataset(output)
    _write_json(output / "dataset_audit.json", audit)
    if not audit["valid"]:
        raise ValueError("built dataset failed strong audit: " + "; ".join(audit["errors"][:5]))
    summary["dataset_audit_valid"] = True
    summary["dataset_manifest_sha256"] = sha256_file(output / "dataset_manifest.json")
    summary["gallery"] = _draw_gallery(output, dataset_manifest["samples"])  # type: ignore[arg-type]
    _write_json(output / "dataset_summary.json", summary)
    return summary


def make_plan(
    *,
    action: str,
    review_sheet: Path | None,
    source_dir: Path | None,
    snapshot_dir: Path,
    out_dir: Path | None,
    layout: str | None,
    window: int | None,
    split_at: str | None,
    right_contexts: Sequence[int] = DEFAULT_RIGHT_CONTEXTS,
) -> dict[str, object]:
    """Return a no-read/no-write command plan for dry-run review."""
    if action == "snapshot" and (review_sheet is None or source_dir is None):
        raise ValueError("snapshot plan requires review_sheet and source_dir")
    if action == "build" and (out_dir is None or layout is None or window is None):
        raise ValueError("build plan requires out_dir/layout/window")
    if window is not None and window <= 0:
        raise ValueError("window must be positive")
    if split_at is not None:
        timestamp = utc_timestamp(split_at, field="split_at")
        if timestamp >= HOLDOUT_START:
            raise ValueError("split_at must be before holdout")
    return {
        "dry_run": True,
        "action": action,
        "review_sheet": str(review_sheet.resolve()) if review_sheet else None,
        "source_dir": str(source_dir.resolve()) if source_dir else None,
        "snapshot_dir": str(snapshot_dir.resolve()),
        "out_dir": str(out_dir.resolve()) if out_dir else None,
        "layout": layout,
        "window_bars": window,
        "physical_window_minutes": window * BAR_MINUTES if window else None,
        "pixels_per_bar": (
            round((IMG_WIDTH - 2 * MARGIN) / max(window - 1, 1), 6) if window else None
        ),
        "split_at": split_at,
        "right_contexts": list(right_contexts) if layout == "staggered_causal" else [],
        "holdout_cutoff": utc_iso(HOLDOUT_START),
        "training": False,
        "network": False,
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
    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--review-sheet", required=True, type=Path)
    snapshot_parser.add_argument("--source-dir", required=True, type=Path)
    snapshot_parser.add_argument("--out", required=True, type=Path)
    snapshot_parser.add_argument("--dry-run", action="store_true")

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--snapshot-dir", required=True, type=Path)
    build_parser.add_argument("--out", required=True, type=Path)
    build_parser.add_argument(
        "--layout", choices=("original", "staggered_causal"), required=True
    )
    build_parser.add_argument("--window", required=True, type=int)
    build_parser.add_argument("--split-at", default=DEFAULT_SPLIT_AT)
    build_parser.add_argument(
        "--right-contexts", type=_parse_contexts, default=DEFAULT_RIGHT_CONTEXTS
    )
    build_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.action == "snapshot":
        if args.dry_run:
            payload = make_plan(
                action="snapshot",
                review_sheet=args.review_sheet,
                source_dir=args.source_dir,
                snapshot_dir=args.out,
                out_dir=None,
                layout=None,
                window=None,
                split_at=None,
            )
        else:
            payload = create_preholdout_snapshot(
                review_sheet=args.review_sheet,
                source_dir=args.source_dir,
                out_dir=args.out,
            )
    else:
        if args.dry_run:
            payload = make_plan(
                action="build",
                review_sheet=None,
                source_dir=None,
                snapshot_dir=args.snapshot_dir,
                out_dir=args.out,
                layout=args.layout,
                window=args.window,
                split_at=args.split_at,
                right_contexts=args.right_contexts,
            )
        else:
            payload = build_manual_short_dataset(
                snapshot_dir=args.snapshot_dir,
                out_dir=args.out,
                layout=args.layout,
                window=args.window,
                split_at=args.split_at,
                right_contexts=args.right_contexts,
            )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
