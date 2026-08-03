"""Create single-class YOLO boxes for dense moving-average chart regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import ALL_MA_COLS
from .render import ChartTransform

FAST_SPREAD_MAX = 0.0028
FULL_SPREAD_MAX = 0.0055
MIN_DENSE_BARS = 5
MERGE_GAP_BARS = 2
X_PAD_PX = 6
Y_PAD_FRAC = 0.35
MAX_DENSE_BARS = 12
CLASS_ID = 0
CLASS_NAME = "dense_cluster"


@dataclass(frozen=True)
class DenseSegment:
    """Inclusive dense interval inside one rendered chart window."""

    start: int
    end: int


def _tightest_window(
    full_spread: np.ndarray,
    start: int,
    end: int,
    max_bars: int,
) -> DenseSegment:
    length = end - start + 1
    if length <= max_bars:
        return DenseSegment(start, end)
    best_index = start
    best_score = float("inf")
    for index in range(start, end - max_bars + 2):
        values = full_spread[index : index + max_bars]
        if np.isnan(values).all():
            continue
        score = float(np.nanmean(values))
        if score < best_score:
            best_score = score
            best_index = index
    return DenseSegment(best_index, best_index + max_bars - 1)


def find_dense_segments(
    frame: pd.DataFrame,
    *,
    fast_max: float = FAST_SPREAD_MAX,
    full_max: float = FULL_SPREAD_MAX,
    min_bars: int = MIN_DENSE_BARS,
    merge_gap: int = MERGE_GAP_BARS,
    max_bars: int = MAX_DENSE_BARS,
) -> list[DenseSegment]:
    """Find and compact dense MA runs using current-bar spread columns only."""
    if min_bars <= 0 or max_bars <= 0:
        raise ValueError("min_bars and max_bars must be positive")
    full_spread = pd.to_numeric(frame["full_spread"], errors="coerce").to_numpy()
    dense = (
        (pd.to_numeric(frame["fast_spread"], errors="coerce") <= fast_max)
        & (full_spread <= full_max)
    ).to_numpy()
    indices = np.flatnonzero(dense)
    if len(indices) == 0:
        return []
    runs: list[list[int]] = [[int(indices[0]), int(indices[0])]]
    for index in indices[1:]:
        if int(index) - runs[-1][1] <= merge_gap + 1:
            runs[-1][1] = int(index)
        else:
            runs.append([int(index), int(index)])
    return [
        _tightest_window(full_spread, start, end, max_bars)
        for start, end in runs
        if end - start + 1 >= min_bars
    ]


def segment_to_bbox(
    frame: pd.DataFrame,
    segment: DenseSegment,
    transform: ChartTransform,
    *,
    x_pad_px: int = X_PAD_PX,
    y_pad_frac: float = Y_PAD_FRAC,
) -> tuple[float, float, float, float] | None:
    """Map a dense segment to normalized YOLO `(xc, yc, width, height)`."""
    region = frame.iloc[segment.start : segment.end + 1]
    values = [
        float(value)
        for column in ALL_MA_COLS
        if column in region.columns
        for value in region[column]
        if pd.notna(value)
    ]
    if not values:
        return None
    high, low = max(values), min(values)
    pad = max(
        (high - low) * y_pad_frac,
        (transform.price_max - transform.price_min) * 0.004,
    )
    x1 = transform.x_at(segment.start) - transform.candle_half_w - x_pad_px
    x2 = transform.x_at(segment.end) + transform.candle_half_w + x_pad_px
    y1 = transform.y_at(high + pad)
    y2 = transform.y_at(low - pad)
    x1 = float(np.clip(x1, 0, transform.width - 1))
    x2 = float(np.clip(x2, 1, transform.width))
    y1 = float(np.clip(y1, 0, transform.height - 1))
    y2 = float(np.clip(y2, 1, transform.height))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return (
        (x1 + x2) / 2 / transform.width,
        (y1 + y2) / 2 / transform.height,
        (x2 - x1) / transform.width,
        (y2 - y1) / transform.height,
    )


def label_window(
    frame: pd.DataFrame,
    transform: ChartTransform,
) -> list[tuple[float, float, float, float]]:
    """Return every rule-generated box for one rendered chart window."""
    boxes = []
    for segment in find_dense_segments(frame):
        box = segment_to_bbox(frame, segment, transform)
        if box is not None:
            boxes.append(box)
    return boxes


def to_yolo_lines(boxes: list[tuple[float, float, float, float]]) -> str:
    """Serialize normalized boxes in single-class YOLO text format."""
    return "".join(
        f"{CLASS_ID} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}\n"
        for xc, yc, width, height in boxes
    )
