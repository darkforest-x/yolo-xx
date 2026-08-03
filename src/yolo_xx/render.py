"""Render candlesticks and six moving averages into fixed-semantics YOLO images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .data import ALL_MA_COLS

CANDLE_GREEN = (129, 153, 8)
CANDLE_RED = (69, 54, 242)
WICK = (118, 118, 118)
BG = (255, 255, 255)
MA_COLORS = {
    "sma20": (196, 114, 32),
    "sma60": (176, 168, 92),
    "sma120": (140, 110, 110),
    "ema20": (36, 96, 240),
    "ema60": (60, 160, 250),
    "ema120": (150, 70, 200),
}

IMG_WIDTH = 1280
IMG_HEIGHT = 742
MARGIN = 12
MIN_REL_SPAN = 0.06


@dataclass(frozen=True)
class ChartTransform:
    """Map a bar index and price into one rendered image's pixel coordinates."""

    n_bars: int
    width: int
    height: int
    left: int
    top: int
    plot_w: int
    plot_h: int
    price_min: float
    price_max: float
    candle_half_w: int

    def x_at(self, index: int) -> int:
        if self.n_bars <= 1:
            return self.left
        return int(self.left + (index / (self.n_bars - 1)) * self.plot_w)

    def y_at(self, price: float) -> int:
        span = max(self.price_max - self.price_min, 1e-12)
        return int(self.top + (self.price_max - float(price)) / span * self.plot_h)


def _price_bounds(frame: pd.DataFrame, pad: float = 0.06) -> tuple[float, float]:
    series = [frame["low"], frame["high"]]
    series.extend(frame[column] for column in ALL_MA_COLS if column in frame.columns)
    values = pd.concat(series).dropna()
    if values.empty:
        raise ValueError("cannot render a frame without finite price values")
    low, high = float(values.min()), float(values.max())
    middle = (high + low) / 2
    span = max(high - low, abs(middle) * MIN_REL_SPAN, 1e-9)
    low, high = middle - span / 2, middle + span / 2
    return low - span * pad, high + span * pad


def make_chart_transform(
    frame: pd.DataFrame,
    *,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
) -> ChartTransform:
    """Build the deterministic chart transform without drawing the image."""
    if frame.empty:
        raise ValueError("cannot render an empty frame")
    frame = frame.reset_index(drop=True)
    count = len(frame)
    left = top = MARGIN
    plot_w, plot_h = width - 2 * MARGIN, height - 2 * MARGIN
    if plot_w <= 0 or plot_h <= 0:
        raise ValueError("image dimensions must be larger than twice the margin")
    price_min, price_max = _price_bounds(frame)
    candle_half_w = max(1, int(plot_w / count * 0.34))
    return ChartTransform(
        n_bars=count,
        width=width,
        height=height,
        left=left,
        top=top,
        plot_w=plot_w,
        plot_h=plot_h,
        price_min=price_min,
        price_max=price_max,
        candle_half_w=candle_half_w,
    )


def render_chart(
    frame: pd.DataFrame,
    *,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    out_path: str | Path | None = None,
) -> tuple[np.ndarray, ChartTransform]:
    """Render one OHLCV window whose moving-average columns already exist."""
    frame = frame.reset_index(drop=True)
    transform = make_chart_transform(frame, width=width, height=height)
    image = np.full((height, width, 3), BG, dtype=np.uint8)

    for index, row in frame.iterrows():
        x = transform.x_at(index)
        y_high, y_low = transform.y_at(row["high"]), transform.y_at(row["low"])
        y_open, y_close = transform.y_at(row["open"]), transform.y_at(row["close"])
        color = CANDLE_GREEN if float(row["close"]) >= float(row["open"]) else CANDLE_RED
        cv2.line(image, (x, y_high), (x, y_low), WICK, 1, cv2.LINE_AA)
        y1, y2 = min(y_open, y_close), max(y_open, y_close)
        if y2 - y1 < 2:
            y2 = y1 + 2
        cv2.rectangle(
            image,
            (x - transform.candle_half_w, y1),
            (x + transform.candle_half_w, y2),
            color,
            -1,
            cv2.LINE_AA,
        )

    for column in ALL_MA_COLS:
        if column not in frame.columns:
            continue
        points = [
            (transform.x_at(index), transform.y_at(float(value)))
            for index, value in enumerate(frame[column])
            if pd.notna(value)
        ]
        if len(points) >= 2:
            cv2.polylines(
                image,
                [np.asarray(points, dtype=np.int32)],
                False,
                MA_COLORS[column],
                1,
                cv2.LINE_AA,
            )

    if out_path is not None:
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output), image):
            raise OSError(f"failed to write chart: {output}")
    return image, transform
