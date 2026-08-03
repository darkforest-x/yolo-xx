from __future__ import annotations

import numpy as np
import pandas as pd

from yolo_xx.data import ALL_MA_COLS, add_mas
from yolo_xx.labels import DenseSegment, find_dense_segments, segment_to_bbox
from yolo_xx.render import ChartTransform, render_chart


def _frame(count: int = 200) -> pd.DataFrame:
    close = 100 + np.sin(np.arange(count) / 20) * 0.1
    return pd.DataFrame(
        {
            "open": close - 0.02,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
            "volume": np.ones(count),
        }
    )


def test_add_mas_and_render_have_fixed_geometry(tmp_path) -> None:
    frame = add_mas(_frame())
    assert set(ALL_MA_COLS).issubset(frame.columns)
    output = tmp_path / "chart.png"
    image, transform = render_chart(frame, width=320, height=180, out_path=output)
    assert image.shape == (180, 320, 3)
    assert output.is_file()
    assert transform.x_at(0) == transform.left
    assert transform.x_at(len(frame) - 1) == transform.left + transform.plot_w


def test_dense_run_is_trimmed_to_tightest_twelve_bars() -> None:
    count = 40
    frame = pd.DataFrame(
        {
            "fast_spread": [0.001] * count,
            "full_spread": [0.002 + index * 0.00001 for index in range(count)],
        }
    )
    segments = find_dense_segments(frame)
    assert segments == [DenseSegment(0, 11)]


def test_segment_box_is_normalized() -> None:
    frame = _frame(30)
    for column in ALL_MA_COLS:
        frame[column] = 100.0
    transform = ChartTransform(
        n_bars=30,
        width=320,
        height=742,
        left=12,
        top=12,
        plot_w=296,
        plot_h=718,
        price_min=95.0,
        price_max=105.0,
        candle_half_w=3,
    )
    box = segment_to_bbox(frame, DenseSegment(5, 10), transform)
    assert box is not None
    assert all(0 <= value <= 1 for value in box)
    assert box[2] > 0 and box[3] > 0
