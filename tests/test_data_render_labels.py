from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yolo_xx.data import (
    ALL_MA_COLS,
    add_mas,
    cache_symbol,
    list_cache_files,
    load_ohlcv_csv,
)
from yolo_xx.labels import DenseSegment, find_dense_segments, label_segments, segment_to_bbox
from yolo_xx.render import ChartTransform, render_chart
from yolo_xx.specs import DetectionSpec


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


def test_dense_segments_keep_their_real_left_middle_and_right_positions() -> None:
    count = 60
    fast = np.full(count, 0.02)
    full = np.full(count, 0.03)
    expected = [DenseSegment(2, 7), DenseSegment(25, 30), DenseSegment(50, 55)]
    for segment in expected:
        fast[segment.start : segment.end + 1] = 0.001
        full[segment.start : segment.end + 1] = 0.002
    frame = pd.DataFrame({"fast_spread": fast, "full_spread": full})
    assert find_dense_segments(frame) == expected


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


def test_five_minute_spec_preserves_physical_ma_horizons() -> None:
    spec = DetectionSpec(timeframe="5m")
    assert spec.ma_periods == (60, 180, 360)
    assert spec.dense_min_bars == 15
    assert spec.dense_max_bars == 36
    assert spec.merge_gap_bars == 6
    assert spec.warmup_bars == 1080
    assert spec.as_dict()["warmup_minutes"] == 5400


def test_two_minute_dense_minimum_rounds_up_explicitly() -> None:
    spec = DetectionSpec(timeframe="2m")
    assert spec.dense_min_bars == 38
    assert spec.as_dict()["dense_min_resolved_minutes"] == 76


def test_dynamic_ma_periods_render_and_label_real_segment(tmp_path) -> None:
    spec = DetectionSpec(timeframe="5m")
    frame = add_mas(_frame(500), periods=spec.ma_periods)
    assert {"sma360", "ema360"}.issubset(frame.columns)
    image, transform = render_chart(
        frame,
        width=640,
        height=360,
        ma_periods=spec.ma_periods,
        out_path=tmp_path / "five_minute.png",
    )
    assert image.shape == (360, 640, 3)
    labeled = label_segments(
        frame,
        transform,
        ma_periods=spec.ma_periods,
        min_bars=spec.dense_min_bars,
        merge_gap=spec.merge_gap_bars,
        max_bars=spec.dense_max_bars,
    )
    assert all(segment.start <= segment.end for segment, _ in labeled)


def test_cache_discovery_and_cadence_are_timeframe_strict(tmp_path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    frame = _frame(5)
    frame["ts"] = (
        pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC").astype("int64")
        // 1_000_000
    )
    five = cache / "okx_TEST_USDT_SWAP_5m_5.csv"
    frame.to_csv(five, index=False)
    frame.to_csv(cache / "okx_TEST_USDT_SWAP_15m_5.csv", index=False)
    assert cache_symbol(five) == "okx_TEST_USDT_SWAP"
    assert cache_symbol(five, timeframe="5m") == "okx_TEST_USDT_SWAP"
    assert list_cache_files(cache, min_rows=0, timeframe="5m") == [five]
    loaded = load_ohlcv_csv(five, timeframe="5m")
    assert len(loaded) == 5

    broken = frame.drop(index=2)
    broken_path = cache / "okx_BROKEN_USDT_SWAP_5m_4.csv"
    broken.to_csv(broken_path, index=False)
    try:
        load_ohlcv_csv(broken_path, timeframe="5m")
    except ValueError as error:
        assert "non-contiguous 5m cadence" in str(error)
    else:
        raise AssertionError("expected strict cadence rejection")

    shifted = frame.copy()
    shifted["ts"] = shifted["ts"] + 60_000
    shifted_path = cache / "okx_SHIFTED_USDT_SWAP_5m_5.csv"
    shifted.to_csv(shifted_path, index=False)
    try:
        load_ohlcv_csv(shifted_path, timeframe="5m")
    except ValueError as error:
        assert "not aligned to 5m UTC cadence" in str(error)
    else:
        raise AssertionError("expected UTC alignment rejection")


@pytest.mark.parametrize("column", ["ts", "close", "volume"])
def test_strict_loader_rejects_unparseable_required_values(tmp_path, column) -> None:
    frame = _frame(5)
    frame["ts"] = (
        pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC").astype("int64")
        // 1_000_000
    )
    frame[column] = frame[column].astype(object)
    frame.loc[2, column] = "malformed"
    path = tmp_path / f"okx_BAD_{column}_USDT_SWAP_5m_5.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=rf"unparseable required value.*{column}"):
        load_ohlcv_csv(path, timeframe="5m", strict_cadence=True)
