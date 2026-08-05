"""Boxes become wall-clock intervals, and the holdout stamp survives the trip."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from yolo_xx.cross_timeframe import box_interval, clusters, cooccurrence, load_scan
from yolo_xx.render import IMG_WIDTH, MARGIN

START = pd.Timestamp("2026-08-01T00:00:00Z")


def _xywhn(left_bar: float, right_bar: float, bars: int) -> list[float]:
    """Build a normalized box spanning two bar indices, inverting `x_at`."""
    plot_w = IMG_WIDTH - 2 * MARGIN
    spans = bars - 1

    def normalized(bar: float) -> float:
        return (MARGIN + bar / spans * plot_w) / IMG_WIDTH

    low, high = normalized(left_bar), normalized(right_bar)
    return [(low + high) / 2, 0.5, high - low, 0.1]


def test_box_maps_to_the_bars_it_covers() -> None:
    cadence = pd.Timedelta(minutes=5)
    start, end = box_interval(
        _xywhn(10, 20, 96), window_start=START, window_bars=96, cadence=cadence
    )
    assert start == START + 10 * cadence
    # The right edge resolves to bar 20's close, one cadence past its open.
    assert end == START + 21 * cadence


def test_edges_clamp_inside_the_window() -> None:
    cadence = pd.Timedelta(minutes=1)
    start, end = box_interval(
        [0.5, 0.5, 2.0, 0.1], window_start=START, window_bars=96, cadence=cadence
    )
    assert start == START
    assert end == START + 96 * cadence


def test_the_same_pattern_width_is_a_different_duration_per_timeframe() -> None:
    """A fixed bar-count box is not a fixed amount of time."""
    box = _xywhn(40, 52, 96)
    spans = {
        minutes: box_interval(
            box,
            window_start=START,
            window_bars=96,
            cadence=pd.Timedelta(minutes=minutes),
        )
        for minutes in (2, 30)
    }
    fast = spans[2][1] - spans[2][0]
    slow = spans[30][1] - spans[30][0]
    assert slow == 15 * fast


def _scan(
    tmp_path: Path,
    timeframe: str,
    minutes: int,
    *,
    holdout: bool,
    bars: tuple[float, float] = (40, 52),
) -> Path:
    directory = tmp_path / timeframe
    directory.mkdir(parents=True)
    (directory / "predictions.json").write_text(
        json.dumps(
            {
                "timeframe": timeframe,
                "window_bars": 96,
                "holdout_read": holdout,
                "conf": 0.3,
                "iou": 0.7,
                "weights_sha256": "a" * 64,
                "image_count": 1,
                "items": [
                    {
                        "symbol": "okx_BTC_USDT_SWAP",
                        "sample_id": f"btc__{timeframe}",
                        "window_start_time": START.isoformat().replace("+00:00", "Z"),
                        "window_end_close_time": (
                            (START + 96 * pd.Timedelta(minutes=minutes))
                            .isoformat()
                            .replace("+00:00", "Z")
                        ),
                        "overlay": None,
                        "detections": [
                            {
                                "xywhn": _xywhn(bars[0], bars[1], 96),
                                "confidence": 0.5,
                                "available_at": (
                                    (START + 96 * pd.Timedelta(minutes=minutes))
                                    .isoformat()
                                    .replace("+00:00", "Z")
                                ),
                            }
                        ],
                    }
                ],
            }
        )
    )
    return directory


def test_holdout_stamp_is_read_back_from_predictions(tmp_path: Path) -> None:
    scan = load_scan(_scan(tmp_path, "5m", 5, holdout=True))
    assert scan["holdout_read"] is True
    assert load_scan(_scan(tmp_path, "3m", 3, holdout=False))["holdout_read"] is False


def test_cooccurrence_ignores_regions_a_timeframe_never_rendered(tmp_path: Path) -> None:
    scans = {
        "2m": load_scan(_scan(tmp_path, "2m", 2, holdout=True)),
        "30m": load_scan(_scan(tmp_path, "30m", 30, holdout=True)),
    }
    rows = {
        (row["source_timeframe"], row["target_timeframe"]): row
        for row in cooccurrence(scans)
    }
    # The 2m window ends long before the 30m box, so nothing is comparable back.
    assert rows[("30m", "2m")]["comparable_detections"] == 0
    assert rows[("30m", "2m")]["echo_rate"] is None


def test_clusters_need_more_than_one_timeframe(tmp_path: Path) -> None:
    # Same bar indices land on different wall-clock times, so the ranges are
    # chosen per timeframe to actually overlap: 5m bars 40-52 cover 200-265 min,
    # 3m bars 67-87 cover 201-264 min.
    scans = {"5m": load_scan(_scan(tmp_path, "5m", 5, holdout=True))}
    assert clusters(scans, min_timeframes=2) == []
    scans["3m"] = load_scan(
        _scan(tmp_path, "3m", 3, holdout=True, bars=(67, 87))
    )
    found = clusters(scans, min_timeframes=2)
    assert len(found) == 1
    assert found[0]["timeframes"] == ["3m", "5m"]


def test_mixed_holdout_provenance_is_refused(tmp_path: Path) -> None:
    from yolo_xx.cross_timeframe import main

    _scan(tmp_path, "5m", 5, holdout=True)
    _scan(tmp_path, "3m", 3, holdout=False)
    with pytest.raises(ValueError, match="disagree on holdout provenance"):
        main(
            [
                "--scan-results",
                str(tmp_path),
                "--out",
                str(tmp_path / "out"),
                "--timeframes",
                "5m,3m",
            ]
        )
