from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from yolo_xx.audit import audit_dataset
from yolo_xx.manual_short import (
    build_manual_short_dataset,
    create_preholdout_snapshot,
    make_plan,
)


def _write_source(path: Path) -> pd.DatetimeIndex:
    times = pd.date_range("2025-01-01", periods=1000, freq="15min", tz="UTC")
    close = 100 + np.sin(np.arange(len(times)) / 20) * 0.2
    frame = pd.DataFrame(
        {
            "ts": times.astype("int64") // 1_000_000,
            "open": close - 0.02,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
            "volume": np.ones(len(times)),
            "open_time": times,
        }
    )
    boundary = pd.DataFrame(
        [
            {
                "ts": int(pd.Timestamp("2026-05-04T00:00:00Z").value // 1_000_000),
                "open": "DO_NOT_PARSE",
                "high": "DO_NOT_PARSE",
                "low": "DO_NOT_PARSE",
                "close": "DO_NOT_PARSE",
                "volume": "DO_NOT_PARSE",
                "open_time": "2026-05-04T00:00:00Z",
            }
        ]
    )
    pd.concat([frame, boundary], ignore_index=True).to_csv(path, index=False)
    return times


def _write_sheet(path: Path, times: pd.DatetimeIndex) -> None:
    fields = [
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
    ]
    rows = []
    for index, cut_index in enumerate((300, 380, 700, 780)):
        b0, b1 = 70, 80
        rows.append(
            {
                "box_id": f"TEST_{cut_index:06d}__b0",
                "symbol": "TEST_USDT_SWAP",
                "stem": f"TEST_{cut_index:06d}",
                "cut_time": times[cut_index].isoformat(),
                "bar_b0": b0,
                "bar_b1": b1,
                "yolo_xc": 0.377,
                "yolo_yc": 0.5,
                "yolo_w": 0.055,
                "yolo_h": 0.1,
                "owner_side": "short",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_snapshot_stops_before_boundary_ohlcv_and_builds_both_layouts(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    times = _write_source(source_dir / "okx_TEST_USDT_SWAP_15m_1001.csv")
    sheet = tmp_path / "review.csv"
    _write_sheet(sheet, times)
    snapshot = tmp_path / "snapshot"
    summary = create_preholdout_snapshot(
        review_sheet=sheet,
        source_dir=source_dir,
        out_dir=snapshot,
    )
    assert summary["post_cutoff_ohlcv_rows_materialized"] == 0
    copied = pd.read_csv(snapshot / "okx_TEST_USDT_SWAP_15m_1001.csv")
    assert len(copied) == 1000
    assert copied["open"].ne("DO_NOT_PARSE").all()

    baseline = tmp_path / "baseline"
    baseline_summary = build_manual_short_dataset(
        snapshot_dir=snapshot,
        out_dir=baseline,
        layout="original",
        window=200,
        split_at=times[600].isoformat(),
    )
    assert baseline_summary["dataset_audit_valid"] is True
    assert audit_dataset(baseline)["valid"] is True

    short = tmp_path / "short"
    short_summary = build_manual_short_dataset(
        snapshot_dir=snapshot,
        out_dir=short,
        layout="staggered_causal",
        window=96,
        split_at=times[600].isoformat(),
        right_contexts=(0, 8, 16, 24),
    )
    assert short_summary["dataset_audit_valid"] is True
    manifest = json.loads((short / "dataset_manifest.json").read_text())
    assert manifest["window_bars"] == 96
    rights = [
        box["xywhn"][0] + box["xywhn"][2] / 2
        for sample in manifest["samples"]
        for box in sample["boxes"]
    ]
    assert len({round(value, 2) for value in rights}) > 1
    assert all(
        pd.Timestamp(box["box_end_close_time"]) <= pd.Timestamp(box["available_at"])
        for sample in manifest["samples"]
        for box in sample["boxes"]
    )


def test_manual_short_dry_run_is_no_read_no_write(tmp_path: Path) -> None:
    output = tmp_path / "out"
    plan = make_plan(
        action="build",
        review_sheet=None,
        source_dir=None,
        snapshot_dir=tmp_path / "missing_snapshot_is_allowed",
        out_dir=output,
        layout="staggered_causal",
        window=96,
        split_at="2026-02-15T00:00:00Z",
    )
    assert plan["pixels_per_bar"] > 13
    assert plan["training"] is False
    assert not output.exists()
