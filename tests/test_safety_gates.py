from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yolo_xx.audit import audit_dataset
from yolo_xx.dataset import build, make_build_plan


def _write_csv(
    path: Path,
    *,
    start: str = "2025-01-01",
    count: int = 900,
    timeframe: str = "15min",
) -> None:
    timestamps = pd.date_range(start, periods=count, freq=timeframe, tz="UTC")
    close = 100 + np.sin(np.arange(count) / 30) * 0.05
    pd.DataFrame(
        {
            "ts": timestamps.astype("int64") // 1_000_000,
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.ones(count),
        }
    ).to_csv(path, index=False)


def _build_kwargs(cache: Path, source_manifest: Path | None) -> dict[str, object]:
    return {
        "cache_dir": cache,
        "source_manifest": source_manifest,
        "window": 120,
        "stride": 120,
        "train_frac": 0.8,
        "target_bg_frac": 0.35,
        "max_images": 20,
        "seed": 7,
        "end_before": "2026-05-04T00:00:00Z",
        "min_rows": 0,
    }


def test_source_snapshot_failures_happen_before_pd_read_csv(
    tmp_path, monkeypatch, make_source_manifest
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    csv_path = cache / "okx_TEST_USDT_SWAP_15m_900.csv"
    _write_csv(csv_path)
    valid_manifest = make_source_manifest(cache, timeframe="15m")
    late_manifest = make_source_manifest(
        cache,
        timeframe="15m",
        cutoff_exclusive="2026-05-05T00:00:00Z",
        manifest_path=tmp_path / "late.json",
    )
    mismatched_manifest = make_source_manifest(
        cache,
        timeframe="15m",
        manifest_path=tmp_path / "mismatched.json",
    )
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")

    def forbidden_read(*args, **kwargs):
        raise AssertionError("pd.read_csv must not run before snapshot authentication")

    monkeypatch.setattr(pd, "read_csv", forbidden_read)
    with pytest.raises(ValueError, match="requires --source-manifest"):
        build(tmp_path / "missing", **_build_kwargs(cache, None))
    with pytest.raises(ValueError, match="later than holdout start"):
        build(tmp_path / "late", **_build_kwargs(cache, late_manifest))
    with pytest.raises(ValueError, match="snapshot stat mismatch"):
        build(tmp_path / "mismatch", **_build_kwargs(cache, mismatched_manifest))
    # The manifest was valid when written, but its source changed before parse.
    assert valid_manifest.is_file()


def test_global_time_split_is_shared_across_different_symbol_coverages(
    tmp_path, make_source_manifest
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_csv(cache / "okx_A_USDT_SWAP_15m_1200.csv", count=1200)
    _write_csv(
        cache / "okx_B_USDT_SWAP_15m_900.csv",
        start="2025-01-04",
        count=900,
    )
    source_manifest = make_source_manifest(cache, timeframe="15m")
    output = tmp_path / "dataset"
    build(
        output,
        cache_dir=cache,
        source_manifest=source_manifest,
        window=120,
        stride=120,
        train_frac=0.6,
        split_at="2025-01-08T00:00:00Z",
        target_bg_frac=0.35,
        max_images=30,
        seed=7,
        end_before="2026-05-04T00:00:00Z",
        min_rows=0,
    )
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    split_at = pd.Timestamp(manifest["split_at"])
    train = [sample for sample in manifest["samples"] if sample["split"] == "train"]
    val = [sample for sample in manifest["samples"] if sample["split"] == "val"]
    assert train and val
    assert all(pd.Timestamp(sample["available_at"]) < split_at for sample in train)
    assert all(pd.Timestamp(sample["window_start_time"]) >= split_at for sample in val)
    assert max(pd.Timestamp(sample["available_at"]) for sample in train) < min(
        pd.Timestamp(sample["window_start_time"]) for sample in val
    )
    assert manifest["dropped_cross_split"] > 0
    assert {sample["symbol"] for sample in manifest["samples"]} == {
        "okx_A_USDT_SWAP",
        "okx_B_USDT_SWAP",
    }


def test_schema_v2_audit_detects_manifest_label_and_image_tampering(
    tmp_path, make_source_manifest
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    _write_csv(cache / "okx_TEST_USDT_SWAP_15m_900.csv")
    source_manifest = make_source_manifest(cache, timeframe="15m")
    output = tmp_path / "dataset"
    build(output, **_build_kwargs(cache, source_manifest))
    assert audit_dataset(output)["valid"] is True

    manifest_path = output / "dataset_manifest.json"
    original_manifest = manifest_path.read_bytes()
    payload = json.loads(original_manifest)
    payload["samples"][0]["available_at"] = "2025-01-01T00:00:00Z"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert audit_dataset(output)["valid"] is False
    manifest_path.write_bytes(original_manifest)

    payload = json.loads(original_manifest)
    positive = next(sample for sample in payload["samples"] if sample["n_boxes"] > 0)
    label_path = output / positive["label"]
    original_label = label_path.read_bytes()
    label_path.write_bytes(original_label + b"0 0.1 0.1 0.1 0.1\n")
    assert audit_dataset(output)["valid"] is False
    label_path.write_bytes(original_label)

    image_path = output / payload["samples"][0]["image"]
    original_image = image_path.read_bytes()
    image_path.write_bytes(original_image + b"tamper")
    assert audit_dataset(output)["valid"] is False
    image_path.write_bytes(original_image)
    assert audit_dataset(output)["valid"] is True


@pytest.mark.parametrize(
    ("timeframe", "expected_window"),
    [("15m", 200), ("5m", 600), ("3m", 1000), ("2m", 1500), ("1m", 3000)],
)
def test_default_window_is_3000_physical_minutes(
    tmp_path, timeframe, expected_window
) -> None:
    plan = make_build_plan(
        tmp_path / "out",
        cache_dir=tmp_path / "cache",
        source_manifest=None,
        window=None,
        stride=None,
        train_frac=0.8,
        target_bg_frac=0.35,
        max_images=20,
        seed=7,
        end_before="2026-05-04T00:00:00Z",
        timeframe=timeframe,
    )
    assert plan["physical_window_minutes"] == 3000
    assert plan["window_bars"] == expected_window
    assert plan["stride_bars"] == expected_window
    assert plan["pixels_per_bar"] > 0
    if timeframe in {"1m", "2m"}:
        assert plan["resolution_risks"]
    else:
        assert plan["resolution_risks"] == []
