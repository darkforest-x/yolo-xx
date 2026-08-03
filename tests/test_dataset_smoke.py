from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yolo_xx.audit import audit_dataset
from yolo_xx.dataset import build, main


def test_small_local_dataset_build_and_audit(tmp_path, make_source_manifest) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    count = 900
    close = 100 + np.sin(np.arange(count) / 30) * 0.05
    timestamps = pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts": timestamps.astype("int64") // 1_000_000,
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.ones(count),
        }
    )
    frame.to_csv(cache / "okx_TEST_USDT_SWAP_15m_900.csv", index=False)
    source_manifest = make_source_manifest(cache, timeframe="15m")
    output = tmp_path / "dataset"
    summary = build(
        output,
        cache_dir=cache,
        window=120,
        stride=120,
        train_frac=0.8,
        target_bg_frac=0.35,
        max_images=20,
        seed=7,
        end_before="2026-05-04T00:00:00Z",
        source_manifest=source_manifest,
        min_rows=0,
    )
    assert summary["train_images"] >= 1
    assert summary["val_images"] >= 1
    stored_summary = json.loads((output / "dataset_summary.json").read_text())
    assert stored_summary["seed"] == 7
    assert stored_summary["schema_version"] == 2
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["detection_spec"]["timeframe"] == "15m"
    positive_samples = [sample for sample in manifest["samples"] if sample["boxes"]]
    assert positive_samples
    sample = positive_samples[0]
    assert sample["available_at"] == sample["window_end_close_time"]
    assert all(box["available_at"] == sample["available_at"] for box in sample["boxes"])
    assert any(
        pd.Timestamp(box["box_end_close_time"]) < pd.Timestamp(box["available_at"])
        for positive in positive_samples
        for box in positive["boxes"]
    )
    audit = audit_dataset(output)
    assert audit["valid"] is True
    assert audit["errors"] == []


def test_builder_refuses_to_mix_existing_outputs(tmp_path, make_source_manifest) -> None:
    output = tmp_path / "dataset"
    existing = output / "images" / "train"
    existing.mkdir(parents=True)
    (existing / "old.png").write_bytes(b"old")
    cache = tmp_path / "cache"
    cache.mkdir()
    source_manifest = make_source_manifest(cache, timeframe="15m")
    try:
        build(
            output,
            cache_dir=cache,
            window=120,
            stride=120,
            train_frac=0.8,
            target_bg_frac=0.35,
            max_images=20,
            seed=7,
            end_before="2026-05-04T00:00:00Z",
            source_manifest=source_manifest,
            min_rows=0,
        )
    except FileExistsError as error:
        assert "refusing to mix datasets" in str(error)
    else:
        raise AssertionError("expected an existing-output refusal")


def test_five_minute_dataset_records_resolved_spec(tmp_path, make_source_manifest) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    count = 1500
    close = 100 + np.sin(np.arange(count) / 30) * 0.05
    timestamps = pd.date_range("2025-01-01", periods=count, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "ts": timestamps.astype("int64") // 1_000_000,
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.ones(count),
        }
    )
    frame.to_csv(cache / "okx_TEST_USDT_SWAP_5m_1500.csv", index=False)
    source_manifest = make_source_manifest(cache, timeframe="5m")
    output = tmp_path / "dataset"
    summary = build(
        output,
        cache_dir=cache,
        window=120,
        stride=120,
        train_frac=0.8,
        target_bg_frac=0.35,
        max_images=4,
        seed=7,
        end_before="2026-05-04T00:00:00Z",
        source_manifest=source_manifest,
        min_rows=0,
        timeframe="5m",
    )
    assert summary["detection_spec"]["ma_periods"] == [60, 180, 360]
    manifest = json.loads((output / "dataset_manifest.json").read_text())
    assert manifest["samples"][0]["window_end_close_time"].endswith("Z")
    assert {sample["symbol"] for sample in manifest["samples"]} == {
        "okx_TEST_USDT_SWAP"
    }


def test_cli_dry_run_does_not_read_csv_or_write_output(tmp_path, monkeypatch, capsys) -> None:
    def fail_read(*args, **kwargs):
        raise AssertionError("dry-run must not read CSV")

    monkeypatch.setattr(pd, "read_csv", fail_read)
    output = tmp_path / "must_not_exist"
    main(
        [
            "--cache-dir",
            str(tmp_path / "missing_cache_is_allowed_in_dry_run"),
            "--out",
            str(output),
            "--timeframe",
            "3m",
            "--ma-minutes",
            "300,900,1800",
            "--dry-run",
        ]
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True
    assert plan["detection_spec"]["ma_periods"] == [100, 300, 600]
    assert not output.exists()
