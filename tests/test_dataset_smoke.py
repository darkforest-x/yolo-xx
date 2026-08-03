from __future__ import annotations

import json

import numpy as np
import pandas as pd

from yolo_xx.audit import audit_dataset
from yolo_xx.dataset import build


def test_small_local_dataset_build_and_audit(tmp_path) -> None:
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
        min_rows=0,
    )
    assert summary["train_images"] >= 1
    assert summary["val_images"] >= 1
    assert json.loads((output / "dataset_summary.json").read_text())["seed"] == 7
    audit = audit_dataset(output)
    assert audit["valid"] is True
    assert audit["errors"] == []


def test_builder_refuses_to_mix_existing_outputs(tmp_path) -> None:
    output = tmp_path / "dataset"
    existing = output / "images" / "train"
    existing.mkdir(parents=True)
    (existing / "old.png").write_bytes(b"old")
    cache = tmp_path / "cache"
    cache.mkdir()
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
            min_rows=0,
        )
    except FileExistsError as error:
        assert "refusing to mix datasets" in str(error)
    else:
        raise AssertionError("expected an existing-output refusal")
