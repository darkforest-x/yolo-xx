"""The holdout gate is two-way: crossing it needs an opt-in and leaves a stamp.

A snapshot may only reach past the frozen holdout start with an explicit
`allow_holdout`, and the resulting manifest is permanently marked.  Neither
direction may be forged: post-holdout data cannot claim pre-holdout provenance,
and a pre-holdout snapshot cannot claim to be a holdout scan.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yolo_xx.micro_snapshot import create_snapshot, resolve_cutoff
from yolo_xx.scan_set import audit_scan_arm
from yolo_xx.source_manifest import HOLDOUT_START, load_source_manifest

CUTOFF = pd.Timestamp("2026-06-01T00:00:00Z")


def _write_cache(directory: Path, *, end: pd.Timestamp, count: int = 600) -> Path:
    """Write one 5m OHLCV cache file whose last candle closes at `end`."""
    directory.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(end=end - pd.Timedelta(minutes=5), periods=count, freq="5min")
    close = 100 + np.arange(count) * 0.01
    path = directory / f"okx_TEST_USDT_SWAP_5m_{count}.csv"
    pd.DataFrame(
        {
            "ts": times.astype("int64") // 1_000_000,
            "open": close - 0.02,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
            "volume": np.ones(count),
        }
    ).to_csv(path, index=False)
    return path


def _snapshot(tmp_path: Path, name: str, **kwargs) -> Path:
    cache = tmp_path / f"cache_{name}"
    _write_cache(cache, end=kwargs.pop("end"))
    out = tmp_path / name
    create_snapshot(cache_dir=cache, out_dir=out, timeframe="5m", **kwargs)
    return out


def test_preholdout_snapshot_keeps_the_frozen_cutoff_and_no_stamp(tmp_path: Path) -> None:
    out = _snapshot(tmp_path, "pre", end=HOLDOUT_START)
    manifest = json.loads((out / "source_snapshot.json").read_text())
    assert manifest["cutoff_exclusive"] == "2026-05-04T00:00:00Z"
    assert manifest["safety"]["holdout_read"] is False
    snapshot = load_source_manifest(out / "source_snapshot.json", expected_source_dir=out)
    assert snapshot.holdout_read is False


def test_holdout_snapshot_is_stamped_and_needs_an_opt_in_to_load(tmp_path: Path) -> None:
    out = _snapshot(tmp_path, "post", end=CUTOFF, allow_holdout=True, cutoff=CUTOFF)
    manifest = json.loads((out / "source_snapshot.json").read_text())
    assert manifest["cutoff_exclusive"] == "2026-06-01T00:00:00Z"
    assert manifest["safety"]["holdout_read"] is True

    # The default path must refuse it outright.
    with pytest.raises(ValueError, match="explicit holdout opt-in"):
        load_source_manifest(out / "source_snapshot.json", expected_source_dir=out)

    snapshot = load_source_manifest(
        out / "source_snapshot.json", expected_source_dir=out, allow_holdout=True
    )
    assert snapshot.holdout_read is True
    assert snapshot.cutoff_exclusive == CUTOFF


def test_post_holdout_data_cannot_claim_preholdout_provenance(tmp_path: Path) -> None:
    out = _snapshot(tmp_path, "forged_pre", end=CUTOFF, allow_holdout=True, cutoff=CUTOFF)
    path = out / "source_snapshot.json"
    payload = json.loads(path.read_text())
    payload["safety"]["holdout_read"] = False  # drop the stamp, keep the late cutoff
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not declare safety.holdout_read=true"):
        load_source_manifest(path, expected_source_dir=out, allow_holdout=True)


def test_preholdout_data_cannot_claim_a_holdout_stamp(tmp_path: Path) -> None:
    out = _snapshot(tmp_path, "forged_post", end=HOLDOUT_START)
    path = out / "source_snapshot.json"
    payload = json.loads(path.read_text())
    payload["safety"]["holdout_read"] = True  # stamp without a late cutoff
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not past the holdout start"):
        load_source_manifest(path, expected_source_dir=out, allow_holdout=True)


def test_explicit_cutoff_requires_the_opt_in() -> None:
    with pytest.raises(ValueError, match="requires allow_holdout=True"):
        resolve_cutoff("5m", cutoff=CUTOFF)
    with pytest.raises(ValueError, match="not past the holdout start"):
        resolve_cutoff("5m", allow_holdout=True, cutoff=HOLDOUT_START)
    assert resolve_cutoff("5m") == HOLDOUT_START


def test_scan_arm_audit_rejects_a_stamp_that_disagrees_with_the_cutoff(
    tmp_path: Path,
) -> None:
    arm = tmp_path / "w96"
    arm.mkdir()
    (arm / "scan_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_type": "yolo_xx_scan_set",
                "timeframe": "5m",
                "end_before": "2026-06-01T00:00:00Z",
                "holdout_read": False,
                "window_bars": 96,
                "samples": [],
            }
        )
    )
    audit = audit_scan_arm(arm)
    assert audit["valid"] is False
    assert any("end_before and holdout_read disagree" in item for item in audit["errors"])
