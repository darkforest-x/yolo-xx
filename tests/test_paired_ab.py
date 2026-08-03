from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yolo_xx.paired_ab import audit_pair, build_pair, make_plan
from yolo_xx.manual_short import create_preholdout_snapshot
from yolo_xx.portable import create_training_receipt, verify_training_receipt


def _write_source(path: Path) -> pd.DatetimeIndex:
    times = pd.date_range("2025-01-01", periods=3000, freq="15min", tz="UTC")
    # A persistent trend keeps the frozen dense-MA rule false, providing safe
    # synthetic backgrounds without weakening the production matcher.
    close = 100 + np.arange(len(times)) * 0.08 + np.sin(np.arange(len(times)) / 9) * 0.5
    frame = pd.DataFrame(
        {
            "ts": times.astype("int64") // 1_000_000,
            "open": close - 0.03,
            "high": close + 0.09,
            "low": close - 0.09,
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
    for cut_index in (500, 800, 1900, 2300):
        rows.append(
            {
                "box_id": f"TEST_USDT_SWAP_{cut_index:06d}__b0",
                "symbol": "TEST_USDT_SWAP",
                "stem": f"TEST_USDT_SWAP_{cut_index:06d}",
                "cut_time": times[cut_index].isoformat(),
                "bar_b0": 70,
                "bar_b1": 80,
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


def test_paired_ab_build_has_shared_ledger_and_rule_clear_backgrounds(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    times = _write_source(source / "okx_TEST_USDT_SWAP_15m_3001.csv")
    sheet = tmp_path / "review.csv"
    _write_sheet(sheet, times)
    snapshot = tmp_path / "snapshot"
    create_preholdout_snapshot(review_sheet=sheet, source_dir=source, out_dir=snapshot)

    output = tmp_path / "pair"
    summary = build_pair(
        snapshot_dir=snapshot,
        out_dir=output,
        split_at=times[1500].isoformat(),
        max_positive_anchors=4,
    )
    assert summary["dataset_audits_valid"] is True
    assert summary["matched_pairs"] == 4
    audit = audit_pair(output)
    assert audit["valid"] is True

    manifests = {
        name: json.loads((output / name / "dataset_manifest.json").read_text())
        for name in ("w200", "w96")
    }
    ids = {
        name: [sample["id"] for sample in manifest["samples"]]
        for name, manifest in manifests.items()
    }
    assert ids["w200"] == ids["w96"]
    assert len(ids["w200"]) == 8
    assert all(
        sample["n_boxes"] == 0
        for sample in manifests["w200"]["samples"]
        if sample["sample_kind"] == "negative"
    )
    assert "path:" not in (output / "w200" / "data.yaml").read_text()

    receipt_path = output / "w200" / "portable_receipt.json"
    receipt = create_training_receipt(
        data_yaml=output / "w200" / "data.yaml", out=receipt_path
    )
    verified = verify_training_receipt(
        data_yaml=output / "w200" / "data.yaml",
        receipt=receipt_path,
        expected_receipt_sha256=receipt["receipt_sha256"],
    )
    assert verified["valid"] is True
    manifest = manifests["w200"]
    image_path = output / "w200" / manifest["samples"][0]["image"]
    original = image_path.read_bytes()
    image_path.write_bytes(original + b"tamper")
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        verify_training_receipt(
            data_yaml=output / "w200" / "data.yaml",
            receipt=receipt_path,
            expected_receipt_sha256=receipt["receipt_sha256"],
        )
    image_path.write_bytes(original)


def test_paired_ab_dry_run_is_no_read_no_write(tmp_path: Path) -> None:
    output = tmp_path / "not-created"
    plan = make_plan(
        snapshot_dir=tmp_path / "missing-is-not-read",
        out_dir=output,
        split_at="2026-02-15T00:00:00Z",
        right_contexts=(0, 8, 16, 24),
        seed=20260804,
        max_positive_anchors=8,
    )
    assert plan["windows"] == [200, 96]
    assert plan["holdout_read"] is False
    assert not output.exists()
