from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from yolo_xx.scan_predict import build_plan, run
from yolo_xx.scan_set import (
    audit_scan_pair,
    build_scan_pair,
    create_scan_receipt,
    verify_scan_receipt,
)


def _write_source(path: Path, count: int = 1200) -> None:
    times = pd.date_range("2025-01-01", periods=count, freq="5min", tz="UTC")
    close = 100 + np.arange(count) * 0.01 + np.sin(np.arange(count) / 11)
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


def test_scan_pair_receipt_and_directory_batched_prediction(
    tmp_path: Path, monkeypatch, make_source_manifest
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = snapshot / "okx_TEST_USDT_SWAP_5m_1200.csv"
    _write_source(source)
    make_source_manifest(snapshot, timeframe="5m", manifest_path=snapshot / "source_snapshot.json")
    pair = tmp_path / "scan"
    summary = build_scan_pair(snapshot_dir=snapshot, out_dir=pair, max_images=4)
    assert summary["audit_valid"] is True
    assert audit_scan_pair(pair)["valid"] is True

    receipt_path = pair / "w200" / "portable_scan_receipt.json"
    receipt = create_scan_receipt(arm_dir=pair / "w200", out=receipt_path)
    verified = verify_scan_receipt(
        arm_dir=pair / "w200",
        receipt=receipt_path,
        expected_receipt_sha256=receipt["receipt_sha256"],
    )
    assert verified["valid"] is True

    class FakeTensor:
        def __init__(self, values) -> None:
            self.values = np.asarray(values)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class FakeBoxes:
        xywhn = FakeTensor([[0.5, 0.4, 0.2, 0.1]])
        cls = FakeTensor([0.0])
        conf = FakeTensor([0.9])

    class FakeResult:
        boxes = FakeBoxes()
        names = {0: "dense_cluster"}

        def __init__(self, path: Path) -> None:
            self.path = str(path)

    class FakeYOLO:
        calls = []

        def __init__(self, weights: str) -> None:
            self.weights = weights

        def predict(self, **kwargs):
            self.calls.append(kwargs["source"])
            return iter(FakeResult(path) for path in sorted(Path(kwargs["source"]).glob("*.png")))

    fake = types.ModuleType("ultralytics")
    fake.__version__ = "test-version"
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"weights")
    plan = build_plan(
        weights=weights,
        arm_dir=pair / "w200",
        out_dir=tmp_path / "predictions",
        conf=0.30,
        iou=0.70,
        imgsz=960,
        batch=16,
        device="cpu",
        overlay_limit=0,
    )
    result = run(plan)
    assert result["image_count"] == 4
    assert result["detection_count"] == 4
    assert result["images_with_detections"] == 4
    assert len(FakeYOLO.calls) == 1
    assert all(item["detections"][0]["box_right_fraction"] == 0.6 for item in result["items"])
