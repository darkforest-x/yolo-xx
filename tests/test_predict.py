from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from yolo_xx.dataset import build
from yolo_xx.predict import (
    artifact_relative_path,
    build_predict_plan,
    discover_images,
    ensure_clean_output,
    load_dataset_contexts,
    normalize_detections,
    run_prediction,
    sha256_file,
    source_relative_path,
    to_prediction_lines,
)


def test_discover_images_is_sorted_and_can_preserve_nested_paths(tmp_path) -> None:
    source = tmp_path / "images"
    nested = source / "nested"
    nested.mkdir(parents=True)
    (source / "b.PNG").write_bytes(b"image")
    (source / "a.jpg").write_bytes(b"image")
    (source / "a.png").write_bytes(b"image")
    (nested / "a.webp").write_bytes(b"image")
    (source / "ignore.txt").write_text("not an image")

    shallow = discover_images(source)
    assert [path.name for path in shallow] == ["a.jpg", "a.png", "b.PNG"]
    recursive = discover_images(source, recursive=True)
    assert len(recursive) == 4
    assert source_relative_path(nested / "a.webp", source) == Path("nested/a.webp")
    assert artifact_relative_path(Path("a.jpg"), ".txt") == Path("a.jpg.txt")
    assert artifact_relative_path(Path("a.png"), ".txt") == Path("a.png.txt")


def test_prediction_serialization_is_stable() -> None:
    detections = normalize_detections(
        [[0.5, 0.4, 0.2, 0.1]],
        [0.0],
        [0.876543219],
        {0: "dense_cluster"},
    )
    assert detections == [
        {
            "class_id": 0,
            "class_name": "dense_cluster",
            "confidence": 0.87654322,
            "xywhn": [0.5, 0.4, 0.2, 0.1],
        }
    ]
    assert to_prediction_lines(detections) == (
        "0 0.50000000 0.40000000 0.20000000 0.10000000 0.87654322\n"
    )


def test_predict_plan_is_pure_and_validates_ranges(tmp_path) -> None:
    plan = build_predict_plan(
        weights=tmp_path / "missing.pt",
        source=tmp_path / "missing-images",
        output=tmp_path / "out",
        conf=0.25,
        iou=0.7,
        imgsz=960,
        batch=8,
        device="auto",
        recursive=True,
        save_overlays=True,
    )
    assert plan["weights"].endswith("missing.pt")
    assert plan["recursive"] is True
    assert not (tmp_path / "out").exists()
    mapped_plan = build_predict_plan(
        weights=tmp_path / "missing.pt",
        source=tmp_path / "missing-images",
        output=tmp_path / "mapped-out",
        conf=0.25,
        iou=0.7,
        imgsz=960,
        batch=8,
        device="auto",
        recursive=True,
        save_overlays=False,
        dataset_manifest=tmp_path / "missing-manifest.json",
    )
    assert mapped_plan["schema_version"] == 2
    assert mapped_plan["dataset_manifest"].endswith("missing-manifest.json")
    assert not (tmp_path / "mapped-out").exists()
    with pytest.raises(ValueError, match="between zero and one"):
        build_predict_plan(
            weights="model.pt",
            source="images",
            output="out",
            conf=1.1,
            iou=0.7,
            imgsz=960,
            batch=8,
            device="cpu",
            recursive=False,
            save_overlays=False,
        )


def test_dataset_context_rejects_backfilled_availability(tmp_path) -> None:
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "detection_spec": {"timeframe": "5m"},
                "samples": [
                    {
                        "id": "sample",
                        "image": "images/train/sample.png",
                        "symbol": "okx_TEST_USDT_SWAP",
                        "source_file": "/fixture/source.csv",
                        "source_sha256": "placeholder",
                        "window_start_time": "2025-01-01T00:00:00Z",
                        "window_end_close_time": "2025-01-01T01:00:00Z",
                        "available_at": "2025-01-01T00:30:00Z",
                        "image_sha256": "placeholder",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="violates availability contract"):
        load_dataset_contexts(manifest)


def test_prediction_output_must_be_empty(tmp_path) -> None:
    output = tmp_path / "predictions"
    output.mkdir()
    assert ensure_clean_output(output) == output
    (output / "old.json").write_text("{}")
    with pytest.raises(FileExistsError, match="refusing to mix prediction runs"):
        ensure_clean_output(output)


def test_weight_hash_is_exact(tmp_path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"yolo-xx")
    assert sha256_file(weights) == "db7b0d0eeae87be106e7ab9afe7d9d2d0713416101826b9681ee81f5a33cd365"


def test_offline_prediction_writes_complete_artifacts(
    tmp_path, monkeypatch, make_source_manifest
) -> None:
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

        def __init__(self, path: str) -> None:
            self.path = path

        def plot(self):
            return np.zeros((8, 12, 3), dtype=np.uint8)

    class FakeYOLO:
        calls = []

        def __init__(self, weights: str) -> None:
            self.weights = weights

        def predict(self, **kwargs):
            assert kwargs["save"] is False
            assert isinstance(kwargs["source"], str)
            self.calls.append(kwargs["source"])
            return iter([FakeResult(kwargs["source"])])

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.__version__ = "test-version"
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake-weights")
    source = tmp_path / "dataset"
    cache = tmp_path / "cache"
    cache.mkdir()
    count = 900
    close = 100 + np.sin(np.arange(count) / 30) * 0.05
    timestamps = pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC")
    pd.DataFrame(
        {
            "ts": timestamps.astype("int64") // 1_000_000,
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.ones(count),
        }
    ).to_csv(cache / "okx_TEST_USDT_SWAP_15m_900.csv", index=False)
    source_snapshot = make_source_manifest(cache, timeframe="15m")
    build(
        source,
        cache_dir=cache,
        source_manifest=source_snapshot,
        window=120,
        stride=120,
        train_frac=0.5,
        target_bg_frac=0.5,
        max_images=2,
        seed=7,
        end_before="2026-05-04T00:00:00Z",
        min_rows=0,
    )
    dataset_manifest = source / "dataset_manifest.json"
    timeframe, contexts = load_dataset_contexts(dataset_manifest)
    assert timeframe == "15m"
    assert len(contexts) == 2
    output = tmp_path / "predictions"
    plan = build_predict_plan(
        weights=weights,
        source=source,
        output=output,
        conf=0.25,
        iou=0.7,
        imgsz=960,
        batch=8,
        device="cpu",
        recursive=True,
        save_overlays=True,
        dataset_manifest=dataset_manifest,
    )
    manifest = run_prediction(plan)

    assert manifest["image_count"] == 2
    assert manifest["detection_count"] == 2
    assert manifest["ultralytics_version"] == "test-version"
    assert len(FakeYOLO.calls) == 2
    items = json.loads((output / "predictions.json").read_text())["items"]
    assert {item["relative_image"] for item in items} == set(contexts)
    assert [item["relative_image"] for item in items] == sorted(contexts)
    item = items[0]
    label = output / "labels" / f"{item['relative_image']}.txt"
    overlay = output / "overlays" / f"{item['relative_image']}.png"
    assert label.read_text() == "0 0.50000000 0.40000000 0.20000000 0.10000000 0.90000000\n"
    assert overlay.is_file()
    assert item["symbol"] == "okx_TEST_USDT_SWAP"
    assert item["source_file"].endswith("okx_TEST_USDT_SWAP_15m_900.csv")
    assert len(item["source_sha256"]) == 64
    assert len(item["image_sha256"]) == 64
    assert item["weights_sha256"] == manifest["weights_sha256"]
    assert item["dataset_manifest_sha256"] == manifest["dataset_manifest_sha256"]
    assert item["detector_timeframe"] == "15m"
    assert item["detections"][0]["available_at"] == item["available_at"]
    assert "signal_time" not in item["detections"][0]
