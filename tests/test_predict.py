from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from yolo_xx.predict import (
    artifact_relative_path,
    build_predict_plan,
    discover_images,
    ensure_clean_output,
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


def test_offline_prediction_writes_complete_artifacts(tmp_path, monkeypatch) -> None:
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

        def plot(self):
            return np.zeros((8, 12, 3), dtype=np.uint8)

    class FakeYOLO:
        def __init__(self, weights: str) -> None:
            self.weights = weights

        def predict(self, **kwargs):
            assert kwargs["save"] is False
            return iter([FakeResult()])

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.__version__ = "test-version"
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake-weights")
    source = tmp_path / "images"
    source.mkdir()
    (source / "sample.jpg").write_bytes(b"fake-image")
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
        recursive=False,
        save_overlays=True,
    )
    manifest = run_prediction(plan)

    label = output / "labels" / "sample.jpg.txt"
    overlay = output / "overlays" / "sample.jpg.png"
    assert label.read_text() == "0 0.50000000 0.40000000 0.20000000 0.10000000 0.90000000\n"
    assert overlay.is_file()
    assert manifest["image_count"] == 1
    assert manifest["detection_count"] == 1
    assert manifest["ultralytics_version"] == "test-version"
    assert json.loads((output / "predictions.json").read_text())["items"][0]["relative_image"] == "sample.jpg"
