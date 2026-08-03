from __future__ import annotations

import sys
import types

import pytest

from yolo_xx.consistency import iou, match_greedy
from yolo_xx.evaluate import main as evaluate_main
from yolo_xx.train import (
    SAFE_AUG,
    build_training_contract,
    build_train_kwargs,
    ensure_run_output_available,
    infer_finetune,
    parse_cache,
    main as train_main,
)


def test_semantic_augmentations_are_disabled() -> None:
    for name in (
        "fliplr",
        "flipud",
        "mosaic",
        "mixup",
        "cutmix",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "bgr",
        "translate",
        "scale",
    ):
        assert SAFE_AUG[name] == 0.0
    assert SAFE_AUG["multi_scale"] is False


def test_train_plan_is_pure_and_finetune_is_inferred(tmp_path) -> None:
    assert infer_finetune("yolo11n.pt") is False
    assert infer_finetune("runs/detect/round8/weights/best.pt") is True
    assert parse_cache("false") is False
    kwargs = build_train_kwargs(
        data=tmp_path / "data.yaml",
        epochs=3,
        imgsz=640,
        batch=2,
        patience=2,
        device="cpu",
        workers=0,
        cache=False,
        project=tmp_path / "runs",
        name="smoke",
        plots=False,
        resume=False,
        finetune=True,
        seed=42,
    )
    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["lr0"] == 1e-4
    assert kwargs["mosaic"] == 0.0
    assert kwargs["exist_ok"] is False
    assert kwargs["deterministic"] is True
    assert kwargs["amp"] is True
    assert kwargs["close_mosaic"] == 0
    cold_kwargs = build_train_kwargs(
        data=tmp_path / "data.yaml",
        epochs=3,
        imgsz=640,
        batch=2,
        patience=2,
        device="cpu",
        workers=0,
        cache=False,
        project=tmp_path / "runs",
        name="cold",
        plots=False,
        resume=False,
        finetune=False,
        seed=42,
    )
    assert cold_kwargs["optimizer"] == "auto"
    assert cold_kwargs["translate"] == 0.0
    assert cold_kwargs["scale"] == 0.0
    counterpart = dict(cold_kwargs)
    counterpart.update(
        {
            "data": str(tmp_path / "other.yaml"),
            "project": str(tmp_path / "other-runs"),
            "name": "other-name",
        }
    )
    left = build_training_contract(
        model="yolo11n.pt", train_kwargs=cold_kwargs, model_sha256="a" * 64
    )
    right = build_training_contract(
        model="yolo11n.pt", train_kwargs=counterpart, model_sha256="a" * 64
    )
    assert left["contract_sha256"] == right["contract_sha256"]


def test_consistency_matching_is_one_to_one() -> None:
    box = (0.5, 0.5, 0.2, 0.2)
    assert iou(box, box) == 1.0
    assert match_greedy([box, box], [box], iou_threshold=0.5) == 1


def test_new_training_run_refuses_existing_directory(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "duplicate"
    run_dir.mkdir(parents=True)
    try:
        ensure_run_output_available(tmp_path / "runs", "duplicate", resume=False)
    except FileExistsError as error:
        assert "refusing to mix training runs" in str(error)
    else:
        raise AssertionError("expected an existing-run refusal")
    ensure_run_output_available(tmp_path / "runs", "duplicate", resume=True)


def test_train_and_eval_reject_invalid_dataset_before_model_import(
    tmp_path, monkeypatch
) -> None:
    class ForbiddenYOLO:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("YOLO must not initialize before strong dataset audit")

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = ForbiddenYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\n")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"not-loaded")
    with pytest.raises(ValueError, match="failed schema-v2/pre-holdout audit"):
        train_main(
            [
                "--data",
                str(data_yaml),
                "--device",
                "cpu",
                "--project",
                str(tmp_path / "runs"),
            ]
        )
    with pytest.raises(ValueError, match="failed schema-v2/pre-holdout audit"):
        evaluate_main(
            [
                "--weights",
                str(weights),
                "--data",
                str(data_yaml),
                "--out",
                str(tmp_path / "metrics.json"),
                "--device",
                "cpu",
            ]
        )
