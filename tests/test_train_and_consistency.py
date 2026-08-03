from __future__ import annotations

from yolo_xx.consistency import iou, match_greedy
from yolo_xx.train import SAFE_AUG, build_train_kwargs, infer_finetune, parse_cache


def test_semantic_augmentations_are_disabled() -> None:
    for name in ("fliplr", "flipud", "mosaic", "mixup", "hsv_h", "hsv_s", "hsv_v"):
        assert SAFE_AUG[name] == 0.0


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


def test_consistency_matching_is_one_to_one() -> None:
    box = (0.5, 0.5, 0.2, 0.2)
    assert iou(box, box) == 1.0
    assert match_greedy([box, box], [box], iou_threshold=0.5) == 1
