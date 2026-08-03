"""Train the dense-chart YOLO detector with semantics-preserving settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SAFE_AUG: dict[str, Any] = {
    "fliplr": 0.0,
    "flipud": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "translate": 0.02,
    "scale": 0.1,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "erasing": 0.0,
    "auto_augment": None,
}

FINETUNE_OPT: dict[str, Any] = {
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "lrf": 0.01,
    "warmup_epochs": 0.5,
}

DEFAULT_WORKERS = 6
DEFAULT_CACHE = "disk"


def pick_device() -> str:
    """Choose CUDA, then Apple MPS, then CPU without importing torch at startup."""
    import torch

    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def parse_cache(raw: str) -> bool | str:
    """Map a CLI cache spelling to an Ultralytics-compatible value."""
    key = raw.strip().lower()
    if key in {"0", "false", "no", "none", "off"}:
        return False
    if key == "ram":
        return "ram"
    if key in {"1", "true", "yes", "on", "disk"}:
        return "disk"
    return raw


def infer_finetune(model: str | Path) -> bool:
    """Treat official `yolo*.pt` baselines as cold starts and other weights as chains."""
    return not Path(model).name.lower().startswith("yolo")


def build_train_kwargs(
    *,
    data: str | Path,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    device: str,
    workers: int,
    cache: bool | str,
    project: str | Path,
    name: str,
    plots: bool,
    resume: bool,
    finetune: bool,
    seed: int,
) -> dict[str, Any]:
    """Build the complete, inspectable Ultralytics `train` keyword mapping."""
    kwargs: dict[str, Any] = {
        "data": str(Path(data).resolve()),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "patience": patience,
        "device": device,
        "workers": workers,
        "cache": cache,
        "project": str(project),
        "name": name,
        "exist_ok": True,
        "plots": plots,
        "rect": True,
        "resume": resume,
        "seed": seed,
        **SAFE_AUG,
    }
    if finetune:
        kwargs.update(FINETUNE_OPT)
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="dense_15m_smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--finetune", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    finetune = infer_finetune(args.model) if args.finetune is None else args.finetune
    device = args.device or ("auto" if args.dry_run else pick_device())
    kwargs = build_train_kwargs(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=device,
        workers=args.workers,
        cache=parse_cache(args.cache),
        project=args.project,
        name=args.name,
        plots=args.plots,
        resume=args.resume,
        finetune=finetune,
        seed=args.seed,
    )
    plan = {"model": args.model, "finetune": finetune, "train": kwargs}
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset YAML does not exist: {args.data}")

    from ultralytics import YOLO

    result = YOLO(args.model).train(**kwargs)
    print(result)


if __name__ == "__main__":
    main()
