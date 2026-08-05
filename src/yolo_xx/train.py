"""Train the dense-chart YOLO detector with semantics-preserving settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Sequence

from .audit import require_valid_dataset
from .portable import verify_training_receipt
from .source_manifest import sha256_file

SAFE_AUG: dict[str, Any] = {
    "fliplr": 0.0,
    "flipud": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "cutmix": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "bgr": 0.0,
    # Multi-position labels must come from real windows.  Synthetic shifts or
    # scaling can change candle/MA geometry and recreate a positional shortcut.
    "translate": 0.0,
    "scale": 0.0,
    "degrees": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "multi_scale": False,
    "erasing": 0.0,
    "auto_augment": None,
}

FINETUNE_OPT: dict[str, Any] = {
    "optimizer": "AdamW",
    "lr0": 1e-4,
    "lrf": 0.01,
    "warmup_epochs": 0.5,
}

# The Windows 3060 host has 16GB system RAM.  Four workers with caching disabled
# avoids the historical cache/validation memory failures while keeping the GPU
# fed.  Callers can still opt in to disk/RAM cache explicitly.
DEFAULT_WORKERS = 4
DEFAULT_CACHE = "false"


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


def ensure_run_output_available(
    project: str | Path,
    name: str,
    *,
    resume: bool,
) -> None:
    """Reject accidental reuse of a completed/new-run directory."""
    run_dir = Path(project) / name
    if run_dir.exists() and not resume:
        raise FileExistsError(f"refusing to mix training runs; output exists: {run_dir}")


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
    deterministic: bool = True,
    amp: bool = True,
    save_period: int = -1,
    optimizer: str | None = None,
    lr0: float | None = None,
    lrf: float | None = None,
    cos_lr: bool = False,
    warmup_epochs: float | None = None,
) -> dict[str, Any]:
    """Build the complete, inspectable Ultralytics `train` keyword mapping.

    The learning-rate schedule is exposed because the owner-short runs were
    dominated by epoch-to-epoch validation swings rather than by convergence: a
    flat high LR on a 2k-image set keeps knocking the model off its own optimum.
    These knobs are part of the comparison contract, so two runs that differ in
    schedule can never be presented as a single-variable A/B.
    """
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
        # Never mix a new experiment into an existing run directory.
        "exist_ok": False,
        "plots": plots,
        "rect": True,
        "resume": resume,
        "seed": seed,
        "deterministic": deterministic,
        "amp": amp,
        "save": True,
        "save_period": save_period,
        "val": True,
        # Mosaic is already zero; disabling its late close phase keeps the
        # schedule explicit and identical across paired runs.
        "close_mosaic": 0,
        **SAFE_AUG,
    }
    if finetune:
        kwargs.update(FINETUNE_OPT)
    else:
        kwargs["optimizer"] = "auto"
    if optimizer is not None:
        kwargs["optimizer"] = optimizer
    if cos_lr:
        kwargs["cos_lr"] = True
    for key, value in (("lr0", lr0), ("lrf", lrf), ("warmup_epochs", warmup_epochs)):
        if value is not None:
            kwargs[key] = value
    return kwargs


def build_training_contract(
    *,
    model: str | Path,
    train_kwargs: dict[str, Any],
    model_sha256: str | None,
) -> dict[str, object]:
    """Hash all A/B training semantics while excluding dataset/run identity."""
    excluded = {"data", "project", "name"}
    comparable = {key: train_kwargs[key] for key in sorted(train_kwargs) if key not in excluded}
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_type": "yolo_xx_training_ab_contract",
        "model_name": Path(model).name,
        "model_sha256": model_sha256,
        "train": comparable,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def runtime_fingerprint(device: str) -> dict[str, object]:
    """Record the actual framework and CUDA environment used for one fit."""
    import numpy
    import torch
    import ultralytics

    payload: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": numpy.__version__,
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if str(device).lower() not in {"cpu", "mps"}:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device!r} requested but torch.cuda.is_available() is false")
        index = int(str(device).split(",", 1)[0])
        properties = torch.cuda.get_device_properties(index)
        payload.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(index),
                "cuda_total_memory_gb": round(properties.total_memory / 1024**3, 3),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
    return payload


def main(argv: Sequence[str] | None = None) -> None:
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
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--optimizer", help="override the optimizer (default: auto)")
    parser.add_argument("--lr0", type=float, help="initial learning rate")
    parser.add_argument("--lrf", type=float, help="final LR as a fraction of lr0")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR schedule")
    parser.add_argument("--warmup-epochs", type=float)
    parser.add_argument("--portable-receipt", type=Path)
    parser.add_argument("--portable-receipt-sha256")
    parser.add_argument("--contract-out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

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
        deterministic=args.deterministic,
        amp=args.amp,
        save_period=args.save_period,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        warmup_epochs=args.warmup_epochs,
    )
    dry_contract = build_training_contract(
        model=args.model, train_kwargs=kwargs, model_sha256=None
    )
    plan = {
        "model": args.model,
        "finetune": finetune,
        "train": kwargs,
        "comparison_contract": dry_contract,
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return
    if not args.data.is_file():
        raise FileNotFoundError(f"dataset YAML does not exist: {args.data}")
    if (args.portable_receipt is None) != (args.portable_receipt_sha256 is None):
        raise ValueError(
            "--portable-receipt and --portable-receipt-sha256 must be provided together"
        )
    if args.portable_receipt is None:
        require_valid_dataset(args.data)
        audit_mode = "full_source_snapshot"
    else:
        verify_training_receipt(
            data_yaml=args.data,
            receipt=args.portable_receipt,
            expected_receipt_sha256=args.portable_receipt_sha256,
        )
        audit_mode = "portable_payload_with_mac_full_audit_receipt"
    ensure_run_output_available(args.project, args.name, resume=args.resume)

    model_path = Path(args.model)
    model_digest = sha256_file(model_path) if model_path.is_file() else None
    contract = build_training_contract(
        model=args.model,
        train_kwargs=kwargs,
        model_sha256=model_digest,
    )
    validated_plan = {
        "schema_version": 1,
        "audit_mode": audit_mode,
        "contract": contract,
        "runtime": runtime_fingerprint(device),
        "data_yaml": str(args.data.resolve()),
    }
    if args.contract_out is not None:
        args.contract_out.parent.mkdir(parents=True, exist_ok=True)
        args.contract_out.write_text(
            json.dumps(validated_plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps({"validated_training_plan": validated_plan}, indent=2, sort_keys=True))

    from ultralytics import YOLO

    result = YOLO(args.model).train(**kwargs)
    print(result)


if __name__ == "__main__":
    main()
