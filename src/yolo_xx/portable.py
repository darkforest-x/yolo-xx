"""Create and verify source-free receipts for disposable GPU fitting workers.

The Mac remains the source-of-truth machine and performs the full schema-v2
source snapshot audit.  A receipt pins that successful audit to the exact data
YAML, dataset manifest, images, and labels.  A Windows worker may then verify the
copied payload without receiving the 287MB OHLCV snapshot.  The receipt SHA-256
must be carried separately by the launcher command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .audit import audit_dataset
from .source_manifest import HOLDOUT_START, sha256_file, utc_timestamp

RECEIPT_TYPE = "yolo_xx_portable_training_receipt"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _dataset_root(data_yaml: str | Path) -> Path:
    path = Path(data_yaml).resolve()
    if not path.is_file() or path.name not in {"data.yaml", "data.yml"}:
        raise FileNotFoundError(f"dataset YAML does not exist: {path}")
    return path.parent


def create_training_receipt(
    *, data_yaml: str | Path, out: str | Path
) -> dict[str, object]:
    """Run a full local audit and pin the exact portable training payload."""
    root = _dataset_root(data_yaml)
    output = Path(out).resolve()
    audit = audit_dataset(root)
    if not audit["valid"]:
        raise ValueError(
            "cannot create portable receipt from invalid dataset: "
            + "; ".join(audit["errors"][:5])
        )
    manifest_path = root / "dataset_manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    files = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
        }
        for relative in sorted(
            {str(item[field]) for item in samples for field in ("image", "label")}
        )
    ]
    payload = {
        "schema_version": 1,
        "manifest_type": RECEIPT_TYPE,
        "full_source_audit_valid": True,
        "full_source_audit_error_count": 0,
        "holdout_read": False,
        "dataset_root_name": root.name,
        "data_yaml": {"path": Path(data_yaml).name, "sha256": sha256_file(data_yaml)},
        "dataset_manifest": {
            "path": manifest_path.name,
            "sha256": sha256_file(manifest_path),
        },
        "source_snapshot_sha256": manifest["source_snapshot"]["sha256"],
        "end_before": manifest["end_before"],
        "split_at": manifest["split_at"],
        "sample_count": len(samples),
        "files": files,
    }
    if output.exists():
        raise FileExistsError(f"refusing to overwrite portable receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    return {
        "schema_version": 1,
        "manifest_type": RECEIPT_TYPE,
        "receipt": str(output),
        "receipt_sha256": sha256_file(output),
        "dataset_root_name": root.name,
        "sample_count": len(samples),
        "file_count": len(files),
        "dataset_manifest_sha256": payload["dataset_manifest"]["sha256"],
        "source_snapshot_sha256": payload["source_snapshot_sha256"],
        "full_source_audit_valid": True,
        "holdout_read": False,
    }


def _safe_child(root: Path, raw: object, *, field: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a non-empty relative path")
    candidate = Path(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{field} must remain under the dataset root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes the dataset root") from error
    return resolved


def verify_training_receipt(
    *,
    data_yaml: str | Path,
    receipt: str | Path,
    expected_receipt_sha256: str | None,
) -> dict[str, object]:
    """Verify a copied payload without opening the source snapshot or OHLCV."""
    root = _dataset_root(data_yaml)
    receipt_path = Path(receipt).resolve()
    if not isinstance(expected_receipt_sha256, str) or len(expected_receipt_sha256) != 64:
        raise ValueError("expected portable receipt SHA-256 is required")
    actual_receipt_sha = sha256_file(receipt_path)
    if actual_receipt_sha != expected_receipt_sha256.lower():
        raise ValueError("portable receipt SHA-256 mismatch")
    try:
        payload: Any = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("portable receipt is not valid JSON") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("portable receipt schema_version must be 1")
    if payload.get("manifest_type") != RECEIPT_TYPE:
        raise ValueError(f"portable receipt manifest_type must be {RECEIPT_TYPE}")
    if payload.get("full_source_audit_valid") is not True or payload.get("holdout_read") is not False:
        raise ValueError("portable receipt lacks a safe successful full-audit declaration")
    if payload.get("dataset_root_name") != root.name:
        raise ValueError("portable receipt dataset root name mismatch")
    end_before = utc_timestamp(payload.get("end_before"), field="receipt end_before")
    split_at = utc_timestamp(payload.get("split_at"), field="receipt split_at")
    if end_before > HOLDOUT_START or split_at > HOLDOUT_START:
        raise ValueError("portable receipt declares post-holdout data")

    data_info = payload.get("data_yaml")
    manifest_info = payload.get("dataset_manifest")
    if not isinstance(data_info, dict) or not isinstance(manifest_info, dict):
        raise ValueError("portable receipt is missing data/manifest identity")
    declared_yaml = _safe_child(root, data_info.get("path"), field="data_yaml.path")
    if declared_yaml != Path(data_yaml).resolve() or sha256_file(declared_yaml) != data_info.get(
        "sha256"
    ):
        raise ValueError("portable data YAML identity mismatch")
    manifest_path = _safe_child(
        root, manifest_info.get("path"), field="dataset_manifest.path"
    )
    if sha256_file(manifest_path) != manifest_info.get("sha256"):
        raise ValueError("portable dataset manifest identity mismatch")

    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_snapshot", {}).get("sha256") != payload.get(
        "source_snapshot_sha256"
    ):
        raise ValueError("portable source snapshot identity mismatch")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != payload.get("sample_count"):
        raise ValueError("portable sample count mismatch")
    expected_files = {
        str(item[field]): str(item[f"{field}_sha256"])
        for item in samples
        for field in ("image", "label")
    }
    receipt_files = payload.get("files")
    if not isinstance(receipt_files, list):
        raise ValueError("portable receipt files must be a list")
    declared_files: dict[str, str] = {}
    for index, item in enumerate(receipt_files):
        if not isinstance(item, dict):
            raise ValueError(f"portable files[{index}] must be an object")
        relative = item.get("path")
        digest = item.get("sha256")
        path = _safe_child(root, relative, field=f"files[{index}].path")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"portable files[{index}] lacks path/SHA-256")
        if relative in declared_files:
            raise ValueError(f"duplicate portable payload path: {relative}")
        declared_files[relative] = digest
        if sha256_file(path) != digest:
            raise ValueError(f"portable payload SHA-256 mismatch: {relative}")
    if declared_files != expected_files:
        raise ValueError("portable receipt file ledger differs from dataset manifest")
    return {
        "schema_version": 1,
        "audit_type": "yolo_xx_portable_training_receipt_audit",
        "valid": True,
        "dataset": str(root),
        "sample_count": len(samples),
        "file_count": len(declared_files),
        "receipt_sha256": actual_receipt_sha,
        "holdout_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--data", required=True, type=Path)
    create.add_argument("--out", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--data", required=True, type=Path)
    verify.add_argument("--receipt", required=True, type=Path)
    verify.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action == "create":
        payload = create_training_receipt(data_yaml=args.data, out=args.out)
    else:
        payload = verify_training_receipt(
            data_yaml=args.data,
            receipt=args.receipt,
            expected_receipt_sha256=args.receipt_sha256,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
