"""Build paired unlabeled chart sets for offline micro-timeframe scans.

Endpoints stop at the source snapshot's own cutoff.  Scanning past the frozen
holdout start requires an explicit opt-in and is stamped into every manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .data import add_mas, cache_symbol, load_ohlcv_csv
from .render import IMG_WIDTH, MARGIN, min_rel_span_for, render_chart
from .source_manifest import (
    HOLDOUT_START,
    SnapshotFile,
    load_source_manifest,
    sha256_file,
    utc_iso,
    utc_timestamp,
    verify_loaded_frame,
    verify_snapshot_file,
    verify_snapshot_identity,
)
from .specs import canonical_timeframe, timeframe_minutes

WINDOWS = (200, 96)
SCAN_MANIFEST = "scan_manifest.json"
PAIR_MANIFEST = "scan_pair_manifest.json"
RECEIPT_TYPE = "yolo_xx_portable_scan_receipt"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _payload_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _symbol(record: SnapshotFile, timeframe: str) -> str:
    return cache_symbol(record.path, timeframe=timeframe)


def _load_frame(record: SnapshotFile, timeframe: str) -> pd.DataFrame:
    verify_snapshot_file(record)
    frame = load_ohlcv_csv(record.path, timeframe=timeframe, strict_cadence=True)
    verify_loaded_frame(frame, record, timeframe=timeframe)
    frame = add_mas(frame, periods=(20, 60, 120))
    verify_snapshot_file(record)
    return frame


def _even_indices(start: int, end: int, count: int) -> list[int]:
    if start > end or count <= 0:
        return []
    available = end - start + 1
    if count >= available:
        return list(range(start, end + 1))
    if count == 1:
        return [(start + end) // 2]
    values = [round(start + index * (end - start) / (count - 1)) for index in range(count)]
    return sorted(set(values))


def _select_anchors(
    records: Sequence[SnapshotFile],
    *,
    timeframe: str,
    max_images: int,
    since: pd.Timestamp | None = None,
    until: pd.Timestamp | None = None,
) -> list[tuple[SnapshotFile, int, pd.Timestamp]]:
    if max_images <= 0:
        raise ValueError("max_images must be positive")
    per_symbol = max(1, (max_images + len(records) - 1) // len(records))
    candidates: dict[str, list[tuple[SnapshotFile, int, pd.Timestamp]]] = defaultdict(list)
    for record in records:
        frame = _load_frame(record, timeframe)
        # The widest window still needs its full left context, so `since` moves the
        # earliest endpoint forward without ever shortening a rendered window.
        start_index = max(WINDOWS) - 1
        if since is not None:
            opens = pd.to_datetime(frame["open_time"], utc=True)
            eligible = opens >= since
            start_index = max(
                start_index, int(eligible.argmax()) if bool(eligible.any()) else len(frame)
            )
        end_index = len(frame) - 1
        if until is not None:
            opens = pd.to_datetime(frame["open_time"], utc=True)
            eligible = opens <= until
            end_index = int(eligible.to_numpy().nonzero()[0][-1]) if bool(eligible.any()) else -1
        indices = _even_indices(start_index, end_index, per_symbol)
        symbol = _symbol(record, timeframe)
        candidates[symbol] = [
            (record, index, pd.Timestamp(frame.iloc[index]["open_time"])) for index in indices
        ]
    selected: list[tuple[SnapshotFile, int, pd.Timestamp]] = []
    symbols = sorted(candidates)
    depth = 0
    while len(selected) < max_images:
        added = False
        for symbol in symbols:
            pool = candidates[symbol]
            if depth < len(pool):
                selected.append(pool[depth])
                added = True
                if len(selected) == max_images:
                    break
        if not added:
            break
        depth += 1
    return selected


def _contract_rows(samples: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "id",
        "symbol",
        "source_sha256",
        "source_end_index",
        "window_end_open_time",
        "window_end_close_time",
        "available_at",
    )
    return [
        {field: item[field] for field in fields}
        for item in sorted(samples, key=lambda value: str(value["id"]))
    ]


def audit_scan_arm(arm_dir: str | Path) -> dict[str, object]:
    """Authenticate one unlabeled scan set and every rendered chart."""
    root = Path(arm_dir).resolve()
    errors: list[str] = []
    manifest_path = root / SCAN_MANIFEST
    try:
        manifest: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "schema_version": 1,
            "audit_type": "yolo_xx_scan_arm_audit",
            "valid": False,
            "errors": [f"scan manifest unreadable: {error}"],
        }
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        errors.append("scan manifest schema_version must be 1")
    if manifest.get("manifest_type") != "yolo_xx_scan_set":
        errors.append("scan manifest type must be yolo_xx_scan_set")
    timeframe = manifest.get("timeframe")
    try:
        normalized = canonical_timeframe(timeframe)
    except (TypeError, ValueError) as error:
        errors.append(str(error))
        normalized = "15m"
    # A scan arm carries its own provenance stamp.  The cutoff and the stamp must
    # agree in both directions, so post-holdout images can never be audited as if
    # they were pre-holdout evidence.
    declared_holdout = manifest.get("holdout_read") is True
    try:
        arm_cutoff = utc_timestamp(manifest.get("end_before"), field="end_before")
    except (TypeError, ValueError) as error:
        errors.append(str(error))
        arm_cutoff = HOLDOUT_START
    if (arm_cutoff > HOLDOUT_START) != declared_holdout:
        errors.append(
            "scan manifest end_before and holdout_read disagree: "
            f"end_before={manifest.get('end_before')} holdout_read={manifest.get('holdout_read')}"
        )
    snapshot_info = manifest.get("source_snapshot")
    records: dict[str, SnapshotFile] = {}
    if not isinstance(snapshot_info, dict):
        errors.append("source_snapshot must be an object")
    else:
        relative = snapshot_info.get("manifest")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            errors.append("source snapshot manifest path must be relative")
        else:
            snapshot_path = (root / relative).resolve()
            if sha256_file(snapshot_path) != snapshot_info.get("sha256"):
                errors.append("source snapshot manifest SHA-256 mismatch")
            try:
                snapshot = load_source_manifest(
                    snapshot_path,
                    expected_source_dir=manifest.get("source_dir"),
                    expected_timeframe=normalized,
                    end_before=arm_cutoff,
                    allow_holdout=declared_holdout,
                )
                verify_snapshot_identity(snapshot)
                records = {str(record.path): record for record in snapshot.files}
            except (OSError, ValueError) as error:
                errors.append(f"source snapshot invalid: {error}")
    window = manifest.get("window_bars")
    if window not in WINDOWS:
        errors.append(f"window_bars must be one of {WINDOWS}")
        window = 0
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        errors.append("scan samples must be a non-empty list")
        samples = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    cadence = pd.Timedelta(minutes=timeframe_minutes(normalized))
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{prefix} must be an object")
            continue
        sample_id = sample.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
            errors.append(f"{prefix}.id is empty or duplicated")
        else:
            seen_ids.add(sample_id)
        relative = sample.get("image")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen_images
        ):
            errors.append(f"{prefix}.image is invalid or duplicated")
            continue
        seen_images.add(relative)
        image_path = (root / relative).resolve()
        if not image_path.is_file() or sha256_file(image_path) != sample.get("image_sha256"):
            errors.append(f"{prefix}.image missing or SHA-256 mismatch")
        source_file = sample.get("source_file")
        source_sha = sample.get("source_sha256")
        if source_file not in records or records[source_file].sha256 != source_sha:
            errors.append(f"{prefix}.source identity mismatch")
        try:
            start = pd.Timestamp(sample.get("window_start_time"))
            end_open = pd.Timestamp(sample.get("window_end_open_time"))
            end_close = pd.Timestamp(sample.get("window_end_close_time"))
            available = pd.Timestamp(sample.get("available_at"))
        except (TypeError, ValueError) as error:
            errors.append(f"{prefix} has invalid time: {error}")
            continue
        if any(value.tzinfo is None for value in (start, end_open, end_close, available)):
            errors.append(f"{prefix} times must be timezone aware")
        elif not (
            end_open == start + (int(window) - 1) * cadence
            and end_close == end_open + cadence
            and available == end_close
            and available <= arm_cutoff
        ):
            errors.append(f"{prefix} violates window/availability contract")
        start_index = sample.get("source_start_index")
        end_index = sample.get("source_end_index")
        if (
            isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or isinstance(end_index, bool)
            or not isinstance(end_index, int)
            or end_index - start_index + 1 != window
        ):
            errors.append(f"{prefix} source indices disagree with window")
    contract = _payload_hash(_contract_rows(samples))
    if manifest.get("scan_contract_sha256") != contract:
        errors.append("scan contract SHA-256 mismatch")
    return {
        "schema_version": 1,
        "audit_type": "yolo_xx_scan_arm_audit",
        "scan_arm": str(root),
        "valid": not errors,
        "sample_count": len(samples),
        "scan_contract_sha256": contract,
        "errors": errors,
    }


def audit_scan_pair(pair_root: str | Path) -> dict[str, object]:
    """Verify both scan arms share the same source endpoints and identities."""
    root = Path(pair_root).resolve()
    audits = {name: audit_scan_arm(root / name) for name in ("w200", "w96")}
    errors = [f"{name} scan audit failed" for name, audit in audits.items() if not audit["valid"]]
    try:
        manifests = {
            name: json.loads((root / name / SCAN_MANIFEST).read_text(encoding="utf-8"))
            for name in ("w200", "w96")
        }
        if _contract_rows(manifests["w200"]["samples"]) != _contract_rows(
            manifests["w96"]["samples"]
        ):
            errors.append("scan A/B endpoint ledgers differ")
        contract = _payload_hash(_contract_rows(manifests["w200"]["samples"]))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        errors.append(f"scan pair manifests unreadable: {error}")
        contract = None
    if not (root / PAIR_MANIFEST).is_file():
        errors.append(f"missing {PAIR_MANIFEST}")
    return {
        "schema_version": 1,
        "audit_type": "yolo_xx_scan_pair_audit",
        "scan_pair": str(root),
        "valid": not errors,
        "scan_contract_sha256": contract,
        "arms": audits,
        "errors": errors,
    }


def create_scan_receipt(*, arm_dir: str | Path, out: str | Path) -> dict[str, object]:
    """Pin one fully audited scan arm for a source-free GPU worker."""
    root = Path(arm_dir).resolve()
    output = Path(out).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scan receipt: {output}")
    audit = audit_scan_arm(root)
    if not audit["valid"]:
        raise ValueError("cannot receipt an invalid scan arm: " + "; ".join(audit["errors"][:5]))
    manifest_path = root / SCAN_MANIFEST
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [
        {"path": str(item["image"]), "sha256": str(item["image_sha256"])}
        for item in sorted(manifest["samples"], key=lambda value: str(value["image"]))
    ]
    payload = {
        "schema_version": 1,
        "manifest_type": RECEIPT_TYPE,
        "full_source_audit_valid": True,
        "holdout_read": manifest.get("holdout_read") is True,
        "arm_root_name": root.name,
        "timeframe": manifest["timeframe"],
        "window_bars": manifest["window_bars"],
        "end_before": manifest["end_before"],
        "scan_manifest": {"path": SCAN_MANIFEST, "sha256": sha256_file(manifest_path)},
        "source_snapshot_sha256": manifest["source_snapshot"]["sha256"],
        "sample_count": len(manifest["samples"]),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    return {
        "schema_version": 1,
        "manifest_type": RECEIPT_TYPE,
        "receipt": str(output),
        "receipt_sha256": sha256_file(output),
        "timeframe": manifest["timeframe"],
        "window_bars": manifest["window_bars"],
        "sample_count": len(files),
        "file_count": len(files),
        "full_source_audit_valid": True,
        "holdout_read": manifest.get("holdout_read") is True,
    }


def verify_scan_receipt(
    *, arm_dir: str | Path, receipt: str | Path, expected_receipt_sha256: str
) -> dict[str, object]:
    """Verify an unlabeled image payload without reading its source OHLCV."""
    root = Path(arm_dir).resolve()
    receipt_path = Path(receipt).resolve()
    if len(expected_receipt_sha256) != 64 or sha256_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("portable scan receipt SHA-256 mismatch")
    payload: Any = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("portable scan receipt schema_version must be 1")
    if payload.get("manifest_type") != RECEIPT_TYPE:
        raise ValueError(f"portable scan receipt type must be {RECEIPT_TYPE}")
    if payload.get("full_source_audit_valid") is not True:
        raise ValueError("portable scan receipt lacks a safe full-audit declaration")
    receipt_holdout = payload.get("holdout_read") is True
    if payload.get("arm_root_name") != root.name:
        raise ValueError("portable scan receipt arm name mismatch")
    if (pd.Timestamp(payload.get("end_before")) > HOLDOUT_START) != receipt_holdout:
        raise ValueError(
            "portable scan receipt end_before and holdout_read disagree"
        )
    manifest_info = payload.get("scan_manifest")
    if not isinstance(manifest_info, dict) or manifest_info.get("path") != SCAN_MANIFEST:
        raise ValueError("portable scan receipt manifest identity is missing")
    manifest_path = root / SCAN_MANIFEST
    if sha256_file(manifest_path) != manifest_info.get("sha256"):
        raise ValueError("portable scan manifest SHA-256 mismatch")
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("timeframe") != payload.get("timeframe")
        or manifest.get("window_bars") != payload.get("window_bars")
        or manifest.get("source_snapshot", {}).get("sha256")
        != payload.get("source_snapshot_sha256")
    ):
        raise ValueError("portable scan manifest semantics mismatch")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != payload.get("sample_count"):
        raise ValueError("portable scan sample count mismatch")
    expected = {str(item["image"]): str(item["image_sha256"]) for item in samples}
    files = payload.get("files")
    if not isinstance(files, list):
        raise ValueError("portable scan receipt files must be a list")
    declared: dict[str, str] = {}
    for index, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(f"portable scan files[{index}] must be an object")
        relative, digest = item.get("path"), item.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"portable scan files[{index}] lacks path/SHA-256")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in declared:
            raise ValueError(f"portable scan files[{index}] path is unsafe/duplicated")
        actual_path = (root / path).resolve()
        try:
            actual_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"portable scan files[{index}] escapes arm root") from error
        if sha256_file(actual_path) != digest:
            raise ValueError(f"portable scan image SHA-256 mismatch: {relative}")
        declared[relative] = digest
    if declared != expected:
        raise ValueError("portable scan receipt file ledger differs from manifest")
    return {
        "schema_version": 1,
        "audit_type": "yolo_xx_portable_scan_receipt_audit",
        "valid": True,
        "scan_arm": str(root),
        "timeframe": manifest["timeframe"],
        "window_bars": manifest["window_bars"],
        "sample_count": len(samples),
        "receipt_sha256": expected_receipt_sha256,
        "holdout_read": False,
    }


def build_scan_pair(
    *,
    snapshot_dir: str | Path,
    out_dir: str | Path,
    max_images: int = 512,
    allow_holdout: bool = False,
    since: object | None = None,
    until: object | None = None,
) -> dict[str, object]:
    """Render paired w200/w96 unlabeled images at identical window endpoints.

    Endpoints stop at the snapshot's own declared cutoff.  Scanning a holdout
    snapshot requires `allow_holdout=True`, and every emitted manifest keeps the
    `holdout_read` stamp so the audit can enforce it later.
    """
    snapshot_root = Path(snapshot_dir).resolve()
    output = Path(out_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scan set: {output}")
    snapshot_path = snapshot_root / "source_snapshot.json"
    snapshot = load_source_manifest(
        snapshot_path,
        expected_source_dir=snapshot_root,
        allow_holdout=allow_holdout,
    )
    if snapshot.timeframe not in {"1m", "2m", "3m", "5m", "15m", "30m"}:
        raise ValueError("scan snapshot timeframe must be 1m, 2m, 3m, 5m, 15m, or 30m")
    cutoff = snapshot.cutoff_exclusive
    # A 15m-calibrated vertical floor squashes a 1m chart out of the detector's
    # training domain, so the floor follows the timeframe being rendered.
    span_floor = min_rel_span_for(snapshot.timeframe)
    since_ts = utc_timestamp(since, field="since") if since is not None else None
    until_ts = utc_timestamp(until, field="until") if until is not None else None
    verify_snapshot_identity(snapshot)
    anchors = _select_anchors(
        snapshot.files,
        timeframe=snapshot.timeframe,
        max_images=max_images,
        since=since_ts,
        until=until_ts,
    )
    if not anchors:
        raise ValueError("scan snapshot produced no eligible 200-bar endpoints")
    earliest = min(anchor[2] for anchor in anchors)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        arms = {window: staging / f"w{window}" for window in WINDOWS}
        for arm in arms.values():
            (arm / "images").mkdir(parents=True)
        anchors_by_source: dict[Path, list[tuple[SnapshotFile, int, pd.Timestamp]]] = defaultdict(list)
        for anchor in anchors:
            anchors_by_source[anchor[0].path].append(anchor)
        samples_by_window: dict[int, list[dict[str, object]]] = defaultdict(list)
        for source_path in sorted(anchors_by_source, key=str):
            record = anchors_by_source[source_path][0][0]
            frame = _load_frame(record, snapshot.timeframe)
            symbol = _symbol(record, snapshot.timeframe)
            for _, end_index, end_open in anchors_by_source[source_path]:
                stamp = end_open.strftime("%Y%m%dT%H%M%SZ")
                sample_id = f"{symbol}__{stamp}"
                for window in WINDOWS:
                    start_index = end_index - window + 1
                    subframe = frame.iloc[start_index : end_index + 1].reset_index(drop=True)
                    image_path = arms[window] / "images" / f"{sample_id}.png"
                    render_chart(
                        subframe,
                        out_path=image_path,
                        ma_periods=(20, 60, 120),
                        min_rel_span=span_floor,
                    )
                    start_time = pd.Timestamp(frame.iloc[start_index]["open_time"])
                    end_close = end_open + pd.Timedelta(
                        minutes=timeframe_minutes(snapshot.timeframe)
                    )
                    samples_by_window[window].append(
                        {
                            "id": sample_id,
                            "symbol": symbol,
                            "source_file": str(record.path),
                            "source_sha256": record.sha256,
                            "source_start_index": start_index,
                            "source_end_index": end_index,
                            "image": image_path.relative_to(arms[window]).as_posix(),
                            "image_sha256": sha256_file(image_path),
                            "window_start_time": utc_iso(start_time),
                            "window_end_open_time": utc_iso(end_open),
                            "window_end_close_time": utc_iso(end_close),
                            "available_at": utc_iso(end_close),
                        }
                    )
            verify_snapshot_file(record)

        contract_rows = _contract_rows(samples_by_window[max(WINDOWS)])
        if contract_rows != _contract_rows(samples_by_window[min(WINDOWS)]):
            raise AssertionError("rendered scan endpoint ledgers differ")
        contract = _payload_hash(contract_rows)
        for window in WINDOWS:
            arm = arms[window]
            snapshot_copy = arm / "source_snapshot_manifest.json"
            snapshot_copy.write_bytes(snapshot_path.read_bytes())
            manifest = {
                "schema_version": 1,
                "manifest_type": "yolo_xx_scan_set",
                "unlabeled": True,
                "holdout_read": snapshot.holdout_read,
                "source_dir": str(snapshot_root),
                "source_snapshot": {
                    "manifest": snapshot_copy.relative_to(arm).as_posix(),
                    "sha256": snapshot.manifest_sha256,
                },
                "timeframe": snapshot.timeframe,
                "ma_periods_bars": [20, 60, 120],
                "min_rel_span": span_floor,
                "window_bars": window,
                "pixels_per_bar": round((IMG_WIDTH - 2 * MARGIN) / (window - 1), 6),
                "end_before": utc_iso(cutoff),
                "endpoints_since": utc_iso(since_ts) if since_ts is not None else None,
                "endpoints_until": utc_iso(until_ts) if until_ts is not None else None,
                "earliest_window_end_open_time": utc_iso(earliest),
                "scan_contract_sha256": contract,
                "samples": sorted(samples_by_window[window], key=lambda item: str(item["id"])),
            }
            _write_json(arm / SCAN_MANIFEST, manifest)
            audit = audit_scan_arm(arm)
            if not audit["valid"]:
                raise ValueError(
                    f"w{window} scan audit failed: " + "; ".join(audit["errors"][:5])
                )
        pair_manifest = {
            "schema_version": 1,
            "manifest_type": "yolo_xx_scan_pair",
            "unlabeled": True,
            "holdout_read": snapshot.holdout_read,
            "timeframe": snapshot.timeframe,
            "windows": list(WINDOWS),
            "ma_periods_bars": [20, 60, 120],
            "sample_count_per_arm": len(anchors),
            "symbols": len({_symbol(item[0], snapshot.timeframe) for item in anchors}),
            "scan_contract_sha256": contract,
            "source_snapshot_sha256": snapshot.manifest_sha256,
            "arms": {
                f"w{window}": {
                    "manifest": f"w{window}/{SCAN_MANIFEST}",
                    "manifest_sha256": sha256_file(arms[window] / SCAN_MANIFEST),
                }
                for window in WINDOWS
            },
        }
        _write_json(staging / PAIR_MANIFEST, pair_manifest)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    audit = audit_scan_pair(output)
    _write_json(output / "scan_pair_audit.json", audit)
    if not audit["valid"]:
        raise ValueError("final scan pair audit failed: " + "; ".join(audit["errors"][:5]))
    summary = {
        "schema_version": 1,
        "scan_pair": str(output),
        "timeframe": snapshot.timeframe,
        "sample_count_per_arm": len(anchors),
        "symbols": len({_symbol(item[0], snapshot.timeframe) for item in anchors}),
        "scan_contract_sha256": audit["scan_contract_sha256"],
        "source_snapshot_sha256": snapshot.manifest_sha256,
        "holdout_read": snapshot.holdout_read,
        "audit_valid": True,
    }
    _write_json(output / "scan_pair_summary.json", summary)
    return summary


def make_plan(
    *,
    snapshot_dir: Path,
    out_dir: Path,
    max_images: int,
    allow_holdout: bool = False,
    since: object | None = None,
    until: object | None = None,
) -> dict[str, object]:
    """Return a no-read/no-write scan-set plan."""
    return {
        "dry_run": True,
        "snapshot_dir": str(snapshot_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "windows": list(WINDOWS),
        "ma_periods_bars": [20, 60, 120],
        "max_images_per_arm": max_images,
        "endpoints_since": str(since) if since is not None else None,
        "endpoints_until": str(until) if until is not None else None,
        "unlabeled": True,
        "holdout_read": allow_holdout,
        "training": False,
        "network": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--snapshot-dir", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--max-images", type=int, default=512)
    build.add_argument(
        "--allow-holdout",
        action="store_true",
        help="opt in to scanning a snapshot stamped holdout_read=true",
    )
    build.add_argument(
        "--since",
        help="only place window endpoints at or after this UTC timestamp",
    )
    build.add_argument(
        "--until",
        help="only place window endpoints at or before this UTC timestamp",
    )
    build.add_argument("--dry-run", action="store_true")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--pair-root", required=True, type=Path)
    audit_parser.add_argument("--out", type=Path)
    receipt_parser = subparsers.add_parser("receipt")
    receipt_parser.add_argument("--arm-dir", required=True, type=Path)
    receipt_parser.add_argument("--out", required=True, type=Path)
    verify_parser = subparsers.add_parser("verify-receipt")
    verify_parser.add_argument("--arm-dir", required=True, type=Path)
    verify_parser.add_argument("--receipt", required=True, type=Path)
    verify_parser.add_argument("--receipt-sha256", required=True)
    args = parser.parse_args(argv)
    if args.action == "audit":
        payload = audit_scan_pair(args.pair_root)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            _write_json(args.out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0 if payload["valid"] else 1
    if args.action == "receipt":
        payload = create_scan_receipt(arm_dir=args.arm_dir, out=args.out)
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if args.action == "verify-receipt":
        payload = verify_scan_receipt(
            arm_dir=args.arm_dir,
            receipt=args.receipt,
            expected_receipt_sha256=args.receipt_sha256,
        )
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    common = {
        "snapshot_dir": args.snapshot_dir,
        "out_dir": args.out,
        "max_images": args.max_images,
        "allow_holdout": args.allow_holdout,
        "since": args.since,
        "until": args.until,
    }
    payload = make_plan(**common) if args.dry_run else build_scan_pair(**common)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
