"""Fail-closed identity checks for immutable, physically pre-holdout CSV snapshots.

The snapshot manifest is deliberately separate from dataset generation.  A real
build authenticates paths, stat fields, and file hashes before pandas is allowed
to parse a CSV.  This prevents a mixed or later-mutated market file from being
silently truncated after it has already crossed the holdout boundary.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .specs import canonical_timeframe, timeframe_minutes

HOLDOUT_START = pd.Timestamp("2026-05-04T00:00:00Z")
SOURCE_MANIFEST_SCHEMA_VERSION = 1
SOURCE_MANIFEST_TYPE = "yolo_xx_source_snapshot"


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one local artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_timestamp(value: object, *, field: str) -> pd.Timestamp:
    """Parse a timezone-aware UTC timestamp and reject ambiguous local time."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a valid timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must include a UTC timezone")
    return timestamp.tz_convert("UTC")


def utc_iso(value: object, *, field: str = "timestamp") -> str:
    """Serialize one accepted timestamp as canonical UTC ISO-8601."""
    return utc_timestamp(value, field=field).isoformat().replace("+00:00", "Z")


def enforce_preholdout(value: object, *, field: str) -> pd.Timestamp:
    """Return an accepted boundary no later than the immutable holdout start."""
    timestamp = utc_timestamp(value, field=field)
    if timestamp > HOLDOUT_START:
        raise ValueError(
            f"{field} {timestamp.isoformat()} is later than holdout start "
            f"{HOLDOUT_START.isoformat()}"
        )
    return timestamp


def resolve_boundary(
    value: object, *, field: str, allow_holdout: bool = False
) -> pd.Timestamp:
    """Return an accepted boundary; crossing the holdout start needs an opt-in.

    Default behaviour is unchanged and fail-closed: without `allow_holdout` any
    boundary later than `HOLDOUT_START` is rejected.
    """
    if not allow_holdout:
        return enforce_preholdout(value, field=field)
    return utc_timestamp(value, field=field)


@dataclass(frozen=True)
class SnapshotFile:
    """One authenticated CSV and its declared temporal extent."""

    path: Path
    relative_path: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    row_count: int
    first_open_time: pd.Timestamp
    last_open_time: pd.Timestamp
    last_closed_at: pd.Timestamp

    def as_manifest_dict(self) -> dict[str, object]:
        """Return normalized, JSON-safe evidence for a dataset manifest."""
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "first_open_time": utc_iso(self.first_open_time),
            "last_open_time": utc_iso(self.last_open_time),
            "last_closed_at": utc_iso(self.last_closed_at),
        }


@dataclass(frozen=True)
class SourceSnapshot:
    """Validated immutable snapshot declaration; CSV contents are not yet read."""

    manifest_path: Path
    manifest_sha256: str
    source_dir: Path
    timeframe: str
    cutoff_exclusive: pd.Timestamp
    files: tuple[SnapshotFile, ...]
    holdout_read: bool = False


def _safe_child(root: Path, raw_path: object, *, field: str) -> tuple[Path, str]:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{field} must be a non-empty path string")
    candidate = Path(raw_path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes source_dir: {raw_path}") from error
    if not relative.parts:
        raise ValueError(f"{field} must name a file below source_dir")
    return resolved, relative.as_posix()


def _positive_int(value: object, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def load_source_manifest(
    manifest_path: str | Path,
    *,
    expected_source_dir: str | Path | None = None,
    expected_timeframe: str | None = None,
    end_before: object | None = None,
    allow_holdout: bool = False,
) -> SourceSnapshot:
    """Validate manifest schema and time claims without parsing any source CSV."""
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source manifest does not exist: {path}")
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"source manifest is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("source manifest root must be an object")
    if payload.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"source manifest schema_version must be {SOURCE_MANIFEST_SCHEMA_VERSION}"
        )
    if payload.get("manifest_type") != SOURCE_MANIFEST_TYPE:
        raise ValueError(f"source manifest type must be {SOURCE_MANIFEST_TYPE}")
    if payload.get("immutable") is not True:
        raise ValueError("source manifest must declare immutable=true")

    raw_source_dir = payload.get("source_dir")
    if not isinstance(raw_source_dir, str) or not raw_source_dir:
        raise ValueError("source manifest source_dir must be a non-empty path")
    source_dir = Path(raw_source_dir).resolve()
    if expected_source_dir is not None and source_dir != Path(expected_source_dir).resolve():
        raise ValueError(
            f"source manifest source_dir {source_dir} does not match requested "
            f"cache_dir {Path(expected_source_dir).resolve()}"
        )

    timeframe = canonical_timeframe(payload.get("timeframe", ""))
    if expected_timeframe is not None and timeframe != canonical_timeframe(expected_timeframe):
        raise ValueError(
            f"source manifest timeframe {timeframe} does not match requested "
            f"{canonical_timeframe(expected_timeframe)}"
        )
    safety = payload.get("safety")
    declared_holdout = bool(safety.get("holdout_read")) if isinstance(safety, dict) else False
    if declared_holdout and not allow_holdout:
        raise ValueError(
            "source manifest declares holdout_read=true; loading it requires an "
            "explicit holdout opt-in"
        )
    cutoff = resolve_boundary(
        payload.get("cutoff_exclusive"),
        field="cutoff_exclusive",
        allow_holdout=allow_holdout,
    )
    # Both directions are enforced: post-holdout data must say so, and a manifest
    # may not claim holdout provenance it does not have.
    if cutoff > HOLDOUT_START and not declared_holdout:
        raise ValueError(
            f"cutoff_exclusive {cutoff.isoformat()} is past holdout start but the "
            "manifest does not declare safety.holdout_read=true"
        )
    if declared_holdout and cutoff <= HOLDOUT_START:
        raise ValueError(
            "manifest declares safety.holdout_read=true but cutoff_exclusive is "
            "not past the holdout start"
        )
    requested_end = (
        resolve_boundary(end_before, field="end_before", allow_holdout=allow_holdout)
        if end_before is not None
        else cutoff
    )

    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("source manifest files must be a non-empty list")
    cadence = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    files: list[SnapshotFile] = []
    seen: set[Path] = set()
    for index, item in enumerate(raw_files):
        prefix = f"files[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be an object")
        resolved, relative = _safe_child(source_dir, item.get("path"), field=f"{prefix}.path")
        if resolved in seen:
            raise ValueError(f"duplicate source file in manifest: {resolved}")
        seen.add(resolved)
        digest = item.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest.lower())
        ):
            raise ValueError(f"{prefix}.sha256 must be a 64-character hex digest")
        first_open = utc_timestamp(item.get("first_open_time"), field=f"{prefix}.first_open_time")
        last_open = utc_timestamp(item.get("last_open_time"), field=f"{prefix}.last_open_time")
        last_closed = utc_timestamp(item.get("last_closed_at"), field=f"{prefix}.last_closed_at")
        if first_open > last_open:
            raise ValueError(f"{prefix} first_open_time is after last_open_time")
        if last_closed != last_open + cadence:
            raise ValueError(f"{prefix}.last_closed_at must equal last_open_time plus timeframe")
        if last_closed > cutoff:
            raise ValueError(f"{prefix}.last_closed_at exceeds cutoff_exclusive")
        if last_closed > requested_end:
            raise ValueError(f"{prefix}.last_closed_at exceeds requested end_before")
        files.append(
            SnapshotFile(
                path=resolved,
                relative_path=relative,
                size_bytes=_positive_int(
                    item.get("size_bytes"), field=f"{prefix}.size_bytes", allow_zero=True
                ),
                mtime_ns=_positive_int(
                    item.get("mtime_ns"), field=f"{prefix}.mtime_ns", allow_zero=True
                ),
                sha256=digest.lower(),
                row_count=_positive_int(item.get("row_count"), field=f"{prefix}.row_count"),
                first_open_time=first_open,
                last_open_time=last_open,
                last_closed_at=last_closed,
            )
        )
    return SourceSnapshot(
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        source_dir=source_dir,
        timeframe=timeframe,
        cutoff_exclusive=cutoff,
        files=tuple(files),
        holdout_read=declared_holdout,
    )


def verify_snapshot_file(record: SnapshotFile) -> None:
    """Verify path, stat identity, and hash with before/after drift detection."""
    path = record.path
    if path.is_symlink():
        raise ValueError(f"snapshot source must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"snapshot source file does not exist: {path}")
    before = path.stat()
    if before.st_size != record.size_bytes or before.st_mtime_ns != record.mtime_ns:
        raise ValueError(f"snapshot stat mismatch before CSV read: {path}")
    digest = sha256_file(path)
    after = path.stat()
    if (
        after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_size != record.size_bytes
        or after.st_mtime_ns != record.mtime_ns
    ):
        raise ValueError(f"snapshot source drifted during hash verification: {path}")
    if digest != record.sha256:
        raise ValueError(f"snapshot SHA-256 mismatch before CSV read: {path}")


def verify_snapshot_identity(snapshot: SourceSnapshot) -> None:
    """Authenticate every declared source without invoking pandas CSV parsing."""
    for record in snapshot.files:
        verify_snapshot_file(record)


def verify_loaded_frame(
    frame: pd.DataFrame,
    record: SnapshotFile,
    *,
    timeframe: str,
) -> None:
    """Cross-check parsed row count and exact first/last candle availability."""
    if len(frame) != record.row_count:
        raise ValueError(
            f"snapshot row_count mismatch after CSV read: {record.path}: "
            f"expected {record.row_count}, got {len(frame)}"
        )
    if frame.empty:
        raise ValueError(f"snapshot source unexpectedly parsed as empty: {record.path}")
    first_open = utc_timestamp(frame.iloc[0]["open_time"], field="parsed first_open_time")
    last_open = utc_timestamp(frame.iloc[-1]["open_time"], field="parsed last_open_time")
    last_closed = last_open + pd.Timedelta(minutes=timeframe_minutes(timeframe))
    if first_open != record.first_open_time:
        raise ValueError(f"snapshot first_open_time mismatch after CSV read: {record.path}")
    if last_open != record.last_open_time:
        raise ValueError(f"snapshot last_open_time mismatch after CSV read: {record.path}")
    if last_closed != record.last_closed_at:
        raise ValueError(f"snapshot last_closed_at mismatch after CSV read: {record.path}")
