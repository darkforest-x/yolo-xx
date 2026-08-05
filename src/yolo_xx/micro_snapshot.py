"""Materialize immutable hashed prefixes for local 1m/2m/3m/5m/15m/30m OHLCV.

Only timestamps are inspected at the frozen boundary.  OHLCV fields are parsed
and copied exclusively for candles whose close is available before the cutoff;
the first boundary-or-later row is never parsed beyond its timestamp.

The cutoff defaults to the frozen pre-holdout start.  Snapshotting anything
later is a deliberate, one-way decision: it requires `--allow-holdout`, and the
resulting manifest is permanently stamped `holdout_read=true` so no downstream
audit can mistake it for pre-holdout evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd

from .source_manifest import HOLDOUT_START, sha256_file, utc_iso, utc_timestamp
from .specs import canonical_timeframe, timeframe_minutes

REQUIRED = ("ts", "open", "high", "low", "close", "volume")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _safe_float(raw: str, *, field: str) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _select_sources(cache_dir: Path, timeframe: str) -> list[Path]:
    pattern = re.compile(
        rf"^okx_(?P<symbol>.+)_{re.escape(timeframe)}_(?P<rows>[0-9]+)(?:_latest)?\.csv$"
    )
    selected: dict[str, tuple[int, Path]] = {}
    for path in sorted(cache_dir.glob(f"okx_*_{timeframe}_*.csv")):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        symbol, rows = match.group("symbol"), int(match.group("rows"))
        current = selected.get(symbol)
        if current is None or (rows, path.name) > (current[0], current[1].name):
            selected[symbol] = (rows, path)
    return [item[1] for item in sorted(selected.values(), key=lambda item: item[1].name)]


def _copy_prefix(
    source: Path,
    destination: Path,
    *,
    timeframe: str,
    cutoff: pd.Timestamp,
) -> dict[str, object] | None:
    cadence_minutes = timeframe_minutes(timeframe)
    cadence_ms = cadence_minutes * 60 * 1000
    cutoff_ms = int(cutoff.value // 1_000_000)
    prefix_digest = hashlib.sha256()
    first_ms: int | None = None
    last_ms: int | None = None
    previous_ms: int | None = None
    row_count = 0
    boundary_seen = False
    with source.open("rb") as source_handle, destination.open(
        "w", newline="", encoding="utf-8"
    ) as destination_handle:
        header_raw = source_handle.readline()
        if not header_raw:
            raise ValueError(f"empty OHLCV source: {source}")
        prefix_digest.update(header_raw)
        header = next(csv.reader([header_raw.decode("utf-8").rstrip("\r\n")]))
        missing = sorted(set(REQUIRED) - set(header))
        if missing:
            raise ValueError(f"{source}: missing columns: {', '.join(missing)}")
        indices = {name: header.index(name) for name in REQUIRED}
        writer = csv.DictWriter(destination_handle, fieldnames=REQUIRED)
        writer.writeheader()
        for raw_bytes in source_handle:
            fields = next(csv.reader([raw_bytes.decode("utf-8").rstrip("\r\n")]))
            if len(fields) < len(header):
                raise ValueError(f"{source}: short CSV row")
            try:
                timestamp_ms = int(fields[indices["ts"]])
            except ValueError as error:
                raise ValueError(f"{source}: invalid timestamp") from error
            # The candle is eligible only when its full OHLC is closed before
            # the cutoff.  Never parse later OHLC fields.
            if timestamp_ms + cadence_ms > cutoff_ms:
                boundary_seen = True
                break
            if previous_ms is not None and timestamp_ms - previous_ms != cadence_ms:
                raise ValueError(f"{source}: pre-holdout cadence is not contiguous {timeframe}")
            values = {"ts": str(timestamp_ms)}
            for name in REQUIRED[1:]:
                raw_value = fields[indices[name]]
                value = _safe_float(raw_value, field=f"{source.name} {name}")
                if name != "volume" and value <= 0:
                    raise ValueError(f"{source}: non-positive {name}")
                values[name] = raw_value
            writer.writerow(values)
            prefix_digest.update(raw_bytes)
            first_ms = timestamp_ms if first_ms is None else first_ms
            last_ms = timestamp_ms
            previous_ms = timestamp_ms
            row_count += 1
    if row_count == 0 or first_ms is None or last_ms is None:
        destination.unlink(missing_ok=True)
        return None
    stat = destination.stat()
    first = pd.Timestamp(first_ms, unit="ms", tz="UTC")
    last = pd.Timestamp(last_ms, unit="ms", tz="UTC")
    return {
        "path": destination.name,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(destination),
        "row_count": row_count,
        "first_open_time": utc_iso(first),
        "last_open_time": utc_iso(last),
        "last_closed_at": utc_iso(last + pd.Timedelta(minutes=cadence_minutes)),
        "origin_path": str(source.resolve()),
        "origin_size_bytes": source.stat().st_size,
        "origin_mtime_ns": source.stat().st_mtime_ns,
        "origin_preholdout_prefix_sha256": prefix_digest.hexdigest(),
        "boundary_timestamp_checked": boundary_seen,
        "post_cutoff_ohlcv_rows_materialized": 0,
    }


def resolve_cutoff(
    timeframe: str, *, allow_holdout: bool = False, cutoff: object | None = None
) -> pd.Timestamp:
    """Return the exclusive cutoff for a snapshot.

    Pre-holdout snapshots always stop at the frozen `HOLDOUT_START`. A holdout
    scan must opt in explicitly, and then stops at the last fully closed candle.
    """
    if not allow_holdout:
        if cutoff is not None:
            raise ValueError("an explicit cutoff requires allow_holdout=True")
        return HOLDOUT_START
    cadence = pd.Timedelta(minutes=timeframe_minutes(timeframe))
    if cutoff is not None:
        resolved = utc_timestamp(cutoff, field="cutoff")
    else:
        resolved = pd.Timestamp.now(tz="UTC").floor(cadence)
    if resolved <= HOLDOUT_START:
        raise ValueError(
            "allow_holdout=True but the cutoff is not past the holdout start; "
            "use the pre-holdout path instead"
        )
    return resolved


def create_snapshot(
    *,
    cache_dir: str | Path,
    out_dir: str | Path,
    timeframe: str,
    min_preholdout_rows: int = 200,
    allow_holdout: bool = False,
    cutoff: object | None = None,
) -> dict[str, object]:
    """Copy one timeframe's eligible local prefixes into a hashed snapshot."""
    normalized = canonical_timeframe(timeframe)
    if normalized not in {"1m", "2m", "3m", "5m", "15m", "30m"}:
        raise ValueError("snapshot timeframe must be 1m, 2m, 3m, 5m, 15m, or 30m")
    boundary = resolve_cutoff(normalized, allow_holdout=allow_holdout, cutoff=cutoff)
    source_root = Path(cache_dir).resolve()
    output = Path(out_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite snapshot: {output}")
    sources = _select_sources(source_root, normalized)
    if not sources:
        raise FileNotFoundError(f"no local {normalized} source CSVs found")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    entries: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    try:
        for source in sources:
            destination = staging / source.name
            entry = _copy_prefix(
                source, destination, timeframe=normalized, cutoff=boundary
            )
            if entry is None:
                excluded.append({"source": str(source), "reason": "zero_preholdout_rows"})
                continue
            if int(entry["row_count"]) < min_preholdout_rows:
                destination.unlink()
                excluded.append(
                    {
                        "source": str(source),
                        "reason": "below_min_preholdout_rows",
                        "row_count": entry["row_count"],
                    }
                )
                continue
            entries.append(entry)
        if not entries:
            raise ValueError(f"no {normalized} source has {min_preholdout_rows} pre-holdout rows")
        manifest = {
            "schema_version": 1,
            "manifest_type": "yolo_xx_source_snapshot",
            "immutable": True,
            "source_dir": str(output),
            "timeframe": normalized,
            "cutoff_exclusive": utc_iso(boundary),
            "files": entries,
            "excluded_sources": excluded,
            "safety": {
                "holdout_read": allow_holdout,
                "boundary_timestamp_only_checked": True,
                "post_cutoff_ohlcv_rows_materialized": 0,
            },
        }
        _write_json(staging / "source_snapshot.json", manifest)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    summary = {
        "schema_version": 1,
        "snapshot_dir": str(output),
        "timeframe": normalized,
        "files": len(entries),
        "preholdout_rows": sum(int(item["row_count"]) for item in entries),
        "excluded_sources": len(excluded),
        "source_snapshot_sha256": sha256_file(output / "source_snapshot.json"),
        "cutoff_exclusive": utc_iso(boundary),
        "holdout_read": allow_holdout,
        "post_cutoff_ohlcv_rows_materialized": 0,
    }
    _write_json(output / "snapshot_summary.json", summary)
    return summary


def make_plan(
    *,
    cache_dir: Path,
    out_dir: Path,
    timeframe: str,
    min_preholdout_rows: int,
    allow_holdout: bool = False,
    cutoff: object | None = None,
) -> dict[str, object]:
    """Return a no-read/no-write snapshot plan."""
    normalized = canonical_timeframe(timeframe)
    return {
        "dry_run": True,
        "cache_dir": str(cache_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "timeframe": normalized,
        "min_preholdout_rows": min_preholdout_rows,
        "cutoff_exclusive": utc_iso(
            resolve_cutoff(normalized, allow_holdout=allow_holdout, cutoff=cutoff)
        ),
        "holdout_read": allow_holdout,
        "network": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--min-preholdout-rows", type=int, default=200)
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help=(
            "opt in to snapshotting data past the frozen holdout start; every "
            "artifact is stamped holdout_read=true and can never be read back "
            "as a pre-holdout snapshot"
        ),
    )
    parser.add_argument(
        "--cutoff",
        help="explicit exclusive UTC cutoff; requires --allow-holdout",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    common = {
        "cache_dir": args.cache_dir,
        "out_dir": args.out,
        "timeframe": args.timeframe,
        "min_preholdout_rows": args.min_preholdout_rows,
        "allow_holdout": args.allow_holdout,
        "cutoff": args.cutoff,
    }
    payload = make_plan(**common) if args.dry_run else create_snapshot(**common)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
