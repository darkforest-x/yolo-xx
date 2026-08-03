from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from yolo_xx.source_manifest import sha256_file
from yolo_xx.specs import timeframe_minutes


@pytest.fixture
def make_source_manifest():
    """Create an immutable manifest only for synthetic test CSV fixtures."""

    def make(
        cache: Path,
        *,
        timeframe: str,
        cutoff_exclusive: str = "2026-05-04T00:00:00Z",
        files: list[Path] | None = None,
        manifest_path: Path | None = None,
    ) -> Path:
        selected = sorted(files if files is not None else cache.glob("*.csv"))
        entries = []
        cadence = pd.Timedelta(minutes=timeframe_minutes(timeframe))
        for path in selected:
            frame = pd.read_csv(path)
            open_times = pd.to_datetime(frame["ts"], unit="ms", utc=True)
            stat = path.stat()
            entries.append(
                {
                    "path": path.relative_to(cache).as_posix(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(path),
                    "row_count": len(frame),
                    "first_open_time": open_times.iloc[0].isoformat().replace("+00:00", "Z"),
                    "last_open_time": open_times.iloc[-1].isoformat().replace("+00:00", "Z"),
                    "last_closed_at": (open_times.iloc[-1] + cadence)
                    .isoformat()
                    .replace("+00:00", "Z"),
                }
            )
        output = manifest_path or cache / "source_snapshot.json"
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "manifest_type": "yolo_xx_source_snapshot",
                    "immutable": True,
                    "source_dir": str(cache.resolve()),
                    "timeframe": timeframe,
                    "cutoff_exclusive": cutoff_exclusive,
                    "files": entries,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return output

    return make
