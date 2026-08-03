"""Local OHLCV loading and chart-only moving-average calculation.

This module has no network client and no dependency on the parent repository.
The generated columns use only the current and prior close values.  The legacy
default is SMA/EMA 20, 60, and 120 on 15m bars; callers may provide equivalent
periods for another whole-minute timeframe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .specs import (
    DEFAULT_TIMEFRAME,
    canonical_timeframe,
    fast_ma_column_names,
    ma_column_names,
    timeframe_minutes,
)

MA_PERIODS = (20, 60, 120)
SMA_COLS = tuple(f"sma{period}" for period in MA_PERIODS)
EMA_COLS = tuple(f"ema{period}" for period in MA_PERIODS)
ALL_MA_COLS = ma_column_names(MA_PERIODS)
FAST_MA_COLS = fast_ma_column_names(MA_PERIODS)
WARMUP_BARS = max(MA_PERIODS)
REQUIRED_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


def load_ohlcv_csv(
    path: str | Path,
    *,
    end_before: str | pd.Timestamp | None = None,
    timeframe: str = DEFAULT_TIMEFRAME,
    strict_cadence: bool = True,
) -> pd.DataFrame:
    """Load one local candle CSV and enforce an optional exact UTC cadence.

    Strict cadence rejects duplicate, out-of-order, and missing bars instead of
    silently repairing them.  Set ``strict_cadence=False`` only for a deliberate
    audit/repair workflow; dataset builds keep it enabled by default.
    """
    source = Path(path)
    normalized_timeframe = canonical_timeframe(timeframe)
    cadence = pd.Timedelta(minutes=timeframe_minutes(normalized_timeframe))
    frame = pd.read_csv(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {', '.join(missing)}")

    for column in ("ts", "open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["ts"], unit="ms", utc=True, errors="coerce")
    if strict_cadence:
        invalid = frame[list(REQUIRED_COLUMNS)].isna()
        invalid["ts"] = invalid["ts"] | frame["open_time"].isna()
        bad_rows = invalid.any(axis=1)
        if bad_rows.any():
            index = bad_rows.loc[bad_rows].index[0]
            columns = [column for column in REQUIRED_COLUMNS if bool(invalid.loc[index, column])]
            raise ValueError(
                f"{source}: unparseable required value(s) at row {index}: "
                + ", ".join(columns)
            )
    else:
        frame = frame.dropna(subset=["open_time", "open", "high", "low", "close"])

    if end_before is not None:
        cutoff = pd.Timestamp(end_before)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        # Keep only bars whose full OHLC input is available by the boundary.
        frame = frame.loc[frame["open_time"] + cadence <= cutoff]

    if strict_cadence:
        if frame["open_time"].duplicated().any():
            duplicate = frame.loc[frame["open_time"].duplicated(), "open_time"].iloc[0]
            raise ValueError(f"{source}: duplicate candle at {duplicate.isoformat()}")
        if not frame["open_time"].is_monotonic_increasing:
            raise ValueError(f"{source}: candles are not ordered by open time")
        epoch_ns = frame["open_time"].astype("int64")
        cadence_ns = int(cadence.value)
        misaligned = frame.loc[(epoch_ns % cadence_ns) != 0, "open_time"]
        if not misaligned.empty:
            raise ValueError(
                f"{source}: candle {misaligned.iloc[0].isoformat()} is not aligned "
                f"to {normalized_timeframe} UTC cadence"
            )
        deltas = frame["open_time"].diff().iloc[1:]
        invalid = deltas.loc[deltas != cadence]
        if not invalid.empty:
            index = invalid.index[0]
            previous = frame.loc[frame.index[frame.index.get_loc(index) - 1], "open_time"]
            current = frame.loc[index, "open_time"]
            raise ValueError(
                f"{source}: non-contiguous {normalized_timeframe} cadence between "
                f"{previous.isoformat()} and {current.isoformat()}"
            )
    else:
        frame = frame.sort_values("open_time").drop_duplicates("open_time", keep="last")
    return frame.reset_index(drop=True)


def add_mas(
    frame: pd.DataFrame,
    *,
    periods: Iterable[int] = MA_PERIODS,
) -> pd.DataFrame:
    """Add backward-looking SMA/EMA columns and relative bundle spreads."""
    normalized = tuple(int(period) for period in periods)
    if len(normalized) < 2 or any(period <= 0 for period in normalized):
        raise ValueError("periods must contain at least two positive integers")
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    for period in normalized:
        out[f"sma{period}"] = close.rolling(period).mean()
        out[f"ema{period}"] = close.ewm(span=period, adjust=False).mean()
    all_mas = out[list(ma_column_names(normalized))]
    fast_mas = out[list(fast_ma_column_names(normalized))]
    safe_close = close.replace(0, pd.NA)
    out["fast_spread"] = (fast_mas.max(axis=1) - fast_mas.min(axis=1)) / safe_close
    out["full_spread"] = (all_mas.max(axis=1) - all_mas.min(axis=1)) / safe_close
    return out


def cache_symbol(
    path: str | Path,
    *,
    timeframe: str | None = None,
) -> str:
    """Return the symbol key without a row count or timeframe filename suffix."""
    stem = Path(path).stem
    prefix, separator, suffix = stem.rpartition("_")
    symbol = prefix if separator and suffix.isdigit() else stem
    if timeframe is not None:
        timeframe_suffix = f"_{canonical_timeframe(timeframe)}"
        return symbol[: -len(timeframe_suffix)] if symbol.endswith(timeframe_suffix) else symbol
    candidate, separator, token = symbol.rpartition("_")
    if separator and token.endswith("m") and token[:-1].isdigit():
        return candidate
    return symbol


def list_cache_files(
    cache_dir: str | Path,
    *,
    min_rows: int = 10_000,
    timeframe: str = DEFAULT_TIMEFRAME,
) -> list[Path]:
    """Select the longest declared CSV per symbol for one exact timeframe."""
    root = Path(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"cache directory does not exist: {root}")
    normalized_timeframe = canonical_timeframe(timeframe)
    best: dict[str, tuple[int, Path]] = {}
    for path in sorted(root.glob(f"*_{normalized_timeframe}_*.csv")):
        if path.name.startswith("gate_"):
            continue
        try:
            declared_rows = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if declared_rows < min_rows:
            continue
        symbol = cache_symbol(path, timeframe=normalized_timeframe)
        if symbol not in best or declared_rows > best[symbol][0]:
            best[symbol] = (declared_rows, path)
    return [item[1] for item in sorted(best.values(), key=lambda item: item[1].name)]
