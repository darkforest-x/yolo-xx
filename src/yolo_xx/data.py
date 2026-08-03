"""Local OHLCV loading and chart-only moving-average calculation.

This module has no network client and no dependency on the parent repository.
The generated columns use only the current and prior close values: SMA/EMA
20, 60, and 120 plus within-bar relative bundle spreads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MA_PERIODS = (20, 60, 120)
SMA_COLS = tuple(f"sma{period}" for period in MA_PERIODS)
EMA_COLS = tuple(f"ema{period}" for period in MA_PERIODS)
ALL_MA_COLS = SMA_COLS + EMA_COLS
FAST_MA_COLS = ("sma20", "ema20", "sma60", "ema60")
WARMUP_BARS = max(MA_PERIODS)
REQUIRED_COLUMNS = ("ts", "open", "high", "low", "close", "volume")


def load_ohlcv_csv(
    path: str | Path,
    *,
    end_before: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load one local candle CSV and apply an optional strict UTC upper bound."""
    source = Path(path)
    frame = pd.read_csv(source)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {', '.join(missing)}")

    for column in ("ts", "open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["open_time"] = pd.to_datetime(frame["ts"], unit="ms", utc=True, errors="coerce")
    frame = frame.dropna(subset=["open_time", "open", "high", "low", "close"])
    frame = frame.sort_values("open_time").drop_duplicates("open_time", keep="last")

    if end_before is not None:
        cutoff = pd.Timestamp(end_before)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        frame = frame.loc[frame["open_time"] < cutoff]
    return frame.reset_index(drop=True)


def add_mas(frame: pd.DataFrame) -> pd.DataFrame:
    """Add backward-looking SMA/EMA columns and relative bundle spreads."""
    out = frame.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    for period in MA_PERIODS:
        out[f"sma{period}"] = close.rolling(period).mean()
        out[f"ema{period}"] = close.ewm(span=period, adjust=False).mean()
    all_mas = out[list(ALL_MA_COLS)]
    fast_mas = out[list(FAST_MA_COLS)]
    safe_close = close.replace(0, pd.NA)
    out["fast_spread"] = (fast_mas.max(axis=1) - fast_mas.min(axis=1)) / safe_close
    out["full_spread"] = (all_mas.max(axis=1) - all_mas.min(axis=1)) / safe_close
    return out


def cache_symbol(path: str | Path) -> str:
    """Return the filename prefix before a conventional trailing row count."""
    stem = Path(path).stem
    prefix, separator, suffix = stem.rpartition("_")
    return prefix if separator and suffix.isdigit() else stem


def list_cache_files(cache_dir: str | Path, *, min_rows: int = 10_000) -> list[Path]:
    """Select the longest declared 15m CSV per symbol from an explicit directory."""
    root = Path(cache_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"cache directory does not exist: {root}")
    best: dict[str, tuple[int, Path]] = {}
    for path in sorted(root.glob("*_15m_*.csv")):
        if path.name.startswith("gate_"):
            continue
        try:
            declared_rows = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if declared_rows < min_rows:
            continue
        symbol = cache_symbol(path)
        if symbol not in best or declared_rows > best[symbol][0]:
            best[symbol] = (declared_rows, path)
    return [item[1] for item in sorted(best.values(), key=lambda item: item[1].name)]
