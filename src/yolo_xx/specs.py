"""Timeframe-aware, chart-only specifications for YOLO dataset generation.

The values in this module describe rendering and rule-label geometry only.  MA
durations are expressed in real minutes and converted to candle counts without
looking at future rows or any outcome/trading data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

DEFAULT_TIMEFRAME = "15m"
DEFAULT_MA_MINUTES = (300, 900, 1800)
DEFAULT_DENSE_MIN_MINUTES = 75
DEFAULT_DENSE_MAX_MINUTES = 180
DEFAULT_MERGE_GAP_MINUTES = 30
DEFAULT_PHYSICAL_WINDOW_MINUTES = 3000
_TIMEFRAME_RE = re.compile(r"^([1-9][0-9]*)m$")


def timeframe_minutes(value: str | int) -> int:
    """Return a positive whole-minute cadence from ``1m``-style input."""
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("timeframe minutes must be positive")
        return value
    match = _TIMEFRAME_RE.fullmatch(value.strip().lower())
    if match is None:
        raise ValueError(f"unsupported timeframe {value!r}; expected e.g. '5m'")
    return int(match.group(1))


def canonical_timeframe(value: str | int) -> str:
    """Normalize a supported timeframe to its filename token."""
    return f"{timeframe_minutes(value)}m"


def resolve_window_bars(
    timeframe: str | int,
    window: int | None = None,
    *,
    physical_minutes: int = DEFAULT_PHYSICAL_WINDOW_MINUTES,
) -> int:
    """Resolve a fixed physical chart span unless bars are explicitly supplied."""
    if window is not None:
        if isinstance(window, bool) or window <= 0:
            raise ValueError("window bars must be positive")
        return int(window)
    cadence = timeframe_minutes(timeframe)
    if physical_minutes <= 0 or physical_minutes % cadence != 0:
        raise ValueError("physical window minutes must divide exactly by timeframe")
    return physical_minutes // cadence


def parse_minute_list(raw: str | Iterable[int]) -> tuple[int, ...]:
    """Parse comma-separated physical MA durations into an ordered tuple."""
    if isinstance(raw, str):
        try:
            values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
        except ValueError as error:
            raise ValueError("MA minutes must be comma-separated positive integers") from error
    else:
        values = tuple(int(item) for item in raw)
    if not values or any(value <= 0 for value in values):
        raise ValueError("MA minutes must contain positive integers")
    if tuple(sorted(set(values))) != values:
        raise ValueError("MA minutes must be unique and strictly increasing")
    return values


def ma_column_names(periods: Iterable[int]) -> tuple[str, ...]:
    """Return SMA columns followed by EMA columns for the requested periods."""
    normalized = tuple(int(period) for period in periods)
    return tuple(f"sma{period}" for period in normalized) + tuple(
        f"ema{period}" for period in normalized
    )


def fast_ma_column_names(periods: Iterable[int]) -> tuple[str, ...]:
    """Return SMA/EMA columns for the first two physical MA horizons."""
    normalized = tuple(int(period) for period in periods)
    fast = normalized[:2]
    return tuple(name for period in fast for name in (f"sma{period}", f"ema{period}"))


@dataclass(frozen=True)
class DetectionSpec:
    """Resolved physical-duration contract for one detector dataset.

    MA durations must map exactly to bars so their meaning is unchanged across
    timeframes.  A minimum duration rounds up (never becomes shorter), while a
    maximum duration and merge gap round down.  Requested and resolved values
    are both written to the dataset manifest.
    """

    timeframe: str = DEFAULT_TIMEFRAME
    ma_minutes: tuple[int, ...] = DEFAULT_MA_MINUTES
    dense_min_minutes: int = DEFAULT_DENSE_MIN_MINUTES
    dense_max_minutes: int = DEFAULT_DENSE_MAX_MINUTES
    merge_gap_minutes: int = DEFAULT_MERGE_GAP_MINUTES

    def __post_init__(self) -> None:
        canonical = canonical_timeframe(self.timeframe)
        minutes = parse_minute_list(self.ma_minutes)
        object.__setattr__(self, "timeframe", canonical)
        object.__setattr__(self, "ma_minutes", minutes)
        if len(minutes) != 3:
            raise ValueError("the dense-chart contract requires exactly three MA horizons")
        if self.dense_min_minutes <= 0 or self.dense_max_minutes <= 0:
            raise ValueError("dense durations must be positive")
        if self.dense_min_minutes > self.dense_max_minutes:
            raise ValueError("dense minimum minutes cannot exceed maximum minutes")
        if self.merge_gap_minutes < 0:
            raise ValueError("merge gap minutes cannot be negative")
        cadence = self.bar_minutes
        incompatible = [value for value in minutes if value % cadence != 0]
        if incompatible:
            raise ValueError(
                f"MA minutes {incompatible} are not divisible by {self.timeframe} cadence"
            )
        if self.dense_min_bars > self.dense_max_bars:
            raise ValueError("resolved dense minimum bars exceed maximum bars")

    @property
    def bar_minutes(self) -> int:
        return timeframe_minutes(self.timeframe)

    @property
    def ma_periods(self) -> tuple[int, ...]:
        return tuple(value // self.bar_minutes for value in self.ma_minutes)

    @property
    def dense_min_bars(self) -> int:
        return max(1, math.ceil(self.dense_min_minutes / self.bar_minutes))

    @property
    def dense_max_bars(self) -> int:
        return max(1, self.dense_max_minutes // self.bar_minutes)

    @property
    def merge_gap_bars(self) -> int:
        return self.merge_gap_minutes // self.bar_minutes

    @property
    def warmup_bars(self) -> int:
        """Use the legacy three-longest-MA warmup at every cadence."""
        return 3 * max(self.ma_periods)

    def as_dict(self) -> dict[str, object]:
        """Return requested physical durations and their explicit bar resolution."""
        return {
            "timeframe": self.timeframe,
            "bar_minutes": self.bar_minutes,
            "ma_minutes": list(self.ma_minutes),
            "ma_periods": list(self.ma_periods),
            "dense_min_minutes": self.dense_min_minutes,
            "dense_min_bars": self.dense_min_bars,
            "dense_min_resolved_minutes": self.dense_min_bars * self.bar_minutes,
            "dense_max_minutes": self.dense_max_minutes,
            "dense_max_bars": self.dense_max_bars,
            "dense_max_resolved_minutes": self.dense_max_bars * self.bar_minutes,
            "merge_gap_minutes": self.merge_gap_minutes,
            "merge_gap_bars": self.merge_gap_bars,
            "merge_gap_resolved_minutes": self.merge_gap_bars * self.bar_minutes,
            "warmup_bars": self.warmup_bars,
            "warmup_minutes": self.warmup_bars * self.bar_minutes,
        }
