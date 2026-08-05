"""Simulate what happens after a small-timeframe detection.

The detector is trained on owner-labelled short setups.  A completed dense
cluster on 2m finishes in about 25 minutes, versus about 3 hours on 15m, so the
"after the fact" detection is still early in wall-clock terms.  This module
takes each small-timeframe detection as a short signal at its `available_at`
and simulates an entry on a slower timeframe.

No lookahead: entry is the first entry-timeframe bar that OPENS at or after the
signal time, and the ATR that sizes the barriers is computed strictly from bars
before that entry.  When a single bar touches both barriers the stop is assumed
to fill first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data import load_ohlcv_csv
from .source_manifest import utc_iso
from .specs import canonical_timeframe, timeframe_minutes


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def average_true_range(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ATR; index-aligned so a row only ever sees its own past."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def load_signals(scan_root: Path, timeframes: Sequence[str]) -> list[dict[str, Any]]:
    """Read detections and treat each one as a short signal at `available_at`."""
    signals: list[dict[str, Any]] = []
    for timeframe in timeframes:
        path = scan_root / timeframe / "predictions.json"
        if not path.is_file():
            continue
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["items"]:
            for detection in item["detections"]:
                signals.append(
                    {
                        "signal_timeframe": str(payload["timeframe"]),
                        "symbol": str(item["symbol"]),
                        "sample_id": str(item["sample_id"]),
                        "signal_time": pd.Timestamp(detection["available_at"]),
                        "confidence": round(float(detection["confidence"]), 6),
                    }
                )
    signals.sort(key=lambda item: (item["signal_time"], item["symbol"]))
    return signals


def _entry_frames(price_dir: Path, timeframe: str) -> dict[str, pd.DataFrame]:
    """Load one entry timeframe's OHLCV per symbol, with ATR attached."""
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(price_dir.glob(f"okx_*_{timeframe}_*.csv")):
        symbol = "okx_" + path.name[len("okx_") :].rsplit(f"_{timeframe}_", 1)[0]
        frame = load_ohlcv_csv(path, timeframe=timeframe, strict_cadence=True)
        frame["atr"] = average_true_range(frame)
        frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
        frames[symbol] = frame.reset_index(drop=True)
    return frames


def simulate_short(
    signal: dict[str, Any],
    frame: pd.DataFrame,
    *,
    take_profit_atr: float,
    stop_loss_atr: float,
    max_bars: int,
    round_trip_cost: float,
) -> dict[str, Any] | None:
    """Walk one short trade forward bar by bar and record how it ended."""
    # pandas keeps the UTC tz here; numpy's searchsorted would drop it.
    position = int(frame["open_time"].searchsorted(signal["signal_time"], side="left"))
    if position >= len(frame):
        return None  # signal lands past the end of local price data
    entry_row = frame.iloc[position]
    atr = float(entry_row["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return None  # not enough history yet to size the barriers
    entry = float(entry_row["open"])
    take_profit = entry - take_profit_atr * atr
    stop_loss = entry + stop_loss_atr * atr

    outcome, exit_price, exit_time, held = "open", None, None, 0
    for offset in range(1, max_bars + 1):
        index = position + offset
        if index >= len(frame):
            break
        row = frame.iloc[index]
        held = offset
        # Pessimistic: if one bar spans both barriers, assume the stop filled.
        if float(row["high"]) >= stop_loss:
            outcome, exit_price, exit_time = "stop", stop_loss, row["open_time"]
            break
        if float(row["low"]) <= take_profit:
            outcome, exit_price, exit_time = "target", take_profit, row["open_time"]
            break
    else:
        index = min(position + max_bars, len(frame) - 1)
        row = frame.iloc[index]
        outcome, exit_price, exit_time = "timeout", float(row["close"]), row["open_time"]

    if exit_price is None:
        return {**signal, "outcome": "truncated", "note": "price data ended first"}
    gross = (entry - exit_price) / entry  # short: profit when price falls
    return {
        **signal,
        "outcome": outcome,
        "entry_time": entry_row["open_time"],
        "entry_price": entry,
        "atr": atr,
        "take_profit": take_profit,
        "stop_loss": stop_loss,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "bars_held": held,
        "gross_return": gross,
        "net_return": gross - round_trip_cost,
    }


def _stats(trades: Sequence[dict[str, Any]]) -> dict[str, Any]:
    closed = [t for t in trades if t["outcome"] in {"target", "stop", "timeout"}]
    if not closed:
        return {"closed": 0}
    net = [t["net_return"] for t in closed]
    wins = [value for value in net if value > 0]
    losses = [value for value in net if value <= 0]
    gain, pain = sum(wins), -sum(losses)
    return {
        "closed": len(closed),
        "target": sum(1 for t in closed if t["outcome"] == "target"),
        "stop": sum(1 for t in closed if t["outcome"] == "stop"),
        "timeout": sum(1 for t in closed if t["outcome"] == "timeout"),
        "win_rate": round(len(wins) / len(closed), 6),
        "mean_net_return": round(float(np.mean(net)), 6),
        "median_net_return": round(float(np.median(net)), 6),
        "total_net_return": round(float(np.sum(net)), 6),
        "profit_factor": round(gain / pain, 4) if pain > 0 else None,
        "median_bars_held": float(np.median([t["bars_held"] for t in closed])),
    }


def run(
    *,
    scan_root: Path,
    price_dir: Path,
    signal_timeframes: Sequence[str],
    entry_timeframes: Sequence[str],
    take_profit_atr: float,
    stop_loss_atr: float,
    max_bars: int,
    round_trip_cost: float,
    cooldown_minutes: float,
) -> dict[str, Any]:
    signals = load_signals(scan_root, signal_timeframes)
    results: dict[str, Any] = {
        "schema_version": 1,
        "report_type": "yolo_xx_signal_outcome",
        "direction": "short",
        "signal_timeframes": list(signal_timeframes),
        "entry_timeframes": list(entry_timeframes),
        "take_profit_atr": take_profit_atr,
        "stop_loss_atr": stop_loss_atr,
        "max_bars": max_bars,
        "round_trip_cost": round_trip_cost,
        "cooldown_minutes": cooldown_minutes,
        "signal_count": len(signals),
        "by_entry_timeframe": {},
    }
    for entry_timeframe in entry_timeframes:
        frames = _entry_frames(price_dir, canonical_timeframe(entry_timeframe))
        trades: list[dict[str, Any]] = []
        for signal in signals:
            frame = frames.get(signal["symbol"])
            if frame is None:
                continue
            trade = simulate_short(
                signal,
                frame,
                take_profit_atr=take_profit_atr,
                stop_loss_atr=stop_loss_atr,
                max_bars=max_bars,
                round_trip_cost=round_trip_cost,
            )
            if trade is not None:
                trades.append(trade)
        # One position per symbol at a time, so overlapping signals do not
        # multiply the same move into several "independent" wins.
        deduped: list[dict[str, Any]] = []
        busy_until: dict[str, pd.Timestamp] = {}
        for trade in sorted(trades, key=lambda item: item["signal_time"]):
            free = busy_until.get(trade["symbol"])
            if free is not None and trade["signal_time"] < free:
                continue
            deduped.append(trade)
            exit_time = trade.get("exit_time")
            busy_until[trade["symbol"]] = (
                pd.Timestamp(exit_time)
                if exit_time is not None
                else trade["signal_time"]
            ) + pd.Timedelta(minutes=cooldown_minutes)
        per_signal_timeframe = {
            timeframe: _stats([t for t in deduped if t["signal_timeframe"] == timeframe])
            for timeframe in signal_timeframes
        }
        results["by_entry_timeframe"][entry_timeframe] = {
            "all_signals": _stats(trades),
            "one_position_per_symbol": _stats(deduped),
            "by_signal_timeframe": per_signal_timeframe,
            "trades": [
                {
                    key: (utc_iso(value) if isinstance(value, pd.Timestamp) else value)
                    for key, value in trade.items()
                }
                for trade in deduped
            ],
        }
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-results", required=True, type=Path)
    parser.add_argument("--price-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--signal-timeframes", default="2m,3m,5m")
    parser.add_argument("--entry-timeframes", default="15m,30m")
    parser.add_argument("--take-profit-atr", type=float, default=5.0)
    parser.add_argument("--stop-loss-atr", type=float, default=2.0)
    parser.add_argument("--max-bars", type=int, default=96)
    parser.add_argument("--round-trip-cost", type=float, default=0.0006)
    parser.add_argument("--cooldown-minutes", type=float, default=0.0)
    args = parser.parse_args(argv)

    payload = run(
        scan_root=args.scan_results.resolve(),
        price_dir=args.price_dir.resolve(),
        signal_timeframes=args.signal_timeframes.split(","),
        entry_timeframes=args.entry_timeframes.split(","),
        take_profit_atr=args.take_profit_atr,
        stop_loss_atr=args.stop_loss_atr,
        max_bars=args.max_bars,
        round_trip_cost=args.round_trip_cost,
        cooldown_minutes=args.cooldown_minutes,
    )
    _write_json(args.out / "signal_outcome.json", payload)
    summary = {
        entry: {
            "all_signals": value["all_signals"],
            "one_position_per_symbol": value["one_position_per_symbol"],
            "by_signal_timeframe": value["by_signal_timeframe"],
        }
        for entry, value in payload["by_entry_timeframe"].items()
    }
    print(json.dumps({**{k: v for k, v in payload.items() if k != "by_entry_timeframe"},
                      "results": summary}, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
