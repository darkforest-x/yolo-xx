"""The trade simulator must never see the future and must resolve barriers the
pessimistic way.  These are the assumptions every profitability number rests on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from yolo_xx.outcome import _stats, average_true_range, simulate_short

START = pd.Timestamp("2026-08-01T00:00:00Z")


def _frame(highs, lows, opens=None, closes=None, minutes: int = 15) -> pd.DataFrame:
    count = len(highs)
    opens = list(opens if opens is not None else [100.0] * count)
    closes = list(closes if closes is not None else opens)
    return pd.DataFrame(
        {
            "open_time": pd.date_range(START, periods=count, freq=f"{minutes}min", tz="UTC"),
            "open": opens,
            "high": list(highs),
            "low": list(lows),
            "close": closes,
            "volume": np.ones(count),
            "atr": [1.0] * count,
        }
    )


def _signal(when: pd.Timestamp) -> dict:
    return {"symbol": "okx_BTC_USDT_SWAP", "signal_time": when, "confidence": 0.5}


def _run(frame: pd.DataFrame, when: pd.Timestamp, **kwargs) -> dict | None:
    defaults = dict(
        take_profit_atr=5.0, stop_loss_atr=2.0, max_bars=10, round_trip_cost=0.0
    )
    return simulate_short(_signal(when), frame, **{**defaults, **kwargs})


def test_entry_is_the_first_bar_opening_at_or_after_the_signal() -> None:
    # Long enough that the trade times out rather than running past the data.
    frame = _frame(highs=[100] * 20, lows=[100] * 20)
    # A signal between bar 1 and bar 2 must not fill on bar 1's open.
    trade = _run(frame, START + pd.Timedelta(minutes=20))
    assert trade["entry_time"] == START + pd.Timedelta(minutes=30)
    # A signal exactly on a bar open fills on that bar.
    assert _run(frame, START + pd.Timedelta(minutes=30))["entry_time"] == START + pd.Timedelta(
        minutes=30
    )


def test_short_take_profit_pays_and_stop_loss_costs() -> None:
    # entry 100, atr 1 -> target 95, stop 102
    win = _run(_frame(highs=[100, 100, 100], lows=[100, 100, 94]), START)
    assert win["outcome"] == "target"
    assert win["gross_return"] == pytest.approx(0.05)

    loss = _run(_frame(highs=[100, 100, 103], lows=[100, 100, 100]), START)
    assert loss["outcome"] == "stop"
    assert loss["gross_return"] == pytest.approx(-0.02)


def test_a_bar_touching_both_barriers_is_scored_as_a_stop() -> None:
    trade = _run(_frame(highs=[100, 103], lows=[100, 94]), START)
    assert trade["outcome"] == "stop"


def test_timeout_marks_to_close_and_cost_is_subtracted() -> None:
    frame = _frame(highs=[100] * 6, lows=[100] * 6, closes=[100, 100, 100, 99, 99, 99])
    trade = _run(frame, START, max_bars=3, round_trip_cost=0.001)
    assert trade["outcome"] == "timeout"
    assert trade["net_return"] == pytest.approx(trade["gross_return"] - 0.001)


def test_a_signal_past_the_end_of_local_prices_is_dropped() -> None:
    frame = _frame(highs=[100] * 3, lows=[100] * 3)
    assert _run(frame, START + pd.Timedelta(days=5)) is None


def test_a_bar_without_enough_history_for_atr_is_dropped() -> None:
    frame = _frame(highs=[100] * 4, lows=[100] * 4)
    frame.loc[0, "atr"] = np.nan
    assert _run(frame, START) is None


def test_atr_only_uses_past_bars() -> None:
    frame = _frame(highs=[101] * 30, lows=[99] * 30, closes=[100] * 30)
    atr = average_true_range(frame, period=14)
    # Wilder's ATR needs a full period before it reports anything.
    assert atr.iloc[:13].isna().all()
    assert atr.iloc[-1] == pytest.approx(2.0, abs=1e-6)
    # Injecting a huge future bar must not change any earlier value.
    shocked = frame.copy()
    shocked.loc[29, "high"] = 500.0
    assert average_true_range(shocked, period=14).iloc[20] == pytest.approx(atr.iloc[20])


def test_profit_factor_matches_gross_gain_over_gross_pain() -> None:
    trades = [
        {"outcome": "target", "net_return": 0.05, "bars_held": 3},
        {"outcome": "stop", "net_return": -0.02, "bars_held": 2},
        {"outcome": "stop", "net_return": -0.02, "bars_held": 4},
    ]
    stats = _stats(trades)
    assert stats["closed"] == 3
    assert stats["win_rate"] == pytest.approx(1 / 3)
    assert stats["profit_factor"] == pytest.approx(0.05 / 0.04, rel=1e-4)
