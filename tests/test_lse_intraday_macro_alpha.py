from __future__ import annotations

import numpy as np
import pandas as pd

import lse_intraday_macro_alpha as research


def synthetic_market(periods: int = 2400) -> dict[str, pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=periods, freq="15min", tz="UTC")
    symbols = ["EUR/USD", "GBP/USD", "USD/JPY", "ES", "NQ", "ZN"]
    rng = np.random.default_rng(17)
    innovations = rng.normal(0.0, 0.0007, (periods, len(symbols)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(innovations, axis=0)), index=index, columns=symbols)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.DataFrame(np.maximum(open_, close) * 1.0005, index=index, columns=symbols)
    low = pd.DataFrame(np.minimum(open_, close) * 0.9995, index=index, columns=symbols)
    volume = pd.DataFrame(rng.lognormal(12.0, 0.4, (periods, len(symbols))), index=index, columns=symbols)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def test_candidate_protocol_is_small_and_frozen() -> None:
    definitions = research.candidates()
    assert len(definitions) == 7
    assert len({candidate.key for candidate in definitions}) == 7
    assert {candidate.family for candidate in definitions} == {
        "macro_surprise_drift",
        "macro_overreaction_reversal",
        "liquidity_shock_reversal",
    }


def test_parse_numeric_handles_calendar_units() -> None:
    assert research.parse_numeric("3.2%") == 0.032
    assert research.parse_numeric("250K") == 250_000.0
    assert research.parse_numeric("1.5M") == 1_500_000.0
    assert np.isnan(research.parse_numeric("N/A"))


def test_event_impulse_enters_only_after_completed_reaction_bar() -> None:
    market = synthetic_market()
    event_time = market["close"].index[2000]
    calendar = pd.DataFrame(
        [{
            "timestamp": event_time,
            "currency": "USD",
            "event": "CPI",
            "family": "inflation",
            "surprise_z": 2.0,
        }]
    )
    candidate = research.Candidate("macro_surprise_drift", 4, 0.75)
    backtest = research.build_backtest(candidate, market, calendar)
    reaction_bar = market["close"].index.searchsorted(event_time + research.BAR_DELTA, side="left")
    reaction_time = market["close"].index[reaction_bar]
    next_time = market["close"].index[reaction_bar + 1]
    assert backtest.positions.loc[reaction_time].abs().sum() == 0.0
    assert backtest.positions.loc[next_time].abs().sum() > 0.0


def test_future_calendar_mutation_does_not_change_past_liquidity_signal() -> None:
    market = synthetic_market()
    cutoff = market["close"].index[2200]
    calendar = pd.DataFrame(
        [{"timestamp": market["close"].index[1800], "currency": "USD", "family": "inflation", "surprise_z": 2.0}]
    )
    original = research.liquidity_impulses(market, calendar, 2.0)
    mutated = pd.concat(
        [
            calendar,
            pd.DataFrame([{"timestamp": market["close"].index[2300], "currency": "USD", "family": "growth", "surprise_z": 5.0}]),
        ],
        ignore_index=True,
    )
    changed = research.liquidity_impulses(market, mutated, 2.0)
    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_costs_equal_turnover_times_class_cost_when_returns_are_flat() -> None:
    market = synthetic_market()
    for field in ("open", "high", "low", "close"):
        market[field].loc[:, :] = 100.0
    event_time = market["close"].index[2000]
    calendar = pd.DataFrame(
        [{"timestamp": event_time, "currency": "USD", "family": "inflation", "surprise_z": 2.0}]
    )
    candidate = research.Candidate("macro_surprise_drift", 1, 0.75)
    backtest = research.build_backtest(candidate, market, calendar)
    assert backtest.bar_returns.sum() <= 0.0
    assert backtest.costs.sum() > 0.0
    np.testing.assert_allclose(backtest.bar_returns.to_numpy(), -backtest.costs.to_numpy())
