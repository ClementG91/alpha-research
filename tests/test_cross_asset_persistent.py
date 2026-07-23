from __future__ import annotations

import numpy as np
import pandas as pd

import cross_asset_persistent as persistent


def test_persistent_candidate_grid_has_multiple_horizons() -> None:
    candidates = persistent.make_candidates()
    assert len(candidates) >= 150
    assert {candidate.family for candidate in candidates} == {
        "residual_trend",
        "residual_reversion",
        "dual_horizon_trend",
    }
    assert {int(candidate.values()["rebalance"]) for candidate in candidates} == {5, 10, 20}


def test_turnover_cost_is_charged_only_when_positions_change() -> None:
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    positions = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    positions.loc[dates[5]:dates[14], "A"] = 0.5
    positions.loc[dates[5]:dates[14], "B"] = -0.5
    turnover = positions.sub(positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    assert turnover.gt(0).sum() == 2
    assert turnover.sum() == 2.0


def test_positions_are_lagged_after_target_construction() -> None:
    dates = pd.date_range("2018-01-01", periods=500, freq="B")
    symbols = ["SPY", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
    rng = np.random.default_rng(22)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(symbols))), axis=0)),
        index=dates,
        columns=symbols,
    )
    panel = {
        "close": close,
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": close * 0 + 1_000_000,
    }
    classes = pd.Series({symbol: ("benchmark" if symbol == "SPY" else ("x" if symbol < "G" else "y")) for symbol in symbols})
    candidate = persistent.PersistentCandidate(
        "residual_trend",
        tuple(sorted({
            "lookback": 20.0, "skip": 0.0, "quantile": 0.2,
            "rebalance": 5.0, "beta_window": 126.0, "vol_window": 60.0,
        }.items())),
    )
    positions, _ = persistent.target_weights(candidate, panel, classes)
    assert positions.iloc[0].abs().sum() == 0
    assert positions.abs().sum(axis=1).gt(0).any()
    changes = positions.ne(positions.shift(1)).any(axis=1)
    assert changes.sum() < len(positions) / 2
