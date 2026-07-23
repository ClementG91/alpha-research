from __future__ import annotations

import numpy as np
import pandas as pd

import cross_asset_ensemble as ensemble


def synthetic_panel(periods: int = 700) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    dates = pd.date_range("2018-01-01", periods=periods, freq="B")
    symbols = ["SPY"] + [f"A{i}" for i in range(1, 25)]
    rng = np.random.default_rng(44)
    shocks = rng.normal(0, 0.009, (periods, len(symbols)))
    market = rng.normal(0, 0.007, periods)[:, None]
    loadings = np.linspace(0.4, 1.4, len(symbols))[None, :]
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(shocks + market * loadings, axis=0)),
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
    class_names = ["benchmark", "equity", "international", "rates", "real_assets", "themes", "styles"]
    classes = pd.Series({
        symbol: class_names[0] if symbol == "SPY" else class_names[1 + (index - 1) % 6]
        for index, symbol in enumerate(symbols)
    })
    return panel, classes


def test_ensemble_has_exactly_three_fixed_sleeves() -> None:
    panel, classes = synthetic_panel()
    _, sleeves = ensemble.build(panel, classes)
    assert set(sleeves) == {"trend", "low_vol", "residual"}


def test_ensemble_is_equal_weighted() -> None:
    panel, classes = synthetic_panel()
    result, sleeves = ensemble.build(panel, classes)
    expected = pd.concat({name: sleeve.returns for name, sleeve in sleeves.items()}, axis=1).mean(axis=1)
    pd.testing.assert_series_equal(result.returns, expected)


def test_turnover_cost_is_only_charged_on_position_changes() -> None:
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    positions = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    positions.loc[dates[5]:dates[14], "A"] = 0.5
    positions.loc[dates[5]:dates[14], "B"] = -0.5
    returns = pd.DataFrame(0.0, index=dates, columns=["A", "B"])
    result = ensemble.from_positions(positions, returns, 5.0)
    expected_turnover = positions.sub(positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    pd.testing.assert_series_equal(result.turnover, expected_turnover)
    assert result.returns.sum() == -(expected_turnover.sum() * 5.0 / 10_000.0)


def test_inverting_positions_does_not_refund_costs() -> None:
    dates = pd.date_range("2025-01-01", periods=30, freq="B")
    positions = pd.DataFrame(
        np.tile([0.5, -0.5], (len(dates), 1)),
        index=dates,
        columns=["A", "B"],
    )
    returns = pd.DataFrame({"A": 0.002, "B": -0.002}, index=dates)
    direct = ensemble.from_positions(positions, returns, 5.0)
    inverted = ensemble.from_positions(-positions, returns, 5.0)
    assert direct.returns.iloc[0] + inverted.returns.iloc[0] < 0
    assert direct.returns.mean() > inverted.returns.mean()


def test_hold_function_lags_rebalance_targets() -> None:
    dates = pd.date_range("2025-01-01", periods=15, freq="B")
    targets = pd.DataFrame({"A": np.arange(15, dtype=float)}, index=dates)
    positions = ensemble.hold(targets, 5)
    assert positions.iloc[0, 0] == 0.0
    assert positions.iloc[1, 0] == targets.iloc[0, 0]
    assert positions.iloc[6, 0] == targets.iloc[5, 0]
