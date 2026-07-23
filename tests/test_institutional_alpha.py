from __future__ import annotations

import numpy as np
import pandas as pd

import institutional_alpha as research


def synthetic_prices(periods: int = 240) -> pd.DataFrame:
    dates = pd.date_range("2000-01-31", periods=periods, freq="ME")
    rng = np.random.default_rng(42)
    returns = rng.normal(0.005, 0.04, size=(periods, len(research.ASSET_CLASS)))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=list(research.ASSET_CLASS))


def test_future_mutation_does_not_change_existing_tsmom_weights() -> None:
    prices = synthetic_prices()
    original = research.tsmom_weights(prices.pct_change(fill_method=None))
    mutated = prices.copy()
    cutoff = 170
    mutated.iloc[cutoff:] *= np.linspace(1.0, 3.0, len(mutated) - cutoff)[:, None]
    changed = research.tsmom_weights(mutated.pct_change(fill_method=None))
    pd.testing.assert_frame_equal(original.iloc[:cutoff], changed.iloc[:cutoff])


def test_execution_uses_one_full_period_lag() -> None:
    prices = synthetic_prices(80)
    returns = prices.pct_change(fill_method=None)
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    weights.loc[weights.index[30], "SPY"] = 0.25
    backtest = research.apply_execution(weights, returns, "test", 5.0)
    assert backtest.weights is not None
    assert backtest.weights.loc[weights.index[30], "SPY"] == 0.0
    assert backtest.weights.loc[weights.index[31], "SPY"] == 0.25


def test_metrics_recovers_small_beta() -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2000-01-31", periods=240, freq="ME")
    benchmark = pd.Series(rng.normal(0.005, 0.04, len(dates)), index=dates)
    strategy = pd.Series(0.002 + 0.03 * benchmark.to_numpy() + rng.normal(0.0, 0.01, len(dates)), index=dates)
    result = research.metrics(strategy, benchmark)
    assert abs(float(result["beta"]) - 0.03) < 0.05
    assert float(result["alpha"]) > 0.0
