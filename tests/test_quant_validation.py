from __future__ import annotations

import numpy as np
import pandas as pd

import quant_validation as validation


def test_block_bootstrap_detects_persistent_positive_mean() -> None:
    rng = np.random.default_rng(11)
    dates = pd.date_range("2000-01-01", periods=1800, freq="B")
    returns = pd.Series(0.0005 + rng.normal(0.0, 0.004, len(dates)), index=dates)
    assert validation.block_bootstrap_sharpe_pvalue(returns, paths=500) < 0.05


def test_pbo_is_bounded() -> None:
    rng = np.random.default_rng(9)
    dates = pd.date_range("2000-01-01", periods=1000, freq="B")
    frame = pd.DataFrame(rng.normal(0.0, 0.01, (len(dates), 5)), index=dates)
    result = validation.probability_of_backtest_overfitting(frame)
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["combinations"] > 0


def test_hac_alpha_recovers_low_beta() -> None:
    rng = np.random.default_rng(3)
    dates = pd.date_range("2000-01-01", periods=1200, freq="B")
    benchmark = pd.Series(rng.normal(0.0, 0.01, len(dates)), index=dates)
    strategy = pd.Series(0.0003 + 0.04 * benchmark.to_numpy() + rng.normal(0.0, 0.004, len(dates)), index=dates)
    result = validation.hac_alpha(strategy, benchmark)
    assert abs(result["beta"] - 0.04) < 0.03
    assert result["alpha_t"] > 2.0
