from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.core import ResearchConfig, run_walk_forward


GROUPS = {
    "SPY": "equity",
    "QQQ": "equity",
    "IEF": "bond",
    "TLT": "bond",
    "GLD": "gold",
    "DBC": "commodity",
    "BTC-USD": "crypto",
    "ETH-USD": "crypto",
}


def synthetic_prices(seed: int = 7, months: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2008-01-31", periods=months, freq="ME")
    common = rng.normal(0.006, 0.035, size=months)
    data: dict[str, np.ndarray] = {}
    for i, symbol in enumerate(GROUPS):
        beta = 0.15 + 0.1 * (i % 4)
        drift = 0.002 + 0.0008 * i
        noise = rng.normal(drift, 0.025 + 0.004 * i, size=months)
        returns = beta * common + noise
        data[symbol] = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(data, index=dates)


def fast_config() -> ResearchConfig:
    return ResearchConfig(
        min_train_months=48,
        lookback_months=72,
        embargo_months=1,
        covariance_months=24,
        holdout_months=24,
        monte_carlo_paths=100,
        permutation_trials=50,
        model_mode="ridge",
        random_seed=11,
    )


def test_walk_forward_anti_lookahead_audit_passes() -> None:
    result = run_walk_forward(synthetic_prices(), GROUPS, fast_config())
    assert not result.audit.empty
    assert result.audit["anti_lookahead_pass"].all()
    assert (result.audit["train_label_end"] < result.audit.index).all()


def test_future_mutation_cannot_change_past_weights() -> None:
    prices = synthetic_prices()
    cutoff = prices.index[-36]
    original = run_walk_forward(prices, GROUPS, fast_config())

    mutated = prices.copy()
    rng = np.random.default_rng(123)
    future_mask = mutated.index >= cutoff
    shocks = rng.lognormal(mean=0.0, sigma=0.8, size=(future_mask.sum(), mutated.shape[1]))
    mutated.loc[future_mask] = mutated.loc[future_mask].to_numpy() * shocks
    changed = run_walk_forward(mutated, GROUPS, fast_config())

    common_dates = original.weights.index.intersection(changed.weights.index)
    past_dates = common_dates[common_dates < cutoff]
    pd.testing.assert_frame_equal(
        original.weights.loc[past_dates],
        changed.weights.loc[past_dates],
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_portfolio_constraints_are_enforced() -> None:
    config = fast_config()
    result = run_walk_forward(synthetic_prices(), GROUPS, config)
    assert (result.weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (result.weights.max(axis=1) <= config.per_asset_cap + 1e-9).all()
    crypto = result.weights[["BTC-USD", "ETH-USD"]].sum(axis=1)
    assert (crypto <= config.crypto_cap + 1e-9).all()


def test_returns_start_after_first_signal() -> None:
    result = run_walk_forward(synthetic_prices(), GROUPS, fast_config())
    assert result.returns.index.min() > result.weights.index.min()
    assert result.diagnostics["anti_lookahead_pass"] is True
