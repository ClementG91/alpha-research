from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research.advanced_v2 import run_advanced_walk_forward
from quant_research.core import ResearchConfig


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


def synthetic_prices(seed: int = 19, months: int = 132) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-31", periods=months, freq="ME")
    risk = rng.normal(0.005, 0.035, size=months)
    defensive = rng.normal(0.002, 0.015, size=months)
    data: dict[str, np.ndarray] = {}
    for i, (symbol, group) in enumerate(GROUPS.items()):
        common = defensive if group in {"bond", "gold"} else risk
        noise = rng.normal(0.001 + i * 0.0002, 0.018 + i * 0.002, months)
        returns = 0.35 * common + noise
        data[symbol] = 100.0 * np.cumprod(1.0 + returns)
    return pd.DataFrame(data, index=dates)


def config() -> ResearchConfig:
    return ResearchConfig(
        min_train_months=48,
        lookback_months=72,
        embargo_months=1,
        covariance_months=24,
        holdout_months=24,
        monte_carlo_paths=50,
        permutation_trials=20,
        model_mode="ridge",
        trial_count=16,
        random_seed=17,
    )


def test_advanced_sleeves_are_purged_and_constrained() -> None:
    cfg = config()
    result = run_advanced_walk_forward(synthetic_prices(), GROUPS, cfg)

    assert not result.audit.empty
    assert result.audit["anti_lookahead_pass"].all()
    ml = result.audit[result.audit["sleeve"].str.startswith("ml_")]
    assert (ml["train_label_end"] < ml.index).all()
    assert set(ml["horizon_months"].unique()) == {1, 3, 6}
    assert (result.weights.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (result.weights.max(axis=1) <= cfg.per_asset_cap + 1e-9).all()
    crypto = result.weights[["BTC-USD", "ETH-USD"]].sum(axis=1)
    assert (crypto <= cfg.crypto_cap + 1e-9).all()
    assert np.isclose(
        result.diagnostics["average_gross_exposure"],
        result.weights.sum(axis=1).mean(),
    )
    assert set(result.diagnostics["sleeve_metrics"]) == {
        "ml_1m",
        "ml_3m",
        "ml_6m",
        "statistical_trend",
    }
