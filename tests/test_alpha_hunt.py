from __future__ import annotations

import numpy as np
import pandas as pd

import alpha_hunt
import alpha_hunt_runner  # noqa: F401  # installs the hedged target-weight implementation
import quant_validation


def synthetic_hedge_fixture() -> tuple[list[str], pd.Series, pd.Series]:
    classes = pd.Series(
        {
            "SPY": "benchmark",
            "QQQ": "us_equity",
            "IWM": "us_equity",
            "EFA": "international_equity",
            "AGG": "rates_credit",
            "DBC": "real_assets",
            "XLE": "us_sectors",
            "QUAL": "equity_factors",
            "US_ALPHA": "us_equity",
            "INTL_ALPHA": "international_equity",
            "RATE_ALPHA": "rates_credit",
            "REAL_ALPHA": "real_assets",
            "SECTOR_ALPHA": "us_sectors",
            "FACTOR_ALPHA": "equity_factors",
        }
    )
    beta = pd.Series(
        {
            "SPY": 1.00,
            "QQQ": 1.20,
            "IWM": 1.40,
            "EFA": 0.80,
            "AGG": 0.10,
            "DBC": 0.30,
            "XLE": 1.10,
            "QUAL": 0.90,
            "US_ALPHA": 1.10,
            "INTL_ALPHA": 0.70,
            "RATE_ALPHA": 0.20,
            "REAL_ALPHA": 0.40,
            "SECTOR_ALPHA": 1.00,
            "FACTOR_ALPHA": 0.85,
        }
    )
    return list(classes.index), classes, beta


def test_candidate_protocol_is_small_and_frozen() -> None:
    candidates = alpha_hunt.make_candidates()
    assert len(candidates) == 12
    assert len({candidate.key for candidate in candidates}) == 12


def test_prior_zscore_is_unchanged_by_future_mutation() -> None:
    dates = pd.date_range("2000-01-01", periods=500, freq="B")
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(len(dates), 3)), index=dates, columns=list("ABC"))
    original = alpha_hunt.prior_zscore(frame)
    mutated = frame.copy()
    mutated.iloc[420:] += 1000.0
    changed = alpha_hunt.prior_zscore(mutated)
    pd.testing.assert_frame_equal(original.iloc[:420], changed.iloc[:420])


def test_sparse_alpha_legs_survive_factor_hedging() -> None:
    symbols, classes, beta = synthetic_hedge_fixture()
    score = pd.Series(np.nan, index=symbols)
    score.loc["US_ALPHA"] = 2.0
    score.loc["INTL_ALPHA"] = -2.0
    weights = alpha_hunt_runner.hedged_sparse_weights(score, beta, classes, 0.20)
    assert weights.abs().sum() > 0.99
    assert abs(float(weights.sum())) < 1e-10
    assert abs(float((weights * beta).sum())) < 1e-10
    for class_name in classes.unique():
        if class_name == "benchmark":
            continue
        assert abs(float(weights.loc[classes == class_name].sum())) < 1e-10
    assert weights.loc["US_ALPHA"] > 0.0
    assert weights.loc["INTL_ALPHA"] < 0.0


def test_signal_enters_at_next_open_and_costs_use_turnover() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    symbols, classes, beta_row = synthetic_hedge_fixture()
    score = pd.DataFrame(np.nan, index=dates, columns=symbols)
    score.loc[dates[10], "US_ALPHA"] = 2.0
    score.loc[dates[10], "INTL_ALPHA"] = -2.0
    beta = pd.DataFrame(
        np.repeat(beta_row.reindex(symbols).to_numpy()[None, :], len(dates), axis=0),
        index=dates,
        columns=symbols,
    )
    execution = pd.DataFrame(0.0, index=dates, columns=symbols)
    candidate = alpha_hunt.Candidate("auction_flow_reversal", 1, 0.20, 1.0)
    features = {
        "auction_flow_reversal": score,
        "trade_beta": beta,
        "execution_returns": execution,
    }
    result = alpha_hunt.backtest_candidate(candidate, features, classes, cost_bps=5.0)
    assert result.positions.loc[dates[10]].abs().sum() == 0.0
    assert result.positions.loc[dates[11]].abs().sum() > 0.0
    pd.testing.assert_series_equal(
        result.returns,
        -result.turnover * 5.0 / 10_000.0,
        check_names=False,
    )


def test_pbo_treats_degenerate_folds_as_failure_instead_of_crashing() -> None:
    dates = pd.date_range("2010-01-01", periods=1000, freq="B")
    rng = np.random.default_rng(91)
    candidates = pd.DataFrame(
        {
            "active": rng.normal(0.0, 0.01, len(dates)),
            "flat_a": 0.0,
            "flat_b": 0.0,
        },
        index=dates,
    )
    result = quant_validation.probability_of_backtest_overfitting(candidates)
    assert result["combinations"] > 0
    assert 0.0 <= result["pbo"] <= 1.0
