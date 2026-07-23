from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cross_asset_research as research
from cross_asset_research_v3 import vectorized_strategy_from_score


def test_projection_enforces_class_and_beta_neutrality() -> None:
    idx = pd.Index(["A", "B", "C", "D", "E", "F", "G", "H"])
    raw = pd.Series([0.25, -0.25, 0, 0, 0.25, -0.25, 0, 0], index=idx)
    beta = pd.Series([1.2, 0.8, 1.0, 1.1, -0.2, 0.3, 0.1, 0.2], index=idx)
    classes = pd.Series(["x"] * 4 + ["y"] * 4, index=idx)
    result = research.project_constraints(raw, beta, classes)
    assert abs(result.groupby(classes).sum()).max() < 1e-10
    assert abs(float((result * beta).sum())) < 1e-10
    assert result.abs().sum() == pytest.approx(1.0)


def test_vectorized_portfolio_is_neutral_each_day() -> None:
    dates = pd.date_range("2022-01-01", periods=80, freq="B")
    symbols = list("ABCDEFGHIJKL")
    rng = np.random.default_rng(12)
    score = pd.DataFrame(rng.normal(size=(len(dates), len(symbols))), index=dates, columns=symbols)
    execution = pd.DataFrame(rng.normal(0, 0.01, size=score.shape), index=dates, columns=symbols)
    beta = pd.DataFrame(rng.normal(1, 0.3, size=score.shape), index=dates, columns=symbols)
    classes = pd.Series({symbol: ("x", "y", "z")[i // 4] for i, symbol in enumerate(symbols)})
    result = vectorized_strategy_from_score(score, execution, beta, classes, 0.25, 10.0)
    for class_name in classes.unique():
        columns = classes.index[classes == class_name]
        assert result.positions[columns].sum(axis=1).abs().max() < 1e-10
    assert (result.positions * beta).sum(axis=1).abs().max() < 1e-10
    assert result.positions.abs().sum(axis=1).replace(0, np.nan).dropna().eq(1.0).all()


def test_close_reversal_signal_is_lagged() -> None:
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    symbols = ["SPY", "A", "B", "C", "D", "E", "F", "G", "H"]
    rng = np.random.default_rng(4)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(symbols))), axis=0)),
        index=dates,
        columns=symbols,
    )
    open_ = close.shift(1).fillna(close.iloc[0])
    panel = {
        "close": close,
        "open": open_,
        "high": close * 1.01,
        "low": close * 0.99,
        "volume": close * 0 + 1_000_000,
    }
    classes = pd.Series({symbol: "one" if i < 5 else "two" for i, symbol in enumerate(symbols)})
    candidate = research.Candidate(
        "residual_reversal",
        tuple(sorted({"lookback": 1.0, "quantile": 0.2, "vol_window": 20.0, "beta_window": 63.0}.items())),
    )
    result = research.build_candidate(candidate, panel, classes, 5.0)
    assert result.positions.iloc[:64].abs().sum().sum() == 0


def test_hac_regression_recovers_low_beta() -> None:
    rng = np.random.default_rng(3)
    spy = pd.Series(rng.normal(0, 0.01, 1000))
    strategy = pd.Series(0.0002 + 0.02 * spy + rng.normal(0, 0.003, 1000))
    metrics = research.regression_metrics(strategy, spy)
    assert metrics["beta"] == pytest.approx(0.02, abs=0.04)
    assert metrics["alpha"] > 0


def test_candidate_grid_has_breadth() -> None:
    candidates = research.make_candidates()
    assert len(candidates) >= 50
    assert {candidate.family for candidate in candidates} == {
        "residual_reversal",
        "gap_reversal",
        "dispersion_reversal",
    }
