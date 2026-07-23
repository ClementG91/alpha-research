from __future__ import annotations

import numpy as np
import pandas as pd

import alpha_hunt


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


def test_signal_enters_at_next_open_and_costs_use_turnover() -> None:
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    symbols = [f"S{i:02d}" for i in range(20)]
    classes = pd.Series({symbol: f"class_{index // 5}" for index, symbol in enumerate(symbols)})
    score = pd.DataFrame(np.nan, index=dates, columns=symbols)
    score.loc[dates[10]] = np.arange(len(symbols), dtype=float)
    beta = pd.DataFrame(0.0, index=dates, columns=symbols)
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
