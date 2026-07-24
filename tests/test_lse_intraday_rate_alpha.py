from __future__ import annotations

import numpy as np
import pandas as pd

import lse_intraday_macro_alpha as base
import lse_intraday_rate_alpha as research
import lse_intraday_rate_runner as runner  # noqa: F401  # installs class-neutral targets


def synthetic_yields() -> dict[str, pd.DataFrame]:
    index = pd.date_range("2020-01-01", periods=120, freq="B", tz="UTC")
    us2 = pd.Series(1.5, index=index)
    us10 = pd.Series(1.8, index=index)
    us2.iloc[70] += 0.30
    return {
        "US2Y": pd.DataFrame({"close": us2}, index=index),
        "US10Y": pd.DataFrame({"close": us10}, index=index),
    }


def test_rate_candidate_protocol_is_small_and_frozen() -> None:
    definitions = research.candidates()
    assert len(definitions) == 8
    assert len({candidate.key for candidate in definitions}) == 8
    assert {candidate.family for candidate in definitions} == {
        "unconditional_reversal",
        "rate_calm_reversal",
        "rate_stress_continuation",
        "us_open_rate_calm_reversal",
        "us_open_rate_stress_continuation",
    }


def test_rate_regime_is_lagged_one_full_observation() -> None:
    yields = synthetic_yields()
    dates = yields["US2Y"].index
    intraday = pd.date_range(dates[68], dates[73] + pd.Timedelta(hours=23), freq="15min", tz="UTC")
    regimes = research.rate_regimes(intraday, yields)
    shock_day = dates[70].floor("D")
    next_observation = dates[71].floor("D")
    assert not regimes.loc[regimes.index.floor("D") == shock_day, "stress"].any()
    assert regimes.loc[regimes.index.floor("D") == next_observation, "stress"].all()


def test_future_yield_mutation_does_not_change_past_regimes() -> None:
    yields = synthetic_yields()
    intraday = pd.date_range("2020-01-01", periods=9000, freq="15min", tz="UTC")
    original = research.rate_regimes(intraday, yields)
    mutated = {name: frame.copy() for name, frame in yields.items()}
    mutated["US2Y"].iloc[-10:, mutated["US2Y"].columns.get_loc("close")] += 10.0
    changed = research.rate_regimes(intraday, mutated)
    cutoff = mutated["US2Y"].index[-11]
    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_relative_value_targets_are_zero_net_by_class() -> None:
    index = pd.DatetimeIndex([pd.Timestamp("2025-01-02 14:30", tz="UTC")])
    columns = ["EUR/USD", "GBP/USD", "USD/JPY", "ES", "NQ", "GC"]
    raw = pd.DataFrame([[3.0, 0.0, 0.0, -2.5, 0.0, 4.0]], index=index, columns=columns)
    classes = pd.Series({symbol: base.asset_class(symbol) for symbol in columns})
    weights = runner.relative_class_neutral_targets(raw, classes)
    assert weights.abs().sum(axis=1).iloc[0] > 0.99
    for class_name in classes.unique():
        members = classes.index[classes == class_name]
        assert abs(float(weights.loc[index[0], members].sum())) < 1e-12
    assert weights.loc[index[0], "GC"] == 0.0  # no second commodity hedge exists


def test_us_open_mask_handles_new_york_dst() -> None:
    winter = pd.DatetimeIndex([pd.Timestamp("2025-01-02 14:30", tz="UTC")])
    summer = pd.DatetimeIndex([pd.Timestamp("2025-07-02 13:30", tz="UTC")])
    assert bool(research.us_open_mask(winter).iloc[0])
    assert bool(research.us_open_mask(summer).iloc[0])


def test_am_pm_release_times_are_exact() -> None:
    index_value = pd.Timestamp("2025-01-01", tz="UTC")
    from lse_intraday_macro_runner import release_timestamp

    assert release_timestamp(index_value, "2025-01-01", "11:00 PM") == pd.Timestamp(
        "2025-01-01 23:00:00", tz="UTC"
    )
    assert release_timestamp(index_value, "2025-01-01", "01:30 AM") == pd.Timestamp(
        "2025-01-01 01:30:00", tz="UTC"
    )
