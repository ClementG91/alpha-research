from __future__ import annotations

import numpy as np
import pandas as pd

import cross_asset_momentum as momentum


def test_external_tradable_universe_is_disjoint_from_core() -> None:
    core = set(momentum.CORE_UNIVERSE)
    external = {symbol for symbol, class_name in momentum.EXTERNAL_UNIVERSE.items() if class_name != "benchmark"}
    assert not core.intersection(external)
    assert len(external) >= 60
    assert len(set(momentum.EXTERNAL_UNIVERSE.values()) - {"benchmark"}) >= 5


def test_momentum_grid_is_predeclared_and_broad() -> None:
    candidates = momentum.make_candidates()
    assert len(candidates) >= 100
    assert {candidate.family for candidate in candidates} == {
        "residual_momentum",
        "intraday_momentum",
        "overnight_momentum",
        "dispersion_momentum",
    }


def test_inverted_positions_pay_costs_in_both_directions() -> None:
    dates = pd.date_range("2025-01-01", periods=40, freq="B")
    columns = list("ABCD")
    positions = pd.DataFrame(
        np.tile([0.25, 0.25, -0.25, -0.25], (len(dates), 1)),
        index=dates,
        columns=columns,
    )
    execution = pd.DataFrame(0.0, index=dates, columns=columns)
    execution["A"] = 0.01
    execution["B"] = 0.005
    execution["C"] = -0.005
    execution["D"] = -0.01
    spy = pd.Series(0.0, index=dates)
    positive, _ = momentum.evaluate_positions(
        positions, execution, spy, (str(dates.min().date()), str(dates.max().date())), 10.0,
    )
    inverted, _ = momentum.evaluate_positions(
        -positions, execution, spy, (str(dates.min().date()), str(dates.max().date())), 10.0,
    )
    expected_round_trip_cost = 10.0 / 10_000.0
    assert np.allclose(positive + inverted, -2.0 * expected_round_trip_cost)
    assert positive.mean() > 0
    assert inverted.mean() < 0
