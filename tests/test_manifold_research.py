from __future__ import annotations

import json
from pathlib import Path

import pytest

from manifold_research import curve, daily_returns, drawdown, families, plots
from run_manifold_research import normalise_stability


def fixture_report() -> dict:
    metrics = {"sharpe": 1.2, "total_return": 0.3, "max_drawdown": -0.1, "trade_stats": {"total_trades": 40}}
    return {
        "accepted": False,
        "winner": {
            "family": "fixture",
            "train": {"metrics": {**metrics, "sharpe": 0.8}},
            "validation": {"metrics": {**metrics, "sharpe": 0.9}},
            "holdout": {"metrics": metrics, "equity_curve": [10000, 10100, 9900, 10300], "daily_returns": [0.01, -0.02, 0.04]},
            "test_2025": {"metrics": {**metrics, "sharpe": 1.1}},
            "test_2026_ytd": {"metrics": {**metrics, "sharpe": 0.3}},
            "heatmap": {"metric_grid": [[0.2, 0.4], [0.5, 0.8]], "x_values": [8, 12], "y_values": [96, 168]},
            "stability": {"values": [{"Int64": 48}, {"Float64": 72.0}, {"Int64": 120}], "metric_values": [0.7, 0.9, 0.8]},
            "monte_carlo": {"n_paths": 1000, "final_return": {"percentiles": [[0.9, 0.4], [0.95, 0.5], [0.99, 0.7]]}, "max_drawdown": {"percentiles": [[0.9, 0.2], [0.95, 0.25], [0.99, 0.35]]}},
        },
    }


def test_strategy_families_are_json() -> None:
    docs = families()
    assert len(docs) == 3
    for item in docs:
        assert "position_sizing" in item["strategy"]
        json.dumps(item["strategy"])


def test_payload_normalisation() -> None:
    assert curve({"equity_curve": [{"equity": 100}, {"value": 90}]}) == [100.0, 90.0]
    assert daily_returns({"daily_returns": [{"return": 0.1}, -0.2]}) == [0.1, -0.2]
    assert drawdown([100, 110, 99]) == pytest.approx([0.0, 0.0, -0.1])


def test_typed_stability_normalisation() -> None:
    report = fixture_report()
    normalise_stability(report)
    assert report["winner"]["stability"]["values"] == [48.0, 72.0, 120.0]
    assert report["winner"]["stability"]["metrics"] == [0.7, 0.9, 0.8]


def test_plots(tmp_path: Path) -> None:
    report = fixture_report()
    normalise_stability(report)
    plots(report, tmp_path)
    expected = {"equity_curve.svg", "drawdown.svg", "period_metrics.svg", "parameter_heatmap.svg", "stability.svg", "monte_carlo.svg"}
    assert {path.name for path in (tmp_path / "plots").glob("*.svg")} == expected
