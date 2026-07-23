from __future__ import annotations

from pathlib import Path

import pytest

from manifold_research import correlation, curve, daily_returns, drawdown, families, plots, selection_score, validation_gate
from run_manifold_research import normalise_stability


def metrics(sharpe: float = 1.2, alpha: float = 0.2, beta: float = 0.1) -> dict:
    return {
        "sharpe": sharpe,
        "alpha": alpha,
        "beta": beta,
        "tstat_alpha": 1.8,
        "total_return": 0.3,
        "max_drawdown": -0.1,
        "trade_stats": {"total_trades": 40},
    }


def fixture_report() -> dict:
    return {
        "accepted": False,
        "hypotheses": [{"selected": {"validation": {"metrics": metrics(0.8, 0.1, 0.12)}}}],
        "winner": {
            "family": "fixture",
            "interval": "4h",
            "train": {"metrics": metrics(0.8)},
            "validation": {"metrics": metrics(0.9)},
            "holdout": {"metrics": metrics(), "equity_curve": [10000, 10100, 9900, 10300], "daily_returns": [0.01, -0.02, 0.04]},
            "test_2025": {"metrics": metrics(1.1)},
            "test_2026_ytd": {"metrics": metrics(0.3)},
            "double_cost": {"metrics": metrics(0.9)},
            "extra_delay": {"metrics": metrics(0.8)},
            "heatmap": {"metric_grid": [[0.2, 0.4], [0.5, 0.8]], "x_values": [8, 12], "y_values": [96, 168]},
            "stability": {"values": [{"Int64": 48}, {"Float64": 72.0}, {"Int64": 120}], "metric_values": [0.7, 0.9, 0.8]},
            "monte_carlo": {"n_paths": 1000, "final_return": {"percentiles": [[0.9, 0.4], [0.95, 0.5], [0.99, 0.7]]}, "max_drawdown": {"percentiles": [[0.9, 0.2], [0.95, 0.25], [0.99, 0.35]]}},
        },
    }


def test_mean_reversion_families() -> None:
    docs = families()
    assert len(docs) == 5
    assert all("reversion" in item["name"] or "snapback" in item["name"] or "fade" in item["name"] for item in docs)
    assert all("symbol_ref('BTCUSDT'" in " ".join(item["signals"].values()) for item in docs)


def test_payload_helpers() -> None:
    assert curve({"equity_curve": [{"equity": 100}, {"value": 90}]}) == [100.0, 90.0]
    assert daily_returns({"daily_returns": [{"return": 0.1}, -0.2]}) == [0.1, -0.2]
    assert drawdown([100, 110, 99]) == pytest.approx([0.0, 0.0, -0.1])
    assert correlation([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_alpha_beta_gate_and_score() -> None:
    good = {"train": {"metrics": metrics(0.8)}, "validation": {"metrics": metrics(0.7, 0.2, 0.1)}}
    bad_beta = {"train": {"metrics": metrics(0.8)}, "validation": {"metrics": metrics(0.7, 0.2, 0.6)}}
    assert validation_gate(good)
    assert not validation_gate(bad_beta)
    assert selection_score(good["train"], good["validation"], 0.5) > selection_score(bad_beta["train"], bad_beta["validation"], 0.5)


def test_typed_stability_normalisation() -> None:
    report = fixture_report()
    normalise_stability(report)
    assert report["winner"]["stability"]["values"] == [48.0, 72.0, 120.0]
    assert report["winner"]["stability"]["metrics"] == [0.7, 0.9, 0.8]


def test_plots(tmp_path: Path) -> None:
    report = fixture_report()
    normalise_stability(report)
    plots(report, tmp_path)
    expected = {
        "equity_curve.svg",
        "drawdown.svg",
        "period_metrics.svg",
        "alpha_beta.svg",
        "beta_frontier.svg",
        "parameter_heatmap.svg",
        "stability.svg",
        "stress_tests.svg",
        "monte_carlo.svg",
    }
    assert {path.name for path in (tmp_path / "plots").glob("*.svg")} == expected
