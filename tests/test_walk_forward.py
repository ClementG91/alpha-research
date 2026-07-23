from __future__ import annotations

from run_walk_forward_mean_reversion import (
    FOLDS,
    LEGACY_AUDIT,
    aggregate_fold_metrics,
    consensus_params,
    freeze,
)
from run_walk_forward_mean_reversion_v2 import conditional_families


def result(sharpe: float, alpha: float, beta: float, trades: int = 30) -> dict:
    return {
        "metrics": {
            "sharpe": sharpe,
            "alpha": alpha,
            "beta": beta,
            "tstat_alpha": 1.8,
            "max_drawdown": -0.10,
            "trade_stats": {"total_trades": trades},
        }
    }


def selection(params: dict, sharpe: float = 0.8, alpha: float = 0.1, beta: float = 0.05) -> dict:
    return {
        "params": params,
        "train": result(0.9, 0.1, 0.05),
        "test": result(sharpe, alpha, beta),
    }


def test_conditional_families_use_garch() -> None:
    families = conditional_families()
    assert len(families) == 5
    assert all("garch(" in " ".join(family["signals"].values()) for family in families)
    assert all("risk_size" in family["signals"] for family in families)
    assert all(family["grid"] for family in families)


def test_fold_boundaries_precede_legacy_audit() -> None:
    assert [fold["name"] for fold in FOLDS] == ["2022", "2023", "2024"]
    assert all(fold["train"][1] < fold["test"][0] for fold in FOLDS)
    assert max(fold["test"][1] for fold in FOLDS) < LEGACY_AUDIT[0]


def test_consensus_and_freeze_preserve_types() -> None:
    strategy = {
        "name": "fixture",
        "parameters": {
            "window": {"default": {"Int64": 48}},
            "entry": {"default": {"Float64": 2.5}},
        },
    }
    selections = [
        selection({"window": 24, "entry": 1.5}),
        selection({"window": 48, "entry": 2.5}),
        selection({"window": 96, "entry": 4.0}),
    ]
    params = consensus_params(selections, strategy)
    assert params == {"window": 48, "entry": 2.5}
    frozen = freeze(strategy, params, "locked")
    assert frozen["parameters"]["window"]["default"] == {"Int64": 48}
    assert frozen["parameters"]["entry"]["default"] == {"Float64": 2.5}


def test_multi_fold_gate_rewards_consistency() -> None:
    good = [
        selection({"entry": 2.0}, 0.7, 0.08, 0.05),
        selection({"entry": 2.5}, 0.5, 0.06, 0.08),
        selection({"entry": 2.5}, 0.4, 0.04, 0.10),
    ]
    bad = [
        selection({"entry": 1.0}, 1.5, 0.20, 0.60),
        selection({"entry": 4.0}, -0.8, -0.05, 0.50),
        selection({"entry": 8.0}, -0.2, -0.02, 0.40),
    ]
    good_metrics = aggregate_fold_metrics(good)
    bad_metrics = aggregate_fold_metrics(bad)
    assert good_metrics["passed"]
    assert not bad_metrics["passed"]
    assert good_metrics["score"] > bad_metrics["score"]
