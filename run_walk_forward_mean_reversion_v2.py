from __future__ import annotations

from typing import Any

import manifold_research as engine
import run_walk_forward_mean_reversion as walk


def conditional_families() -> list[dict[str, Any]]:
    """Active mean reversion with GARCH-normalised position risk."""
    return [
        {
            "name": "garch_shock_fade",
            "signals": {
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "risk_size": "param('size', default=0.08) / (1.0 + conditional_vol * param('vol_penalty', default=0.10))",
                "shock": "roc(close, param('horizon', default=3))",
                "range_vol": "natr(14)",
            },
            "size": "when((shock < (0.0 - param('entry', default=0.04))) & (range_vol > param('min_natr', default=0.0)), risk_size, when((shock > param('entry', default=0.04)) & (range_vol > param('min_natr', default=0.0)), 0.0 - risk_size, 0.0))",
            "grid": {
                "horizon": [2, 4, 6], "entry": [0.02, 0.04, 0.08],
                "min_natr": [0.0, 2.0], "vol_penalty": [0.05, 0.50],
                "size": [0.08],
            },
        },
        {
            "name": "garch_kalman_fade",
            "signals": {
                "log_price": "log(close)",
                "state": "kalman(log_price)",
                "innovation": "log_price - state",
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "risk_size": "param('size', default=0.10) / (1.0 + conditional_vol * param('vol_penalty', default=0.10))",
            },
            "size": "when(innovation < (0.0 - param('entry', default=0.015)), risk_size, when(innovation > param('entry', default=0.015), 0.0 - risk_size, 0.0))",
            "grid": {
                "entry": [0.005, 0.015, 0.03, 0.06],
                "vol_penalty": [0.05, 0.50], "size": [0.08, 0.12],
            },
        },
        {
            "name": "garch_median_fade",
            "signals": {
                "median": "rolling_median(close, param('window', default=48))",
                "deviation": "close / (median + 0.000000001) - 1.0",
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "risk_size": "param('size', default=0.10) / (1.0 + conditional_vol * param('vol_penalty', default=0.10))",
            },
            "size": "when(deviation < (0.0 - param('entry', default=0.03)), risk_size, when(deviation > param('entry', default=0.03), 0.0 - risk_size, 0.0))",
            "grid": {
                "window": [24, 48, 96], "entry": [0.01, 0.03, 0.06],
                "vol_penalty": [0.05, 0.50], "size": [0.08, 0.12],
            },
        },
        {
            "name": "garch_kama_fade",
            "signals": {
                "anchor": "kama(close, param('anchor_period', default=30))",
                "deviation": "close / (anchor + 0.000000001) - 1.0",
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "risk_size": "param('size', default=0.10) / (1.0 + conditional_vol * param('vol_penalty', default=0.10))",
            },
            "size": "when(deviation < (0.0 - param('entry', default=0.03)), risk_size, when(deviation > param('entry', default=0.03), 0.0 - risk_size, 0.0))",
            "grid": {
                "anchor_period": [15, 30, 60], "entry": [0.01, 0.03, 0.06],
                "vol_penalty": [0.05, 0.50], "size": [0.08, 0.12],
            },
        },
        {
            "name": "garch_volatility_exhaustion",
            "signals": {
                "ret1": "roc(close, 1)",
                "fast_vol": "garch(ret1, 0.000001, 0.18, 0.75)",
                "slow_vol": "garch(ret1, 0.000001, 0.05, 0.93)",
                "vol_ratio": "fast_vol / (slow_vol + 0.000000001)",
                "risk_size": "param('size', default=0.08) / (1.0 + slow_vol * param('vol_penalty', default=0.10))",
                "shock": "roc(close, param('horizon', default=3))",
            },
            "size": "when((shock < (0.0 - param('entry', default=0.04))) & (vol_ratio > param('min_vol_ratio', default=0.0)), risk_size, when((shock > param('entry', default=0.04)) & (vol_ratio > param('min_vol_ratio', default=0.0)), 0.0 - risk_size, 0.0))",
            "grid": {
                "horizon": [2, 4, 6], "entry": [0.02, 0.04, 0.08],
                "min_vol_ratio": [0.0, 0.8, 1.1],
                "vol_penalty": [0.05, 0.50], "size": [0.08],
            },
        },
    ]


def fold_score(train: dict[str, Any], test: dict[str, Any], probability: float) -> float:
    if engine.trades(train) == 0 or engine.trades(test) == 0:
        return -1_000_000.0
    return walk.fold_score(train, test, probability)


async def evaluate_fold(session: Any, family: dict[str, Any], strategy: dict[str, Any], fold: dict[str, Any]) -> dict[str, Any]:
    sweep = await engine.call(session, "run_sweep", {
        "strategy_json": strategy,
        "param_grid": family["grid"],
        "config": engine.cfg(*fold["train"], walk.INTERVAL),
        "lite": True,
        "top_k": 64,
        "rank_metric": "sharpe",
        "device": "auto",
        "precision": "fp64",
    })
    correction = sweep.get("overfitting_correction") or {}
    probability = engine.num(correction.get("probability_edge_is_real"), 0.0)
    train_rows = [row for row in sweep.get("top", []) if row.get("params") and engine.trades(row) > 0]
    docs = [walk.freeze(strategy, row["params"], f"{family['name']}_{fold['name']}_{rank}") for rank, row in enumerate(train_rows)]
    test_rows = engine.rows(await engine.call(session, "run_batch", {
        "strategies": [{"strategy_json": doc} for doc in docs],
        "config": engine.cfg(*fold["test"], walk.INTERVAL),
        "lite": False,
        "max_parallelism": 0,
    })) if docs else []
    candidates = [
        {
            "params": train_row["params"],
            "train": train_row,
            "test": test_row,
            "strategy_json": doc,
            "score": fold_score(train_row, test_row, probability),
        }
        for train_row, test_row, doc in zip(train_rows, test_rows, docs, strict=False)
        if engine.trades(test_row) > 0
    ]
    candidates.sort(key=lambda row: row["score"], reverse=True)
    if not candidates:
        raise RuntimeError(f"No active fold candidate for {family['name']} {fold['name']}")
    return {
        "name": fold["name"],
        "train_period": fold["train"],
        "test_period": fold["test"],
        "overfitting_correction": correction,
        **candidates[0],
    }


def main() -> int:
    walk.conditional_families = conditional_families
    walk.fold_score = fold_score
    walk.evaluate_fold = evaluate_fold
    return walk.main()


if __name__ == "__main__":
    raise SystemExit(main())
