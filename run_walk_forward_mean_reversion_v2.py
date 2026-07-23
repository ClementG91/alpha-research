from __future__ import annotations

from typing import Any

import manifold_research as engine
import run_walk_forward_mean_reversion as walk


def conditional_families() -> list[dict[str, Any]]:
    """Mean reversion with GARCH-based position-risk normalisation."""
    return [
        {
            "name": "garch_liquidity_exhaustion",
            "signals": {
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "risk_size": "param('size', default=0.08) / (1.0 + conditional_vol * param('vol_penalty', default=0.10))",
                "shock": "roc(close, param('horizon', default=3))",
                "money_flow": "mfi(14)",
                "range_vol": "natr(14)",
                "rebound": "roc(close, 1)",
            },
            "size": "when((shock < (0.0 - param('entry', default=0.08))) & (money_flow < param('mfi_low', default=25.0)) & (range_vol > param('min_natr', default=2.0)) & (rebound > (0.0 - param('rebound_floor', default=0.015))), risk_size, when((shock > param('entry', default=0.08)) & (money_flow > param('mfi_high', default=75.0)) & (range_vol > param('min_natr', default=2.0)) & (rebound < param('rebound_floor', default=0.015)), 0.0 - risk_size, 0.0))",
            "grid": {
                "horizon": [2, 6], "entry": [0.05, 0.10],
                "mfi_low": [25.0], "mfi_high": [75.0],
                "min_natr": [2.0, 5.0], "rebound_floor": [0.015],
                "vol_penalty": [0.05, 0.50], "size": [0.08],
            },
        },
        {
            "name": "garch_kalman_innovation",
            "signals": {
                "log_price": "log(close)",
                "state": "kalman(log_price)",
                "innovation": "log_price - state",
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "risk_size": "param('size', default=0.10) / (1.0 + conditional_vol * param('vol_penalty', default=0.10))",
                "state_slope": "linreg_slope(state, 48)",
                "abs_slope": "abs_val(state_slope)",
                "trend_strength": "adx(14)",
            },
            "size": "when((innovation < (0.0 - param('entry', default=0.04))) & (abs_slope < param('slope_limit', default=0.002)) & (trend_strength < param('max_adx', default=28.0)), risk_size, when((innovation > param('entry', default=0.04)) & (abs_slope < param('slope_limit', default=0.002)) & (trend_strength < param('max_adx', default=28.0)), 0.0 - risk_size, 0.0))",
            "grid": {
                "entry": [0.02, 0.05, 0.08], "slope_limit": [0.001, 0.003],
                "max_adx": [24.0, 32.0], "vol_penalty": [0.05, 0.50],
                "size": [0.10],
            },
        },
        {
            "name": "garch_median_reversion",
            "signals": {
                "median": "rolling_median(close, param('window', default=48))",
                "deviation": "close / (median + 0.000000001) - 1.0",
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "risk_size": "param('size', default=0.10) / (1.0 + conditional_vol * param('vol_penalty', default=0.10))",
                "oscillator": "rsi(close, param('rsi_period', default=10))",
                "trend_strength": "adx(14)",
            },
            "size": "when((deviation < (0.0 - param('entry', default=0.05))) & (oscillator < param('rsi_low', default=25.0)) & (trend_strength < param('max_adx', default=28.0)), risk_size, when((deviation > param('entry', default=0.05)) & (oscillator > param('rsi_high', default=75.0)) & (trend_strength < param('max_adx', default=28.0)), 0.0 - risk_size, 0.0))",
            "grid": {
                "window": [24, 72], "entry": [0.03, 0.07],
                "rsi_period": [7, 14], "rsi_low": [25.0], "rsi_high": [75.0],
                "max_adx": [24.0, 32.0], "vol_penalty": [0.05, 0.50],
                "size": [0.10],
            },
        },
        {
            "name": "garch_kama_cci_reversion",
            "signals": {
                "anchor": "kama(close, param('anchor_period', default=30))",
                "deviation": "close / (anchor + 0.000000001) - 1.0",
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "risk_size": "param('size', default=0.10) / (1.0 + conditional_vol * param('vol_penalty', default=0.10))",
                "commodity_channel": "cci(20)",
                "trend_strength": "adx(14)",
            },
            "size": "when((deviation < (0.0 - param('entry', default=0.05))) & (commodity_channel < (0.0 - param('cci_level', default=120.0))) & (trend_strength < param('max_adx', default=28.0)), risk_size, when((deviation > param('entry', default=0.05)) & (commodity_channel > param('cci_level', default=120.0)) & (trend_strength < param('max_adx', default=28.0)), 0.0 - risk_size, 0.0))",
            "grid": {
                "anchor_period": [20, 50], "entry": [0.03, 0.07],
                "cci_level": [100.0, 150.0], "max_adx": [24.0, 32.0],
                "vol_penalty": [0.05, 0.50], "size": [0.10],
            },
        },
        {
            "name": "conditional_volatility_exhaustion",
            "signals": {
                "ret1": "roc(close, 1)",
                "fast_vol": "garch(ret1, 0.000001, 0.18, 0.75)",
                "slow_vol": "garch(ret1, 0.000001, 0.05, 0.93)",
                "vol_ratio": "fast_vol / (slow_vol + 0.000000001)",
                "risk_size": "param('size', default=0.08) / (1.0 + slow_vol * param('vol_penalty', default=0.10))",
                "shock": "roc(close, param('horizon', default=3))",
                "money_flow": "mfi(14)",
                "rebound": "roc(close, 1)",
            },
            "size": "when((shock < (0.0 - param('entry', default=0.08))) & (vol_ratio > param('min_vol_ratio', default=1.0)) & (money_flow < param('mfi_low', default=25.0)) & (rebound > (0.0 - param('rebound_floor', default=0.015))), risk_size, when((shock > param('entry', default=0.08)) & (vol_ratio > param('min_vol_ratio', default=1.0)) & (money_flow > param('mfi_high', default=75.0)) & (rebound < param('rebound_floor', default=0.015)), 0.0 - risk_size, 0.0))",
            "grid": {
                "horizon": [2, 6], "entry": [0.05, 0.10],
                "min_vol_ratio": [0.8, 1.1], "mfi_low": [25.0],
                "mfi_high": [75.0], "rebound_floor": [0.015],
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
        "top_k": walk.TOP_PER_FOLD,
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
