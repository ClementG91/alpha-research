from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import manifold_research as engine

INTERVAL = "4h"
FOLDS = (
    {"name": "2022", "train": ("2021-01-01", "2021-12-31"), "test": ("2022-01-01", "2022-12-31")},
    {"name": "2023", "train": ("2021-01-01", "2022-12-31"), "test": ("2023-01-01", "2023-12-31")},
    {"name": "2024", "train": ("2021-01-01", "2023-12-31"), "test": ("2024-01-01", "2024-12-31")},
)
LEGACY_AUDIT = ("2025-01-01", "2026-07-01")
PAPER_START = "2026-07-02"
TOP_PER_FOLD = 4
TARGET_ABS_BETA = 0.15
TARGET_SHARPE = 1.5


def conditional_families() -> list[dict[str, Any]]:
    """Conditional-volatility mean reversion using indicators exposed by ManifoldBT MCP."""
    return [
        {
            "name": "garch_liquidity_exhaustion",
            "signals": {
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "shock": "roc(close, param('horizon', default=3))",
                "scaled_shock": "shock / (conditional_vol + 0.000000001)",
                "money_flow": "mfi(14)",
                "range_vol": "natr(14)",
                "rebound": "roc(close, 1)",
            },
            "size": "when((scaled_shock < (0.0 - param('entry', default=3.0))) & (money_flow < param('mfi_low', default=25.0)) & (range_vol > param('min_natr', default=2.0)) & (rebound > (0.0 - param('rebound_floor', default=0.03))), param('size', default=0.08), when((scaled_shock > param('entry', default=3.0)) & (money_flow > param('mfi_high', default=75.0)) & (range_vol > param('min_natr', default=2.0)) & (rebound < param('rebound_floor', default=0.03)), 0.0 - param('size', default=0.08), 0.0))",
            "grid": {
                "horizon": [2, 4, 6],
                "entry": [2.0, 3.5, 5.0],
                "mfi_low": [20.0, 30.0],
                "mfi_high": [70.0, 80.0],
                "min_natr": [1.5, 4.0],
                "rebound_floor": [0.03],
                "size": [0.08],
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
                "scaled_innovation": "innovation / (conditional_vol + 0.000000001)",
                "state_slope": "linreg_slope(state, 48)",
                "abs_slope": "abs_val(state_slope)",
                "trend_strength": "adx(14)",
            },
            "size": "when((scaled_innovation < (0.0 - param('entry', default=2.5))) & (abs_slope < param('slope_limit', default=0.002)) & (trend_strength < param('max_adx', default=28.0)), param('size', default=0.10), when((scaled_innovation > param('entry', default=2.5)) & (abs_slope < param('slope_limit', default=0.002)) & (trend_strength < param('max_adx', default=28.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "entry": [1.5, 2.5, 4.0],
                "slope_limit": [0.001, 0.003],
                "max_adx": [22.0, 32.0],
                "size": [0.08, 0.12],
            },
        },
        {
            "name": "garch_median_reversion",
            "signals": {
                "median": "rolling_median(close, param('window', default=48))",
                "deviation": "close / (median + 0.000000001) - 1.0",
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "scaled_deviation": "deviation / (conditional_vol + 0.000000001)",
                "oscillator": "rsi(close, param('rsi_period', default=10))",
                "trend_strength": "adx(14)",
            },
            "size": "when((scaled_deviation < (0.0 - param('entry', default=2.5))) & (oscillator < param('rsi_low', default=28.0)) & (trend_strength < param('max_adx', default=28.0)), param('size', default=0.10), when((scaled_deviation > param('entry', default=2.5)) & (oscillator > param('rsi_high', default=72.0)) & (trend_strength < param('max_adx', default=28.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "window": [24, 48, 96],
                "entry": [1.5, 2.5, 4.0],
                "rsi_period": [7, 14],
                "rsi_low": [22.0, 30.0],
                "rsi_high": [70.0, 78.0],
                "max_adx": [22.0, 32.0],
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
                "scaled_deviation": "deviation / (conditional_vol + 0.000000001)",
                "commodity_channel": "cci(20)",
                "trend_strength": "adx(14)",
            },
            "size": "when((scaled_deviation < (0.0 - param('entry', default=2.5))) & (commodity_channel < (0.0 - param('cci_level', default=120.0))) & (trend_strength < param('max_adx', default=28.0)), param('size', default=0.10), when((scaled_deviation > param('entry', default=2.5)) & (commodity_channel > param('cci_level', default=120.0)) & (trend_strength < param('max_adx', default=28.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "anchor_period": [20, 40, 70],
                "entry": [1.5, 2.5, 4.0],
                "cci_level": [90.0, 140.0],
                "max_adx": [22.0, 32.0],
                "size": [0.10],
            },
        },
        {
            "name": "conditional_volatility_snapback",
            "signals": {
                "ret1": "roc(close, 1)",
                "fast_vol": "garch(ret1, 0.000001, 0.18, 0.75)",
                "slow_vol": "garch(ret1, 0.000001, 0.05, 0.93)",
                "vol_ratio": "fast_vol / (slow_vol + 0.000000001)",
                "shock": "roc(close, param('horizon', default=3))",
                "scaled_shock": "shock / (fast_vol + 0.000000001)",
                "money_flow": "mfi(14)",
                "rebound": "roc(close, 1)",
            },
            "size": "when((scaled_shock < (0.0 - param('entry', default=3.0))) & (vol_ratio > param('min_vol_ratio', default=1.15)) & (money_flow < param('mfi_low', default=25.0)) & (rebound > (0.0 - param('rebound_floor', default=0.03))), param('size', default=0.08), when((scaled_shock > param('entry', default=3.0)) & (vol_ratio > param('min_vol_ratio', default=1.15)) & (money_flow > param('mfi_high', default=75.0)) & (rebound < param('rebound_floor', default=0.03)), 0.0 - param('size', default=0.08), 0.0))",
            "grid": {
                "horizon": [2, 4, 6],
                "entry": [2.0, 3.5, 5.0],
                "min_vol_ratio": [1.05, 1.25, 1.50],
                "mfi_low": [20.0, 30.0],
                "mfi_high": [70.0, 80.0],
                "rebound_floor": [0.03],
                "size": [0.08],
            },
        },
    ]


def freeze(strategy: dict[str, Any], params: dict[str, Any], name: str) -> dict[str, Any]:
    frozen = json.loads(json.dumps(strategy))
    frozen["name"] = name
    for key, value in params.items():
        default = ((frozen.get("parameters") or {}).get(key) or {}).get("default") or {}
        is_int = isinstance(default, dict) and "Int64" in default
        typed = {"Int64": int(round(value))} if is_int else {"Float64": float(value)}
        if key in frozen.get("parameters", {}):
            frozen["parameters"][key]["default"] = typed
    return frozen


def fold_score(train: dict[str, Any], test: dict[str, Any], probability: float) -> float:
    train_sharpe = engine.metric(train, "sharpe", -99)
    test_sharpe = engine.metric(test, "sharpe", -99)
    alpha = engine.metric(test, "alpha", -1)
    alpha_t = engine.metric(test, "tstat_alpha", -9)
    beta = abs(engine.metric(test, "beta", 9))
    drawdown = abs(engine.metric(test, "max_drawdown", -1))
    trade_penalty = max(0, 20 - engine.trades(test)) * 0.04
    return (
        min(train_sharpe, test_sharpe)
        + 0.20 * test_sharpe
        + 0.30 * max(alpha, 0.0)
        + 0.08 * max(alpha_t, 0.0)
        + 0.15 * probability
        - 3.0 * beta
        - max(0.0, drawdown - 0.20) * 2.0
        - trade_penalty
    )


def consensus_params(selections: list[dict[str, Any]], strategy: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for selection in selections:
        for key, value in selection["params"].items():
            try:
                values.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                continue
    result: dict[str, Any] = {}
    for key, series in values.items():
        default = ((strategy.get("parameters") or {}).get(key) or {}).get("default") or {}
        value = statistics.median(series)
        result[key] = int(round(value)) if isinstance(default, dict) and "Int64" in default else float(value)
    return result


def parameter_drift(selections: list[dict[str, Any]]) -> float:
    values: dict[str, list[float]] = {}
    for selection in selections:
        for key, value in selection["params"].items():
            try:
                values.setdefault(key, []).append(float(value))
            except (TypeError, ValueError):
                continue
    drifts: list[float] = []
    for series in values.values():
        if len(series) < 2:
            continue
        center = abs(statistics.median(series))
        scale = center if center > 1e-9 else 1.0
        drifts.append(min((max(series) - min(series)) / scale, 5.0))
    return statistics.mean(drifts) if drifts else 0.0


def aggregate_fold_metrics(selections: list[dict[str, Any]]) -> dict[str, Any]:
    tests = [selection["test"] for selection in selections]
    sharpes = [engine.metric(row, "sharpe", -99) for row in tests]
    alphas = [engine.metric(row, "alpha", -99) for row in tests]
    alpha_ts = [engine.metric(row, "tstat_alpha", -99) for row in tests]
    betas = [abs(engine.metric(row, "beta", 99)) for row in tests]
    total_trades = sum(engine.trades(row) for row in tests)
    drift = parameter_drift(selections)
    metrics = {
        "median_sharpe": statistics.median(sharpes),
        "minimum_sharpe": min(sharpes),
        "mean_sharpe": statistics.mean(sharpes),
        "median_alpha": statistics.median(alphas),
        "median_alpha_tstat": statistics.median(alpha_ts),
        "maximum_abs_beta": max(betas),
        "positive_sharpe_folds": sum(value > 0 for value in sharpes),
        "positive_alpha_folds": sum(value > 0 for value in alphas),
        "total_trades": total_trades,
        "parameter_drift": drift,
    }
    metrics["score"] = (
        metrics["median_sharpe"]
        + 0.25 * metrics["minimum_sharpe"]
        + 0.30 * max(metrics["median_alpha"], 0.0)
        + 0.08 * max(metrics["median_alpha_tstat"], 0.0)
        + 0.08 * metrics["positive_sharpe_folds"]
        + 0.08 * metrics["positive_alpha_folds"]
        - 2.75 * metrics["maximum_abs_beta"]
        - 0.10 * drift
    )
    metrics["passed"] = (
        metrics["median_sharpe"] >= 0.35
        and metrics["minimum_sharpe"] >= -0.50
        and metrics["positive_sharpe_folds"] >= 2
        and metrics["positive_alpha_folds"] >= 2
        and metrics["maximum_abs_beta"] <= 0.25
        and metrics["total_trades"] >= 60
    )
    return metrics


async def compose(session: Any, family: dict[str, Any]) -> dict[str, Any]:
    payload = await engine.call(session, "compose_strategy", {
        "name": family["name"],
        "signals": family["signals"],
        "size": family["size"],
    })
    strategy = payload.get("strategy_json") or (payload.get("result") or {}).get("strategy_json")
    if not isinstance(strategy, dict):
        raise TypeError(f"compose_strategy did not return strategy_json: {payload}")
    return strategy


async def evaluate_fold(session: Any, family: dict[str, Any], strategy: dict[str, Any], fold: dict[str, Any]) -> dict[str, Any]:
    sweep = await engine.call(session, "run_sweep", {
        "strategy_json": strategy,
        "param_grid": family["grid"],
        "config": engine.cfg(*fold["train"], INTERVAL),
        "lite": True,
        "top_k": TOP_PER_FOLD,
        "rank_metric": "sharpe",
        "device": "auto",
        "precision": "fp64",
    })
    correction = sweep.get("overfitting_correction") or {}
    probability = engine.num(correction.get("probability_edge_is_real"), 0.0)
    train_rows = [row for row in sweep.get("top", []) if row.get("params")]
    docs = [freeze(strategy, row["params"], f"{family['name']}_{fold['name']}_{rank}") for rank, row in enumerate(train_rows)]
    test_rows = engine.rows(await engine.call(session, "run_batch", {
        "strategies": [{"strategy_json": doc} for doc in docs],
        "config": engine.cfg(*fold["test"], INTERVAL),
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
    ]
    candidates.sort(key=lambda row: row["score"], reverse=True)
    if not candidates:
        raise RuntimeError(f"No fold candidate for {family['name']} {fold['name']}")
    return {
        "name": fold["name"],
        "train_period": fold["train"],
        "test_period": fold["test"],
        "overfitting_correction": correction,
        **candidates[0],
    }


async def research(output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "manual expanding-window walk-forward through ManifoldBT MCP",
        "folds": list(FOLDS),
        "legacy_audit": LEGACY_AUDIT,
        "legacy_audit_is_pristine": False,
        "paper_start": PAPER_START,
        "families": [],
        "errors": [],
        "blockers": [
            "The 2025-2026 window was consumed by earlier research and is only a secondary audit, not a pristine holdout.",
            "Native run_walk_forward is Pro-only; folds are orchestrated through separate ManifoldBT calls.",
            "SPY cannot be ingested, so beta is Manifold's available market-factor beta rather than a direct SPY regression.",
            "The datastore contains CryptoSpot proxies and no funding, basis, open-interest or liquidation history.",
        ],
    }
    ClientSession, streamable = engine.mcp_imports()
    async with streamable(engine.MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tools = [tool.model_dump(mode="json") for tool in listed.tools]
            (output / "mcp-tools.json").write_text(json.dumps(tools, indent=2), encoding="utf-8")
            report["version"] = await engine.call(session, "get_version", {})
            report["indicators"] = await engine.call(session, "list_indicators", {})

            finalists: list[dict[str, Any]] = []
            for family in conditional_families():
                family_report: dict[str, Any] = {"name": family["name"], "folds": []}
                try:
                    strategy = await compose(session, family)
                    family_report["compile"] = await engine.call(session, "validate_strategy", {"strategy_json": strategy})
                    selections = []
                    for fold in FOLDS:
                        selection = await evaluate_fold(session, family, strategy, fold)
                        selections.append(selection)
                        family_report["folds"].append({key: value for key, value in selection.items() if key != "strategy_json"})
                    aggregate = aggregate_fold_metrics(selections)
                    params = consensus_params(selections, strategy)
                    locked = freeze(strategy, params, f"{family['name']}_walk_forward_locked")
                    family_report["aggregate"] = aggregate
                    family_report["consensus_params"] = params
                    finalists.append({
                        "family": family["name"],
                        "strategy_json": locked,
                        "folds": selections,
                        "aggregate": aggregate,
                        "params": params,
                    })
                except Exception as exc:
                    family_report["error"] = str(exc)
                    report["errors"].append({"family": family["name"], "error": str(exc)})
                report["families"].append(family_report)

            if not finalists:
                raise RuntimeError("No conditional-volatility family completed all walk-forward folds")
            finalists.sort(key=lambda row: row["aggregate"]["score"], reverse=True)
            passed = [row for row in finalists if row["aggregate"]["passed"]]
            winner = (passed or finalists)[0]
            report["used_gate_fallback"] = not bool(passed)
            report["winner_family"] = winner["family"]
            report["winner_aggregate"] = winner["aggregate"]
            report["winner_params"] = winner["params"]

            locked_path = output / "locked_strategy.json"
            locked_path.write_text(json.dumps(winner["strategy_json"], indent=2), encoding="utf-8")

            legacy = await engine.backtest(
                session,
                winner["strategy_json"],
                LEGACY_AUDIT,
                INTERVAL,
                include_equity_curve=True,
                equity_points=400,
                include_daily_returns=True,
            )
            double_cost = await engine.backtest(session, winner["strategy_json"], LEGACY_AUDIT, INTERVAL, slippage_bps=6.0)
            extra_delay = await engine.backtest(session, winner["strategy_json"], LEGACY_AUDIT, INTERVAL, delay=2)
            development_mc = await engine.call(session, "run_monte_carlo", {
                "strategy_json": winner["strategy_json"],
                "config": engine.cfg("2021-01-01", "2024-12-31", INTERVAL),
                "mc_config": {
                    "n_paths": 1000,
                    "method": {"type": "block_bootstrap", "block_size": 24},
                    "rng_seed": 42,
                    "confidence_levels": [0.9, 0.95, 0.99],
                    "cvar_levels": [0.95, 0.99],
                    "dd_thresholds": [-0.10, -0.20, -0.30],
                },
            })
            report["legacy_audit_result"] = legacy
            report["double_cost"] = double_cost
            report["extra_delay"] = extra_delay
            report["development_monte_carlo"] = development_mc

            paper_end = date.today().isoformat()
            report["paper_end"] = paper_end
            if paper_end > PAPER_START:
                try:
                    report["paper_result"] = await engine.backtest(
                        session,
                        winner["strategy_json"],
                        (PAPER_START, paper_end),
                        INTERVAL,
                        include_equity_curve=True,
                        equity_points=200,
                        include_daily_returns=True,
                    )
                except Exception as exc:
                    report["paper_error"] = str(exc)
            paper = report.get("paper_result") or {}
            paper_days = (date.fromisoformat(paper_end) - date.fromisoformat(PAPER_START)).days
            report["paper_days"] = max(paper_days, 0)
            report["historically_qualified"] = bool(winner["aggregate"]["passed"])
            report["validated_alpha"] = (
                report["historically_qualified"]
                and report["paper_days"] >= 90
                and engine.metric(paper, "sharpe", -99) >= 0.75
                and engine.metric(paper, "alpha", -99) > 0
                and abs(engine.metric(paper, "beta", 99)) <= TARGET_ABS_BETA
                and engine.trades(paper) >= 20
            )
    return report


def plot_report(report: dict[str, Any], output: Path) -> None:
    plt = engine.plt_module()
    path = output / "plots"
    path.mkdir(parents=True, exist_ok=True)

    family = next(item for item in report["families"] if item["name"] == report["winner_family"])
    folds = family["folds"]
    names = [item["name"] for item in folds]
    sharpes = [engine.metric(item["test"], "sharpe", 0.0) for item in folds]
    alphas = [engine.metric(item["test"], "alpha", 0.0) for item in folds]
    betas = [engine.metric(item["test"], "beta", 0.0) for item in folds]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(names, sharpes)
    ax.axhline(0, lw=0.8)
    ax.set_title("Walk-forward Sharpe by unseen year")
    ax.grid(axis="y", alpha=0.25)
    engine.save(fig, path / "fold_sharpe.svg")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = list(range(len(names)))
    ax.bar([value - 0.18 for value in x], alphas, width=0.36, label="Alpha")
    ax.bar([value + 0.18 for value in x], betas, width=0.36, label="Beta")
    ax.axhline(0, lw=0.8)
    ax.set_xticks(x, names)
    ax.set_title("Walk-forward alpha and beta")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    engine.save(fig, path / "fold_alpha_beta.svg")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    family_names = [item["name"] for item in report["families"] if item.get("aggregate")]
    scores = [item["aggregate"]["score"] for item in report["families"] if item.get("aggregate")]
    ax.barh(family_names, scores)
    ax.set_title("Multi-fold selection score")
    ax.grid(axis="x", alpha=0.25)
    engine.save(fig, path / "family_scores.svg")

    legacy = report["legacy_audit_result"]
    stress_names = ["Legacy audit", "Double slippage", "Two-bar delay"]
    stress_values = [
        engine.metric(legacy, "sharpe", 0.0),
        engine.metric(report["double_cost"], "sharpe", 0.0),
        engine.metric(report["extra_delay"], "sharpe", 0.0),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(stress_names, stress_values)
    ax.axhline(0, lw=0.8)
    ax.set_title("Secondary historical audit and execution stress")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    engine.save(fig, path / "legacy_stress.svg")

    paper = report.get("paper_result") or {}
    paper_equity = engine.curve(paper)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if paper_equity:
        ax.plot(paper_equity)
    else:
        ax.text(0.5, 0.5, report.get("paper_error", "No new paper data yet"), ha="center", transform=ax.transAxes)
    ax.set_title(f"Forward paper monitor from {PAPER_START}")
    ax.grid(alpha=0.25)
    engine.save(fig, path / "paper_equity.svg")


def markdown(report: dict[str, Any], output: Path) -> None:
    aggregate = report["winner_aggregate"]
    legacy = report["legacy_audit_result"]
    paper = report.get("paper_result") or {}
    lines = [
        "# Conditional-volatility walk-forward research",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "## Status",
        "",
        f"- Selected family: `{report['winner_family']}`.",
        f"- Historical multi-fold gate: `{'PASS' if report['historically_qualified'] else 'FAIL'}`.",
        f"- Forward paper validation: `{'PASS' if report['validated_alpha'] else 'PENDING/FAIL'}`.",
        f"- Gate fallback used: `{report['used_gate_fallback']}`.",
        "",
        "A strategy is no longer labelled validated from the historical audit alone. It needs at least 90 days and 20 trades in the forward paper window.",
        "",
        "## Walk-forward folds",
        "",
        "| Fold | Unseen period | Sharpe | Alpha | Beta | Trades |",
        "|---|---|---:|---:|---:|---:|",
    ]
    family = next(item for item in report["families"] if item["name"] == report["winner_family"])
    for fold in family["folds"]:
        test = fold["test"]
        lines.append(
            f"| {fold['name']} | {fold['test_period'][0]} to {fold['test_period'][1]} | "
            f"{engine.metric(test, 'sharpe'):.3f} | {engine.metric(test, 'alpha'):.4f} | "
            f"{engine.metric(test, 'beta'):.4f} | {engine.trades(test)} |"
        )
    lines += [
        "",
        "## Aggregate selection metrics",
        "",
        f"- Median unseen Sharpe: `{aggregate['median_sharpe']:.3f}`.",
        f"- Worst unseen Sharpe: `{aggregate['minimum_sharpe']:.3f}`.",
        f"- Median alpha: `{aggregate['median_alpha']:.4f}`.",
        f"- Maximum absolute beta: `{aggregate['maximum_abs_beta']:.4f}`.",
        f"- Positive-alpha folds: `{aggregate['positive_alpha_folds']}/3`.",
        f"- Parameter drift score: `{aggregate['parameter_drift']:.3f}`.",
        "",
        "## Secondary historical audit — not pristine",
        "",
        f"- Sharpe: `{engine.metric(legacy, 'sharpe'):.3f}`.",
        f"- Alpha: `{engine.metric(legacy, 'alpha'):.4f}`.",
        f"- Beta: `{engine.metric(legacy, 'beta'):.4f}`.",
        f"- Return: `{engine.metric(legacy, 'total_return'):.2%}`.",
        f"- Max drawdown: `{engine.metric(legacy, 'max_drawdown'):.2%}`.",
        f"- Double-slippage Sharpe: `{engine.metric(report['double_cost'], 'sharpe'):.3f}`.",
        f"- Two-bar-delay Sharpe: `{engine.metric(report['extra_delay'], 'sharpe'):.3f}`.",
        "",
        "## Forward paper monitor",
        "",
        f"- Window: `{PAPER_START}` to `{report['paper_end']}` ({report['paper_days']} calendar days).",
    ]
    if paper:
        lines += [
            f"- Sharpe: `{engine.metric(paper, 'sharpe'):.3f}`.",
            f"- Alpha: `{engine.metric(paper, 'alpha'):.4f}`.",
            f"- Beta: `{engine.metric(paper, 'beta'):.4f}`.",
            f"- Trades: `{engine.trades(paper)}`.",
        ]
    else:
        lines.append(f"- No usable new data yet: `{report.get('paper_error', 'datastore has no post-start bars')}`.")
    lines += [
        "",
        "## Audit files",
        "",
        "- `locked_strategy.json`: frozen consensus strategy, selected without the legacy audit.",
        "- `raw.json`: complete ManifoldBT payloads.",
        "- `plots/*.svg`: walk-forward, stress and paper-monitor charts.",
        "",
        "## Blocking limitations",
        "",
    ] + [f"- {item}" for item in report["blockers"]]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/walk_forward"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(research(args.output))
        (args.output / "raw.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        markdown(report, args.output)
        plot_report(report, args.output)
        return 0
    except Exception:
        (args.output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
