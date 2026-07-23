from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import manifold_research as engine
import run_manifold_research as family_source

SYMBOL_IDS = {
    "BTCUSDT": 1, "ETHUSDT": 2, "SOLUSDT": 3, "BNBUSDT": 4,
    "XRPUSDT": 5, "ADAUSDT": 6, "DOGEUSDT": 7, "LINKUSDT": 10,
    "LTCUSDT": 11, "TRXUSDT": 12,
}
INTERVAL = "4h"
TOP_TRAIN = 5
SLEEVES = 3


def config(start: str, end: str, *, slippage_bps: float = 3.0, delay: int = 1) -> dict[str, Any]:
    cfg = engine.cfg(start, end, INTERVAL, slippage_bps=slippage_bps, delay=delay)
    cfg["universe"] = list(SYMBOL_IDS.values())
    cfg["symbol_names"] = dict(SYMBOL_IDS)
    return cfg


def families() -> list[dict[str, Any]]:
    docs = family_source.compiler_safe_families()
    grids = {
        "btc_ratio_deviation_reversion": {
            "ratio_window": [72, 144], "resid_window": [96], "horizon": [6, 18],
            "entry_dev": [0.035, 0.065, 0.10], "confirm_dev": [0.025], "max_natr": [8.0],
        },
        "dual_anchor_kalman_reversion": {
            "horizon": [6, 18], "btc_weight": [0.45, 0.70],
            "entry_dev": [0.025, 0.05, 0.085], "slope_limit": [0.002], "max_natr": [9.0],
        },
        "liquidity_shock_snapback": {
            "shock_horizon": [2, 6], "shock_window": [72, 144], "confirm_horizon": [2],
            "entry_dev": [0.04, 0.075, 0.12], "min_natr": [3.0], "rebound_floor": [0.025],
        },
        "relative_rsi_reversion": {
            "rsi_period": [7, 14], "ratio_window": [72, 168], "rsi_low": [25.0],
            "rsi_high": [75.0], "entry_dev": [0.025, 0.05, 0.085], "max_natr": [9.0],
        },
        "multi_horizon_residual_fade": {
            "fast_horizon": [3, 9], "mid_horizon": [18, 48], "fast_window": [96],
            "mid_window": [168], "fast_entry": [0.03, 0.065, 0.10],
            "mid_limit": [0.08], "max_natr": [10.0],
        },
    }
    for doc in docs:
        doc["grid"] = grids[doc["name"]]
    return docs


def parameter_sets(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[key] for key in keys))]


def freeze(doc: dict[str, Any], params: dict[str, Any], name: str) -> dict[str, Any]:
    frozen = copy.deepcopy(doc)
    frozen["name"] = name
    for key, value in params.items():
        if key in frozen.get("parameters", {}):
            frozen["parameters"][key]["default"] = (
                {"Int64": value} if isinstance(value, int) and not isinstance(value, bool)
                else {"Float64": float(value)}
            )
    return frozen


def train_rank(row: dict[str, Any]) -> float:
    return (
        engine.metric(row, "sharpe", -99)
        + 0.25 * max(engine.metric(row, "alpha", 0), 0)
        - 1.75 * abs(engine.metric(row, "beta", 9))
        - 1.5 * max(0, abs(engine.metric(row, "max_drawdown", -1)) - 0.30)
    )


async def logged_call(session: Any, name: str, args: dict[str, Any], out: Path) -> Any:
    try:
        return await engine.call(session, name, args)
    except Exception as exc:
        with (out / "call-errors.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"tool": name, "error": str(exc)}, default=str) + "\n")
        raise


async def batch(session: Any, docs: list[dict[str, Any]], cfg: dict[str, Any], out: Path) -> list[dict[str, Any]]:
    if not docs:
        return []
    payload = await logged_call(session, "run_batch", {
        "strategies": [{"strategy_json": doc} for doc in docs],
        "config": cfg,
        "lite": False,
        "max_parallelism": 0,
    }, out)
    return engine.rows(payload)


async def single(session: Any, doc: dict[str, Any], period: tuple[str, str], out: Path, **kwargs: Any) -> dict[str, Any]:
    cfg = config(*period, slippage_bps=kwargs.pop("slippage_bps", 3.0), delay=kwargs.pop("delay", 1))
    return await logged_call(session, "run_backtest", {"strategy_json": doc, "config": cfg, **kwargs}, out)


async def heatmap(session: Any, candidate: dict[str, Any], family: dict[str, Any], out: Path) -> dict[str, Any]:
    x_name, x_values, y_name, y_values = family["heat"]
    docs, coords = [], []
    for x_value in x_values:
        for y_value in y_values:
            params = {**candidate["params"], x_name: x_value, y_name: y_value}
            docs.append(freeze(candidate["base_strategy"], params, f"heat_{len(docs)}"))
            coords.append((x_value, y_value))
    results = await batch(session, docs, config(*engine.TRAIN), out)
    lookup = {coord: engine.metric(row, "sharpe") for coord, row in zip(coords, results, strict=False)}
    return {
        "x_values": x_values, "y_values": y_values,
        "metric_grid": [[lookup.get((x, y), float("nan")) for y in y_values] for x in x_values],
    }


async def stability(session: Any, candidate: dict[str, Any], family: dict[str, Any], out: Path) -> dict[str, Any]:
    name, values = family["stability"]
    docs = [freeze(candidate["base_strategy"], {**candidate["params"], name: value}, f"stability_{i}") for i, value in enumerate(values)]
    results = await batch(session, docs, config(*engine.VALID), out)
    return {"values": values, "metric_values": [engine.metric(row, "sharpe") for row in results]}


async def research(out: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mcp_url": engine.MCP_URL,
        "engine_only": True,
        "research_direction": "market-neutral residual mean reversion",
        "train": engine.TRAIN, "validation": engine.VALID, "holdout": engine.TEST,
        "target_sharpe": engine.TARGET_SHARPE, "target_abs_beta": engine.TARGET_ABS_BETA,
        "hypotheses": [], "errors": [], "blockers": [], "declared_trials": 0,
    }
    ClientSession, streamable = engine.mcp_imports()
    async with streamable(engine.MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_docs = [tool.model_dump(mode="json") for tool in tools.tools]
            (out / "mcp-tools.json").write_text(json.dumps(tool_docs, indent=2))
            tool_map = {tool["name"]: tool for tool in tool_docs}
            report["version"] = await logged_call(session, "get_version", {}, out)
            report["symbols"] = engine.rows(await logged_call(session, "list_symbols", {}, out))
            indicators = await logged_call(session, "list_indicators", {}, out)
            (out / "indicators.json").write_text(json.dumps(indicators, indent=2))
            report["blockers"] = [
                "SPY is unavailable and ingest_data is not exposed: zero S&P 500 beta cannot be proven on this server.",
                "run_sweep cannot orchestrate SymbolRef; combinations are enumerated in Python and evaluated by Manifold run_batch.",
                "Native run_walk_forward is Pro-only; train, validation and holdout use separate MCP calls.",
                "The datastore contains CryptoSpot proxies without funding, basis, open interest or liquidations.",
                "Community Monte Carlo is capped at 1,000 paths.",
            ]

            candidates: list[dict[str, Any]] = []
            family_by_name: dict[str, dict[str, Any]] = {}
            for family in families():
                family_by_name[family["name"]] = family
                entry: dict[str, Any] = {"family": family["name"], "interval": INTERVAL}
                try:
                    base = await engine.compose(session, family)
                    entry["compile"] = await logged_call(session, "validate_strategy", {"strategy_json": base}, out)
                    params = parameter_sets(family["grid"])
                    report["declared_trials"] += len(params)
                    docs = [freeze(base, values, f"{family['name']}_train_{i}") for i, values in enumerate(params)]
                    train_rows = await batch(session, docs, config(*engine.TRAIN), out)
                    ranked = sorted(zip(params, docs, train_rows, strict=False), key=lambda item: train_rank(item[2]), reverse=True)[:TOP_TRAIN]
                    valid_rows = await batch(session, [item[1] for item in ranked], config(*engine.VALID), out)
                    local = []
                    for (values, doc, train_row), valid_row in zip(ranked, valid_rows, strict=False):
                        row = {
                            "family": family["name"], "interval": INTERVAL,
                            "params": values, "base_strategy": base, "strategy_json": doc,
                            "train": train_row, "validation": valid_row,
                            "score": engine.selection_score(train_row, valid_row, 0.0),
                        }
                        row["passed_validation_gate"] = engine.validation_gate(row)
                        local.append(row)
                    local.sort(key=lambda row: row["score"], reverse=True)
                    entry["selected"] = local[0] if local else None
                    candidates.extend(local[:2])
                except Exception as exc:
                    entry["error"] = str(exc)
                    report["errors"].append({"family": family["name"], "error": str(exc)})
                report["hypotheses"].append(entry)

            if not candidates:
                raise RuntimeError("No mean-reversion candidate completed train and validation")
            candidates.sort(key=lambda row: row["score"], reverse=True)
            for candidate in candidates[:engine.MAX_FINALISTS]:
                candidate["validation_full"] = await single(session, candidate["strategy_json"], engine.VALID, out, include_daily_returns=True)
                candidate["validation_returns"] = engine.daily_returns(candidate["validation_full"])
            strict = [row for row in candidates if row["passed_validation_gate"]]
            pool = strict or candidates
            diverse = engine.choose_diverse(pool[:engine.MAX_FINALISTS], SLEEVES)
            winner = diverse[0]
            family = family_by_name[winner["family"]]

            holdout = await single(session, winner["strategy_json"], engine.TEST, out, include_equity_curve=True, equity_points=500, include_daily_returns=True)
            winner.update({
                "holdout": holdout,
                "test_2025": await single(session, winner["strategy_json"], ("2025-01-01", "2025-12-31"), out),
                "test_2026_ytd": await single(session, winner["strategy_json"], ("2026-01-01", engine.TEST[1]), out),
                "double_cost": await single(session, winner["strategy_json"], engine.TEST, out, slippage_bps=6.0),
                "extra_delay": await single(session, winner["strategy_json"], engine.TEST, out, delay=2),
                "heatmap": await heatmap(session, winner, family, out),
                "stability": await stability(session, winner, family, out),
                "monte_carlo": await logged_call(session, "run_monte_carlo", {
                    "strategy_json": winner["strategy_json"], "config": config(*engine.TEST),
                    "mc_config": {"n_paths": 1000, "method": {"type": "block_bootstrap", "block_size": 24}, "rng_seed": 42,
                                  "confidence_levels": [0.9, 0.95, 0.99], "cvar_levels": [0.95, 0.99],
                                  "dd_thresholds": [-0.15, -0.25, -0.35]},
                }, out),
            })

            portfolio = None
            portfolio_error = None
            if "run_portfolio" in tool_map and len(diverse) >= 2:
                try:
                    portfolio = await logged_call(session, "run_portfolio", {
                        "portfolio": {"strategies": [{"strategy_json": row["strategy_json"], "weight": 1 / len(diverse)} for row in diverse],
                                      "risk_rules": [], "rebalance": {}},
                        "config": config(*engine.TEST),
                    }, out)
                except Exception as exc:
                    portfolio_error = str(exc)

            report["winner"] = winner
            report["selected_sleeves"] = [
                {key: value for key, value in row.items() if key not in {"base_strategy", "strategy_json", "validation_returns"}}
                for row in diverse
            ]
            report["portfolio"] = portfolio
            report["portfolio_error"] = portfolio_error
            report["validation_gate_passes"] = len(strict)
            report["used_gate_fallback"] = not bool(strict)
            report["accepted"] = (
                engine.metric(holdout, "sharpe", -99) >= engine.TARGET_SHARPE
                and engine.metric(holdout, "alpha", -99) > 0
                and engine.metric(holdout, "tstat_alpha", -99) >= 1.5
                and abs(engine.metric(holdout, "beta", 99)) <= engine.TARGET_ABS_BETA
                and engine.metric(winner["validation"], "sharpe", -99) >= 0.5
                and engine.metric(winner["test_2026_ytd"], "sharpe", -99) > 0
                and engine.metric(winner["double_cost"], "sharpe", -99) >= 0.7
                and engine.metric(winner["extra_delay"], "sharpe", -99) >= 0.7
                and engine.trades(holdout) >= 40
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/manifold"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(research(args.output))
        family_source.normalise_stability(report)
        (args.output / "raw.json").write_text(json.dumps(report, indent=2, default=str))
        engine.markdown(report, args.output)
        engine.plots(report, args.output)
        return 0
    except Exception:
        (args.output / "FATAL.txt").write_text(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
