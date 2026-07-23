from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MCP_URL = os.getenv("MANIFOLDBT_MCP_URL", "https://mcp.manifoldbt.com/mcp")
TRAIN = ("2021-01-01", "2023-12-31")
VALID = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-07-01")
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT",
]
INTERVALS = ("1h", "4h")
TARGET_SHARPE = 1.5
TARGET_ABS_BETA = 0.15
TOP_TRAIN = 8
MAX_FINALISTS = 8
PORTFOLIO_SLEEVES = 3


def families() -> list[dict[str, Any]]:
    """Mean-reversion families expressed in ManifoldBT's natural DSL."""
    return [
        {
            "name": "btc_ratio_zscore_reversion",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "ratio": "close / (btc + 0.000000000001)",
                "ratio_z": "ratio.zscore(param('ratio_window', default=96))",
                "residual": "roc(close, param('horizon', default=12)) - roc(btc, param('horizon', default=12))",
                "residual_z": "residual.zscore(param('resid_window', default=96))",
                "vol": "close.pct_change(1).rolling_std(param('vol_window', default=48))",
                "vol_z": "vol.zscore(param('vol_regime', default=168))",
            },
            "size": "when((ratio_z < (0.0 - param('entry_z', default=1.5))) & (residual_z < (0.0 - param('confirm_z', default=0.75))) & (vol_z < param('max_vol_z', default=1.5)), param('size', default=0.10), when((ratio_z > param('entry_z', default=1.5)) & (residual_z > param('confirm_z', default=0.75)) & (vol_z < param('max_vol_z', default=1.5)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "ratio_window": [72, 120], "resid_window": [72, 120],
                "horizon": [6, 18], "entry_z": [1.25, 1.75, 2.25],
                "confirm_z": [0.5, 1.0], "max_vol_z": [1.0, 1.75], "size": [0.10],
            },
            "heat": ("entry_z", [1.0, 1.25, 1.5, 1.75, 2.0, 2.25], "ratio_window", [48, 72, 96, 144, 192]),
            "stability": ("confirm_z", [0.25, 0.5, 0.75, 1.0, 1.25]),
        },
        {
            "name": "dual_anchor_residual_reversion",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "eth": "symbol_ref('ETHUSDT', 'close')",
                "asset_ret": "roc(close, param('horizon', default=12))",
                "market_ret": "roc(btc, param('horizon', default=12)) * param('btc_weight', default=0.65) + roc(eth, param('horizon', default=12)) * (1.0 - param('btc_weight', default=0.65))",
                "residual": "asset_ret - market_ret",
                "residual_z": "residual.zscore(param('window', default=120))",
                "slow_residual": "roc(close, param('slow_horizon', default=72)) - (roc(btc, param('slow_horizon', default=72)) * param('btc_weight', default=0.65) + roc(eth, param('slow_horizon', default=72)) * (1.0 - param('btc_weight', default=0.65)))",
                "slow_z": "slow_residual.zscore(param('slow_window', default=240))",
            },
            "size": "when((residual_z < (0.0 - param('entry_z', default=1.75))) & (slow_z > (0.0 - param('trend_limit', default=1.25))), param('size', default=0.10), when((residual_z > param('entry_z', default=1.75)) & (slow_z < param('trend_limit', default=1.25)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "horizon": [6, 18], "window": [72, 144],
                "slow_horizon": [48, 96], "slow_window": [168, 240],
                "btc_weight": [0.5, 0.7], "entry_z": [1.25, 1.75, 2.25],
                "trend_limit": [0.75, 1.5], "size": [0.10],
            },
            "heat": ("entry_z", [1.0, 1.25, 1.5, 1.75, 2.0, 2.25], "window", [48, 72, 96, 120, 168]),
            "stability": ("btc_weight", [0.35, 0.5, 0.65, 0.8]),
        },
        {
            "name": "liquidity_shock_snapback",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "shock": "roc(close, param('shock_horizon', default=3)) - roc(btc, param('shock_horizon', default=3))",
                "shock_z": "shock.zscore(param('shock_window', default=120))",
                "range_pct": "(high - low) / (close + 0.000000000001)",
                "range_z": "range_pct.zscore(param('range_window', default=120))",
                "rebound": "roc(close, param('confirm_horizon', default=2)) - roc(btc, param('confirm_horizon', default=2))",
            },
            "size": "when((shock_z < (0.0 - param('entry_z', default=2.0))) & (range_z > param('min_range_z', default=0.75)) & (rebound > (0.0 - param('rebound_floor', default=0.04))), param('size', default=0.08), when((shock_z > param('entry_z', default=2.0)) & (range_z > param('min_range_z', default=0.75)) & (rebound < param('rebound_floor', default=0.04)), 0.0 - param('size', default=0.08), 0.0))",
            "grid": {
                "shock_horizon": [2, 6], "shock_window": [72, 144],
                "range_window": [72, 144], "confirm_horizon": [1, 3],
                "entry_z": [1.75, 2.25, 2.75], "min_range_z": [0.25, 1.0],
                "rebound_floor": [0.04], "size": [0.08],
            },
            "heat": ("entry_z", [1.5, 1.75, 2.0, 2.25, 2.5, 2.75], "min_range_z", [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]),
            "stability": ("shock_window", [48, 72, 96, 120, 168, 240]),
        },
        {
            "name": "relative_rsi_reversion",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "ratio": "close / (btc + 0.000000000001)",
                "osc": "rsi(ratio, param('rsi_period', default=10))",
                "ratio_z": "ratio.zscore(param('ratio_window', default=120))",
                "btc_vol": "btc.pct_change(1).rolling_std(param('btc_vol_window', default=48))",
                "btc_vol_z": "btc_vol.zscore(param('btc_vol_regime', default=168))",
            },
            "size": "when((osc < param('rsi_low', default=25.0)) & (ratio_z < (0.0 - param('entry_z', default=1.25))) & (btc_vol_z < param('max_btc_vol_z', default=1.5)), param('size', default=0.10), when((osc > param('rsi_high', default=75.0)) & (ratio_z > param('entry_z', default=1.25)) & (btc_vol_z < param('max_btc_vol_z', default=1.5)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "rsi_period": [7, 14], "ratio_window": [72, 144],
                "btc_vol_window": [24, 72], "btc_vol_regime": [120, 240],
                "rsi_low": [20.0, 27.0], "rsi_high": [73.0, 80.0],
                "entry_z": [1.0, 1.5, 2.0], "max_btc_vol_z": [1.0, 2.0], "size": [0.10],
            },
            "heat": ("rsi_low", [15, 20, 25, 30, 35], "entry_z", [0.75, 1.0, 1.25, 1.5, 2.0]),
            "stability": ("rsi_period", [5, 7, 10, 14, 21]),
        },
        {
            "name": "multi_horizon_residual_fade",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "fast_resid": "roc(close, param('fast_horizon', default=6)) - roc(btc, param('fast_horizon', default=6))",
                "mid_resid": "roc(close, param('mid_horizon', default=24)) - roc(btc, param('mid_horizon', default=24))",
                "fast_z": "fast_resid.zscore(param('fast_window', default=96))",
                "mid_z": "mid_resid.zscore(param('mid_window', default=168))",
                "vol": "close.pct_change(1).rolling_std(param('vol_window', default=48))",
                "vol_z": "vol.zscore(param('vol_regime', default=168))",
            },
            "size": "when((fast_z < (0.0 - param('fast_entry', default=1.75))) & (mid_z > (0.0 - param('mid_limit', default=1.0))) & (vol_z < param('max_vol_z', default=1.75)), param('size', default=0.10), when((fast_z > param('fast_entry', default=1.75)) & (mid_z < param('mid_limit', default=1.0)) & (vol_z < param('max_vol_z', default=1.75)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "fast_horizon": [3, 9], "mid_horizon": [18, 48],
                "fast_window": [72, 144], "mid_window": [120, 240],
                "fast_entry": [1.25, 1.75, 2.25], "mid_limit": [0.5, 1.25],
                "max_vol_z": [1.0, 2.0], "size": [0.10],
            },
            "heat": ("fast_entry", [1.0, 1.25, 1.5, 1.75, 2.0, 2.25], "fast_window", [48, 72, 96, 120, 168]),
            "stability": ("mid_limit", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]),
        },
    ]


def cfg(start: str, end: str, interval: str, *, slippage_bps: float = 3.0, delay: int = 1) -> dict[str, Any]:
    return {
        "universe": UNIVERSE,
        "start": start,
        "end": end,
        "bar_interval": interval,
        "initial_capital": 10_000,
        "fees": "binance_perps",
        "slippage": {"kind": "fixed_bps", "bps": slippage_bps},
        "warmup_bars": 1200 if interval == "1h" else 500,
        "execution": {
            "signal_delay": delay,
            "execution_price": "AtOpen",
            "max_position_pct": 0.12,
            "allow_short": True,
            "allow_fractional": True,
            "position_sizing_mode": "FractionOfEquity",
            "pyramiding": False,
        },
    }


def unpack(result: Any) -> Any:
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    text = "\n".join(getattr(x, "text", "") for x in getattr(result, "content", []))
    try:
        return json.loads(text)
    except Exception:
        return {"text": text}


async def call(session: Any, name: str, args: dict[str, Any]) -> Any:
    result = await session.call_tool(name, args)
    payload = unpack(result)
    if getattr(result, "isError", False):
        raise RuntimeError(f"{name} failed: {payload}")
    return payload


def rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("result", "results", "rows"):
            if isinstance(value.get(key), list):
                return value[key]
    raise TypeError(f"expected list, got {type(value).__name__}")


def num(value: Any, default: float = float("nan")) -> float:
    if isinstance(value, dict):
        for key in ("Float64", "Int64", "Float32", "Int32", "value"):
            if key in value:
                return num(value[key], default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def metric(result: dict[str, Any] | None, key: str, default: float = float("nan")) -> float:
    return num(((result or {}).get("metrics") or {}).get(key), default)


def trades(result: dict[str, Any] | None) -> int:
    result = result or {}
    value = result.get("trade_count") or (((result.get("metrics") or {}).get("trade_stats") or {}).get("total_trades")) or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def curve(result: dict[str, Any]) -> list[float]:
    raw = result.get("equity_curve") or ((result.get("result") or {}).get("equity_curve") if isinstance(result.get("result"), dict) else None)
    if not isinstance(raw, list):
        return []
    values: list[float] = []
    for item in raw:
        value = num(item.get("equity") or item.get("value") or item.get("capital")) if isinstance(item, dict) else num(item)
        if math.isfinite(value):
            values.append(value)
    return values


def daily_returns(result: dict[str, Any]) -> list[float]:
    raw = result.get("daily_returns") or []
    values: list[float] = []
    for item in raw:
        value = num(item.get("return") or item.get("value")) if isinstance(item, dict) else num(item)
        if math.isfinite(value):
            values.append(value)
    return values


def drawdown(values: list[float]) -> list[float]:
    peak = float("-inf")
    result: list[float] = []
    for value in values:
        peak = max(peak, value)
        result.append(value / peak - 1 if peak > 0 else 0.0)
    return result


def correlation(left: list[float], right: list[float]) -> float:
    n = min(len(left), len(right))
    if n < 3:
        return float("nan")
    x, y = left[-n:], right[-n:]
    mx, my = sum(x) / n, sum(y) / n
    dx, dy = [v - mx for v in x], [v - my for v in y]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / den if den else float("nan")


def selection_score(train: dict[str, Any], valid: dict[str, Any], probability: float) -> float:
    train_sharpe = metric(train, "sharpe", -99)
    valid_sharpe = metric(valid, "sharpe", -99)
    alpha = metric(valid, "alpha", -1)
    beta = abs(metric(valid, "beta", 9))
    alpha_t = metric(valid, "tstat_alpha", -9)
    drawdown_penalty = max(0.0, abs(metric(valid, "max_drawdown", -1)) - 0.25) * 2.5
    trade_penalty = max(0, 30 - trades(valid)) * 0.03
    return (
        min(train_sharpe, valid_sharpe)
        + 0.15 * (train_sharpe + valid_sharpe)
        + 0.35 * max(alpha, 0.0)
        + 0.08 * max(alpha_t, 0.0)
        + 0.20 * probability
        - 2.75 * beta
        - drawdown_penalty
        - trade_penalty
    )


def validation_gate(row: dict[str, Any]) -> bool:
    return (
        metric(row["train"], "sharpe", -99) > 0
        and metric(row["validation"], "sharpe", -99) >= 0.35
        and metric(row["validation"], "alpha", -99) > 0
        and abs(metric(row["validation"], "beta", 99)) <= 0.30
        and trades(row["validation"]) >= 25
    )


def choose_diverse(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_families: set[str] = set()
    anchor_interval = candidates[0]["interval"] if candidates else None
    for candidate in candidates:
        if candidate["interval"] != anchor_interval:
            continue
        if candidate["family"] in used_families:
            continue
        if any(abs(correlation(candidate.get("validation_returns", []), other.get("validation_returns", []))) > 0.70 for other in selected):
            continue
        selected.append(candidate)
        used_families.add(candidate["family"])
        if len(selected) >= count:
            break
    if len(selected) < count:
        for candidate in candidates:
            if candidate["interval"] != anchor_interval:
                continue
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= count:
                break
    return selected


def mcp_imports() -> tuple[Any, Any]:
    from mcp import ClientSession
    try:
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        from mcp.client.streamable_http import streamable_http_client as streamablehttp_client
    return ClientSession, streamablehttp_client


async def compose(session: Any, family: dict[str, Any]) -> dict[str, Any]:
    payload = await call(session, "compose_strategy", {
        "name": family["name"],
        "signals": family["signals"],
        "size": family["size"],
    })
    strategy_json = payload.get("strategy_json") or (payload.get("result") or {}).get("strategy_json")
    if not isinstance(strategy_json, dict):
        raise TypeError(f"compose_strategy did not return strategy_json for {family['name']}: {payload}")
    return strategy_json


async def backtest(session: Any, strategy_json: dict[str, Any], period: tuple[str, str], interval: str, **kwargs: Any) -> dict[str, Any]:
    config = cfg(*period, interval, slippage_bps=kwargs.pop("slippage_bps", 3.0), delay=kwargs.pop("delay", 1))
    return await call(session, "run_backtest", {
        "strategy_json": strategy_json,
        "config": config,
        **kwargs,
    })


async def research(out: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mcp_url": MCP_URL,
        "engine_only": True,
        "research_direction": "market-neutral residual mean reversion",
        "train": TRAIN,
        "validation": VALID,
        "holdout": TEST,
        "target_sharpe": TARGET_SHARPE,
        "target_abs_beta": TARGET_ABS_BETA,
        "hypotheses": [],
        "errors": [],
        "blockers": [],
    }
    ClientSession, streamable = mcp_imports()
    async with streamable(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            tool_docs = [tool.model_dump(mode="json") for tool in listed.tools]
            (out / "mcp-tools.json").write_text(json.dumps(tool_docs, indent=2))
            tool_map = {tool["name"]: tool for tool in tool_docs}
            report["version"] = await call(session, "get_version", {})
            report["symbols"] = rows(await call(session, "list_symbols", {}))
            report["indicators"] = await call(session, "list_indicators", {})
            if "ingest_data" not in tool_map:
                report["blockers"].append("No ingest_data tool: SPY cannot be loaded, so zero S&P 500 beta cannot be proven inside this MCP server.")
            if "run_walk_forward" in tool_map and str((report["version"] or {}).get("license_tier", "")).lower() == "community":
                report["blockers"].append("Native run_walk_forward is Pro-only; chronological splits are orchestrated with separate MCP calls.")
            report["blockers"] += [
                "The datastore exposes Binance CryptoSpot proxies rather than full perpetual funding/basis/open-interest history.",
                "Manifold's beta metric is used as the available market-factor proxy; it is not a direct regression against SPY.",
                "Community Monte Carlo is capped at 1,000 paths.",
            ]

            candidates: list[dict[str, Any]] = []
            for family in families():
                for interval in INTERVALS:
                    entry: dict[str, Any] = {"family": family["name"], "interval": interval}
                    try:
                        strategy_json = await compose(session, family)
                        entry["compile"] = await call(session, "validate_strategy", {"strategy_json": strategy_json})
                        sweep = await call(session, "run_sweep", {
                            "strategy_json": strategy_json,
                            "param_grid": family["grid"],
                            "config": cfg(*TRAIN, interval),
                            "lite": True,
                            "top_k": TOP_TRAIN,
                            "rank_metric": "sharpe",
                            "device": "auto",
                            "precision": "fp64",
                        })
                        entry["sweep"] = sweep
                        correction = sweep.get("overfitting_correction") or {}
                        probability = num(correction.get("probability_edge_is_real"), 0.0)
                        train_rows = [row for row in sweep.get("top", []) if row.get("params")]
                        frozen_docs: list[dict[str, Any]] = []
                        for rank, row in enumerate(train_rows):
                            frozen = json.loads(json.dumps(strategy_json))
                            frozen["name"] = f"{family['name']}_{interval}_{rank}"
                            for key, value in row["params"].items():
                                if key in frozen.get("parameters", {}):
                                    frozen["parameters"][key]["default"] = {"Int64": value} if isinstance(value, int) else {"Float64": float(value)}
                            frozen_docs.append(frozen)
                        valid_rows = rows(await call(session, "run_batch", {
                            "strategies": [{"strategy_json": doc} for doc in frozen_docs],
                            "config": cfg(*VALID, interval),
                            "lite": False,
                            "max_parallelism": 0,
                        })) if frozen_docs else []
                        local_candidates = []
                        for train_row, valid_row, doc in zip(train_rows, valid_rows, frozen_docs, strict=False):
                            local_candidates.append({
                                "family": family["name"],
                                "interval": interval,
                                "params": train_row["params"],
                                "strategy_json": doc,
                                "train": train_row,
                                "validation": valid_row,
                                "score": selection_score(train_row, valid_row, probability),
                                "overfitting_correction": correction,
                                "heat": family["heat"],
                                "stability_spec": family["stability"],
                            })
                        local_candidates.sort(key=lambda row: row["score"], reverse=True)
                        entry["selected"] = local_candidates[0] if local_candidates else None
                        candidates.extend(row for row in local_candidates[:2] if validation_gate(row))
                    except Exception as exc:
                        entry["error"] = str(exc)
                        report["errors"].append({"family": family["name"], "interval": interval, "error": str(exc)})
                    report["hypotheses"].append(entry)

            candidates.sort(key=lambda row: row["score"], reverse=True)
            if not candidates:
                raise RuntimeError("No mean-reversion candidate passed the validation alpha/beta gate")

            for candidate in candidates[:MAX_FINALISTS]:
                candidate["validation_full"] = await backtest(
                    session, candidate["strategy_json"], VALID, candidate["interval"],
                    include_daily_returns=True,
                )
                candidate["validation_returns"] = daily_returns(candidate["validation_full"])

            diverse = choose_diverse(candidates[:MAX_FINALISTS], PORTFOLIO_SLEEVES)
            report["selected_sleeves"] = [
                {key: value for key, value in sleeve.items() if key not in {"strategy_json", "validation_returns"}}
                for sleeve in diverse
            ]

            portfolio_result: dict[str, Any] | None = None
            portfolio_error: str | None = None
            if "run_portfolio" in tool_map and len(diverse) >= 2:
                portfolio = {
                    "strategies": [
                        {"strategy_json": sleeve["strategy_json"], "weight": 1.0 / len(diverse)}
                        for sleeve in diverse
                    ],
                    "risk_rules": [],
                    "rebalance": {},
                }
                try:
                    portfolio_result = await call(session, "run_portfolio", {
                        "portfolio": portfolio,
                        "config": cfg(*TEST, diverse[0]["interval"]),
                    })
                except Exception as exc:
                    portfolio_error = str(exc)
                    report["errors"].append({"stage": "portfolio", "error": portfolio_error})

            winner = diverse[0]
            holdout = await backtest(
                session, winner["strategy_json"], TEST, winner["interval"],
                include_equity_curve=True, equity_points=500, include_daily_returns=True,
            )
            x_name, x_values, y_name, y_values = winner["heat"]
            winner.update({
                "holdout": holdout,
                "test_2025": await backtest(session, winner["strategy_json"], ("2025-01-01", "2025-12-31"), winner["interval"]),
                "test_2026_ytd": await backtest(session, winner["strategy_json"], ("2026-01-01", TEST[1]), winner["interval"]),
                "double_cost": await backtest(session, winner["strategy_json"], TEST, winner["interval"], slippage_bps=6.0),
                "extra_delay": await backtest(session, winner["strategy_json"], TEST, winner["interval"], delay=2),
                "heatmap": await call(session, "run_sweep_2d", {
                    "strategy_json": winner["strategy_json"],
                    "sweep_config": {"x_param": x_name, "x_values": x_values, "y_param": y_name, "y_values": y_values, "metric": "sharpe", "max_parallelism": 0},
                    "config": cfg(*TRAIN, winner["interval"]),
                }),
                "stability": await call(session, "run_stability", {
                    "strategy_json": winner["strategy_json"],
                    "stability_config": {"param_name": winner["stability_spec"][0], "values": winner["stability_spec"][1], "metric": "sharpe", "max_parallelism": 0},
                    "config": cfg(*VALID, winner["interval"]),
                }),
                "monte_carlo": await call(session, "run_monte_carlo", {
                    "strategy_json": winner["strategy_json"],
                    "config": cfg(*TEST, winner["interval"]),
                    "mc_config": {
                        "n_paths": 1000,
                        "method": {"type": "block_bootstrap", "block_size": 24},
                        "rng_seed": 42,
                        "confidence_levels": [0.9, 0.95, 0.99],
                        "cvar_levels": [0.95, 0.99],
                        "dd_thresholds": [-0.15, -0.25, -0.35],
                    },
                }),
            })
            report["winner"] = winner
            report["portfolio"] = portfolio_result
            report["portfolio_error"] = portfolio_error
            report["accepted"] = (
                metric(holdout, "sharpe", -99) >= TARGET_SHARPE
                and metric(holdout, "alpha", -99) > 0
                and metric(holdout, "tstat_alpha", -99) >= 1.5
                and abs(metric(holdout, "beta", 99)) <= TARGET_ABS_BETA
                and metric(winner["validation"], "sharpe", -99) >= 0.5
                and metric(winner["test_2026_ytd"], "sharpe", -99) > 0
                and metric(winner["double_cost"], "sharpe", -99) >= 0.7
                and metric(winner["extra_delay"], "sharpe", -99) >= 0.7
                and trades(holdout) >= 40
            )
    return report


def plt_module() -> Any:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save(fig: Any, path: Path) -> None:
    plt = plt_module()
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def plots(report: dict[str, Any], out: Path) -> None:
    plt = plt_module()
    path = out / "plots"
    path.mkdir(parents=True, exist_ok=True)
    winner = report["winner"]
    equity = curve(winner["holdout"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(equity) if equity else ax.text(0.5, 0.5, "No MCP equity curve returned", ha="center", transform=ax.transAxes)
    ax.set_title("Frozen holdout equity — residual mean reversion")
    ax.grid(alpha=0.25)
    save(fig, path / "equity_curve.svg")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    dd = [value * 100 for value in drawdown(equity)]
    ax.fill_between(range(len(dd)), dd, 0, alpha=0.45) if dd else ax.text(0.5, 0.5, "No MCP equity curve returned", ha="center", transform=ax.transAxes)
    ax.set_title("Frozen holdout drawdown")
    ax.set_ylabel("%")
    ax.grid(alpha=0.25)
    save(fig, path / "drawdown.svg")

    periods = [
        ("Train", winner["train"]), ("Validation", winner["validation"]),
        ("Holdout", winner["holdout"]), ("2025", winner["test_2025"]),
        ("2026 YTD", winner["test_2026_ytd"]),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(periods)), [metric(row, "sharpe", 0) for _, row in periods])
    ax.axhline(TARGET_SHARPE, ls="--", label="Target 1.5")
    ax.set_xticks(range(len(periods)), [name for name, _ in periods])
    ax.set_title("Sharpe by chronological segment")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, path / "period_metrics.svg")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = list(range(len(periods)))
    ax.bar([i - 0.18 for i in x], [metric(row, "alpha", 0) for _, row in periods], width=0.36, label="Alpha")
    ax.bar([i + 0.18 for i in x], [metric(row, "beta", 0) for _, row in periods], width=0.36, label="Beta")
    ax.axhline(0, lw=0.8)
    ax.axhline(TARGET_ABS_BETA, ls="--", lw=0.8)
    ax.axhline(-TARGET_ABS_BETA, ls="--", lw=0.8)
    ax.set_xticks(x, [name for name, _ in periods])
    ax.set_title("Manifold alpha and market-factor beta")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save(fig, path / "alpha_beta.svg")

    candidates = [entry.get("selected") for entry in report.get("hypotheses", []) if entry.get("selected")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([abs(metric(row["validation"], "beta", 0)) for row in candidates], [metric(row["validation"], "sharpe", 0) for row in candidates])
    ax.axvline(TARGET_ABS_BETA, ls="--", label="|beta| target")
    ax.set_xlabel("Absolute validation beta")
    ax.set_ylabel("Validation Sharpe")
    ax.set_title("Validation frontier: Sharpe versus beta")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, path / "beta_frontier.svg")

    heat = winner.get("heatmap") or {}
    grid = heat.get("metric_grid") or (heat.get("result") or {}).get("metric_grid")
    x_values = heat.get("x_values") or (heat.get("result") or {}).get("x_values") or []
    y_values = heat.get("y_values") or (heat.get("result") or {}).get("y_values") or []
    fig, ax = plt.subplots(figsize=(7.5, 5))
    if grid:
        image = ax.imshow(grid, aspect="auto", origin="lower")
        ax.set_xticks(range(len(y_values)), y_values)
        ax.set_yticks(range(len(x_values)), x_values)
        fig.colorbar(image, ax=ax, label="Sharpe")
    else:
        ax.text(0.5, 0.5, "No run_sweep_2d grid returned", ha="center", transform=ax.transAxes)
    ax.set_title("Training parameter surface")
    save(fig, path / "parameter_heatmap.svg")

    stability = winner.get("stability") or {}
    values = stability.get("values") or stability.get("param_values") or (stability.get("result") or {}).get("values") or []
    scores = stability.get("metrics") or stability.get("metric_values") or (stability.get("result") or {}).get("metrics") or []
    if scores and isinstance(scores[0], dict):
        scores = [num(row.get("value") or row.get("sharpe")) for row in scores]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(values[:len(scores)], scores[:len(values)], marker="o") if values and scores else ax.text(0.5, 0.5, "No plottable stability series", ha="center", transform=ax.transAxes)
    ax.set_title("Validation parameter stability")
    ax.grid(alpha=0.25)
    save(fig, path / "stability.svg")

    stress = [("Base", winner["holdout"]), ("Double cost", winner["double_cost"]), ("2-bar delay", winner["extra_delay"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([name for name, _ in stress], [metric(row, "sharpe", 0) for _, row in stress])
    ax.set_title("Execution stress tests")
    ax.set_ylabel("Sharpe")
    ax.grid(axis="y", alpha=0.25)
    save(fig, path / "stress_tests.svg")

    mc = winner.get("monte_carlo") or {}
    return_percentiles = dict((mc.get("final_return") or {}).get("percentiles") or [])
    dd_percentiles = dict((mc.get("max_drawdown") or {}).get("percentiles") or [])
    labels = ["Return p90", "Return p95", "Return p99", "DD p90", "DD p95", "DD p99"]
    data = [num(return_percentiles.get(level), 0) * 100 for level in (0.9, 0.95, 0.99)] + [-num(dd_percentiles.get(level), 0) * 100 for level in (0.9, 0.95, 0.99)]
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.bar(labels, data)
    ax.axhline(0, lw=0.8)
    ax.tick_params(axis="x", rotation=20)
    ax.set_title(f"Monte Carlo risk ({mc.get('n_paths', 1000)} paths)")
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.25)
    save(fig, path / "monte_carlo.svg")


def markdown(report: dict[str, Any], out: Path) -> None:
    winner = report["winner"]
    lines = [
        "# ManifoldBT residual mean-reversion research",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "The research target is genuine alpha with low market-factor beta, not directional crypto exposure.",
        "All backtests, alpha, beta, equity and Monte Carlo metrics come from ManifoldBT MCP.",
        "",
        f"- Selected family: `{winner['family']}` on `{winner['interval']}`",
        f"- Sharpe target: `{TARGET_SHARPE}`",
        f"- Absolute beta target: `{TARGET_ABS_BETA}`",
        f"- Accepted: `{'YES' if report['accepted'] else 'NO'}`",
        "",
        "| Period | Sharpe | Alpha | Beta | Alpha t-stat | Return | Max DD | Trades |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in [
        ("Train", winner["train"]), ("Validation", winner["validation"]),
        ("Frozen holdout", winner["holdout"]), ("2025", winner["test_2025"]),
        ("2026 YTD", winner["test_2026_ytd"]),
    ]:
        lines.append(
            f"| {name} | {metric(row, 'sharpe'):.3f} | {metric(row, 'alpha'):.4f} | {metric(row, 'beta'):.4f} | "
            f"{metric(row, 'tstat_alpha'):.3f} | {metric(row, 'total_return'):.2%} | {metric(row, 'max_drawdown'):.2%} | {trades(row)} |"
        )
    lines += [
        "",
        "## Execution stress",
        "",
        f"- Double-slippage Sharpe: `{metric(winner['double_cost'], 'sharpe'):.3f}`.",
        f"- Two-bar-delay Sharpe: `{metric(winner['extra_delay'], 'sharpe'):.3f}`.",
        "",
        "## Selected sleeves",
        "",
    ]
    for sleeve in report.get("selected_sleeves", []):
        lines.append(
            f"- `{sleeve['family']}` / `{sleeve['interval']}`: validation Sharpe "
            f"`{metric(sleeve['validation'], 'sharpe'):.3f}`, alpha `{metric(sleeve['validation'], 'alpha'):.4f}`, "
            f"beta `{metric(sleeve['validation'], 'beta'):.4f}`."
        )
    if report.get("portfolio"):
        lines += ["", "## Manifold portfolio", "", "`run_portfolio` completed; the full payload is preserved in `raw.json`."]
    elif report.get("portfolio_error"):
        lines += ["", "## Portfolio blocker", "", f"`run_portfolio` failed: `{report['portfolio_error']}`"]
    lines += ["", "## Blocking limitations", ""] + [f"- {item}" for item in report["blockers"]]
    lines += ["", "Raw MCP responses: `raw.json`. Tool schemas: `mcp-tools.json`. Plots: `plots/*.svg`."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/manifold"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(research(args.output))
        (args.output / "raw.json").write_text(json.dumps(report, indent=2, default=str))
        markdown(report, args.output)
        plots(report, args.output)
        return 0
    except Exception:
        (args.output / "FATAL.txt").write_text(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
