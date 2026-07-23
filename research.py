from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "https://mcp.manifoldbt.com/mcp"
OUT = Path("results")
STORE = {"data_root": "data", "metadata_db": "metadata/metadata.sqlite"}

ASSETS = [
    {"symbol": "BTCUSDT", "id": 1},
    {"symbol": "ETHUSDT", "id": 2},
    {"symbol": "SOLUSDT", "id": 3},
    {"symbol": "BNBUSDT", "id": 4},
    {"symbol": "XRPUSDT", "id": 5},
]
ASSET_BY_SYMBOL = {row["symbol"]: row for row in ASSETS}

TRAIN = ("2021-01-01", "2023-12-31")
VALID = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-07-01")
INTERVALS = ("4h", "12h", "1d")

MIN_OOS_SHARPE = 1.5
MIN_OOS_TRADES = 18
MAX_FINAL_HYPOTHESES = 18
TOP_TRAIN_CANDIDATES = 12

STRATEGIES: dict[str, dict[str, Any]] = {
    "regime_switch": {
        "assets": [row["symbol"] for row in ASSETS],
        "code": """
fast = ema(close, param('fast', default=24))
slow = ema(close, param('slow', default=168))
trend_mom = roc(close, param('mom', default=24))
trend_strength = adx(param('adx_period', default=14))
mean_z = close.zscore(param('z_window', default=72))
trend_long = (fast > slow) & (trend_mom > lit(0.0)) & (trend_strength > lit(20.0))
trend_short = (fast < slow) & (trend_mom < lit(0.0)) & (trend_strength > lit(20.0))
range_long = (trend_strength < lit(18.0)) & (mean_z < lit(-1.6)) & (close > slow * lit(0.72))
range_short = (trend_strength < lit(18.0)) & (mean_z > lit(1.6)) & (close < slow * lit(1.28))
position = when(trend_long, lit(0.42), when(trend_short, lit(-0.42), when(range_long, lit(0.28), when(range_short, lit(-0.28), lit(0.0)))))
strategy = (
    Strategy.create('regime_switch')
    .signal('fast', fast)
    .signal('slow', slow)
    .signal('trend_mom', trend_mom)
    .signal('trend_strength', trend_strength)
    .signal('mean_z', mean_z)
    .size(position)
    .stop_loss(pct=5.0)
    .trailing_stop(pct=3.5)
)
""",
        "grid": {
            "fast": [12, 24, 36],
            "slow": [96, 168, 240],
            "mom": [12, 24, 48],
            "adx_period": [10, 14, 20],
            "z_window": [48, 72, 120],
        },
    },
    "volatility_adjusted_momentum": {
        "assets": [row["symbol"] for row in ASSETS],
        "code": """
mom = roc(close, param('mom', default=48))
ret = close.pct_change(1)
vol = ret.rolling_std(param('vol_window', default=72))
trend = ema(close, param('trend_span', default=168))
vol_z = vol.zscore(param('vol_regime', default=168))
long_cond = (mom > lit(0.0)) & (close > trend) & (vol_z < lit(1.25))
short_cond = (mom < lit(0.0)) & (close < trend) & (vol_z < lit(1.25))
quiet = vol_z < lit(-0.6)
position = when(long_cond & quiet, lit(0.50), when(short_cond & quiet, lit(-0.50), when(long_cond, lit(0.28), when(short_cond, lit(-0.28), lit(0.0)))))
strategy = (
    Strategy.create('volatility_adjusted_momentum')
    .signal('mom', mom)
    .signal('vol', vol)
    .signal('trend', trend)
    .signal('vol_z', vol_z)
    .size(position)
    .stop_loss(pct=6.0)
    .trailing_stop(pct=4.0)
)
""",
        "grid": {
            "mom": [12, 24, 48, 96],
            "vol_window": [24, 48, 72, 120],
            "trend_span": [96, 168, 240],
            "vol_regime": [96, 168, 240],
        },
    },
    "compression_breakout": {
        "assets": [row["symbol"] for row in ASSETS],
        "code": """
width = bollinger_width(close, period=param('bb_period', default=24), num_std=2.0)
width_z = width.zscore(param('width_window', default=120))
upper = close.rolling_max(param('breakout', default=48)).lag(1)
lower = close.rolling_min(param('breakout', default=48)).lag(1)
trend = ema(close, param('trend_span', default=168))
compressed = width_z < lit(-0.55)
long_cond = compressed.lag(1) & (close > upper) & (close > trend)
short_cond = compressed.lag(1) & (close < lower) & (close < trend)
position = when(long_cond, lit(0.45), when(short_cond, lit(-0.45), lit(0.0)))
strategy = (
    Strategy.create('compression_breakout')
    .signal('width', width)
    .signal('width_z', width_z)
    .signal('upper', upper)
    .signal('lower', lower)
    .signal('trend', trend)
    .size(position)
    .stop_loss(pct=4.5)
    .take_profit(pct=11.0)
)
""",
        "grid": {
            "bb_period": [18, 24, 36],
            "width_window": [72, 120, 168],
            "breakout": [24, 48, 72, 120],
            "trend_span": [96, 168, 240],
        },
    },
    "panic_rebound": {
        "assets": [row["symbol"] for row in ASSETS],
        "code": """
z = close.zscore(param('z_window', default=72))
osc = rsi(close, param('rsi_period', default=10))
regime = ema(close, param('regime_span', default=240))
rebound = roc(close, param('confirmation', default=6))
long_cond = (z < lit(-2.0)) & (osc < lit(28.0)) & (close > regime * lit(0.64)) & (rebound > lit(-0.08))
short_cond = (z > lit(2.2)) & (osc > lit(76.0)) & (close < regime * lit(1.36)) & (rebound < lit(0.08))
position = when(long_cond, lit(0.36), when(short_cond, lit(-0.26), lit(0.0)))
strategy = (
    Strategy.create('panic_rebound')
    .signal('z', z)
    .signal('osc', osc)
    .signal('regime', regime)
    .signal('rebound', rebound)
    .size(position)
    .stop_loss(pct=4.0)
    .take_profit(pct=7.5)
)
""",
        "grid": {
            "z_window": [36, 72, 120, 168],
            "rsi_period": [7, 10, 14],
            "regime_span": [120, 240, 360],
            "confirmation": [3, 6, 12, 24],
        },
    },
    "btc_relative_strength": {
        "assets": ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"],
        "cross_asset": True,
        "code": """
btc = symbol_ref('BTCUSDT', 'close')
ratio = close / (btc + lit(1e-12))
ratio_mom = roc(ratio, param('ratio_mom', default=24))
ratio_z = ratio.zscore(param('ratio_window', default=120))
btc_trend = ema(btc, param('btc_trend', default=168))
asset_trend = ema(close, param('asset_trend', default=96))
btc_up = btc > btc_trend
btc_down = btc < btc_trend
long_cond = btc_up & (close > asset_trend) & (ratio_mom > lit(0.0)) & (ratio_z > lit(0.25))
short_cond = btc_down & (close < asset_trend) & (ratio_mom < lit(0.0)) & (ratio_z < lit(-0.25))
position = when(long_cond, lit(0.38), when(short_cond, lit(-0.32), lit(0.0)))
strategy = (
    Strategy.create('btc_relative_strength')
    .signal('btc', btc)
    .signal('ratio', ratio)
    .signal('ratio_mom', ratio_mom)
    .signal('ratio_z', ratio_z)
    .signal('btc_trend', btc_trend)
    .signal('asset_trend', asset_trend)
    .size(position)
    .stop_loss(pct=5.0)
    .trailing_stop(pct=3.5)
)
""",
        "grid": {
            "ratio_mom": [12, 24, 48, 72],
            "ratio_window": [72, 120, 168, 240],
            "btc_trend": [96, 168, 240],
            "asset_trend": [48, 96, 168],
        },
    },
}


def unpack(result: Any) -> Any:
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    texts = [getattr(item, "text", "") for item in getattr(result, "content", [])]
    text = "\n".join(part for part in texts if part)
    try:
        return json.loads(text)
    except Exception:
        return {"text": text, "isError": getattr(result, "isError", False)}


async def call(session: ClientSession, name: str, arguments: dict[str, Any]) -> Any:
    result = await session.call_tool(name, arguments)
    payload = unpack(result)
    if getattr(result, "isError", False):
        raise RuntimeError(f"{name} failed: {payload}")
    return payload


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def metric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    return safe_float((row.get("metrics") or {}).get(key), default)


def trades(row: dict[str, Any]) -> int:
    try:
        return int(row.get("trade_count") or 0)
    except (TypeError, ValueError):
        return 0


def universe_for(asset: dict[str, Any], spec: dict[str, Any]) -> tuple[list[int], dict[str, int]]:
    if spec.get("cross_asset") and asset["symbol"] != "BTCUSDT":
        return [asset["id"], ASSET_BY_SYMBOL["BTCUSDT"]["id"]], {
            asset["symbol"]: asset["id"],
            "BTCUSDT": ASSET_BY_SYMBOL["BTCUSDT"]["id"],
        }
    return [asset["id"]], {asset["symbol"]: asset["id"]}


def config(asset: dict[str, Any], spec: dict[str, Any], start: str, end: str, interval: str) -> dict[str, Any]:
    universe, symbol_names = universe_for(asset, spec)
    return {
        "universe": universe,
        "symbol_names": symbol_names,
        "start": start,
        "end": end,
        "bar_interval": interval,
        "initial_capital": 10_000,
        "fees": "binance_perps",
        "slippage": {"kind": "fixed_bps", "bps": 2.0},
        "warmup_bars": 520,
        "execution": {
            "signal_delay": 1,
            "execution_price": "AtOpen",
            "max_position_pct": 0.50,
            "allow_short": True,
            "allow_fractional": True,
            "position_sizing_mode": "FractionOfEquity",
            "pyramiding": False,
        },
    }


def concretize(code: str, params: dict[str, Any]) -> str:
    concrete = code
    for name, value in params.items():
        pattern = rf"param\(\s*(['\"])({re.escape(name)})\1\s*,\s*default\s*=\s*[^\)]+\)"
        concrete, count = re.subn(pattern, repr(value), concrete)
        if count == 0:
            raise ValueError(f"parameter {name!r} was not found in strategy source")
    return concrete


def candidate_score(train_row: dict[str, Any], valid_row: dict[str, Any]) -> float:
    train_sharpe = metric(train_row, "sharpe", -99.0)
    valid_sharpe = metric(valid_row, "sharpe", -99.0)
    valid_dd = abs(metric(valid_row, "max_drawdown", -1.0))
    trade_penalty = max(0, 12 - trades(valid_row)) * 0.08
    drawdown_penalty = max(0.0, valid_dd - 0.35) * 2.0
    return min(train_sharpe, valid_sharpe) + 0.20 * (train_sharpe + valid_sharpe) - trade_penalty - drawdown_penalty


def perturb(value: Any, factor: float) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(2, int(round(value * factor)))
    if isinstance(value, float):
        return value * factor
    return value


def neighbourhood(code: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [{"label": "center", "params": dict(params), "code": concretize(code, params)}]
    for name, value in params.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        for factor in (0.8, 1.2):
            changed = dict(params)
            changed[name] = perturb(value, factor)
            variants.append({"label": f"{name}_{factor:.1f}x", "params": changed, "code": concretize(code, changed)})
    return variants


async def batch_backtest(session: ClientSession, codes: list[str], cfg: dict[str, Any], *, lite: bool) -> list[dict[str, Any]]:
    if not codes:
        return []
    return await call(session, "run_batch", {"strategies": [{"strategy_code": code} for code in codes], "config": cfg, "store": STORE, "lite": lite, "max_parallelism": 0})


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mcp_url": MCP_URL,
        "criterion": f"net final OOS Sharpe >= {MIN_OOS_SHARPE}",
        "train": TRAIN,
        "validation": VALID,
        "test": TEST,
        "hypotheses": [],
        "finalists": [],
        "errors": [],
    }

    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            schemas = [tool.model_dump(mode="json") for tool in tools.tools]
            (OUT / "mcp-tools.json").write_text(json.dumps(schemas, indent=2))
            report["version"] = await call(session, "get_version", {})

            for asset in ASSETS:
                try:
                    await call(session, "ingest_data", {"provider": "binance", "symbol": asset["symbol"], "symbol_id": asset["id"], "start": f"{TRAIN[0]}T00:00:00Z", "end": f"{TEST[1]}T00:00:00Z", "interval": "1h", "data_root": STORE["data_root"], "metadata_db": STORE["metadata_db"], "exchange": "binance", "asset_class": "crypto_perp"})
                except Exception as exc:
                    report["errors"].append({"stage": "ingest", "asset": asset["symbol"], "error": str(exc)})

            report["symbols"] = await call(session, "list_symbols", {"store": STORE})

            compiled: dict[str, Any] = {}
            for strategy_name, spec in STRATEGIES.items():
                try:
                    compiled[strategy_name] = await call(session, "build_strategy", {"strategy_code": spec["code"]})
                except Exception as exc:
                    report["errors"].append({"stage": "compile", "strategy": strategy_name, "error": str(exc)})
            report["compiled"] = compiled

            for strategy_name, spec in STRATEGIES.items():
                if strategy_name not in compiled:
                    continue
                for symbol in spec["assets"]:
                    asset = ASSET_BY_SYMBOL[symbol]
                    for interval in INTERVALS:
                        hypothesis_id = f"{strategy_name}-{symbol}-{interval}"
                        try:
                            sweep = await call(session, "run_sweep", {"strategy_code": spec["code"], "param_grid": spec["grid"], "config": config(asset, spec, TRAIN[0], TRAIN[1], interval), "store": STORE, "lite": True, "top_k": TOP_TRAIN_CANDIDATES, "rank_metric": "sharpe", "device": "auto", "precision": "fp64"})
                            train_candidates = [row for row in sweep.get("top", []) if row.get("params")]
                            concrete_codes = [concretize(spec["code"], row["params"]) for row in train_candidates]
                            validation_rows = await batch_backtest(session, concrete_codes, config(asset, spec, VALID[0], VALID[1], interval), lite=False)
                            candidates: list[dict[str, Any]] = []
                            for train_row, validation_row, concrete_code in zip(train_candidates, validation_rows, concrete_codes, strict=False):
                                candidates.append({"params": train_row["params"], "train": train_row, "validation": validation_row, "concrete_code": concrete_code, "selection_score": candidate_score(train_row, validation_row)})
                            candidates.sort(key=lambda row: row["selection_score"], reverse=True)
                            report["hypotheses"].append({"id": hypothesis_id, "strategy": strategy_name, "asset": symbol, "interval": interval, "grid_size": sweep.get("total"), "selected": candidates[0] if candidates else None, "validation_rank": None})
                        except Exception as exc:
                            report["errors"].append({"stage": "hypothesis", "id": hypothesis_id, "error": str(exc)})

            ranked_hypotheses = [row for row in report["hypotheses"] if row.get("selected")]
            ranked_hypotheses.sort(key=lambda row: row["selected"]["selection_score"], reverse=True)
            for rank, hypothesis in enumerate(ranked_hypotheses, 1):
                hypothesis["validation_rank"] = rank

            for hypothesis in ranked_hypotheses[:MAX_FINAL_HYPOTHESES]:
                strategy_name = hypothesis["strategy"]
                spec = STRATEGIES[strategy_name]
                asset = ASSET_BY_SYMBOL[hypothesis["asset"]]
                selected = hypothesis["selected"]
                try:
                    final_row = (await batch_backtest(session, [selected["concrete_code"]], config(asset, spec, TEST[0], TEST[1], hypothesis["interval"]), lite=False))[0]
                    final_sharpe = metric(final_row, "sharpe", -99.0)
                    passes_basic = final_sharpe >= MIN_OOS_SHARPE and trades(final_row) >= MIN_OOS_TRADES and metric(selected["train"], "sharpe", -99.0) > 0.0 and metric(selected["validation"], "sharpe", -99.0) > 0.4

                    variants = neighbourhood(spec["code"], selected["params"])
                    neighbour_rows = await batch_backtest(session, [variant["code"] for variant in variants], config(asset, spec, TEST[0], TEST[1], hypothesis["interval"]), lite=False)
                    neighbour_results = []
                    neighbour_sharpes = []
                    for variant, row in zip(variants, neighbour_rows, strict=False):
                        value = metric(row, "sharpe")
                        if math.isfinite(value):
                            neighbour_sharpes.append(value)
                        neighbour_results.append({"label": variant["label"], "params": variant["params"], "result": row})
                    neighbour_median = median(neighbour_sharpes) if neighbour_sharpes else float("nan")
                    robust = bool(math.isfinite(neighbour_median) and neighbour_median >= 0.9)

                    monte_carlo = None
                    if passes_basic or final_sharpe >= 1.2:
                        monte_carlo = await call(session, "run_monte_carlo", {"strategy_code": selected["concrete_code"], "config": config(asset, spec, TEST[0], TEST[1], hypothesis["interval"]), "store": STORE, "mc_config": {"n_paths": 1000, "method": {"type": "block_bootstrap", "block_size": 24}, "rng_seed": 42}})

                    report["finalists"].append({"id": hypothesis["id"], "strategy": strategy_name, "asset": hypothesis["asset"], "interval": hypothesis["interval"], "params": selected["params"], "selection_score": selected["selection_score"], "train": selected["train"], "validation": selected["validation"], "test": final_row, "neighbours": neighbour_results, "neighbour_median_sharpe": neighbour_median, "monte_carlo": monte_carlo, "passes_basic": passes_basic, "passes": bool(passes_basic and robust), "concrete_strategy_code": selected["concrete_code"]})
                except Exception as exc:
                    report["errors"].append({"stage": "final", "id": hypothesis["id"], "error": str(exc)})

    report["finalists"].sort(key=lambda row: metric(row.get("test") or {}, "sharpe", -99.0), reverse=True)
    report["passes"] = [row for row in report["finalists"] if row.get("passes")]
    report["best"] = report["finalists"][0] if report["finalists"] else None
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Crypto alpha research report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"Final acceptance: Sharpe >= {MIN_OOS_SHARPE}, at least {MIN_OOS_TRADES} trades, positive train/validation and neighbourhood median Sharpe >= 0.9.",
        "",
        "Final test was not used for parameter selection. Parameters were selected on 2021-2023 training and 2024 validation data, then frozen for 2025-2026 testing.",
        "",
        "| Rank | Strategy | Asset | TF | Train Sharpe | Validation Sharpe | OOS Sharpe | Return | Max DD | Trades | Neighbour median | Pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for index, row in enumerate(report["finalists"], 1):
        train_row = row.get("train") or {}
        valid_row = row.get("validation") or {}
        test_row = row.get("test") or {}
        lines.append(f"| {index} | {row['strategy']} | {row['asset']} | {row['interval']} | {metric(train_row, 'sharpe'):.3f} | {metric(valid_row, 'sharpe'):.3f} | {metric(test_row, 'sharpe'):.3f} | {metric(test_row, 'total_return'):.3f} | {metric(test_row, 'max_drawdown'):.3f} | {trades(test_row)} | {safe_float(row.get('neighbour_median_sharpe')):.3f} | {'YES' if row.get('passes') else 'NO'} |")

    if report["passes"]:
        winner = report["passes"][0]
        lines += ["", "## Validated candidate", "", f"- Strategy: `{winner['strategy']}`", f"- Asset: `{winner['asset']}`", f"- Timeframe: `{winner['interval']}`", f"- Parameters: `{json.dumps(winner['params'], sort_keys=True)}`", f"- OOS Sharpe: `{metric(winner['test'], 'sharpe'):.3f}`", f"- OOS return: `{metric(winner['test'], 'total_return'):.3f}`", f"- OOS max drawdown: `{metric(winner['test'], 'max_drawdown'):.3f}`", f"- OOS trades: `{trades(winner['test'])}`", f"- Neighbour median Sharpe: `{safe_float(winner['neighbour_median_sharpe']):.3f}`", "", "### Frozen strategy code", "", "```python", winner["concrete_strategy_code"].strip(), "```"]
    else:
        lines += ["", "## Conclusion", "", "No candidate met the full robustness threshold in this run. The highest OOS Sharpe is reported above, but it must not be treated as a validated alpha unless Pass is YES."]

    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{json.dumps(error, sort_keys=True)}`" for error in report["errors"]]

    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
