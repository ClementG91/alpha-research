from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "https://mcp.manifoldbt.com/mcp"
OUT = Path("results")
TRAIN = ("2021-01-01", "2023-12-31")
VALID = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-07-01")
INTERVALS = ("4h", "12h")
MIN_SHARPE = 1.5
MIN_TRADES = 20
TOP_TRAIN = 5
MAX_FINALISTS = 12
CALL_INTERVAL_SECONDS = 1.6

UNIVERSES = [
    {"name": "BTCUSDT", "tickers": ["BTCUSDT"]},
    {"name": "ETHUSDT", "tickers": ["ETHUSDT"]},
    {"name": "SOLUSDT", "tickers": ["SOLUSDT"]},
    {"name": "MAJORS5", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]},
    {"name": "BROAD10", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT"]},
]
RELATIVE_UNIVERSES = [
    {"name": "ETH_VS_BTC", "tickers": ["ETHUSDT", "BTCUSDT"]},
    {"name": "SOL_VS_BTC", "tickers": ["SOLUSDT", "BTCUSDT"]},
    {"name": "BNB_VS_BTC", "tickers": ["BNBUSDT", "BTCUSDT"]},
    {"name": "LINK_VS_BTC", "tickers": ["LINKUSDT", "BTCUSDT"]},
]

STRATEGIES: dict[str, dict[str, Any]] = {
    "trend_rsi_vol": {
        "signals": {
            "fast": "ema(close, param('fast', default=24))",
            "slow": "ema(close, param('slow', default=168))",
            "osc": "rsi(close, param('rsi_period', default=14))",
            "atr_pct": "atr(param('atr_period', default=14)) / close",
        },
        "size": "when((fast > slow) & (osc > 54) & (atr_pct < 0.12), 0.35, when((fast < slow) & (osc < 46) & (atr_pct < 0.12), -0.35, 0.0))",
        "grid": {"fast": [12, 24, 36], "slow": [96, 168, 240], "rsi_period": [10, 14, 21], "atr_period": [10, 14, 21]},
    },
    "trend_pullback": {
        "signals": {
            "fast": "ema(close, param('fast', default=24))",
            "regime": "ema(close, param('regime', default=168))",
            "osc": "rsi(close, param('rsi_period', default=14))",
        },
        "size": "when((close > regime) & (close < fast) & (osc < 44), 0.32, when((close < regime) & (close > fast) & (osc > 56), -0.32, 0.0))",
        "grid": {"fast": [12, 24, 36, 48], "regime": [96, 168, 240], "rsi_period": [7, 10, 14, 21]},
    },
    "donchian_breakout": {
        "signals": {
            "upper": "highest(close, param('breakout', default=48))",
            "lower": "lowest(close, param('breakout', default=48))",
            "regime": "ema(close, param('regime', default=168))",
            "osc": "rsi(close, param('rsi_period', default=14))",
        },
        "size": "when((close >= upper) & (close > regime) & (osc > 55), 0.38, when((close <= lower) & (close < regime) & (osc < 45), -0.38, 0.0))",
        "grid": {"breakout": [24, 48, 72, 120], "regime": [96, 168, 240], "rsi_period": [10, 14, 21]},
    },
    "volatility_expansion": {
        "signals": {
            "fast": "ema(close, param('fast', default=24))",
            "slow": "ema(close, param('slow', default=168))",
            "atr_pct": "atr(param('atr_period', default=14)) / close",
            "atr_base": "sma(atr_pct, param('vol_window', default=72))",
        },
        "size": "when((fast > slow) & (atr_pct > atr_base * 1.15), 0.34, when((fast < slow) & (atr_pct > atr_base * 1.15), -0.34, 0.0))",
        "grid": {"fast": [12, 24, 36], "slow": [96, 168, 240], "atr_period": [10, 14, 21], "vol_window": [48, 72, 120]},
    },
    "tail_reversal": {
        "signals": {
            "osc": "rsi(close, param('rsi_period', default=10))",
            "regime": "ema(close, param('regime', default=240))",
            "fast": "ema(close, param('fast', default=24))",
        },
        "size": "when((osc < 27) & (close > regime * 0.65) & (close < fast), 0.30, when((osc > 73) & (close < regime * 1.35) & (close > fast), -0.26, 0.0))",
        "grid": {"rsi_period": [7, 10, 14, 21], "regime": [120, 240, 360], "fast": [12, 24, 48]},
    },
    "btc_relative_momentum": {
        "relative_only": True,
        "signals": {
            "btc": "symbol_ref('BTCUSDT', 'close')",
            "ratio": "close / (btc + 0.000000000001)",
            "ratio_fast": "ema(ratio, param('ratio_fast', default=24))",
            "ratio_slow": "ema(ratio, param('ratio_slow', default=120))",
            "btc_regime": "ema(btc, param('btc_regime', default=168))",
        },
        "size": "when((ratio_fast > ratio_slow) & (btc > btc_regime), 0.30, when((ratio_fast < ratio_slow) & (btc < btc_regime), -0.30, 0.0))",
        "grid": {"ratio_fast": [12, 24, 48], "ratio_slow": [72, 120, 168], "btc_regime": [96, 168, 240]},
    },
}

_last_call = 0.0


def unpack(result: Any) -> Any:
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    text = "\n".join(getattr(x, "text", "") for x in getattr(result, "content", []))
    try:
        return json.loads(text)
    except Exception:
        return {"text": text, "isError": getattr(result, "isError", False)}


async def call(session: ClientSession, name: str, args: dict[str, Any]) -> Any:
    global _last_call
    delay = CALL_INTERVAL_SECONDS - (time.monotonic() - _last_call)
    if delay > 0:
        await asyncio.sleep(delay)
    result = await session.call_tool(name, args)
    _last_call = time.monotonic()
    payload = unpack(result)
    if getattr(result, "isError", False):
        raise RuntimeError(f"{name} failed: {payload}")
    return payload


def as_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("result"), list):
        return payload["result"]
    raise TypeError(f"expected list result, got {payload}")


def num(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def metric(row: dict[str, Any], name: str, default: float = float("nan")) -> float:
    return num((row.get("metrics") or {}).get(name), default)


def trades(row: dict[str, Any]) -> int:
    raw = row.get("trade_count") or (row.get("metrics") or {}).get("trade_stats", {}).get("total_trades") or 0
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def config(universe: dict[str, Any], start: str, end: str, interval: str) -> dict[str, Any]:
    tickers = universe["tickers"]
    return {
        "universe": tickers[0] if len(tickers) == 1 else tickers,
        "start": start,
        "end": end,
        "bar_interval": interval,
        "initial_capital": 10000,
        "fees": "binance_perps",
        "slippage": {"kind": "fixed_bps", "bps": 2.0},
        "warmup_bars": 400,
        "execution": {
            "signal_delay": 1,
            "execution_price": "AtOpen",
            "max_position_pct": 0.15 if len(tickers) > 1 else 0.40,
            "allow_short": True,
            "allow_fractional": True,
            "position_sizing_mode": "FractionOfEquity",
            "pyramiding": False,
        },
    }


async def compose(session: ClientSession, name: str, spec: dict[str, Any]) -> dict[str, Any]:
    payload = await call(session, "compose_strategy", {"name": name, "signals": spec["signals"], "size": spec["size"]})
    return payload["strategy_json"]


def typed_default(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"Bool": value}
    if isinstance(value, int):
        return {"Int64": value}
    return {"Float64": float(value)}


def freeze(base: dict[str, Any], params: dict[str, Any], name: str) -> dict[str, Any]:
    doc = copy.deepcopy(base)
    doc["name"] = name
    specs = doc.get("parameters") or {}
    for key, value in params.items():
        if key not in specs:
            raise KeyError(f"missing parameter {key}")
        specs[key]["default"] = typed_default(value)
    doc["parameters"] = specs
    return doc


async def run_batch(session: ClientSession, docs: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    payload = await call(session, "run_batch", {"strategies": [{"strategy_json": x} for x in docs], "config": cfg, "lite": False, "max_parallelism": 0})
    return as_list(payload)


def score(train: dict[str, Any], valid: dict[str, Any], dsr: float) -> float:
    ts, vs = metric(train, "sharpe", -99), metric(valid, "sharpe", -99)
    dd = abs(metric(valid, "max_drawdown", -1))
    penalty = max(0, 12 - trades(valid)) * 0.06 + max(0.0, dd - 0.35) * 1.5
    return min(ts, vs) + 0.15 * (ts + vs) + 0.30 * dsr - penalty


def perturb(value: Any, factor: float) -> Any:
    if isinstance(value, int):
        return max(2, int(round(value * factor)))
    return float(value) * factor


def checkpoint(report: dict[str, Any]) -> None:
    (OUT / "checkpoint.json").write_text(json.dumps(report, indent=2, default=str))


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
            report["version"] = await call(session, "get_version", {})
            report["symbols"] = as_list(await call(session, "list_symbols", {}))

            bases: dict[str, dict[str, Any]] = {}
            for name, spec in STRATEGIES.items():
                try:
                    bases[name] = await compose(session, name, spec)
                except Exception as exc:
                    report["errors"].append({"stage": "compose", "strategy": name, "error": str(exc)})
            checkpoint(report)

            for name, spec in STRATEGIES.items():
                if name not in bases:
                    continue
                universes = RELATIVE_UNIVERSES if spec.get("relative_only") else UNIVERSES
                for universe in universes:
                    for interval in INTERVALS:
                        hid = f"{name}-{universe['name']}-{interval}"
                        try:
                            sweep = await call(session, "run_sweep", {
                                "strategy_json": bases[name],
                                "param_grid": spec["grid"],
                                "config": config(universe, TRAIN[0], TRAIN[1], interval),
                                "lite": True,
                                "top_k": TOP_TRAIN,
                                "rank_metric": "sharpe",
                            })
                            correction = sweep.get("overfitting_correction") or {}
                            dsr = num(correction.get("probability_edge_is_real"), 0)
                            train_rows = [x for x in sweep.get("top", []) if x.get("params")]
                            docs = [freeze(bases[name], x["params"], f"{name}_{i}") for i, x in enumerate(train_rows)]
                            valid_rows = await run_batch(session, docs, config(universe, VALID[0], VALID[1], interval))
                            candidates = []
                            for train, valid, doc in zip(train_rows, valid_rows, docs, strict=False):
                                candidates.append({"params": train["params"], "train": train, "validation": valid, "strategy_json": doc, "score": score(train, valid, dsr)})
                            candidates.sort(key=lambda x: x["score"], reverse=True)
                            report["hypotheses"].append({
                                "id": hid, "strategy": name, "universe": universe, "interval": interval,
                                "grid_trials": sweep.get("total"), "overfitting_correction": correction,
                                "sweep_warnings": sweep.get("warnings", []), "selected": candidates[0] if candidates else None,
                            })
                        except Exception as exc:
                            report["errors"].append({"stage": "hypothesis", "id": hid, "error": str(exc)})
                        checkpoint(report)

            ranked = [x for x in report["hypotheses"] if x.get("selected")]
            ranked.sort(key=lambda x: x["selected"]["score"], reverse=True)
            for h in ranked[:MAX_FINALISTS]:
                selected = h["selected"]
                base = bases[h["strategy"]]
                try:
                    test = await call(session, "run_backtest", {"strategy_json": selected["strategy_json"], "config": config(h["universe"], TEST[0], TEST[1], h["interval"])})
                    variants = [("center", selected["params"], selected["strategy_json"])]
                    for key, value in selected["params"].items():
                        for factor in (0.8, 1.2):
                            changed = dict(selected["params"])
                            changed[key] = perturb(value, factor)
                            variants.append((f"{key}_{factor:.1f}x", changed, freeze(base, changed, f"{h['strategy']}_{key}_{factor}")))
                    neighbour_rows = await run_batch(session, [x[2] for x in variants], config(h["universe"], TEST[0], TEST[1], h["interval"]))
                    sharpes = [metric(x, "sharpe") for x in neighbour_rows if math.isfinite(metric(x, "sharpe"))]
                    neighbour_median = median(sharpes) if sharpes else float("nan")
                    dsr = num((h.get("overfitting_correction") or {}).get("probability_edge_is_real"), 0)
                    passed = bool(metric(test, "sharpe", -99) >= MIN_SHARPE and metric(test, "alpha", -99) > 0 and trades(test) >= MIN_TRADES and metric(selected["validation"], "sharpe", -99) >= 0.6 and math.isfinite(neighbour_median) and neighbour_median >= 1.0 and dsr >= 0.75)
                    mc = None
                    if passed or metric(test, "sharpe", -99) >= 1.25:
                        mc = await call(session, "run_monte_carlo", {"strategy_json": selected["strategy_json"], "config": config(h["universe"], TEST[0], TEST[1], h["interval"]), "mc_config": {"n_paths": 1000, "method": {"type": "block_bootstrap", "block_size": 24}, "rng_seed": 42}})
                    report["finalists"].append({
                        "id": h["id"], "strategy": h["strategy"], "universe": h["universe"], "interval": h["interval"],
                        "params": selected["params"], "train": selected["train"], "validation": selected["validation"], "test": test,
                        "neighbour_median_sharpe": neighbour_median,
                        "neighbours": [{"label": label, "params": params, "result": row} for (label, params, _), row in zip(variants, neighbour_rows, strict=False)],
                        "overfitting_correction": h["overfitting_correction"], "monte_carlo": mc, "passes": passed,
                        "strategy_json": selected["strategy_json"],
                    })
                except Exception as exc:
                    report["errors"].append({"stage": "final", "id": h["id"], "error": str(exc)})
                checkpoint(report)

    report["finalists"].sort(key=lambda x: metric(x.get("test") or {}, "sharpe", -99), reverse=True)
    report["passes"] = [x for x in report["finalists"] if x.get("passes")]
    report["best"] = report["finalists"][0] if report["finalists"] else None
    (OUT / "campaign.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Crypto alpha campaign", "", f"Generated: {report['generated_at']}", "",
        "2021-2023 train, 2024 validation, frozen 2025-2026 OOS. Binance-perps fees, 2 bps slippage, one-bar delay; Binance spot bars are used as the price proxy.", "",
        "| # | Strategy | Universe | TF | Train S | Valid S | OOS S | Alpha | Return | DD | Trades | Neighbour S | DSR | Pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for i, row in enumerate(report["finalists"], 1):
        dsr = num((row.get("overfitting_correction") or {}).get("probability_edge_is_real"), 0)
        test = row["test"]
        lines.append(f"| {i} | {row['strategy']} | {row['universe']['name']} | {row['interval']} | {metric(row['train'],'sharpe'):.2f} | {metric(row['validation'],'sharpe'):.2f} | {metric(test,'sharpe'):.2f} | {metric(test,'alpha'):.3f} | {metric(test,'total_return'):.2%} | {metric(test,'max_drawdown'):.2%} | {trades(test)} | {num(row['neighbour_median_sharpe']):.2f} | {dsr:.0%} | {'YES' if row['passes'] else 'NO'} |")
    if report["passes"]:
        w = report["passes"][0]
        lines += ["", "## Validated candidate", "", f"- Strategy: `{w['strategy']}`", f"- Universe: `{w['universe']['name']}`", f"- Timeframe: `{w['interval']}`", f"- Parameters: `{json.dumps(w['params'], sort_keys=True)}`", f"- OOS Sharpe: `{metric(w['test'],'sharpe'):.3f}`", f"- OOS alpha: `{metric(w['test'],'alpha'):.4f}`", f"- Return: `{metric(w['test'],'total_return'):.2%}`", f"- Max drawdown: `{metric(w['test'],'max_drawdown'):.2%}`", f"- Trades: `{trades(w['test'])}`", f"- Neighbour median Sharpe: `{num(w['neighbour_median_sharpe']):.3f}`"]
    else:
        lines += ["", "## Conclusion", "", "No candidate passed every robustness gate. Raw high-Sharpe rows are not validated unless Pass is YES."]
    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{json.dumps(x, sort_keys=True)}`" for x in report["errors"]]
    (OUT / "CAMPAIGN.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
