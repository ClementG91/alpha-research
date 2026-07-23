from __future__ import annotations

import asyncio
import json
import math
import re
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
TOP_TRAIN = 6
MAX_FINALISTS = 24

SINGLES = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]
UNIVERSES = [{"name": s, "tickers": [s]} for s in SINGLES] + [
    {"name": "MAJORS5", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]},
    {"name": "LEGACY6", "tickers": ["BTCUSDT", "ETHUSDT", "LTCUSDT", "BCHUSDT", "LINKUSDT", "TRXUSDT"]},
    {"name": "BROAD10", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT"]},
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
    "donchian_regime_breakout": {
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
    "tail_reversal_regime": {
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
RELATIVE_UNIVERSES = [{"name": f"{s}_VS_BTC", "tickers": [s, "BTCUSDT"]} for s in SINGLES if s != "BTCUSDT"]


def unpack(result: Any) -> Any:
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    text = "\n".join(getattr(x, "text", "") for x in getattr(result, "content", []))
    try:
        return json.loads(text)
    except Exception:
        return {"text": text, "isError": getattr(result, "isError", False)}


async def call(session: ClientSession, name: str, args: dict[str, Any]) -> Any:
    result = await session.call_tool(name, args)
    payload = unpack(result)
    if getattr(result, "isError", False):
        raise RuntimeError(f"{name} failed: {payload}")
    return payload


def as_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("result"), list):
        return payload["result"]
    raise TypeError(f"expected list result, got {type(payload).__name__}: {payload}")


def f(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def metric(row: dict[str, Any], name: str, default: float = float("nan")) -> float:
    return f((row.get("metrics") or {}).get(name), default)


def trade_count(row: dict[str, Any]) -> int:
    try:
        return int(row.get("trade_count") or (row.get("metrics") or {}).get("trade_stats", {}).get("total_trades") or 0)
    except (TypeError, ValueError):
        return 0


def universe_value(tickers: list[str]) -> str | list[str]:
    return tickers[0] if len(tickers) == 1 else tickers


def config(universe: dict[str, Any], start: str, end: str, interval: str) -> dict[str, Any]:
    basket = len(universe["tickers"]) > 1
    return {
        "universe": universe_value(universe["tickers"]),
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
            "max_position_pct": 0.15 if basket else 0.40,
            "allow_short": True,
            "allow_fractional": True,
            "position_sizing_mode": "FractionOfEquity",
            "pyramiding": False,
        },
    }


PARAM_RE_TEMPLATE = r"param\(\s*(['\"]){name}\1\s*,\s*default\s*=\s*[^\)]+\)"


def concrete_text(text: str, params: dict[str, Any]) -> str:
    out = text
    for name, value in params.items():
        pattern = PARAM_RE_TEMPLATE.format(name=re.escape(name))
        out, count = re.subn(pattern, repr(value), out)
        if count == 0:
            raise ValueError(f"parameter {name!r} not found in {text!r}")
    return out


async def compose(session: ClientSession, name: str, spec: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    signals = spec["signals"]
    size = spec["size"]
    if params:
        signals = {key: concrete_text(expr, params) for key, expr in signals.items()}
        size = concrete_text(size, params)
    payload = await call(session, "compose_strategy", {"name": name, "signals": signals, "size": size})
    return payload["strategy_json"]


async def run_batch(session: ClientSession, strategy_jsons: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    payload = await call(session, "run_batch", {"strategies": [{"strategy_json": s} for s in strategy_jsons], "config": cfg, "lite": False, "max_parallelism": 0})
    return as_list(payload)


def selection_score(train: dict[str, Any], valid: dict[str, Any], dsr: float) -> float:
    ts = metric(train, "sharpe", -99)
    vs = metric(valid, "sharpe", -99)
    dd = abs(metric(valid, "max_drawdown", -1))
    penalty = max(0, 15 - trade_count(valid)) * 0.05 + max(0.0, dd - 0.35) * 1.5
    return min(ts, vs) + 0.15 * (ts + vs) + 0.30 * dsr - penalty


def perturb(value: Any, factor: float) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(2, int(round(value * factor)))
    if isinstance(value, float):
        return value * factor
    return value


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train": TRAIN,
        "validation": VALID,
        "test": TEST,
        "criterion": "OOS Sharpe >= 1.5 with positive alpha and robustness checks",
        "hypotheses": [],
        "finalists": [],
        "errors": [],
    }
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            report["version"] = await call(session, "get_version", {})
            report["symbols"] = as_list(await call(session, "list_symbols", {}))

            parameterized: dict[str, dict[str, Any]] = {}
            for strategy_name, spec in STRATEGIES.items():
                try:
                    parameterized[strategy_name] = await compose(session, strategy_name, spec)
                except Exception as exc:
                    report["errors"].append({"stage": "compose", "strategy": strategy_name, "error": str(exc)})

            for strategy_name, spec in STRATEGIES.items():
                if strategy_name not in parameterized:
                    continue
                universes = RELATIVE_UNIVERSES if spec.get("relative_only") else UNIVERSES
                for universe in universes:
                    for interval in INTERVALS:
                        hid = f"{strategy_name}-{universe['name']}-{interval}"
                        try:
                            sweep = await call(session, "run_sweep", {
                                "strategy_json": parameterized[strategy_name],
                                "param_grid": spec["grid"],
                                "config": config(universe, TRAIN[0], TRAIN[1], interval),
                                "lite": True,
                                "top_k": TOP_TRAIN,
                                "rank_metric": "sharpe",
                                "device": "auto",
                                "precision": "fp64",
                            })
                            correction = sweep.get("overfitting_correction") or {}
                            dsr = f(correction.get("probability_edge_is_real"), 0.0)
                            train_rows = [row for row in sweep.get("top", []) if row.get("params")]
                            concrete_jsons = [await compose(session, f"{strategy_name}_{i}", spec, row["params"]) for i, row in enumerate(train_rows)]
                            valid_rows = await run_batch(session, concrete_jsons, config(universe, VALID[0], VALID[1], interval))
                            candidates = []
                            for train_row, valid_row, strat_json in zip(train_rows, valid_rows, concrete_jsons, strict=False):
                                candidates.append({
                                    "params": train_row["params"],
                                    "train": train_row,
                                    "validation": valid_row,
                                    "strategy_json": strat_json,
                                    "score": selection_score(train_row, valid_row, dsr),
                                })
                            candidates.sort(key=lambda x: x["score"], reverse=True)
                            report["hypotheses"].append({
                                "id": hid,
                                "strategy": strategy_name,
                                "universe": universe,
                                "interval": interval,
                                "grid_trials": sweep.get("total"),
                                "overfitting_correction": correction,
                                "sweep_warnings": sweep.get("warnings", []),
                                "selected": candidates[0] if candidates else None,
                            })
                        except Exception as exc:
                            report["errors"].append({"stage": "hypothesis", "id": hid, "error": str(exc)})

            ranked = [h for h in report["hypotheses"] if h.get("selected")]
            ranked.sort(key=lambda h: h["selected"]["score"], reverse=True)
            for h in ranked[:MAX_FINALISTS]:
                selected = h["selected"]
                spec = STRATEGIES[h["strategy"]]
                try:
                    test = await call(session, "run_backtest", {
                        "strategy_json": selected["strategy_json"],
                        "config": config(h["universe"], TEST[0], TEST[1], h["interval"]),
                    })
                    variants = [("center", selected["params"], selected["strategy_json"])]
                    for pname, pvalue in selected["params"].items():
                        for factor in (0.8, 1.2):
                            changed = dict(selected["params"])
                            changed[pname] = perturb(pvalue, factor)
                            variants.append((f"{pname}_{factor:.1f}x", changed, await compose(session, f"{h['strategy']}_{pname}_{factor}", spec, changed)))
                    neighbour_rows = await run_batch(session, [v[2] for v in variants], config(h["universe"], TEST[0], TEST[1], h["interval"]))
                    neighbour_sharpes = [metric(row, "sharpe") for row in neighbour_rows if math.isfinite(metric(row, "sharpe"))]
                    neighbour_median = median(neighbour_sharpes) if neighbour_sharpes else float("nan")
                    dsr = f((h.get("overfitting_correction") or {}).get("probability_edge_is_real"), 0)
                    passes = bool(
                        metric(test, "sharpe", -99) >= MIN_SHARPE
                        and metric(test, "alpha", -99) > 0
                        and trade_count(test) >= MIN_TRADES
                        and metric(selected["validation"], "sharpe", -99) >= 0.6
                        and math.isfinite(neighbour_median)
                        and neighbour_median >= 1.0
                        and dsr >= 0.75
                    )
                    mc = None
                    if passes or metric(test, "sharpe", -99) >= 1.25:
                        mc = await call(session, "run_monte_carlo", {
                            "strategy_json": selected["strategy_json"],
                            "config": config(h["universe"], TEST[0], TEST[1], h["interval"]),
                            "mc_config": {"n_paths": 1000, "method": {"type": "block_bootstrap", "block_size": 24}, "rng_seed": 42},
                        })
                    report["finalists"].append({
                        "id": h["id"], "strategy": h["strategy"], "universe": h["universe"], "interval": h["interval"],
                        "params": selected["params"], "score": selected["score"], "train": selected["train"], "validation": selected["validation"],
                        "test": test, "neighbour_median_sharpe": neighbour_median,
                        "neighbours": [{"label": label, "params": params, "result": row} for (label, params, _), row in zip(variants, neighbour_rows, strict=False)],
                        "overfitting_correction": h["overfitting_correction"], "monte_carlo": mc, "passes": passes,
                        "strategy_json": selected["strategy_json"],
                    })
                except Exception as exc:
                    report["errors"].append({"stage": "final", "id": h["id"], "error": str(exc)})

    report["finalists"].sort(key=lambda x: metric(x.get("test") or {}, "sharpe", -99), reverse=True)
    report["passes"] = [x for x in report["finalists"] if x.get("passes")]
    report["best"] = report["finalists"][0] if report["finalists"] else None
    (OUT / "remote-report.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Remote crypto alpha research", "", f"Generated: {report['generated_at']}", "",
        "Selection: 2021-2023 train, 2024 validation, frozen 2025-2026 OOS. Binance-perps fees, 2 bps slippage and one-bar delay are applied to Binance spot proxy bars.", "",
        "| # | Strategy | Universe | TF | Train S | Valid S | OOS S | Alpha | Return | DD | Trades | Neighbour S | DSR prob. | Pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for i, row in enumerate(report["finalists"], 1):
        test = row["test"]
        dsr = f((row.get("overfitting_correction") or {}).get("probability_edge_is_real"), 0)
        lines.append(
            f"| {i} | {row['strategy']} | {row['universe']['name']} | {row['interval']} | {metric(row['train'],'sharpe'):.2f} | {metric(row['validation'],'sharpe'):.2f} | {metric(test,'sharpe'):.2f} | {metric(test,'alpha'):.3f} | {metric(test,'total_return'):.2%} | {metric(test,'max_drawdown'):.2%} | {trade_count(test)} | {f(row['neighbour_median_sharpe']):.2f} | {dsr:.0%} | {'YES' if row['passes'] else 'NO'} |"
        )
    if report["passes"]:
        w = report["passes"][0]
        lines += ["", "## Validated candidate", "", f"- `{w['strategy']}` on `{w['universe']['name']}` at `{w['interval']}`", f"- Parameters: `{json.dumps(w['params'], sort_keys=True)}`", f"- OOS Sharpe: `{metric(w['test'],'sharpe'):.3f}`", f"- OOS alpha: `{metric(w['test'],'alpha'):.4f}`", f"- OOS return: `{metric(w['test'],'total_return'):.2%}`", f"- OOS max drawdown: `{metric(w['test'],'max_drawdown'):.2%}`", f"- Trades: `{trade_count(w['test'])}`", f"- Neighbour median Sharpe: `{f(w['neighbour_median_sharpe']):.3f}`"]
    else:
        lines += ["", "## Conclusion", "", "No candidate passed every robustness gate. A high raw OOS Sharpe in the table is not considered validated unless Pass is YES."]
    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{json.dumps(e, sort_keys=True)}`" for e in report["errors"]]
    (OUT / "REMOTE_REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
