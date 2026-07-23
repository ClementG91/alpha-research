from __future__ import annotations

import asyncio
import copy
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import campaign

OUT = Path("results")
TRAIN = ("2021-01-01", "2023-12-31")
VALID = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-07-01")
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
INTERVAL = "4h"
TOP_TRAIN = 10


def lit(value: float) -> dict[str, Any]:
    return {"Literal": {"Float64": float(value)}}


def col(name: str) -> dict[str, Any]:
    return {"Column": name}


def param(name: str) -> dict[str, Any]:
    return {"Parameter": name}


def add(a: Any, b: Any) -> dict[str, Any]:
    return {"Add": [a, b]}


def sub(a: Any, b: Any) -> dict[str, Any]:
    return {"Sub": [a, b]}


def mul(a: Any, b: Any) -> dict[str, Any]:
    return {"Mul": [a, b]}


def div(a: Any, b: Any) -> dict[str, Any]:
    return {"Div": [a, b]}


def gt(a: Any, b: Any) -> dict[str, Any]:
    return {"Gt": [a, b]}


def lt(a: Any, b: Any) -> dict[str, Any]:
    return {"Lt": [a, b]}


def iff(cond: Any, yes: Any, no: Any) -> dict[str, Any]:
    return {"IfElse": [cond, yes, no]}


def parameter_spec(name: str, default: Any) -> dict[str, Any]:
    typed = {"Int64": default} if isinstance(default, int) else {"Float64": float(default)}
    return {"name": name, "default": typed, "description": "", "range": None}


def build_strategy(name: str, vol_expr: dict[str, Any]) -> dict[str, Any]:
    close = col("close")
    fast = {"EwmMean": [close, 8.0]}
    slow = {"EwmMean": [close, 168.0]}
    vol_z = {"ZScore": [vol_expr, "vol_window"]}

    high_vol = gt(vol_z, param("risk_z"))
    long_size = iff(high_vol, param("risk_long"), lit(0.15))
    short_size = iff(high_vol, mul(lit(-1.0), param("risk_short")), lit(-0.15))
    position = iff(gt(fast, slow), long_size, iff(lt(fast, slow), short_size, lit(0.0)))

    return {
        "name": name,
        "signals": {
            "fast": fast,
            "slow": slow,
            "volatility": vol_expr,
            "vol_z": vol_z,
        },
        "position_sizing": position,
        "parameters": {
            "vol_window": parameter_spec("vol_window", 72),
            "risk_z": parameter_spec("risk_z", 1.0),
            "risk_long": parameter_spec("risk_long", 0.05),
            "risk_short": parameter_spec("risk_short", 0.05),
        },
        "constraints": [],
        "metadata": {"description": "EMA 8/168 with native volatility-state exposure overlay"},
    }


def strategies() -> dict[str, dict[str, Any]]:
    close, high, low = col("close"), col("high"), col("low")
    range_vol = div({"RollingMean": [sub(high, low), 14]}, add(close, lit(1e-12)))
    returns = {"PctChange": [close, 1]}
    return_vol = {"RollingStd": [returns, 14]}
    return {
        "ema_range_vol_overlay": build_strategy("ema_range_vol_overlay", range_vol),
        "ema_return_vol_overlay": build_strategy("ema_return_vol_overlay", return_vol),
    }


GRID = {
    "vol_window": [48, 72, 120, 168],
    "risk_z": [0.5, 1.0, 1.5, 2.0],
    "risk_long": [0.0, 0.05, 0.10, 0.15],
    "risk_short": [0.0, 0.05, 0.10, 0.15],
}


def cfg(start: str, end: str) -> dict[str, Any]:
    return {
        "universe": UNIVERSE,
        "start": start,
        "end": end,
        "bar_interval": INTERVAL,
        "initial_capital": 10_000,
        "fees": "binance_perps",
        "slippage": {"kind": "fixed_bps", "bps": 2.0},
        "warmup_bars": 500,
        "execution": {
            "signal_delay": 1,
            "execution_price": "AtOpen",
            "max_position_pct": 0.15,
            "allow_short": True,
            "allow_fractional": True,
            "position_sizing_mode": "FractionOfEquity",
            "pyramiding": False,
        },
    }


def metric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    return campaign.metric(row, key, default)


def trades(row: dict[str, Any]) -> int:
    return campaign.trades(row)


def freeze(base: dict[str, Any], values: dict[str, Any], name: str) -> dict[str, Any]:
    doc = copy.deepcopy(base)
    doc["name"] = name
    for key, value in values.items():
        doc["parameters"][key]["default"] = {"Int64": value} if isinstance(value, int) else {"Float64": float(value)}
    return doc


def score(train: dict[str, Any], valid: dict[str, Any], probability: float) -> float:
    ts, vs = metric(train, "sharpe", -99), metric(valid, "sharpe", -99)
    dd = abs(metric(valid, "max_drawdown", -1))
    return min(ts, vs) + 0.20 * (ts + vs) + 0.25 * probability - max(0.0, dd - 0.35) * 1.5


def perturb(value: Any, factor: float) -> Any:
    if isinstance(value, int):
        return max(2, int(round(value * factor)))
    return max(0.0, float(value) * factor)


def checkpoint(report: dict[str, Any]) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "generation7-checkpoint.json").write_text(json.dumps(report, indent=2, default=str))


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
    bases = strategies()

    async with streamablehttp_client(campaign.MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            report["version"] = await campaign.call(session, "get_version", {})

            for name, base in bases.items():
                try:
                    await campaign.call(session, "validate_strategy", {"strategy_json": json.dumps(base)})
                    sweep = await campaign.call(session, "run_sweep", {
                        "strategy_json": base,
                        "param_grid": GRID,
                        "config": cfg(TRAIN[0], TRAIN[1]),
                        "lite": True,
                        "top_k": TOP_TRAIN,
                        "rank_metric": "sharpe",
                    })
                    correction = sweep.get("overfitting_correction") or {}
                    probability = campaign.num(correction.get("probability_edge_is_real"), 0.0)
                    train_rows = [row for row in sweep.get("top", []) if row.get("params")]
                    docs = [freeze(base, row["params"], f"{name}_{idx}") for idx, row in enumerate(train_rows)]
                    valid_rows = await campaign.run_batch(session, docs, cfg(VALID[0], VALID[1]))
                    candidates = []
                    for train, valid, doc in zip(train_rows, valid_rows, docs, strict=False):
                        candidates.append({
                            "params": train["params"],
                            "train": train,
                            "validation": valid,
                            "strategy_json": doc,
                            "score": score(train, valid, probability),
                        })
                    candidates.sort(key=lambda row: row["score"], reverse=True)
                    report["hypotheses"].append({
                        "strategy": name,
                        "trials": sweep.get("total"),
                        "warnings": sweep.get("warnings", []),
                        "overfitting_correction": correction,
                        "selected": candidates[0] if candidates else None,
                    })
                except Exception as exc:
                    report["errors"].append({"stage": "hypothesis", "strategy": name, "error": str(exc)})
                checkpoint(report)

            ranked = [row for row in report["hypotheses"] if row.get("selected")]
            ranked.sort(key=lambda row: row["selected"]["score"], reverse=True)
            for hypothesis in ranked:
                name = hypothesis["strategy"]
                selected = hypothesis["selected"]
                try:
                    test = await campaign.call(session, "run_backtest", {"strategy_json": selected["strategy_json"], "config": cfg(TEST[0], TEST[1])})
                    variants = [("center", selected["params"], selected["strategy_json"])]
                    for key in ("vol_window", "risk_z", "risk_long", "risk_short"):
                        value = selected["params"][key]
                        for factor in (0.8, 1.2):
                            changed = dict(selected["params"])
                            changed[key] = perturb(value, factor)
                            variants.append((f"{key}_{factor}", changed, freeze(bases[name], changed, f"{name}_{key}_{factor}")))
                    neighbours = await campaign.run_batch(session, [row[2] for row in variants], cfg(TEST[0], TEST[1]))
                    sharpes = [metric(row, "sharpe") for row in neighbours if math.isfinite(metric(row, "sharpe"))]
                    neighbour_median = median(sharpes) if sharpes else float("nan")
                    probability = campaign.num((hypothesis.get("overfitting_correction") or {}).get("probability_edge_is_real"), 0.0)
                    passed = bool(
                        metric(test, "sharpe", -99) >= 1.5
                        and metric(test, "alpha", -99) > 0
                        and trades(test) >= 30
                        and metric(selected["train"], "sharpe", -99) >= 0.70
                        and metric(selected["validation"], "sharpe", -99) >= 0.70
                        and neighbour_median >= 1.0
                        and probability >= 0.75
                    )
                    monte_carlo = None
                    if passed or metric(test, "sharpe", -99) >= 1.30:
                        monte_carlo = await campaign.call(session, "run_monte_carlo", {
                            "strategy_json": selected["strategy_json"],
                            "config": cfg(TEST[0], TEST[1]),
                            "mc_config": {"n_paths": 3000, "method": {"type": "block_bootstrap", "block_size": 24}, "rng_seed": 42},
                        })
                    report["finalists"].append({
                        "strategy": name,
                        "params": selected["params"],
                        "train": selected["train"],
                        "validation": selected["validation"],
                        "test": test,
                        "neighbour_median_sharpe": neighbour_median,
                        "neighbours": [{"label": label, "params": params, "result": result} for (label, params, _), result in zip(variants, neighbours, strict=False)],
                        "overfitting_correction": hypothesis["overfitting_correction"],
                        "monte_carlo": monte_carlo,
                        "passes": passed,
                        "strategy_json": selected["strategy_json"],
                    })
                except Exception as exc:
                    report["errors"].append({"stage": "final", "strategy": name, "error": str(exc)})
                checkpoint(report)

    report["finalists"].sort(key=lambda row: metric(row.get("test") or {}, "sharpe", -99), reverse=True)
    report["passes"] = [row for row in report["finalists"] if row.get("passes")]
    report["best"] = report["finalists"][0] if report["finalists"] else None
    (OUT / "generation7.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Crypto alpha generation 7",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Native StrategyDef volatility overlays on the EMA 8/168 core. MAJORS5, 4h. Train 2021-2023, validation 2024, frozen OOS 2025-2026. Binance-perps fees, 2 bps slippage and one-bar delay; spot bars proxy prices and funding is unavailable.",
        "",
        "| # | Strategy | Params | Train S | Valid S | OOS S | Alpha | Return | DD | Trades | Neighbour S | DSR | Pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for idx, row in enumerate(report["finalists"], 1):
        test = row["test"]
        probability = campaign.num((row.get("overfitting_correction") or {}).get("probability_edge_is_real"), 0.0)
        lines.append(
            f"| {idx} | {row['strategy']} | `{json.dumps(row['params'], sort_keys=True)}` | "
            f"{metric(row['train'],'sharpe'):.2f} | {metric(row['validation'],'sharpe'):.2f} | {metric(test,'sharpe'):.2f} | "
            f"{metric(test,'alpha'):.3f} | {metric(test,'total_return'):.2%} | {metric(test,'max_drawdown'):.2%} | "
            f"{trades(test)} | {row['neighbour_median_sharpe']:.2f} | {probability:.0%} | {'YES' if row['passes'] else 'NO'} |"
        )
    if report["passes"]:
        winner = report["passes"][0]
        lines += [
            "", "## Validated candidate", "",
            f"- Strategy: `{winner['strategy']}`",
            f"- Parameters: `{json.dumps(winner['params'], sort_keys=True)}`",
            f"- OOS Sharpe: `{metric(winner['test'],'sharpe'):.3f}`",
            f"- OOS alpha: `{metric(winner['test'],'alpha'):.4f}`",
            f"- Return: `{metric(winner['test'],'total_return'):.2%}`",
            f"- Max drawdown: `{metric(winner['test'],'max_drawdown'):.2%}`",
            f"- Trades: `{trades(winner['test'])}`",
            f"- Neighbour median Sharpe: `{winner['neighbour_median_sharpe']:.3f}`",
        ]
    else:
        lines += ["", "## Conclusion", "", "No candidate reached and retained OOS Sharpe >= 1.5 through all gates."]
    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{json.dumps(error, sort_keys=True)}`" for error in report["errors"]]
    (OUT / "GENERATION7.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
