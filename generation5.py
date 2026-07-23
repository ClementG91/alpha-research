from __future__ import annotations

import asyncio
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
INTERVALS = ("4h", "12h")
TOP_TRAIN = 8
MAX_FINALISTS = 14

UNIVERSES = [
    {"name": "MAJORS5", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]},
    {"name": "BROAD10", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT"]},
]

STRATEGIES: dict[str, dict[str, Any]] = {
    "vol_mom_binary_ls": {
        "signals": {
            "mom": "ema(roc(close, param('mom_period', default=14)), param('smooth', default=6))",
            "avg_range": "sma(high - low, param('vol_window', default=14))",
            "norm_vol": "avg_range / (close + 0.000000000001)",
            "safe_vol": "when(norm_vol > 0.003, norm_vol, 0.003)",
            "score": "mom / safe_vol",
        },
        "size": "when(score > param('entry', default=1.0), 0.40, when(score < (0 - param('entry', default=1.0)), -0.40, 0.0))",
        "grid": {
            "mom_period": [7, 14, 28, 42],
            "smooth": [3, 6, 12],
            "vol_window": [10, 14, 20],
            "entry": [0.5, 1.0, 1.5, 2.0],
        },
    },
    "vol_mom_binary_long": {
        "signals": {
            "mom": "ema(roc(close, param('mom_period', default=14)), param('smooth', default=6))",
            "avg_range": "sma(high - low, param('vol_window', default=14))",
            "norm_vol": "avg_range / (close + 0.000000000001)",
            "safe_vol": "when(norm_vol > 0.003, norm_vol, 0.003)",
            "score": "mom / safe_vol",
            "regime": "ema(close, param('regime', default=168))",
        },
        "size": "when((score > param('entry', default=1.0)) & (close > regime), 0.45, 0.0)",
        "grid": {
            "mom_period": [7, 14, 28, 42],
            "smooth": [3, 6, 12],
            "vol_window": [10, 14, 20],
            "entry": [0.5, 1.0, 1.5, 2.0],
            "regime": [72, 168, 240],
        },
    },
    "vol_mom_tiered_ls": {
        "signals": {
            "mom": "ema(roc(close, param('mom_period', default=14)), param('smooth', default=6))",
            "avg_range": "sma(high - low, param('vol_window', default=14))",
            "norm_vol": "avg_range / (close + 0.000000000001)",
            "safe_vol": "when(norm_vol > 0.003, norm_vol, 0.003)",
            "score": "mom / safe_vol",
        },
        "size": "when(score > (param('entry', default=1.0) * 2), 0.42, when(score > param('entry', default=1.0), 0.22, when(score < (0 - (param('entry', default=1.0) * 2)), -0.42, when(score < (0 - param('entry', default=1.0)), -0.22, 0.0))))",
        "grid": {
            "mom_period": [7, 14, 28, 42],
            "smooth": [3, 6, 12],
            "vol_window": [10, 14, 20],
            "entry": [0.5, 1.0, 1.5, 2.0],
        },
    },
    "vol_mom_trend_gate": {
        "signals": {
            "mom": "ema(roc(close, param('mom_period', default=14)), param('smooth', default=6))",
            "avg_range": "sma(high - low, param('vol_window', default=14))",
            "norm_vol": "avg_range / (close + 0.000000000001)",
            "safe_vol": "when(norm_vol > 0.003, norm_vol, 0.003)",
            "score": "mom / safe_vol",
            "regime": "ema(close, param('regime', default=168))",
        },
        "size": "when((score > param('entry', default=1.0)) & (close > regime), 0.40, when((score < (0 - param('entry', default=1.0))) & (close < regime), -0.40, 0.0))",
        "grid": {
            "mom_period": [7, 14, 28, 42],
            "smooth": [3, 6, 12],
            "vol_window": [10, 14, 20],
            "entry": [0.5, 1.0, 1.5, 2.0],
            "regime": [72, 168, 240],
        },
    },
}


def cfg(universe: dict[str, Any], start: str, end: str, interval: str) -> dict[str, Any]:
    return {
        "universe": universe["tickers"],
        "start": start,
        "end": end,
        "bar_interval": interval,
        "initial_capital": 10_000,
        "fees": "binance_perps",
        "slippage": {"kind": "fixed_bps", "bps": 2.0},
        "warmup_bars": 500,
        "execution": {
            "signal_delay": 1,
            "execution_price": "AtOpen",
            "max_position_pct": 0.15 if len(universe["tickers"]) <= 5 else 0.09,
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


def score(train: dict[str, Any], valid: dict[str, Any], probability: float) -> float:
    ts = metric(train, "sharpe", -99)
    vs = metric(valid, "sharpe", -99)
    vdd = abs(metric(valid, "max_drawdown", -1))
    turnover_penalty = max(0, trades(valid) - 1200) / 2500
    return min(ts, vs) + 0.20 * (ts + vs) + 0.25 * probability - max(0.0, vdd - 0.35) * 1.5 - turnover_penalty


def checkpoint(report: dict[str, Any]) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "generation5-checkpoint.json").write_text(json.dumps(report, indent=2, default=str))


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

    async with streamablehttp_client(campaign.MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            report["version"] = await campaign.call(session, "get_version", {})

            bases: dict[str, dict[str, Any]] = {}
            for name, spec in STRATEGIES.items():
                try:
                    payload = await campaign.call(session, "compose_strategy", {"name": name, "signals": spec["signals"], "size": spec["size"]})
                    bases[name] = payload["strategy_json"]
                except Exception as exc:
                    report["errors"].append({"stage": "compose", "strategy": name, "error": str(exc)})
            checkpoint(report)

            for name, spec in STRATEGIES.items():
                if name not in bases:
                    continue
                for universe in UNIVERSES:
                    for interval in INTERVALS:
                        hid = f"{name}:{universe['name']}:{interval}"
                        try:
                            sweep = await campaign.call(session, "run_sweep", {
                                "strategy_json": bases[name],
                                "param_grid": spec["grid"],
                                "config": cfg(universe, TRAIN[0], TRAIN[1], interval),
                                "lite": True,
                                "top_k": TOP_TRAIN,
                                "rank_metric": "sharpe",
                            })
                            correction = sweep.get("overfitting_correction") or {}
                            probability = campaign.num(correction.get("probability_edge_is_real"), 0.0)
                            train_rows = [row for row in sweep.get("top", []) if row.get("params")]
                            docs = [campaign.freeze(bases[name], row["params"], f"{name}_{idx}") for idx, row in enumerate(train_rows)]
                            valid_rows = await campaign.run_batch(session, docs, cfg(universe, VALID[0], VALID[1], interval))
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
                                "id": hid,
                                "strategy": name,
                                "universe": universe,
                                "interval": interval,
                                "trials": sweep.get("total"),
                                "overfitting_correction": correction,
                                "warnings": sweep.get("warnings", []),
                                "selected": candidates[0] if candidates else None,
                            })
                        except Exception as exc:
                            report["errors"].append({"stage": "hypothesis", "id": hid, "error": str(exc)})
                        checkpoint(report)

            ranked = [row for row in report["hypotheses"] if row.get("selected")]
            ranked.sort(key=lambda row: row["selected"]["score"], reverse=True)
            for hypothesis in ranked[:MAX_FINALISTS]:
                selected = hypothesis["selected"]
                try:
                    test = await campaign.call(session, "run_backtest", {
                        "strategy_json": selected["strategy_json"],
                        "config": cfg(hypothesis["universe"], TEST[0], TEST[1], hypothesis["interval"]),
                    })
                    base = bases[hypothesis["strategy"]]
                    variants = [("center", selected["params"], selected["strategy_json"])]
                    for key in ("entry", "mom_period", "smooth"):
                        if key not in selected["params"]:
                            continue
                        value = selected["params"][key]
                        for factor in (0.8, 1.2):
                            changed = dict(selected["params"])
                            changed[key] = campaign.perturb(value, factor)
                            variants.append((f"{key}_{factor}", changed, campaign.freeze(base, changed, f"{hypothesis['strategy']}_{key}_{factor}")))
                    neighbour_rows = await campaign.run_batch(session, [item[2] for item in variants], cfg(hypothesis["universe"], TEST[0], TEST[1], hypothesis["interval"]))
                    neighbour_sharpes = [metric(row, "sharpe") for row in neighbour_rows if math.isfinite(metric(row, "sharpe"))]
                    neighbour_median = median(neighbour_sharpes) if neighbour_sharpes else float("nan")
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
                            "config": cfg(hypothesis["universe"], TEST[0], TEST[1], hypothesis["interval"]),
                            "mc_config": {"n_paths": 3000, "method": {"type": "block_bootstrap", "block_size": 24}, "rng_seed": 42},
                        })
                    report["finalists"].append({
                        "id": hypothesis["id"],
                        "strategy": hypothesis["strategy"],
                        "universe": hypothesis["universe"],
                        "interval": hypothesis["interval"],
                        "params": selected["params"],
                        "train": selected["train"],
                        "validation": selected["validation"],
                        "test": test,
                        "neighbour_median_sharpe": neighbour_median,
                        "neighbours": [{"label": label, "params": params, "result": result} for (label, params, _), result in zip(variants, neighbour_rows, strict=False)],
                        "overfitting_correction": hypothesis["overfitting_correction"],
                        "monte_carlo": monte_carlo,
                        "passes": passed,
                        "strategy_json": selected["strategy_json"],
                    })
                except Exception as exc:
                    report["errors"].append({"stage": "final", "id": hypothesis["id"], "error": str(exc)})
                checkpoint(report)

    report["finalists"].sort(key=lambda row: metric(row.get("test") or {}, "sharpe", -99), reverse=True)
    report["passes"] = [row for row in report["finalists"] if row.get("passes")]
    report["best"] = report["finalists"][0] if report["finalists"] else None
    (OUT / "generation5.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Crypto alpha generation 5",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Thresholded volatility-adjusted momentum. Train 2021-2023, validation 2024, frozen OOS 2025-2026. Binance-perps fees, 2 bps slippage and one-bar delay; spot bars proxy prices and funding is unavailable.",
        "",
        "| # | Strategy | Universe | TF | Params | Train S | Valid S | OOS S | Alpha | Return | DD | Trades | Neighbour S | DSR | Pass |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for idx, row in enumerate(report["finalists"], 1):
        test = row["test"]
        probability = campaign.num((row.get("overfitting_correction") or {}).get("probability_edge_is_real"), 0.0)
        lines.append(
            f"| {idx} | {row['strategy']} | {row['universe']['name']} | {row['interval']} | `{json.dumps(row['params'], sort_keys=True)}` | "
            f"{metric(row['train'],'sharpe'):.2f} | {metric(row['validation'],'sharpe'):.2f} | {metric(test,'sharpe'):.2f} | "
            f"{metric(test,'alpha'):.3f} | {metric(test,'total_return'):.2%} | {metric(test,'max_drawdown'):.2%} | "
            f"{trades(test)} | {row['neighbour_median_sharpe']:.2f} | {probability:.0%} | {'YES' if row['passes'] else 'NO'} |"
        )
    if report["passes"]:
        winner = report["passes"][0]
        lines += [
            "", "## Validated candidate", "",
            f"- Strategy: `{winner['strategy']}`",
            f"- Universe: `{winner['universe']['name']}`",
            f"- Timeframe: `{winner['interval']}`",
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
    (OUT / "GENERATION5.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
