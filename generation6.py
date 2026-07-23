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
INTERVAL = "4h"
TOP_TRAIN = 8
MAX_FINALISTS = 12
UNIVERSE = {"name": "MAJORS5", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]}

STRATEGIES: dict[str, dict[str, Any]] = {
    "ema_asymmetric_size": {
        "signals": {
            "fast": "ema(close, param('fast', default=8))",
            "slow": "ema(close, param('slow', default=168))",
        },
        "size": "when(fast > slow, param('long_size', default=0.10), 0 - param('short_size', default=0.15))",
        "grid": {
            "fast": [6, 8, 10],
            "slow": [144, 168, 192],
            "long_size": [0.05, 0.10, 0.15],
            "short_size": [0.05, 0.10, 0.15],
        },
    },
    "ema_volatility_bucket": {
        "signals": {
            "fast": "ema(close, 8)",
            "slow": "ema(close, 168)",
            "raw_vol": "sma(high - low, 14) / (close + 0.000000000001)",
            "vol_z": "raw_vol.zscore(param('vol_window', default=72))",
        },
        "size": "when(fast > slow, when(vol_z > param('risk_z', default=1.0), param('risk_size', default=0.05), 0.15), when(fast < slow, when(vol_z > param('risk_z', default=1.0), 0 - param('risk_size', default=0.05), -0.15), 0.0))",
        "grid": {
            "vol_window": [48, 72, 120],
            "risk_z": [0.5, 1.0, 1.5, 2.0],
            "risk_size": [0.0, 0.05, 0.10],
        },
    },
    "ema_spread_quality": {
        "signals": {
            "fast": "ema(close, param('fast', default=8))",
            "slow": "ema(close, param('slow', default=168))",
            "range": "atr(14)",
            "quality": "abs_val(fast - slow) / (range + 0.000000000001)",
        },
        "size": "when((fast > slow) & (quality > param('quality_min', default=0.5)), 0.15, when((fast < slow) & (quality > param('quality_min', default=0.5)), -0.15, 0.0))",
        "grid": {
            "fast": [6, 8, 10],
            "slow": [144, 168, 192],
            "quality_min": [0.25, 0.5, 1.0, 1.5],
        },
    },
    "dual_horizon_balanced": {
        "signals": {
            "long_fast": "ema(close, param('long_fast', default=8))",
            "long_slow": "ema(close, param('long_slow', default=120))",
            "short_fast": "ema(close, param('short_fast', default=8))",
            "short_slow": "ema(close, param('short_slow', default=168))",
        },
        "size": "when(long_fast > long_slow, 0.15, when(short_fast < short_slow, -0.15, 0.0))",
        "grid": {
            "long_fast": [6, 8, 12],
            "long_slow": [72, 120, 168],
            "short_fast": [6, 8, 12],
            "short_slow": [120, 168, 240],
        },
    },
    "dual_horizon_short_bias": {
        "signals": {
            "long_fast": "ema(close, param('long_fast', default=8))",
            "long_slow": "ema(close, param('long_slow', default=120))",
            "short_fast": "ema(close, param('short_fast', default=8))",
            "short_slow": "ema(close, param('short_slow', default=168))",
        },
        "size": "when(long_fast > long_slow, 0.10, when(short_fast < short_slow, -0.15, 0.0))",
        "grid": {
            "long_fast": [6, 8, 12],
            "long_slow": [72, 120, 168],
            "short_fast": [6, 8, 12],
            "short_slow": [120, 168, 240],
        },
    },
    "dual_horizon_long_bias": {
        "signals": {
            "long_fast": "ema(close, param('long_fast', default=8))",
            "long_slow": "ema(close, param('long_slow', default=120))",
            "short_fast": "ema(close, param('short_fast', default=8))",
            "short_slow": "ema(close, param('short_slow', default=168))",
        },
        "size": "when(long_fast > long_slow, 0.15, when(short_fast < short_slow, -0.10, 0.0))",
        "grid": {
            "long_fast": [6, 8, 12],
            "long_slow": [72, 120, 168],
            "short_fast": [6, 8, 12],
            "short_slow": [120, 168, 240],
        },
    },
}


def cfg(start: str, end: str) -> dict[str, Any]:
    return {
        "universe": UNIVERSE["tickers"],
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


def score(train: dict[str, Any], valid: dict[str, Any], probability: float) -> float:
    ts = metric(train, "sharpe", -99)
    vs = metric(valid, "sharpe", -99)
    vdd = abs(metric(valid, "max_drawdown", -1))
    return min(ts, vs) + 0.20 * (ts + vs) + 0.25 * probability - max(0.0, vdd - 0.35) * 1.5


def checkpoint(report: dict[str, Any]) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "generation6-checkpoint.json").write_text(json.dumps(report, indent=2, default=str))


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
                try:
                    sweep = await campaign.call(session, "run_sweep", {
                        "strategy_json": bases[name],
                        "param_grid": spec["grid"],
                        "config": cfg(TRAIN[0], TRAIN[1]),
                        "lite": True,
                        "top_k": TOP_TRAIN,
                        "rank_metric": "sharpe",
                    })
                    correction = sweep.get("overfitting_correction") or {}
                    probability = campaign.num(correction.get("probability_edge_is_real"), 0.0)
                    train_rows = [row for row in sweep.get("top", []) if row.get("params")]
                    docs = [campaign.freeze(bases[name], row["params"], f"{name}_{idx}") for idx, row in enumerate(train_rows)]
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
                        "id": name,
                        "strategy": name,
                        "trials": sweep.get("total"),
                        "overfitting_correction": correction,
                        "warnings": sweep.get("warnings", []),
                        "selected": candidates[0] if candidates else None,
                    })
                except Exception as exc:
                    report["errors"].append({"stage": "hypothesis", "id": name, "error": str(exc)})
                checkpoint(report)

            ranked = [row for row in report["hypotheses"] if row.get("selected")]
            ranked.sort(key=lambda row: row["selected"]["score"], reverse=True)
            for hypothesis in ranked[:MAX_FINALISTS]:
                selected = hypothesis["selected"]
                name = hypothesis["strategy"]
                try:
                    test = await campaign.call(session, "run_backtest", {"strategy_json": selected["strategy_json"], "config": cfg(TEST[0], TEST[1])})
                    base = bases[name]
                    variants = [("center", selected["params"], selected["strategy_json"])]
                    keys = list(selected["params"])
                    for key in keys[:4]:
                        value = selected["params"][key]
                        for factor in (0.8, 1.2):
                            changed = dict(selected["params"])
                            changed[key] = campaign.perturb(value, factor)
                            variants.append((f"{key}_{factor}", changed, campaign.freeze(base, changed, f"{name}_{key}_{factor}")))
                    neighbour_rows = await campaign.run_batch(session, [item[2] for item in variants], cfg(TEST[0], TEST[1]))
                    sharpes = [metric(row, "sharpe") for row in neighbour_rows if math.isfinite(metric(row, "sharpe"))]
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
                        "neighbours": [{"label": label, "params": params, "result": result} for (label, params, _), result in zip(variants, neighbour_rows, strict=False)],
                        "overfitting_correction": hypothesis["overfitting_correction"],
                        "monte_carlo": monte_carlo,
                        "passes": passed,
                        "strategy_json": selected["strategy_json"],
                    })
                except Exception as exc:
                    report["errors"].append({"stage": "final", "id": name, "error": str(exc)})
                checkpoint(report)

    report["finalists"].sort(key=lambda row: metric(row.get("test") or {}, "sharpe", -99), reverse=True)
    report["passes"] = [row for row in report["finalists"] if row.get("passes")]
    report["best"] = report["finalists"][0] if report["finalists"] else None
    (OUT / "generation6.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Crypto alpha generation 6",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Asymmetric EMA horizons/sizing and volatility risk overlays. MAJORS5, 4h. Train 2021-2023, validation 2024, frozen OOS 2025-2026. Binance-perps fees, 2 bps slippage, one-bar delay; spot bars proxy prices and funding is unavailable.",
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
    (OUT / "GENERATION6.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
