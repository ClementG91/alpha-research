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
INTERVALS = ("1h", "4h")
UNIVERSE = {"name": "MAJORS5", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]}


def cfg(start: str, end: str, interval: str) -> dict[str, Any]:
    return {
        "universe": UNIVERSE["tickers"],
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
            "max_position_pct": 0.18,
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


def score(train: dict[str, Any], valid: dict[str, Any]) -> float:
    ts = metric(train, "sharpe", -99)
    vs = metric(valid, "sharpe", -99)
    vdd = abs(metric(valid, "max_drawdown", -1))
    turnover_penalty = max(0, trades(valid) - 900) / 2000
    return min(ts, vs) + 0.20 * (ts + vs) - max(0.0, vdd - 0.35) * 1.5 - turnover_penalty


def checkpoint(report: dict[str, Any]) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "generation4-checkpoint.json").write_text(json.dumps(report, indent=2, default=str))


SPECS: dict[str, dict[str, Any]] = {
    "vol_mom_long_14": {
        "family": "vol_mom",
        "signals": {
            "mom": "ema(roc(close, 14), 6)",
            "avg_range": "sma(high - low, 14)",
            "norm_vol": "avg_range / (close + 0.000000000001)",
            "safe_vol": "when(norm_vol > 0.005, norm_vol, 0.005)",
        },
        "size": "when(mom > 0, (mom / safe_vol) * 0.025, 0.0)",
    },
    "vol_mom_long_28": {
        "family": "vol_mom",
        "signals": {
            "mom": "ema(roc(close, 28), 8)",
            "avg_range": "sma(high - low, 20)",
            "norm_vol": "avg_range / (close + 0.000000000001)",
            "safe_vol": "when(norm_vol > 0.005, norm_vol, 0.005)",
        },
        "size": "when(mom > 0, (mom / safe_vol) * 0.020, 0.0)",
    },
    "vol_mom_ls_14": {
        "family": "vol_mom",
        "signals": {
            "mom": "ema(roc(close, 14), 6)",
            "avg_range": "sma(high - low, 14)",
            "norm_vol": "avg_range / (close + 0.000000000001)",
            "safe_vol": "when(norm_vol > 0.005, norm_vol, 0.005)",
        },
        "size": "(mom / safe_vol) * 0.020",
    },
    "linreg_quality_48": {
        "family": "linreg",
        "signals": {"slope": "linreg_slope(close, 48)", "quality": "linreg_r2(close, 48)"},
        "size": "when((quality > 0.35) & (slope > 0), 0.38, when((quality > 0.35) & (slope < 0), -0.38, 0.0))",
    },
    "linreg_quality_96": {
        "family": "linreg",
        "signals": {"slope": "linreg_slope(close, 96)", "quality": "linreg_r2(close, 96)"},
        "size": "when((quality > 0.45) & (slope > 0), 0.38, when((quality > 0.45) & (slope < 0), -0.38, 0.0))",
    },
    "kalman_cross_fast": {
        "family": "kalman",
        "signals": {"smooth": "kalman(close, 0.00001, 0.01)", "fast": "ema(smooth, 8)", "slow": "ema(smooth, 72)"},
        "size": "when(crossover(fast, slow), 0.40, when(crossunder(fast, slow), -0.40, hold()))",
    },
    "kalman_cross_slow": {
        "family": "kalman",
        "signals": {"smooth": "kalman(close, 0.00001, 0.02)", "fast": "ema(smooth, 12)", "slow": "ema(smooth, 168)"},
        "size": "when(crossover(fast, slow), 0.40, when(crossunder(fast, slow), -0.40, hold()))",
    },
    "supertrend_10_3": {
        "family": "supertrend",
        "signals": {"st": "supertrend(10, 3.0)"},
        "size": "when(close > st, 0.38, when(close < st, -0.38, 0.0))",
    },
    "supertrend_20_4": {
        "family": "supertrend",
        "signals": {"st": "supertrend(20, 4.0)"},
        "size": "when(close > st, 0.38, when(close < st, -0.38, 0.0))",
    },
    "adx_ema_14": {
        "family": "adx",
        "signals": {"fast": "ema(close, 12)", "slow": "ema(close, 120)", "strength": "adx(14)"},
        "size": "when((strength > 22) & (fast > slow), 0.40, when((strength > 22) & (fast < slow), -0.40, 0.0))",
    },
    "adx_ema_24": {
        "family": "adx",
        "signals": {"fast": "ema(close, 24)", "slow": "ema(close, 168)", "strength": "adx(24)"},
        "size": "when((strength > 25) & (fast > slow), 0.40, when((strength > 25) & (fast < slow), -0.40, 0.0))",
    },
    "mfi_trend": {
        "family": "flow",
        "signals": {"fast": "ema(close, 12)", "slow": "ema(close, 120)", "flow": "mfi(14)"},
        "size": "when((fast > slow) & (flow > 55), 0.38, when((fast < slow) & (flow < 45), -0.38, 0.0))",
    },
    "asia_session_trend": {
        "family": "session",
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 168)", "h": "hour()"},
        "size": "when((h >= 0) & (h < 8) & (fast > slow), 0.40, when((h >= 0) & (h < 8) & (fast < slow), -0.40, 0.0))",
    },
    "europe_session_trend": {
        "family": "session",
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 168)", "h": "hour()"},
        "size": "when((h >= 8) & (h < 16) & (fast > slow), 0.40, when((h >= 8) & (h < 16) & (fast < slow), -0.40, 0.0))",
    },
    "us_session_trend": {
        "family": "session",
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 168)", "h": "hour()"},
        "size": "when((h >= 16) & (h < 24) & (fast > slow), 0.40, when((h >= 16) & (h < 24) & (fast < slow), -0.40, 0.0))",
    },
    "weekday_trend": {
        "family": "calendar",
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 168)", "dow": "day_of_week()"},
        "size": "when((dow < 5) & (fast > slow), 0.40, when((dow < 5) & (fast < slow), -0.40, 0.0))",
    },
    "weekend_trend": {
        "family": "calendar",
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 168)", "dow": "day_of_week()"},
        "size": "when((dow >= 5) & (fast > slow), 0.40, when((dow >= 5) & (fast < slow), -0.40, 0.0))",
    },
    "funding_window_trend": {
        "family": "funding_time",
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 168)", "h": "hour()"},
        "size": "when(((h < 2) | ((h >= 8) & (h < 10)) | ((h >= 16) & (h < 18))) & (fast > slow), 0.40, when(((h < 2) | ((h >= 8) & (h < 10)) | ((h >= 16) & (h < 18))) & (fast < slow), -0.40, 0.0))",
    },
    "turn_month_trend": {
        "family": "calendar",
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 168)", "dom": "day_of_month()"},
        "size": "when(((dom <= 3) | (dom >= 27)) & (fast > slow), 0.40, when(((dom <= 3) | (dom >= 27)) & (fast < slow), -0.40, 0.0))",
    },
}


async def run_batch(session: ClientSession, docs: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    payload = await campaign.call(session, "run_batch", {"strategies": [{"strategy_json": doc} for doc in docs], "config": config, "lite": False, "max_parallelism": 0})
    return campaign.as_list(payload)


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train": TRAIN,
        "validation": VALID,
        "test": TEST,
        "compiled": {},
        "candidates": [],
        "finalists": [],
        "errors": [],
    }

    async with streamablehttp_client(campaign.MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            report["version"] = await campaign.call(session, "get_version", {})

            docs: dict[str, dict[str, Any]] = {}
            for name, spec in SPECS.items():
                try:
                    payload = await campaign.call(session, "compose_strategy", {"name": name, "signals": spec["signals"], "size": spec["size"]})
                    docs[name] = payload["strategy_json"]
                    report["compiled"][name] = True
                except Exception as exc:
                    report["compiled"][name] = False
                    report["errors"].append({"stage": "compose", "name": name, "error": str(exc)})
            checkpoint(report)

            names = list(docs)
            documents = [docs[name] for name in names]
            for interval in INTERVALS:
                try:
                    train_rows = await run_batch(session, documents, cfg(TRAIN[0], TRAIN[1], interval))
                    valid_rows = await run_batch(session, documents, cfg(VALID[0], VALID[1], interval))
                    for name, train, valid in zip(names, train_rows, valid_rows, strict=False):
                        report["candidates"].append({
                            "id": f"{name}:{interval}",
                            "name": name,
                            "family": SPECS[name]["family"],
                            "interval": interval,
                            "strategy_json": docs[name],
                            "train": train,
                            "validation": valid,
                            "score": score(train, valid),
                        })
                except Exception as exc:
                    report["errors"].append({"stage": "batch", "interval": interval, "error": str(exc)})
                checkpoint(report)

            ranked = sorted(report["candidates"], key=lambda row: row["score"], reverse=True)
            selected = ranked[:14]
            for interval in INTERVALS:
                group = [row for row in selected if row["interval"] == interval]
                if not group:
                    continue
                try:
                    test_rows = await run_batch(session, [row["strategy_json"] for row in group], cfg(TEST[0], TEST[1], interval))
                    for row, test in zip(group, test_rows, strict=False):
                        report["finalists"].append({**row, "test": test, "monte_carlo": None, "passes": False})
                except Exception as exc:
                    report["errors"].append({"stage": "oos_batch", "interval": interval, "error": str(exc)})
                checkpoint(report)

            family_sharpes: dict[str, list[float]] = {}
            for row in report["finalists"]:
                s = metric(row["test"], "sharpe")
                if math.isfinite(s):
                    family_sharpes.setdefault(row["family"], []).append(s)
            for row in report["finalists"]:
                values = family_sharpes.get(row["family"], [])
                row["family_median_oos_sharpe"] = median(values) if values else float("nan")
                row["passes"] = bool(
                    metric(row["test"], "sharpe", -99) >= 1.5
                    and metric(row["test"], "alpha", -99) > 0
                    and trades(row["test"]) >= 30
                    and metric(row["train"], "sharpe", -99) >= 0.70
                    and metric(row["validation"], "sharpe", -99) >= 0.70
                    and row["family_median_oos_sharpe"] >= 1.0
                )
                if metric(row["test"], "sharpe", -99) >= 1.30:
                    try:
                        row["monte_carlo"] = await campaign.call(session, "run_monte_carlo", {
                            "strategy_json": row["strategy_json"],
                            "config": cfg(TEST[0], TEST[1], row["interval"]),
                            "mc_config": {"n_paths": 2000, "method": {"type": "block_bootstrap", "block_size": 24}, "rng_seed": 42},
                        })
                    except Exception as exc:
                        report["errors"].append({"stage": "monte_carlo", "id": row["id"], "error": str(exc)})
                checkpoint(report)

    report["finalists"].sort(key=lambda row: metric(row.get("test") or {}, "sharpe", -99), reverse=True)
    report["passes"] = [row for row in report["finalists"] if row.get("passes")]
    report["best"] = report["finalists"][0] if report["finalists"] else None
    (OUT / "generation4.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Crypto alpha generation 4",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Crypto-specific anomaly and volatility campaign. Train 2021-2023, validation 2024, frozen OOS 2025-2026. Binance-perps fees, 2 bps slippage and one-bar delay; spot bars proxy prices and funding is unavailable.",
        "",
        "| # | Candidate | Family | TF | Train S | Valid S | OOS S | Alpha | Return | DD | Trades | Family median | Pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for idx, row in enumerate(report["finalists"], 1):
        test = row["test"]
        lines.append(
            f"| {idx} | {row['name']} | {row['family']} | {row['interval']} | "
            f"{metric(row['train'],'sharpe'):.2f} | {metric(row['validation'],'sharpe'):.2f} | "
            f"{metric(test,'sharpe'):.2f} | {metric(test,'alpha'):.3f} | "
            f"{metric(test,'total_return'):.2%} | {metric(test,'max_drawdown'):.2%} | "
            f"{trades(test)} | {row['family_median_oos_sharpe']:.2f} | {'YES' if row['passes'] else 'NO'} |"
        )
    if report["passes"]:
        winner = report["passes"][0]
        lines += [
            "", "## Validated candidate", "",
            f"- Strategy: `{winner['name']}`",
            f"- Family: `{winner['family']}`",
            f"- Timeframe: `{winner['interval']}`",
            f"- OOS Sharpe: `{metric(winner['test'],'sharpe'):.3f}`",
            f"- OOS alpha: `{metric(winner['test'],'alpha'):.4f}`",
            f"- Return: `{metric(winner['test'],'total_return'):.2%}`",
            f"- Max drawdown: `{metric(winner['test'],'max_drawdown'):.2%}`",
            f"- Trades: `{trades(winner['test'])}`",
            f"- Family median OOS Sharpe: `{winner['family_median_oos_sharpe']:.3f}`",
        ]
    else:
        lines += ["", "## Conclusion", "", "No candidate reached and retained OOS Sharpe >= 1.5 through all gates."]
    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{json.dumps(error, sort_keys=True)}`" for error in report["errors"]]
    (OUT / "GENERATION4.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
