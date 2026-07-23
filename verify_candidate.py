from __future__ import annotations

import asyncio
import json
import math
import traceback
from pathlib import Path
from statistics import median

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import campaign
import generation7

OUT = Path("results")
PARAMS = {
    "vol_window": 72,
    "risk_z": 1.0,
    "risk_short": 0.0,
    "risk_long": 0.15,
}


def metric(row, key, default=float("nan")):
    return campaign.metric(row, key, default)


def trades(row):
    return campaign.trades(row)


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    base = generation7.strategies()["ema_return_vol_overlay"]
    strategy = generation7.freeze(base, PARAMS, "ema_return_vol_overlay_verified")
    report = {"params": PARAMS, "errors": []}

    async with streamablehttp_client(campaign.MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            report["version"] = await campaign.call(session, "get_version", {})
            report["validation_plan"] = {
                "train": generation7.TRAIN,
                "validation": generation7.VALID,
                "oos": generation7.TEST,
                "subperiods": [["2025-01-01", "2025-12-31"], ["2026-01-01", "2026-07-01"]],
            }
            report["compile"] = await campaign.call(session, "validate_strategy", {"strategy_json": json.dumps(strategy)})
            report["train"] = await campaign.call(session, "run_backtest", {"strategy_json": strategy, "config": generation7.cfg(*generation7.TRAIN)})
            report["validation"] = await campaign.call(session, "run_backtest", {"strategy_json": strategy, "config": generation7.cfg(*generation7.VALID)})
            report["test"] = await campaign.call(session, "run_backtest", {"strategy_json": strategy, "config": generation7.cfg(*generation7.TEST)})
            report["test_2025"] = await campaign.call(session, "run_backtest", {"strategy_json": strategy, "config": generation7.cfg("2025-01-01", "2025-12-31")})
            report["test_2026_ytd"] = await campaign.call(session, "run_backtest", {"strategy_json": strategy, "config": generation7.cfg("2026-01-01", "2026-07-01")})

            neighbour_params = [
                {**PARAMS, "vol_window": 58},
                {**PARAMS, "vol_window": 86},
                {**PARAMS, "risk_z": 0.8},
                {**PARAMS, "risk_z": 1.2},
                {**PARAMS, "risk_short": 0.025},
                {**PARAMS, "risk_short": 0.05},
                {**PARAMS, "risk_long": 0.12},
            ]
            neighbour_docs = [generation7.freeze(base, params, f"neighbour_{index}") for index, params in enumerate(neighbour_params)]
            neighbour_rows = await campaign.run_batch(session, neighbour_docs, generation7.cfg(*generation7.TEST))
            report["neighbours"] = [
                {"params": params, "result": result}
                for params, result in zip(neighbour_params, neighbour_rows, strict=False)
            ]
            sharpes = [metric(row, "sharpe") for row in neighbour_rows if math.isfinite(metric(row, "sharpe"))]
            report["neighbour_median_sharpe"] = median(sharpes) if sharpes else float("nan")

            report["monte_carlo"] = await campaign.call(session, "run_monte_carlo", {
                "strategy_json": strategy,
                "config": generation7.cfg(*generation7.TEST),
                "mc_config": {
                    "n_paths": 1000,
                    "method": {"type": "block_bootstrap", "block_size": 24},
                    "rng_seed": 42,
                    "confidence_levels": [0.90, 0.95, 0.99],
                    "dd_thresholds": [-0.20, -0.30, -0.40],
                },
            })

    report["passes_sharpe_target"] = bool(
        metric(report["test"], "sharpe", -99) >= 1.5
        and metric(report["test"], "alpha", -99) > 0
        and trades(report["test"]) >= 30
        and metric(report["train"], "sharpe", -99) >= 0.70
        and metric(report["validation"], "sharpe", -99) >= 0.70
        and report["neighbour_median_sharpe"] >= 1.0
    )
    (OUT / "verified-candidate.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Verified crypto candidate",
        "",
        "EMA 8/168 on BTC, ETH, SOL, BNB and XRP (4h), with short exposure reduced to zero when 14-bar return volatility is more than 1 standard deviation above its 72-bar baseline.",
        "",
        f"- Parameters: `{json.dumps(PARAMS, sort_keys=True)}`",
        f"- Train Sharpe: `{metric(report['train'], 'sharpe'):.3f}`",
        f"- Validation Sharpe: `{metric(report['validation'], 'sharpe'):.3f}`",
        f"- OOS Sharpe: `{metric(report['test'], 'sharpe'):.3f}`",
        f"- OOS alpha: `{metric(report['test'], 'alpha'):.4f}`",
        f"- OOS return: `{metric(report['test'], 'total_return'):.2%}`",
        f"- OOS max drawdown: `{metric(report['test'], 'max_drawdown'):.2%}`",
        f"- OOS trades: `{trades(report['test'])}`",
        f"- 2025 Sharpe: `{metric(report['test_2025'], 'sharpe'):.3f}`",
        f"- 2026 YTD Sharpe: `{metric(report['test_2026_ytd'], 'sharpe'):.3f}`",
        f"- Neighbour median Sharpe: `{report['neighbour_median_sharpe']:.3f}`",
        f"- Passes Sharpe >= 1.5 gate: `{'YES' if report['passes_sharpe_target'] else 'NO'}`",
        "",
        "The Monte Carlo payload is stored in `verified-candidate.json`.",
    ]
    (OUT / "VERIFIED_CANDIDATE.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
