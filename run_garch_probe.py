from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any

import manifold_research as engine

PERIOD = ("2021-01-01", "2022-12-31")
INTERVAL = "4h"


def variants() -> list[dict[str, Any]]:
    base_signals = {"shock": "roc(close, 3)"}
    return [
        {
            "name": "baseline_shock_fade",
            "signals": base_signals,
            "size": "when(shock < -0.02, 0.08, when(shock > 0.02, -0.08, 0.0))",
        },
        {
            "name": "garch_unused_shock_fade",
            "signals": {
                **base_signals,
                "conditional_vol": "garch(close, 0.000001, 0.10, 0.85)",
            },
            "size": "when(shock < -0.02, 0.08, when(shock > 0.02, -0.08, 0.0))",
        },
        {
            "name": "garch_ratio_unused_shock_fade",
            "signals": {
                **base_signals,
                "conditional_vol": "garch(close, 0.000001, 0.10, 0.85)",
                "vol_anchor": "ema(conditional_vol, 72)",
                "vol_ratio": "conditional_vol / (vol_anchor + 0.000000001)",
            },
            "size": "when(shock < -0.02, 0.08, when(shock > 0.02, -0.08, 0.0))",
        },
        {
            "name": "garch_regime_sized_shock_fade",
            "signals": {
                **base_signals,
                "conditional_vol": "garch(close, 0.000001, 0.10, 0.85)",
                "vol_anchor": "ema(conditional_vol, 72)",
                "vol_ratio": "conditional_vol / (vol_anchor + 0.000000001)",
                "risk_multiplier": "when(vol_ratio > 1.25, 0.50, 1.0)",
                "risk_size": "0.08 * risk_multiplier",
            },
            "size": "when(shock < -0.02, risk_size, when(shock > 0.02, 0.0 - risk_size, 0.0))",
        },
        {
            "name": "garch_returns_regime_sized_shock_fade",
            "signals": {
                **base_signals,
                "ret1": "roc(close, 1)",
                "conditional_vol": "garch(ret1, 0.000001, 0.10, 0.85)",
                "vol_anchor": "ema(conditional_vol, 72)",
                "vol_ratio": "conditional_vol / (vol_anchor + 0.000000001)",
                "risk_multiplier": "when(vol_ratio > 1.25, 0.50, 1.0)",
                "risk_size": "0.08 * risk_multiplier",
            },
            "size": "when(shock < -0.02, risk_size, when(shock > 0.02, 0.0 - risk_size, 0.0))",
        },
    ]


async def run(output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"period": PERIOD, "interval": INTERVAL, "variants": []}
    ClientSession, streamable = engine.mcp_imports()
    async with streamable(engine.MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            report["version"] = await engine.call(session, "get_version", {})
            for item in variants():
                row: dict[str, Any] = {"name": item["name"]}
                try:
                    composed = await engine.call(session, "compose_strategy", item)
                    strategy = composed.get("strategy_json") or (composed.get("result") or {}).get("strategy_json")
                    row["compile"] = await engine.call(session, "validate_strategy", {"strategy_json": strategy})
                    row["result"] = await engine.call(session, "run_backtest", {
                        "strategy_json": strategy,
                        "config": engine.cfg(*PERIOD, INTERVAL),
                        "include_equity_curve": False,
                        "include_daily_returns": False,
                    })
                except Exception as exc:
                    row["error"] = str(exc)
                report["variants"].append(row)
    (output / "raw.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# GARCH execution probe", "", "| Variant | Compile | Trades | Sharpe | Error |", "|---|---:|---:|---:|---|"]
    for row in report["variants"]:
        result = row.get("result") or {}
        lines.append(
            f"| `{row['name']}` | `{bool(row.get('compile'))}` | {engine.trades(result)} | "
            f"{engine.metric(result, 'sharpe', 0.0):.3f} | {row.get('error', '')} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    output = Path("results/garch_probe")
    output.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(run(output))
        return 0
    except Exception:
        (output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
