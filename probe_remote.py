from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_URL = "https://mcp.manifoldbt.com/mcp"
OUT = Path("results")


def unpack(result: Any) -> Any:
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    text = "\n".join(getattr(x, "text", "") for x in getattr(result, "content", []))
    try:
        return json.loads(text)
    except Exception:
        return {"text": text, "isError": getattr(result, "isError", False)}


async def raw_call(session: ClientSession, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await session.call_tool(name, arguments)
    return {"ok": not bool(getattr(result, "isError", False)), "payload": unpack(result)}


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    report: dict[str, Any] = {}
    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            report["symbols"] = await raw_call(session, "list_symbols", {})

            probes = {
                "literal": {
                    "signals": {"fast": "ema(close, 12)", "slow": "ema(close, 48)"},
                    "size": "when(fast > slow, 0.35, when(fast < slow, -0.35, 0.0))",
                    "name": "literal_ema",
                },
                "param_default": {
                    "signals": {"fast": "ema(close, param('fast', default=12))", "slow": "ema(close, param('slow', default=48))"},
                    "size": "when(fast > slow, 0.35, when(fast < slow, -0.35, 0.0))",
                    "name": "param_default",
                },
                "param_dict_identifier": {
                    "signals": {"fast": "ema(close, fast)", "slow": "ema(close, slow)"},
                    "size": "when(fast > slow, 0.35, when(fast < slow, -0.35, 0.0))",
                    "name": "param_dict_identifier",
                    "parameters": {"fast": 12, "slow": 48},
                },
                "param_call_dict": {
                    "signals": {"fast": "ema(close, param('fast'))", "slow": "ema(close, param('slow'))"},
                    "size": "when(fast > slow, 0.35, when(fast < slow, -0.35, 0.0))",
                    "name": "param_call_dict",
                    "parameters": {"fast": 12, "slow": 48},
                },
            }
            report["compose"] = {}
            for name, args in probes.items():
                report["compose"][name] = await raw_call(session, "compose_strategy", args)

            literal = report["compose"]["literal"]
            if literal["ok"]:
                strategy_json = literal["payload"].get("strategy_json") or literal["payload"].get("result", {}).get("strategy_json")
                report["literal_strategy_json"] = strategy_json
                report["backtest"] = await raw_call(
                    session,
                    "run_backtest",
                    {
                        "strategy_json": strategy_json,
                        "config": {
                            "universe": "BTCUSDT",
                            "start": "2022-01-01",
                            "end": "2024-01-01",
                            "bar_interval": "4h",
                            "initial_capital": 10000,
                            "fees": "binance_perps",
                            "slippage": {"kind": "fixed_bps", "bps": 2.0},
                            "warmup_bars": 100,
                            "execution": {"signal_delay": 1, "execution_price": "AtOpen", "allow_short": True, "max_position_pct": 0.5},
                        },
                    },
                )

            for key in ("param_default", "param_dict_identifier", "param_call_dict"):
                item = report["compose"][key]
                if not item["ok"]:
                    continue
                strategy_json = item["payload"].get("strategy_json") or item["payload"].get("result", {}).get("strategy_json")
                report[f"sweep_{key}"] = await raw_call(
                    session,
                    "run_sweep",
                    {
                        "strategy_json": strategy_json,
                        "param_grid": {"fast": [8, 12, 18], "slow": [36, 48, 72]},
                        "config": {
                            "universe": "BTCUSDT",
                            "start": "2022-01-01",
                            "end": "2024-01-01",
                            "bar_interval": "4h",
                            "initial_capital": 10000,
                            "fees": "binance_perps",
                            "slippage": {"kind": "fixed_bps", "bps": 2.0},
                            "warmup_bars": 100,
                            "execution": {"signal_delay": 1, "execution_price": "AtOpen", "allow_short": True, "max_position_pct": 0.5},
                        },
                        "top_k": 3,
                        "rank_metric": "sharpe",
                    },
                )

    (OUT / "probe.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
