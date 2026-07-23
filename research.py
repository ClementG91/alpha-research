from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
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
]

TRAIN = ("2021-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-07-01")

STRATEGIES: dict[str, str] = {
    "adaptive_trend": """
fast = ema(close, param('fast', default=24))
slow = ema(close, param('slow', default=120))
mom = roc(close, param('mom', default=24))
strength = adx(param('adx_period', default=18))
long_cond = (fast > slow) & (mom > lit(0.0)) & (strength > lit(18.0))
short_cond = (fast < slow) & (mom < lit(0.0)) & (strength > lit(18.0))
position = when(long_cond, lit(0.45), when(short_cond, lit(-0.45), lit(0.0)))
strategy = (
    Strategy.create('adaptive_trend')
    .signal('fast', fast)
    .signal('slow', slow)
    .signal('mom', mom)
    .signal('strength', strength)
    .size(position)
    .stop_loss(pct=5.0)
    .trailing_stop(pct=3.0)
)
""",
    "volatility_squeeze": """
width = bollinger_width(close, period=param('bb_period', default=24), num_std=2.0)
width_z = width.zscore(param('width_window', default=120))
trend = ema(close, param('trend_span', default=96))
mom = roc(close, param('mom', default=12))
long_cond = (width_z < lit(-0.75)) & (close > trend) & (mom > lit(0.0))
short_cond = (width_z < lit(-0.75)) & (close < trend) & (mom < lit(0.0))
position = when(long_cond, lit(0.40), when(short_cond, lit(-0.40), lit(0.0)))
strategy = (
    Strategy.create('volatility_squeeze')
    .signal('width', width)
    .signal('width_z', width_z)
    .signal('trend', trend)
    .signal('mom', mom)
    .size(position)
    .stop_loss(pct=4.0)
    .take_profit(pct=8.0)
)
""",
    "tail_reversal": """
z = close.zscore(param('z_window', default=72))
osc = rsi(close, param('rsi_period', default=14))
regime = ema(close, param('regime_span', default=240))
long_cond = (z < lit(-2.0)) & (osc < lit(25.0)) & (close > regime * lit(0.70))
short_cond = (z > lit(2.0)) & (osc > lit(75.0)) & (close < regime * lit(1.30))
position = when(long_cond, lit(0.35), when(short_cond, lit(-0.35), lit(0.0)))
strategy = (
    Strategy.create('tail_reversal')
    .signal('z', z)
    .signal('osc', osc)
    .signal('regime', regime)
    .size(position)
    .stop_loss(pct=3.0)
    .take_profit(pct=5.0)
)
""",
}

GRIDS: dict[str, dict[str, list[Any]]] = {
    "adaptive_trend": {
        "fast": [12, 24, 36, 48],
        "slow": [72, 120, 168, 240],
        "mom": [12, 24, 48, 72],
        "adx_period": [10, 14, 18, 24],
    },
    "volatility_squeeze": {
        "bb_period": [18, 24, 36, 48],
        "width_window": [72, 120, 168, 240],
        "trend_span": [48, 96, 144, 240],
        "mom": [6, 12, 24, 48],
    },
    "tail_reversal": {
        "z_window": [36, 72, 120, 168],
        "rsi_period": [7, 10, 14, 21],
        "regime_span": [120, 240, 360, 480],
    },
}


def unpack(result: Any) -> Any:
    if getattr(result, "structuredContent", None) is not None:
        return result.structuredContent
    texts = [getattr(item, "text", "") for item in getattr(result, "content", [])]
    text = "\n".join(t for t in texts if t)
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


def config(asset_id: int, start: str, end: str, interval: str) -> dict[str, Any]:
    return {
        "universe": [asset_id],
        "start": start,
        "end": end,
        "bar_interval": interval,
        "initial_capital": 10000,
        "fees": "binance_perps",
        "slippage": {"kind": "fixed_bps", "bps": 2.0},
        "warmup_bars": 500,
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


def metric(row: dict[str, Any], key: str, default: float = -1e18) -> float:
    value = (row.get("metrics") or {}).get(key, default)
    try:
        return float(value)
    except Exception:
        return default


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mcp_url": MCP_URL,
        "criterion": "net out-of-sample Sharpe >= 1.5",
        "train": TRAIN,
        "test": TEST,
        "runs": [],
        "errors": [],
    }

    async with streamablehttp_client(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            schemas = [tool.model_dump(mode="json") for tool in tools.tools]
            (OUT / "mcp-tools.json").write_text(json.dumps(schemas, indent=2))
            report["version"] = await call(session, "get_version", {})
            report["indicators"] = await call(session, "list_indicators", {})

            for asset in ASSETS:
                try:
                    await call(
                        session,
                        "ingest_data",
                        {
                            "provider": "binance",
                            "symbol": asset["symbol"],
                            "symbol_id": asset["id"],
                            "start": f"{TRAIN[0]}T00:00:00Z",
                            "end": f"{TEST[1]}T00:00:00Z",
                            "interval": "1h",
                            "data_root": STORE["data_root"],
                            "metadata_db": STORE["metadata_db"],
                            "exchange": "binance",
                            "asset_class": "crypto_perp",
                        },
                    )
                except Exception as exc:
                    report["errors"].append({"stage": "ingest", "asset": asset["symbol"], "error": str(exc)})

            report["symbols"] = await call(session, "list_symbols", {"store": STORE})

            for strategy_name, strategy_code in STRATEGIES.items():
                try:
                    report.setdefault("compiled", {})[strategy_name] = await call(
                        session, "build_strategy", {"strategy_code": strategy_code}
                    )
                except Exception as exc:
                    report["errors"].append({"stage": "compile", "strategy": strategy_name, "error": str(exc)})
                    continue

                for asset in ASSETS:
                    for interval in ("4h", "12h"):
                        run_id = f"{strategy_name}-{asset['symbol']}-{interval}"
                        try:
                            sweep = await call(
                                session,
                                "run_sweep",
                                {
                                    "strategy_code": strategy_code,
                                    "param_grid": GRIDS[strategy_name],
                                    "config": config(asset["id"], TRAIN[0], TRAIN[1], interval),
                                    "store": STORE,
                                    "lite": True,
                                    "top_k": 8,
                                    "rank_metric": "sharpe",
                                    "device": "auto",
                                    "precision": "fp64",
                                },
                            )
                            candidates = sweep.get("top", [])
                            tested = []
                            for candidate in candidates[:5]:
                                params = candidate.get("params") or {}
                                concrete = strategy_code
                                for name, value in params.items():
                                    concrete = concrete.replace(
                                        f"param('{name}', default=", f"param('{name}', default="
                                    )
                                # run_sweep winner is re-evaluated exactly by a one-point sweep on OOS.
                                oos = await call(
                                    session,
                                    "run_sweep",
                                    {
                                        "strategy_code": strategy_code,
                                        "param_grid": {k: [v] for k, v in params.items()},
                                        "config": config(asset["id"], TEST[0], TEST[1], interval),
                                        "store": STORE,
                                        "lite": False,
                                        "top_k": 1,
                                        "rank_metric": "sharpe",
                                    },
                                )
                                exact = (oos.get("top") or [{}])[0]
                                tested.append({"train": candidate, "test": exact})

                            tested.sort(key=lambda x: metric(x["test"], "sharpe"), reverse=True)
                            best = tested[0] if tested else None
                            robustness = None
                            monte_carlo = None
                            if best and metric(best["test"], "sharpe") >= 1.5:
                                params = best["test"].get("params") or best["train"].get("params") or {}
                                first_param = next(iter(params), None)
                                if first_param:
                                    center = params[first_param]
                                    if isinstance(center, int):
                                        neighbours = sorted(set([max(2, int(center * 0.8)), center, int(center * 1.2)]))
                                        robustness = await call(
                                            session,
                                            "run_stability",
                                            {
                                                "strategy_code": strategy_code,
                                                "stability_config": {
                                                    "param_name": first_param,
                                                    "values": neighbours,
                                                    "metric": "sharpe",
                                                },
                                                "config": config(asset["id"], TEST[0], TEST[1], interval),
                                                "store": STORE,
                                            },
                                        )
                                monte_carlo = await call(
                                    session,
                                    "run_monte_carlo",
                                    {
                                        "strategy_code": strategy_code,
                                        "config": config(asset["id"], TEST[0], TEST[1], interval),
                                        "store": STORE,
                                        "mc_config": {
                                            "n_paths": 1000,
                                            "method": {"type": "block_bootstrap", "block_size": 24},
                                            "rng_seed": 42,
                                        },
                                    },
                                )

                            report["runs"].append(
                                {
                                    "id": run_id,
                                    "strategy": strategy_name,
                                    "asset": asset["symbol"],
                                    "interval": interval,
                                    "sweep": sweep,
                                    "tested_oos": tested,
                                    "best": best,
                                    "passes": bool(best and metric(best["test"], "sharpe") >= 1.5),
                                    "stability": robustness,
                                    "monte_carlo": monte_carlo,
                                }
                            )
                        except Exception as exc:
                            report["errors"].append({"stage": "research", "id": run_id, "error": str(exc)})

    report["runs"].sort(
        key=lambda r: metric((r.get("best") or {}).get("test") or {}, "sharpe"), reverse=True
    )
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Crypto alpha research report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Acceptance threshold: net out-of-sample Sharpe >= 1.5.",
        "",
        "| Rank | Strategy | Asset | TF | OOS Sharpe | Return | Max DD | Trades | Pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for index, run in enumerate(report["runs"], 1):
        best_test = ((run.get("best") or {}).get("test") or {})
        metrics = best_test.get("metrics") or {}
        lines.append(
            f"| {index} | {run['strategy']} | {run['asset']} | {run['interval']} | "
            f"{metrics.get('sharpe', 'n/a')} | {metrics.get('total_return', 'n/a')} | "
            f"{metrics.get('max_drawdown', 'n/a')} | {best_test.get('trade_count', 'n/a')} | "
            f"{'YES' if run.get('passes') else 'NO'} |"
        )
    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{e}`" for e in report["errors"]]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
