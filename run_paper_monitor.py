from __future__ import annotations

import argparse
import asyncio
import json
import traceback
from datetime import date
from pathlib import Path
from typing import Any

import manifold_research as engine

DEFAULT_START = "2026-07-02"
INTERVAL = "4h"


async def monitor(strategy: dict[str, Any], start: str, end: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "start": start,
        "end": end,
        "calendar_days": max((date.fromisoformat(end) - date.fromisoformat(start)).days, 0),
        "minimum_validation_days": 90,
        "minimum_trades": 20,
    }
    ClientSession, streamable = engine.mcp_imports()
    async with streamable(engine.MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            report["version"] = await engine.call(session, "get_version", {})
            try:
                result = await engine.backtest(
                    session,
                    strategy,
                    (start, end),
                    INTERVAL,
                    include_equity_curve=True,
                    equity_points=250,
                    include_daily_returns=True,
                )
                report["result"] = result
                report["status"] = "validated" if (
                    report["calendar_days"] >= 90
                    and engine.metric(result, "sharpe", -99) >= 0.75
                    and engine.metric(result, "alpha", -99) > 0
                    and abs(engine.metric(result, "beta", 99)) <= 0.15
                    and engine.trades(result) >= report["minimum_trades"]
                ) else "collecting"
            except Exception as exc:
                report["status"] = "no_new_data"
                report["error"] = str(exc)
    return report


def render(report: dict[str, Any], output: Path) -> None:
    result = report.get("result") or {}
    lines = [
        "# Forward paper monitor",
        "",
        f"- Window: `{report['start']}` to `{report['end']}`.",
        f"- Calendar days: `{report['calendar_days']}`.",
        f"- Status: `{report['status']}`.",
        "",
        "This monitor never changes parameters. It only re-runs the frozen strategy on bars that arrived after the research cutoff.",
    ]
    if result:
        lines += [
            "",
            "## Current forward metrics",
            "",
            f"- Sharpe: `{engine.metric(result, 'sharpe'):.3f}`.",
            f"- Alpha: `{engine.metric(result, 'alpha'):.4f}`.",
            f"- Beta: `{engine.metric(result, 'beta'):.4f}`.",
            f"- Return: `{engine.metric(result, 'total_return'):.2%}`.",
            f"- Max drawdown: `{engine.metric(result, 'max_drawdown'):.2%}`.",
            f"- Trades: `{engine.trades(result)}`.",
        ]
    else:
        lines += ["", f"No usable new bars yet: `{report.get('error', 'unknown')}`."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plt = engine.plt_module()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    equity = engine.curve(result)
    if equity:
        ax.plot(equity)
    else:
        ax.text(0.5, 0.5, "No new paper equity curve", ha="center", transform=ax.transAxes)
    ax.set_title("Frozen-strategy forward paper equity")
    ax.grid(alpha=0.25)
    engine.save(fig, output / "paper_equity.svg")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=Path, default=Path("results/walk_forward/locked_strategy.json"))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=date.today().isoformat())
    parser.add_argument("--output", type=Path, default=Path("results/paper_monitor"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        strategy = json.loads(args.strategy.read_text(encoding="utf-8"))
        report = asyncio.run(monitor(strategy, args.start, args.end))
        (args.output / "raw.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        render(report, args.output)
        return 0
    except Exception:
        (args.output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
