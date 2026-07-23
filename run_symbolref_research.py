from __future__ import annotations

from pathlib import Path
from typing import Any

import run_market_neutral_research as research


async def symbolref_backtests(
    session: Any,
    docs: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
) -> list[dict[str, Any]]:
    """Manifold Community only handles SymbolRef through individual backtests."""
    results: list[dict[str, Any]] = []
    for doc in docs:
        result = await research.logged_call(
            session,
            "run_backtest",
            {"strategy_json": doc, "config": config},
            output,
        )
        results.append(result)
    return results


def main() -> int:
    research.batch = symbolref_backtests
    return research.main()


if __name__ == "__main__":
    raise SystemExit(main())
