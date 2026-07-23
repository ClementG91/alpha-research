from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_research.advanced_v3 import run_advanced_walk_forward
from quant_research.core import DEFAULT_GROUPS, ResearchConfig
from run_advanced_quant_research import build_advanced_markdown
from run_quant_research import download_prices, enforce_minimum_history


DEFAULT_TICKERS = list(DEFAULT_GROUPS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run diversified advanced quant ensemble"
    )
    parser.add_argument("--output", default="results/advanced_quant")
    parser.add_argument("--mc-paths", type=int, default=5000)
    parser.add_argument("--permutation-trials", type=int, default=1000)
    parser.add_argument("--holdout-months", type=int, default=36)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prices = enforce_minimum_history(download_prices(DEFAULT_TICKERS))
    prices.to_csv(out / "prices_daily.csv")

    config = ResearchConfig(
        monte_carlo_paths=args.mc_paths,
        permutation_trials=args.permutation_trials,
        holdout_months=args.holdout_months,
        trial_count=40,
    )
    result = run_advanced_walk_forward(prices, DEFAULT_GROUPS, config)
    result.weights.to_csv(out / "weights_monthly.csv")
    pd.DataFrame(
        {
            "net_return": result.returns,
            "gross_return": result.gross_returns,
            "cost": result.costs,
        }
    ).to_csv(out / "portfolio_returns.csv")
    result.audit.to_csv(out / "anti_lookahead_audit.csv")
    (out / "report.json").write_text(
        json.dumps(result.serializable(), indent=2, default=str)
    )
    (out / "REPORT.md").write_text(build_advanced_markdown(result))
    print((out / "REPORT.md").read_text())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        Path("results").mkdir(exist_ok=True)
        Path("results/FATAL.txt").write_text(
            f"{type(exc).__name__}: {exc}\n"
        )
        raise
