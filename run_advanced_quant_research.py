from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_research.advanced import run_advanced_walk_forward
from quant_research.core import DEFAULT_GROUPS, ResearchConfig, ResearchResult
from run_quant_research import download_prices, enforce_minimum_history, fmt


DEFAULT_TICKERS = list(DEFAULT_GROUPS)


def build_advanced_markdown(result: ResearchResult) -> str:
    metrics = result.metrics
    holdout = result.holdout_metrics
    diagnostics = result.diagnostics
    mc = result.monte_carlo
    stress = result.stress
    sleeve_metrics = diagnostics.get("sleeve_metrics", {})
    sleeve_weights = diagnostics.get("average_sleeve_weights", {})

    lines = [
        "# Advanced multi-asset quant ensemble",
        "",
        (
            "This is a purged expanding walk-forward ensemble. It contains no "
            "EMA-cross trading rule and no parameter is selected from the final "
            "holdout period."
        ),
        "",
        "## Architecture",
        "",
        "- pooled cross-sectional models forecasting 1-, 3- and 6-month returns;",
        "- Ridge plus shallow histogram gradient boosting inside each ML sleeve;",
        (
            "- independent statistical-trend sleeve using ranked 3/6/12-month "
            "returns and linear-regression slope t-statistics;"
        ),
        "- three-state Gaussian-mixture market-regime model fitted on history only;",
        "- Ledoit-Wolf covariance shrinkage inside asset and sleeve allocation;",
        "- rolling minimum-variance/equal-weight blend across sleeves;",
        "- 25% asset cap, 15% aggregate crypto cap and explicit transaction costs;",
        "- each asset becomes eligible only after two years of live price history.",
        "",
        "## Audit",
        "",
        f"- Data range: `{diagnostics['data_start']}` to `{diagnostics['data_end']}`",
        f"- Assets: `{', '.join(diagnostics['assets'])}`",
        f"- Anti-lookahead: `{'PASS' if diagnostics['anti_lookahead_pass'] else 'FAIL'}`",
        f"- Average monthly turnover: `{fmt(diagnostics['average_monthly_turnover'], True)}`",
        f"- Permutation p-value: `{fmt(diagnostics['permutation_pvalue'])}`",
        "",
        "## Walk-forward performance",
        "",
        "| Metric | Full OOS stream | Final 36-month holdout |",
        "|---|---:|---:|",
        f"| Sharpe | {fmt(metrics.get('sharpe'))} | {fmt(holdout.get('sharpe'))} |",
        (
            "| Deflated Sharpe probability | "
            f"{fmt(metrics.get('deflated_sharpe_ratio'), True)} | "
            f"{fmt(holdout.get('deflated_sharpe_ratio'), True)} |"
        ),
        (
            "| Probabilistic Sharpe probability | "
            f"{fmt(metrics.get('probabilistic_sharpe_ratio'), True)} | "
            f"{fmt(holdout.get('probabilistic_sharpe_ratio'), True)} |"
        ),
        (
            f"| Annual return | {fmt(metrics.get('annual_return'), True)} | "
            f"{fmt(holdout.get('annual_return'), True)} |"
        ),
        (
            f"| Annual volatility | {fmt(metrics.get('annual_volatility'), True)} | "
            f"{fmt(holdout.get('annual_volatility'), True)} |"
        ),
        (
            f"| Maximum drawdown | {fmt(metrics.get('max_drawdown'), True)} | "
            f"{fmt(holdout.get('max_drawdown'), True)} |"
        ),
        (
            f"| Total return | {fmt(metrics.get('total_return'), True)} | "
            f"{fmt(holdout.get('total_return'), True)} |"
        ),
        "",
        "## Sleeve diagnostics",
        "",
        "| Sleeve | Average allocation | Standalone Sharpe | Max drawdown |",
        "|---|---:|---:|---:|",
    ]
    for name, values in sleeve_metrics.items():
        lines.append(
            f"| {name} | {fmt(sleeve_weights.get(name), True)} | "
            f"{fmt(values.get('sharpe'))} | "
            f"{fmt(values.get('max_drawdown'), True)} |"
        )

    lines += [
        "",
        "## Stress tests",
        "",
        "| Scenario | Sharpe | Total return | Max drawdown |",
        "|---|---:|---:|---:|",
    ]
    for name, values in stress.items():
        lines.append(
            f"| {name} | {fmt(values.get('sharpe'))} | "
            f"{fmt(values.get('total_return'), True)} | "
            f"{fmt(values.get('max_drawdown'), True)} |"
        )

    lines += [
        "",
        "## Block-bootstrap Monte Carlo",
        "",
        f"- Paths: `{int(mc.get('paths', 0))}`",
        f"- Median terminal return: `{fmt(mc.get('median_return'), True)}`",
        f"- 5th-percentile terminal return: `{fmt(mc.get('return_p05'), True)}`",
        f"- 1st-percentile terminal return: `{fmt(mc.get('return_p01'), True)}`",
        f"- 5th-percentile Sharpe: `{fmt(mc.get('sharpe_p05'))}`",
        (
            "- Probability of negative terminal return: `"
            f"{fmt(mc.get('probability_negative'), True)}`"
        ),
        (
            "- 99th-percentile drawdown magnitude: `"
            f"{fmt(mc.get('max_drawdown_p99_magnitude'), True)}`"
        ),
        "",
        "## Acceptance gate",
        "",
        (
            "A raw Sharpe above 1.5 is not sufficient. Acceptance also requires "
            "a positive holdout Sharpe, anti-lookahead PASS, positive doubled-cost "
            "and delayed-execution results, a credible permutation p-value and a "
            "deflated-Sharpe probability that survives the declared trial count."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run advanced quant ensemble")
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
        trial_count=32,
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
