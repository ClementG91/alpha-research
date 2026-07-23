from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from quant_research.core import DEFAULT_GROUPS, ResearchConfig, run_walk_forward


DEFAULT_TICKERS = list(DEFAULT_GROUPS)


def download_prices(tickers: list[str], retries: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers=tickers,
                period="max",
                interval="1d",
                auto_adjust=True,
                actions=False,
                progress=False,
                threads=False,
                group_by="column",
                timeout=45,
            )
            if raw.empty:
                raise RuntimeError("Yahoo Finance returned an empty frame")
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"].copy()
            else:
                close = raw[["Close"]].copy()
                close.columns = tickers[:1]
            close = close.sort_index().dropna(how="all")
            missing = [ticker for ticker in tickers if ticker not in close or close[ticker].notna().sum() < 100]
            for ticker in missing:
                single = yf.download(
                    ticker,
                    period="max",
                    interval="1d",
                    auto_adjust=True,
                    actions=False,
                    progress=False,
                    threads=False,
                    timeout=45,
                )
                if not single.empty:
                    series = single["Close"]
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    close[ticker] = series
            close = close.loc[:, [ticker for ticker in tickers if ticker in close]]
            if close.shape[1] < 8:
                raise RuntimeError(f"only {close.shape[1]} assets downloaded successfully")
            return close
        except Exception as exc:  # network/API failures are retried
            last_error = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"price download failed after {retries} attempts: {last_error}")


def fmt(value: object, percentage: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(number):
        return "n/a"
    return f"{number:.2%}" if percentage else f"{number:.3f}"


def build_markdown(result: object) -> str:
    metrics = result.metrics
    holdout = result.holdout_metrics
    diagnostics = result.diagnostics
    mc = result.monte_carlo
    stress = result.stress
    lines = [
        "# Multi-asset quant research",
        "",
        "This report is generated from an expanding, purged walk-forward process. Every monthly position is computed only from information available before the trade month and is applied one month later.",
        "",
        "## Data and audit",
        "",
        f"- Data range: `{diagnostics['data_start']}` to `{diagnostics['data_end']}`",
        f"- Assets: `{', '.join(diagnostics['assets'])}`",
        f"- Anti-lookahead audit: `{'PASS' if diagnostics['anti_lookahead_pass'] else 'FAIL'}`",
        f"- Average monthly turnover: `{fmt(diagnostics['average_monthly_turnover'], True)}`",
        f"- Cross-sectional permutation p-value: `{fmt(diagnostics['permutation_pvalue'])}`",
        "",
        "## Model",
        "",
        "- pooled cross-sectional Ridge + shallow histogram gradient boosting forecasts;",
        "- features include multi-horizon momentum, reversal, realized/downside volatility, drawdown, regression trend t-statistics, skew, kurtosis, benchmark correlation and market breadth;",
        "- three-state Gaussian-mixture regime model fitted only on the historical training window;",
        "- Ledoit-Wolf covariance shrinkage and blended inverse-volatility/mean-variance allocation;",
        "- 10% volatility target, 25% per-asset cap and 15% aggregate crypto cap;",
        "- long/cash implementation because the public price feed does not include borrow or funding costs.",
        "",
        "## Walk-forward performance",
        "",
        "| Metric | Full OOS stream | Final holdout |",
        "|---|---:|---:|",
        f"| Sharpe | {fmt(metrics.get('sharpe'))} | {fmt(holdout.get('sharpe'))} |",
        f"| Deflated Sharpe probability | {fmt(metrics.get('deflated_sharpe_ratio'), True)} | {fmt(holdout.get('deflated_sharpe_ratio'), True)} |",
        f"| Probabilistic Sharpe probability | {fmt(metrics.get('probabilistic_sharpe_ratio'), True)} | {fmt(holdout.get('probabilistic_sharpe_ratio'), True)} |",
        f"| Annual return | {fmt(metrics.get('annual_return'), True)} | {fmt(holdout.get('annual_return'), True)} |",
        f"| Annual volatility | {fmt(metrics.get('annual_volatility'), True)} | {fmt(holdout.get('annual_volatility'), True)} |",
        f"| Maximum drawdown | {fmt(metrics.get('max_drawdown'), True)} | {fmt(holdout.get('max_drawdown'), True)} |",
        f"| Total return | {fmt(metrics.get('total_return'), True)} | {fmt(holdout.get('total_return'), True)} |",
        "",
        "## Stress tests",
        "",
        "| Scenario | Sharpe | Return | Max drawdown |",
        "|---|---:|---:|---:|",
    ]
    for name, values in stress.items():
        lines.append(
            f"| {name} | {fmt(values.get('sharpe'))} | {fmt(values.get('total_return'), True)} | {fmt(values.get('max_drawdown'), True)} |"
        )
    lines += [
        "",
        "## Block-bootstrap Monte Carlo",
        "",
        f"- Paths: `{int(mc.get('paths', 0))}`",
        f"- Median return: `{fmt(mc.get('median_return'), True)}`",
        f"- 5th-percentile return: `{fmt(mc.get('return_p05'), True)}`",
        f"- 1st-percentile return: `{fmt(mc.get('return_p01'), True)}`",
        f"- 5th-percentile Sharpe: `{fmt(mc.get('sharpe_p05'))}`",
        f"- Probability of a 50% loss: `{fmt(mc.get('probability_50pct_loss'), True)}`",
        f"- Probability of a negative terminal return: `{fmt(mc.get('probability_negative'), True)}`",
        "",
        "## Acceptance rule",
        "",
        "A candidate is not accepted merely because its raw Sharpe exceeds 1.5. It must also pass the anti-lookahead audit, maintain a positive holdout Sharpe, survive doubled costs and an extra execution delay, show a low permutation p-value, and retain a defensible deflated Sharpe probability.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run leakage-resistant multi-asset quant research")
    parser.add_argument("--output", default="results/quant", help="output directory")
    parser.add_argument("--mc-paths", type=int, default=5000)
    parser.add_argument("--permutation-trials", type=int, default=1000)
    parser.add_argument("--holdout-months", type=int, default=36)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prices = download_prices(DEFAULT_TICKERS)
    prices.to_csv(out / "prices_daily.csv")

    config = ResearchConfig(
        monte_carlo_paths=args.mc_paths,
        permutation_trials=args.permutation_trials,
        holdout_months=args.holdout_months,
    )
    result = run_walk_forward(prices, DEFAULT_GROUPS, config)
    result.weights.to_csv(out / "weights_monthly.csv")
    pd.DataFrame(
        {
            "net_return": result.returns,
            "gross_return": result.gross_returns,
            "cost": result.costs,
        }
    ).to_csv(out / "portfolio_returns.csv")
    result.audit.to_csv(out / "anti_lookahead_audit.csv")
    (out / "report.json").write_text(json.dumps(result.serializable(), indent=2, default=str))
    (out / "REPORT.md").write_text(build_markdown(result))
    print((out / "REPORT.md").read_text())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        Path("results").mkdir(exist_ok=True)
        Path("results/FATAL.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise
