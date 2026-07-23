# ManifoldBT MCP result

The active research path now uses the remote ManifoldBT MCP engine exclusively for market calculations.

## Verdict

**Rejected.** The selected `return_vol_long_cash` model produced:

- train Sharpe: 1.480;
- validation Sharpe: 0.959;
- frozen holdout Sharpe: **0.320**;
- holdout return: +7.14%;
- holdout maximum drawdown: -22.81%;
- holdout trades: 130;
- 2025 Sharpe: 0.895;
- 2026 YTD Sharpe: **-1.007** with a -4.90% return.

The requested net holdout Sharpe of 1.5 is not validated.

## Monte Carlo

ManifoldBT Community ran 1,000 six-day-block bootstrap paths:

- mean terminal return: +11.32%;
- `prob_of_ruin`: 39.9%;
- 95th-percentile maximum drawdown: 35.20%;
- 99th-percentile maximum drawdown: 42.19%;
- no wipeout observed in the 1,000 paths, which is not evidence of zero wipeout risk.

## Audit trail

- Raw responses: `results/manifold/raw.json`.
- MCP schemas: `results/manifold/mcp-tools.json`.
- Detailed reviewed report: `results/manifold/REPORT.md`.
- Plots: `results/manifold/plots/*.svg`.

## Blocking limitations

- ManifoldBT Community 0.14.0 refuses native `run_walk_forward`; Pro is required.
- No `ingest_data` tool is exposed.
- The server datastore contains Binance CryptoSpot proxies rather than complete perpetual history.
- Funding, basis, open interest and liquidation history are missing.
- Monte Carlo is capped at 1,000 paths.
