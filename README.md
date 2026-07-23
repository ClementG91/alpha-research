# Alpha Research — ManifoldBT MCP only

This repository now uses the remote **ManifoldBT MCP engine** for every market calculation. The previous `yfinance`/scikit-learn multi-asset engine has been removed from the active pipeline.

## What comes from ManifoldBT

- strategy validation;
- Binance-universe discovery;
- parameter sweeps and overfitting correction;
- train, validation and frozen holdout backtests;
- equity curve and daily returns;
- 2D parameter surface;
- parameter stability analysis;
- Monte Carlo risk simulation.

Python only orchestrates MCP calls, preserves raw JSON payloads, and renders SVG files from the numerical series returned by ManifoldBT. It does not recalculate strategy performance.

## Latest research output

The workflow writes the current report to [`results/manifold/REPORT.md`](results/manifold/REPORT.md) and commits the plots to the branch:

![Sharpe by period](results/manifold/plots/period_metrics.svg)

![Parameter stability](results/manifold/plots/stability.svg)

![Monte Carlo risk](results/manifold/plots/monte_carlo.svg)

The frozen holdout equity and drawdown are generated when the new MCP workflow completes:

- [`results/manifold/plots/equity_curve.svg`](results/manifold/plots/equity_curve.svg)
- [`results/manifold/plots/drawdown.svg`](results/manifold/plots/drawdown.svg)
- [`results/manifold/plots/parameter_heatmap.svg`](results/manifold/plots/parameter_heatmap.svg)

## Validation protocol

- Train: 2021–2023.
- Validation: 2024.
- Frozen holdout: 2025 through 1 July 2026.
- Universe: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT and XRPUSDT.
- Timeframe: 4h.
- Binance-perps fee preset, 2 bps slippage and one-bar execution delay.
- Required acceptance threshold: net holdout Sharpe >= 1.5, positive train result, validation Sharpe >= 0.7 and at least 30 holdout trades.

## Real limitations of the public MCP server

- It currently reports ManifoldBT Community and locks native `run_walk_forward` behind Pro.
- It exposes no `ingest_data` tool, so the pipeline cannot add equities, futures or custom datasets.
- Its preloaded datastore contains Binance **CryptoSpot** proxies, not full perpetual history.
- Funding, basis, open interest and liquidation history are unavailable.
- Community Monte Carlo is capped at 1,000 paths.

The workflow therefore performs chronological train/validation/holdout orchestration through separate MCP calls. That is genuinely out of sample, but it is not ManifoldBT's native Pro walk-forward optimiser.
