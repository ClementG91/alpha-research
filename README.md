# Alpha Research

Reproducible crypto strategy research powered by the remote ManifoldBT MCP server at `https://mcp.manifoldbt.com/mcp`.

## Current result

Seven research generations were executed through GitHub Actions with a strict train / validation / frozen out-of-sample process.

- Required target: net OOS Sharpe >= 1.5.
- Best verified result: OOS Sharpe **1.318**.
- Status: **research candidate only — target not validated**.

See [`RESULTS.md`](./RESULTS.md) for the full result and [`strategies/ema_return_vol_overlay_verified.json`](./strategies/ema_return_vol_overlay_verified.json) for the frozen StrategyDef.

## Research standard

A candidate is only considered viable when it meets all of the following:

- net Sharpe ratio >= 1.5 on an untouched out-of-sample period;
- realistic Binance perpetual fees and non-zero slippage;
- one-bar signal delay to avoid same-bar look-ahead;
- enough trades to avoid a small-sample illusion;
- parameter-neighbour robustness rather than a single sharp optimum;
- Monte Carlo stress testing before any paper-trading recommendation.

The repository stores the strategy source, configurations and reproducible research runners for every generation.

> Research only. A historical backtest is not a guarantee of future performance.
