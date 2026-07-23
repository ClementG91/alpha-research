# Alpha Research

Reproducible crypto strategy research powered by the remote ManifoldBT MCP server at `https://mcp.manifoldbt.com/mcp`.

## Research standard

A candidate is only considered viable when it meets all of the following:

- net Sharpe ratio >= 1.5 on an untouched out-of-sample period;
- realistic Binance perpetual fees and non-zero slippage;
- one-bar signal delay to avoid same-bar look-ahead;
- enough trades to avoid a small-sample illusion;
- parameter-neighbour robustness rather than a single sharp optimum;
- Monte Carlo stress testing before any paper-trading recommendation.

The repository stores the exact MCP tool schemas, strategy source, configurations, raw outputs and a generated research report for each run.

> Research only. A historical backtest is not a guarantee of future performance.
