# ManifoldBT MCP research results

Generated: `2026-07-23T14:05:23.773006+00:00`

All performance, equity, sweep, stability and Monte Carlo data come from ManifoldBT MCP. Python only orchestrates and plots its payloads.

- Selected: `return_vol_long_cash`
- Target: `1.5`
- Accepted: `NO`

| Period | Sharpe | Return | Max DD | Trades |
|---|---:|---:|---:|---:|
| Train | 1.480 | 359.19% | -46.43% | 271 |
| Validation | 0.959 | 26.27% | -25.31% | 96 |
| Frozen holdout | 0.320 | 7.14% | -22.81% | 130 |

## Blocking limitations

- No ingest_data tool: only the server's preloaded datastore can be used.
- Native run_walk_forward is Pro-only; train/validation/holdout are separate MCP backtests.
- The datastore exposes Binance CryptoSpot proxies, not complete perpetual funding/basis/OI history.
- Community Monte Carlo is capped at 1,000 paths.

Raw MCP responses: `raw.json`. Tool schemas: `mcp-tools.json`. Plots: `plots/*.svg`.
