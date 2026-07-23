# ManifoldBT status

The active research engine is now exclusively the remote ManifoldBT MCP server.

The previously verified MCP candidate reached an OOS Sharpe of **1.318**, below the required 1.5 threshold. Its 2026 YTD subperiod was negative, so it is not accepted as durable alpha.

The new workflow expands the MCP-only protocol to three volatility-aware EMA families, chronological train/validation/holdout selection, `run_sweep_2d`, `run_stability`, equity/daily-return capture, and Community-limited Monte Carlo. The authoritative result is regenerated in `results/manifold/REPORT.md`.

No new performance claim should be made until the GitHub Actions MCP run finishes and commits its raw payloads and plots.
