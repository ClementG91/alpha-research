# Alpha Research

Leakage-resistant, reproducible research for multi-asset quantitative portfolios.

## Current result

The current pipeline no longer treats an indicator crossover as a quant strategy. It combines:

- pooled cross-sectional Ridge and shallow gradient-boosting forecasts at 1/3/6-month horizons;
- statistical trend measured with regression slope t-statistics;
- PCA residual momentum;
- a defensive macro risk-parity sleeve;
- Gaussian-mixture regime detection;
- Ledoit-Wolf covariance shrinkage and volatility-aware allocation.

The universe combines global equities, real estate, government bonds, gold, commodities, USD and capped BTC/ETH exposure over the maximum available history.

- Required target: net OOS Sharpe >= 1.5.
- Best final 36-month holdout Sharpe: **1.247**.
- Full walk-forward Sharpe: **0.834**.
- Anti-lookahead audit: **PASS**.
- Status: **not validated as alpha >= 1.5**.

See [`QUANT_RESULTS.md`](./QUANT_RESULTS.md) for the methodology, metrics, stress tests, Monte Carlo and rejection reasons.

The previous ManifoldBT EMA experiments are retained only for reproducibility and are deprecated as the primary research result.

## Validation standard

A candidate is accepted only when it passes all of the following:

- purged expanding walk-forward with horizon-specific embargoes;
- point-in-time asset eligibility and next-period execution;
- mutation tests proving future data cannot alter historical weights;
- a genuinely untouched final holdout;
- explicit transaction costs and doubled-cost stress;
- delayed-execution and no-crypto stress tests;
- block-bootstrap Monte Carlo;
- cross-sectional permutation testing;
- Probabilistic and Deflated Sharpe analysis;
- stable performance across multiple independent sleeves.

CI runs deterministic anti-lookahead and allocation tests on Python 3.11 and 3.12. The full market-data research workflow is separate, scheduled weekly and available manually.

> Research only. Historical simulations are not a guarantee of future performance.
