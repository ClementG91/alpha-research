# Conditional-volatility walk-forward research

Generated from ManifoldBT MCP on 23 July 2026.

## Verdict

**Rejected as validated alpha.** The selected `dual_garch_exhaustion` model achieved very low market-factor beta, but its alpha and Sharpe were not consistent across unseen calendar years.

- Historical multi-fold gate: **FAIL**.
- Forward paper validation: **PENDING / FAIL**.
- Parameters were selected without using the previously consumed 2025–2026 audit window.
- The committed strategy is frozen; the paper monitor is not allowed to retune it.

## Expanding walk-forward folds

| Fold | Unseen period | Sharpe | Alpha | Beta | Return | Max DD | Trades |
|---|---|---:|---:|---:|---:|---:|---:|
| 2022 | 2022-01-01 – 2022-12-31 | 0.165 | 0.0031 | -0.0055 | +0.64% | -5.44% | 23 |
| 2023 | 2023-01-01 – 2023-12-31 | 0.985 | 0.0390 | -0.0084 | +3.06% | -2.17% | 44 |
| 2024 | 2024-01-01 – 2024-12-31 | **-0.527** | **-0.0109** | -0.0021 | **-1.31%** | -4.05% | 89 |

Aggregate metrics:

- median unseen Sharpe: **0.165**;
- worst unseen Sharpe: **-0.527**;
- median alpha: **0.0031**;
- maximum absolute beta: **0.0084**;
- positive-alpha folds: **2/3**;
- total unseen trades: **156**;
- parameter-drift score: **0.20**.

The low beta objective was achieved, but the 2024 sign reversal invalidates the alpha hypothesis.

## Locked consensus strategy

- family: `dual_garch_exhaustion`;
- interval: `4h`;
- horizon: `6` bars;
- entry shock: `1.5%`;
- minimum fast/slow GARCH ratio: `1.1`;
- risk-reduction threshold: `1.5`;
- normal position fraction: `8%`, halved in the high conditional-volatility regime.

The compact immutable specification is stored in [`locked_strategy.json`](locked_strategy.json).

## Secondary historical audit — not pristine

The 2025–1 July 2026 window was used during earlier research, so it is only a secondary audit:

- Sharpe: **-0.089**;
- alpha: **-0.0021**;
- beta: **-0.0023**;
- return: **-0.23%**;
- maximum drawdown: **-1.49%**;
- trades: **7**;
- doubled-slippage Sharpe: **-0.098**;
- two-bar-delay Sharpe: **-0.089**.

## Forward paper monitor

Window at snapshot time: **2 July–23 July 2026**.

- elapsed: 21 calendar days;
- trades: 0;
- status: collecting;
- minimum validation requirement: 90 days and 20 trades.

Zero trades over 21 days is not evidence of zero risk or successful validation.

## Development Monte Carlo

ManifoldBT block-bootstrap, 1,000 paths on the 2021–2024 development period:

- mean terminal return: **+1.58%**;
- probability of ruin: **44.7%**;
- 95% return CVaR: **-24.11%**;
- 99% return CVaR: **-31.21%**;
- mean maximum drawdown: **13.17%**;
- 95th-percentile maximum drawdown: **26.33%**;
- 99th-percentile maximum drawdown: **32.63%**.

No wipeout occurred in 1,000 paths; this must not be interpreted as zero wipeout risk.

## Plots

![Unseen-year Sharpe](plots/fold_sharpe.svg)

![Unseen-year alpha and beta](plots/fold_alpha_beta.svg)

![Family selection scores](plots/family_scores.svg)

![Legacy audit and stress](plots/legacy_stress.svg)

![Forward paper monitor](plots/paper_equity.svg)

## Real limitations

- Native `run_walk_forward` is Pro-only; the expanding folds are orchestrated through separate ManifoldBT calls.
- SPY cannot be ingested, so beta is Manifold's available market-factor beta rather than a direct S&P 500 regression.
- The datastore contains Binance CryptoSpot proxies, while perpetual funding, basis, open interest and liquidation history are unavailable.
- The 2025–2026 audit is no longer pristine and cannot be reused as an optimisation target.
