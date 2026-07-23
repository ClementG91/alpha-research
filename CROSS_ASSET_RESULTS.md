# Cross-asset alpha research — final reviewed result

Generated on 23 July 2026.

## Decision

**No strategy tested in this research cycle qualifies as statistically validated alpha.**

The research requirement was not merely to produce a positive backtest. Every candidate had to combine:

- sufficient trade and observation count;
- chronological walk-forward performance;
- low explicit regression beta and correlation to SPY;
- positive HAC/Newey-West alpha significance;
- block-bootstrap significance;
- resistance to higher costs and execution delay;
- replication across a separate ETF universe;
- no reuse of a failed final window as a new untouched holdout.

The beta-neutral construction worked. The alpha requirement did not.

## Data actually used

The CI run loaded adjusted daily OHLCV for:

- **51 core ETFs** used for development;
- **68 ETFs** in the external replication universe;
- **118 unique symbols** after removing the shared SPY benchmark;
- equities, international markets, rates, credit, commodities, currencies, sectors, themes and style/factor ETFs.

Every loaded symbol in the reviewed artifacts came from the public Yahoo chart endpoint with OHLC adjusted consistently by the adjusted-close ratio. No London Strategic Edge credential or export was available in CI. The engine nevertheless supports London Strategic Edge CSV/Parquet exports through `--lse-dir`, and an API/client path through `LSE_API_KEY` when the official client is available.

## Statistical unit

Instrument trades are reported as a breadth and implementation measure. They are **not** treated as independent statistical samples. Significance is computed on daily portfolio returns using HAC standard errors and block bootstrap.

## Experiment 1 — daily residual mean reversion

- Candidates: **52**.
- Core universe: **51 ETFs**, six asset classes.
- Development OOS: **2,513 daily observations**, **10,022 instrument-day trades**.
- Development Sharpe: **-2.862**.
- Development alpha: **-6.04%**, HAC t-stat **-8.71**.
- 2025+ holdout: **389 observations**, **1,551 trades**.
- Holdout Sharpe: **-4.765**.
- Holdout alpha: **-8.17%**, HAC t-stat **-6.20**.
- SPY beta: **0.008**; SPY correlation: **0.088**.
- Bootstrap and multiple-testing p-values: **1.0000**.

**Verdict: rejected.** The portfolio was neutral to SPY but reliably lost money after daily execution costs.

## Experiment 2 — daily residual momentum on different ETFs

Parameters were selected on the core universe, then evaluated on **68 different ETFs**.

- Candidates: **112**.
- External observations: **388**.
- External trades: **2,856**.
- External Sharpe: **-4.121**.
- External alpha: **-7.72%**, HAC t-stat **-5.08**.
- SPY beta: **-0.011**; SPY correlation: **-0.102**.
- Every external asset class lost money.
- Correctly costed inverted positions also lost money: Sharpe **-3.949**.

**Verdict: rejected.** Reversing the signal did not solve the economics because a complete intraday round trip was still paid every active day.

## Experiment 3 — persistent turnover-aware residual strategies

Positions were held for deterministic 5–20 day intervals and costs were charged only on actual changes in weights.

- Candidates: **180**.
- Development folds with positive alpha: **4/5**, but none was individually significant.
- External observations: **388**.
- External trades: **1,252**.
- External Sharpe: **-0.283**.
- External alpha: **-2.21%**, HAC t-stat **-0.58**.
- SPY beta: **0.049**; correlation: **0.186**.
- Block-bootstrap p-value: **0.6225**.
- Multiple-testing p-value: **1.0000**.
- Doubled-cost Sharpe: **-0.529**.

**Verdict: rejected.** Lower turnover removed most of the catastrophic cost drag, but did not produce significant or replicable alpha.

## Experiment 4 — fixed canonical multi-premia ensemble

No parameter search was performed. The portfolio equally weighted:

1. multi-horizon time-series trend;
2. cross-sectional low volatility;
3. a fixed 60-day residual-trend sleeve.

### Five development test folds

| Fold | Sharpe | Alpha t-stat | SPY beta | Correlation | Trades |
|---|---:|---:|---:|---:|---:|
| 2015–2016 | -0.063 | -0.00 | -0.021 | -0.120 | 1,858 |
| 2017–2018 | 0.461 | 0.72 | -0.004 | -0.032 | 1,831 |
| 2019–2020 | 0.672 | 1.10 | -0.003 | -0.026 | 1,943 |
| 2021–2022 | -0.347 | -0.44 | -0.027 | -0.208 | 1,949 |
| 2023–2024 | -0.802 | -0.96 | -0.025 | -0.120 | 1,938 |

### External ETF replication — 2025 onward

- Observations: **388**.
- Trades: **1,959**.
- Sharpe: **-0.204**.
- Annualised return: **-0.47%**.
- Alpha t-stat: **-0.51**.
- SPY beta: **0.028**.
- SPY correlation: **0.215**.
- Maximum drawdown: **-3.21%**.
- Block-bootstrap p-value: **0.6263**.
- Doubled-cost Sharpe: **-0.453**.
- Extra-delay Sharpe: **-0.228**.

Sleeves:

| Sleeve | Sharpe | Alpha t-stat | Annualised return |
|---|---:|---:|---:|
| Multi-horizon trend | 0.689 | 0.41 | +2.19% |
| Low volatility | -0.751 | -0.85 | -2.29% |
| Residual trend | -0.283 | -0.58 | -1.31% |

**Verdict: rejected.** The trend sleeve was positive, but not significant and too correlated with SPY for the requested alpha objective. The ensemble failed the cross-regime, significance, correlation, cost and delay gates.

## Final conclusion

The research now has enough trades to reject the hypotheses for substantive reasons rather than because of a tiny sample. Across the reviewed experiments, the external evaluations contained between **1,252 and 2,856 trades**, plus **388–389 daily observations**.

The honest result is:

- beta neutralisation: **achieved mechanically**;
- sufficient trade breadth: **achieved**;
- significant alpha: **not achieved**;
- robustness after costs and delay: **not achieved**;
- live deployment: **prohibited**.

No further parameter tuning should be performed against the currently inspected 2025–2026 range. A new hypothesis must be frozen before being evaluated on future data or on a genuinely independent dataset with different information content, such as futures term structure, carry, funding, basis, volatility surfaces or macro releases.
