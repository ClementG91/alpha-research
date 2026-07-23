# Causal cross-asset alpha hunt

**Verdict: REJECTED.**

Selected: `auction_flow_reversal[hold=1,q=0.15,threshold=1.25,rebalance=1]` from 12 predeclared candidates.

The strategy uses close-known flow information and enters at the next open. Returns are measured open-to-open, costs are charged to actual turnover, and weights are class- and beta-neutralised.

## Core universe — 2015–2024

- Sharpe: **nan**
- annual alpha: **nan%**
- HAC alpha t-stat: **nan**
- SPY beta: **nan**
- correlation: **nan**
- max drawdown: **0.00%**
- observations: **5032**
- position changes: **0**

## Separate ETF replication — 2015–2024

- Sharpe: **nan**
- annual alpha: **nan%**
- HAC alpha t-stat: **nan**
- SPY beta: **nan**
- correlation: **nan**
- max drawdown: **0.00%**

## Multiple-testing and overfit controls

- core White Reality Check p: **0.0002**
- core deflated-Sharpe probability: **nan**
- core PBO: **1.0000**
- external White Reality Check p: **0.0002**
- external deflated-Sharpe probability: **nan**
- external PBO: **1.0000**

## Stress

- doubled-cost core Sharpe: **nan**
- doubled-cost external Sharpe: **nan**
- extra-delay core Sharpe: **nan**
- extra-delay external Sharpe: **nan**
- remove-best-year core Sharpe: **nan**
- remove-best-year external Sharpe: **nan**

## Gates

- PASS — core observations >= 2000
- FAIL — core trades >= 3000
- FAIL — core Sharpe >= 0.75
- FAIL — core alpha t-stat >= 2.0
- FAIL — core absolute beta <= 0.10
- FAIL — core absolute correlation <= 0.15
- FAIL — positive alpha in >= 4 folds
- FAIL — external Sharpe >= 0.50
- FAIL — external alpha t-stat >= 1.50
- FAIL — external absolute beta <= 0.10
- FAIL — external absolute correlation <= 0.20
- PASS — core White p <= 0.05
- FAIL — core DSR >= 0.95
- FAIL — core PBO <= 0.10
- PASS — external White p <= 0.05
- FAIL — external DSR >= 0.95
- FAIL — external PBO <= 0.10
- FAIL — double-cost core Sharpe > 0.30
- FAIL — double-cost external Sharpe > 0
- FAIL — extra-delay core Sharpe > 0
- FAIL — extra-delay external Sharpe > 0
- FAIL — remove-best-year core Sharpe > 0.30
- FAIL — remove-best-year external Sharpe > 0

The 2025+ interval is diagnostic only because it has already been inspected. Even a passing result remains paper-trading only until genuinely new data arrive.
