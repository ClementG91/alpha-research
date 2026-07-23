# Causal cross-asset alpha hunt

**Verdict: REJECTED.**

Selected: `close_flow_reversal[hold=5,q=0.15,threshold=1.5,rebalance=1]` from 12 predeclared candidates.

The strategy uses close-known flow information and enters at the next open. Returns are measured open-to-open, costs are charged to actual turnover, and weights are class- and beta-neutralised.

## Core universe — 2015–2024

- Sharpe: **-0.602**
- annual alpha: **-1.28%**
- HAC alpha t-stat: **-2.14**
- SPY beta: **0.012**
- correlation: **0.106**
- max drawdown: **-12.96%**
- observations: **2516**
- position changes: **7980**

## Separate ETF replication — 2015–2024

- Sharpe: **-0.057**
- annual alpha: **-0.17%**
- HAC alpha t-stat: **-0.46**
- SPY beta: **0.008**
- correlation: **0.104**
- max drawdown: **-5.41%**

## Multiple-testing and overfit controls

- core White Reality Check p: **1.0000**
- core deflated-Sharpe probability: **0.0000**
- core PBO: **0.6548**
- external White Reality Check p: **1.0000**
- external deflated-Sharpe probability: **0.0000**
- external PBO: **0.0040**

## Stress

- doubled-cost core Sharpe: **-1.203**
- doubled-cost external Sharpe: **-0.723**
- extra-delay core Sharpe: **-0.691**
- extra-delay external Sharpe: **-0.235**
- remove-best-year core Sharpe: **-0.731**
- remove-best-year external Sharpe: **-0.289**

## Gates

- PASS — core observations >= 2000
- PASS — core trades >= 3000
- FAIL — core Sharpe >= 0.75
- FAIL — core alpha t-stat >= 2.0
- PASS — core absolute beta <= 0.10
- PASS — core absolute correlation <= 0.15
- FAIL — positive alpha in >= 4 folds
- FAIL — external Sharpe >= 0.50
- FAIL — external alpha t-stat >= 1.50
- PASS — external absolute beta <= 0.10
- PASS — external absolute correlation <= 0.20
- FAIL — core White p <= 0.05
- FAIL — core DSR >= 0.95
- FAIL — core PBO <= 0.10
- FAIL — external White p <= 0.05
- FAIL — external DSR >= 0.95
- PASS — external PBO <= 0.10
- FAIL — double-cost core Sharpe > 0.30
- FAIL — double-cost external Sharpe > 0
- FAIL — extra-delay core Sharpe > 0
- FAIL — extra-delay external Sharpe > 0
- FAIL — remove-best-year core Sharpe > 0.30
- FAIL — remove-best-year external Sharpe > 0

The 2025+ interval is diagnostic only because it has already been inspected. Even a passing result remains paper-trading only until genuinely new data arrive.
