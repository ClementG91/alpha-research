# Quant research results

## Decision

The requested target was interpreted as a **net, genuinely out-of-sample Sharpe ratio of at least 1.5**.

No candidate has met that target while also passing the statistical and robustness gates.

The earlier EMA-cross experiments are deprecated. They remain in git history for reproducibility but must not be presented as the current research result.

## Final model

The final researched portfolio is a six-sleeve, long/cash, multi-asset ensemble:

1. pooled cross-sectional Ridge + shallow gradient boosting forecasting 1-month returns;
2. the same model family forecasting 3-month returns;
3. the same model family forecasting 6-month returns;
4. statistical trend based on ranked 3/6/12-month returns and linear-regression slope t-statistics;
5. PCA residual momentum after removing the first two common return components;
6. a defensive macro sleeve rotating across equities, bonds, gold, commodities and USD proxies.

Allocation uses Ledoit-Wolf covariance shrinkage both within sleeves and across sleeves. Each asset is capped at 25%, aggregate crypto exposure is capped at 15%, cash is preserved when the volatility target requires less exposure, and explicit transaction costs are charged.

## Point-in-time protocol

- maximum available adjusted daily history is downloaded, then converted to monthly observations;
- research range: January 1993 through July 2026;
- universe: SPY, QQQ, IWM, VEA, EEM, VNQ, IEF, TLT, GLD, DBC, UUP, BTC-USD and ETH-USD;
- an asset is invisible until it has accumulated approximately two years of price history;
- every monthly model is trained only on labels ending before the signal date;
- 1/3/6-month targets use horizon-specific purging plus a one-month embargo;
- positions are applied to the following month, never to the signal month;
- modifying all future prices is tested not to change any historical portfolio weight.

The anti-lookahead audit passes on every generated sleeve and is enforced in CI on Python 3.11 and 3.12.

## Final walk-forward result

| Metric | Full OOS stream | Final 36-month holdout |
|---|---:|---:|
| Sharpe | 0.834 | **1.247** |
| Annual return | 5.40% | 9.06% |
| Annual volatility | 6.58% | 7.18% |
| Maximum drawdown | -19.05% | **-3.20%** |
| Total return | 324.70% | 29.72% |
| Probabilistic Sharpe probability | 100.00% | 97.99% |
| Deflated Sharpe probability | 97.63% | **44.53%** |

Average monthly turnover is 12.93%.

## Sleeve diagnostics

| Sleeve | Average allocation | Standalone Sharpe | Max drawdown |
|---|---:|---:|---:|
| 1-month ML | 12.52% | 0.529 | -27.13% |
| 3-month ML | 14.52% | 0.567 | -26.96% |
| 6-month ML | 10.51% | 0.522 | -34.16% |
| Statistical trend | 25.42% | 0.922 | -17.58% |
| PCA residual momentum | 15.13% | 0.753 | -24.26% |
| Macro defensive risk parity | 21.90% | **1.085** | **-10.22%** |

## Robustness

| Scenario | Sharpe | Total return | Max drawdown |
|---|---:|---:|---:|
| Base | 0.834 | 324.70% | -19.05% |
| Double transaction costs | 0.814 | 310.05% | -19.18% |
| One extra month of execution delay | 0.813 | 325.49% | -20.33% |
| Remove all crypto exposure | 0.702 | 172.71% | -19.05% |

Block-bootstrap Monte Carlo on the final holdout, 5,000 paths and six-month blocks:

- median terminal return: 29.71%;
- fifth-percentile terminal return: 10.28%;
- first-percentile terminal return: 2.85%;
- fifth-percentile Sharpe: 0.478;
- probability of a negative terminal return: 0.28%;
- 99th-percentile maximum-drawdown magnitude: 10.51%.

## Why it is not accepted

The final holdout Sharpe is 1.247, not 1.5. More importantly, the cross-sectional permutation p-value is 0.577 and the holdout Deflated Sharpe probability is only 44.53% after accounting for the declared research trial count.

This means the portfolio is a credible low-volatility diversified allocation, but the available evidence does not prove a persistent alpha-generating ranking model.

## What should be added next

More price-only tuning would increase overfitting risk. The next defensible research step requires genuinely different data:

- continuous futures data with actual carry and roll yield;
- crypto perpetual funding, basis and open-interest history;
- point-in-time macro releases rather than revised macro series;
- options-implied volatility, skew and term structure;
- a survivorship-free security universe including delisted assets;
- paper-trading execution records before any capital allocation.

Until those inputs exist, this model is suitable for research and paper trading only.
