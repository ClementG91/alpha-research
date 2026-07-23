# Crypto alpha research results

## Conclusion

No strategy reached the required **net out-of-sample Sharpe >= 1.5** without failing the robustness gates.

The strongest reproducible candidate is `ema_return_vol_overlay_verified`:

- universe: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT;
- timeframe: 4h;
- signal: EMA(8) versus EMA(168);
- target exposure: +15% per asset in a bullish regime, -15% in a bearish regime;
- volatility overlay: calculate the 14-bar standard deviation of returns and its 72-bar z-score;
- when volatility z-score > 1, keep bullish exposure at +15% but close bearish exposure;
- Binance-perpetual fee schedule, 2 bps slippage and one-bar execution delay.

## Frozen-period metrics

| Period | Sharpe | Return | Max drawdown |
|---|---:|---:|---:|
| Train 2021-2023 | 1.285 | — | — |
| Validation 2024 | 0.934 | — | — |
| OOS 2025-2026-07-01 | **1.318** | **61.47%** | **-25.15%** |
| 2025 only | 1.957 | 62.56% | -11.82% |
| 2026 YTD | -0.253 | -2.55% | -10.91% |

Additional OOS metrics:

- alpha: 0.3511;
- Sortino: 2.053;
- Calmar: 1.502;
- 552 trades;
- neighbouring-parameter median Sharpe: 1.314;
- OOS Sharpe t-stat: 1.612.

## Monte Carlo

Block bootstrap, 1,000 paths, block size 24:

- mean final return: 71.18%;
- minimum simulated final return: -41.95%;
- probability of ruin reported by the engine: 6.5%;
- probability of wipeout: 0%;
- 95th-percentile maximum drawdown magnitude: 31.90%;
- 99th-percentile maximum drawdown magnitude: 40.21%.

## Why it is not validated

The candidate is locally stable and improves the original EMA core, but its OOS Sharpe is 1.318 rather than 1.5. Performance is also regime-dependent: strong in 2025 and negative in 2026 YTD. It should therefore remain a research or paper-trading candidate, not be presented as a proven alpha >= 1.5.

## Research coverage

The repository contains seven generations covering:

- trend following and Donchian breakouts;
- stateful long/short signals;
- multi-strategy portfolios;
- BTC-regime filters;
- Kalman, ADX, MFI, SuperTrend and linear-regression filters;
- UTC session, weekday/weekend and turn-of-month effects;
- volatility-adjusted momentum;
- asymmetric long/short horizons and position sizes;
- native volatility-state exposure overlays.

## Data limitations

The remote datastore uses Binance spot bars as the price proxy while applying Binance-perpetual fees. The funding-rate column is unavailable, so funding P&L is disabled. A production decision requires true perpetual OHLCV/funding data and paper trading.
