from __future__ import annotations

from typing import Any

import manifold_research as engine
import run_manifold_research as runner


def mean_reversion_families() -> list[dict[str, Any]]:
    """Single-asset long/short mean reversion using only supported MCP indicators."""
    return [
        {
            "name": "kalman_innovation_reversion",
            "signals": {
                "log_price": "log(close)",
                "state": "kalman(log_price)",
                "innovation": "log_price - state",
                "state_slope": "linreg_slope(state, 48)",
                "abs_slope": "abs_val(state_slope)",
                "volatility": "natr(14)",
            },
            "size": "when((innovation < (0.0 - param('entry', default=0.04))) & (abs_slope < param('slope_limit', default=0.002)) & (volatility < param('max_natr', default=9.0)), param('size', default=0.10), when((innovation > param('entry', default=0.04)) & (abs_slope < param('slope_limit', default=0.002)) & (volatility < param('max_natr', default=9.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "entry": [0.02, 0.04, 0.07, 0.10],
                "slope_limit": [0.001, 0.003],
                "max_natr": [7.0, 11.0],
                "size": [0.08, 0.12],
            },
            "heat": ("entry", [0.015, 0.025, 0.04, 0.055, 0.075, 0.10], "slope_limit", [0.0005, 0.001, 0.002, 0.004, 0.008]),
            "stability": ("max_natr", [5.0, 7.0, 9.0, 11.0, 14.0]),
        },
        {
            "name": "rolling_median_range_reversion",
            "signals": {
                "median": "rolling_median(close, 48)",
                "deviation": "close / (median + 0.000000000001) - 1.0",
                "oscillator": "rsi(close, param('rsi_period', default=10))",
                "trend_strength": "adx(14)",
                "volatility": "natr(14)",
            },
            "size": "when((deviation < (0.0 - param('entry', default=0.05))) & (oscillator < param('rsi_low', default=28.0)) & (trend_strength < param('max_adx', default=28.0)) & (volatility < param('max_natr', default=10.0)), param('size', default=0.10), when((deviation > param('entry', default=0.05)) & (oscillator > param('rsi_high', default=72.0)) & (trend_strength < param('max_adx', default=28.0)) & (volatility < param('max_natr', default=10.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "rsi_period": [7, 14],
                "entry": [0.03, 0.06, 0.10],
                "rsi_low": [22.0, 30.0],
                "rsi_high": [70.0, 78.0],
                "max_adx": [22.0, 32.0],
                "max_natr": [8.0],
                "size": [0.10],
            },
            "heat": ("entry", [0.02, 0.03, 0.045, 0.06, 0.08, 0.10], "max_adx", [18, 22, 26, 30, 36]),
            "stability": ("rsi_period", [5, 7, 10, 14, 21]),
        },
        {
            "name": "kama_cci_reversion",
            "signals": {
                "anchor": "kama(close, param('anchor_period', default=30))",
                "deviation": "close / (anchor + 0.000000000001) - 1.0",
                "commodity_channel": "cci(20)",
                "anchor_slope": "linreg_slope(anchor, 48)",
                "abs_slope": "abs_val(anchor_slope)",
                "volatility": "natr(14)",
            },
            "size": "when((deviation < (0.0 - param('entry', default=0.045))) & (commodity_channel < (0.0 - param('cci_level', default=120.0))) & (abs_slope < param('slope_limit', default=2.0)) & (volatility < param('max_natr', default=10.0)), param('size', default=0.10), when((deviation > param('entry', default=0.045)) & (commodity_channel > param('cci_level', default=120.0)) & (abs_slope < param('slope_limit', default=2.0)) & (volatility < param('max_natr', default=10.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "anchor_period": [20, 40],
                "entry": [0.025, 0.05, 0.085],
                "cci_level": [90.0, 140.0],
                "slope_limit": [1.0, 3.0],
                "max_natr": [8.0, 12.0],
                "size": [0.10],
            },
            "heat": ("entry", [0.015, 0.025, 0.04, 0.055, 0.07, 0.09], "anchor_period", [10, 20, 30, 40, 60]),
            "stability": ("cci_level", [70, 90, 110, 140, 180]),
        },
        {
            "name": "liquidity_exhaustion_snapback",
            "signals": {
                "short_return": "roc(close, param('horizon', default=3))",
                "return_anchor": "sma(short_return, param('window', default=96))",
                "shock": "short_return - return_anchor",
                "money_flow": "mfi(14)",
                "volatility": "natr(14)",
                "rebound": "roc(close, 1)",
            },
            "size": "when((shock < (0.0 - param('entry', default=0.08))) & (money_flow < param('mfi_low', default=25.0)) & (volatility > param('min_natr', default=3.0)) & (rebound > (0.0 - param('rebound_floor', default=0.03))), param('size', default=0.08), when((shock > param('entry', default=0.08)) & (money_flow > param('mfi_high', default=75.0)) & (volatility > param('min_natr', default=3.0)) & (rebound < param('rebound_floor', default=0.03)), 0.0 - param('size', default=0.08), 0.0))",
            "grid": {
                "horizon": [2, 6],
                "window": [72, 144],
                "entry": [0.05, 0.09, 0.14],
                "mfi_low": [20.0, 30.0],
                "mfi_high": [70.0, 80.0],
                "min_natr": [2.0, 5.0],
                "rebound_floor": [0.03],
                "size": [0.08],
            },
            "heat": ("entry", [0.03, 0.05, 0.07, 0.09, 0.12, 0.15], "min_natr", [1, 2, 3, 4, 6]),
            "stability": ("window", [48, 72, 96, 120, 168, 240]),
        },
        {
            "name": "compression_regime_fade",
            "signals": {
                "anchor": "ema(close, param('anchor_period', default=72))",
                "deviation": "close / (anchor + 0.000000000001) - 1.0",
                "bandwidth": "bollinger_width(close, 20, 2.0)",
                "bandwidth_anchor": "sma(bandwidth, 96)",
                "bandwidth_ratio": "bandwidth / (bandwidth_anchor + 0.000000000001)",
                "oscillator": "rsi(close, param('rsi_period', default=10))",
                "trend_strength": "adx(14)",
            },
            "size": "when((deviation < (0.0 - param('entry', default=0.04))) & (bandwidth_ratio < param('max_width_ratio', default=1.25)) & (oscillator < param('rsi_low', default=30.0)) & (trend_strength < param('max_adx', default=26.0)), param('size', default=0.10), when((deviation > param('entry', default=0.04)) & (bandwidth_ratio < param('max_width_ratio', default=1.25)) & (oscillator > param('rsi_high', default=70.0)) & (trend_strength < param('max_adx', default=26.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "anchor_period": [48, 120],
                "rsi_period": [7, 14],
                "entry": [0.025, 0.05, 0.085],
                "max_width_ratio": [0.9, 1.3],
                "rsi_low": [25.0],
                "rsi_high": [75.0],
                "max_adx": [22.0, 30.0],
                "size": [0.10],
            },
            "heat": ("entry", [0.015, 0.025, 0.04, 0.055, 0.07, 0.09], "max_width_ratio", [0.7, 0.9, 1.1, 1.3, 1.6]),
            "stability": ("anchor_period", [24, 48, 72, 96, 144]),
        },
    ]


def main() -> int:
    engine.INTERVALS = ("4h",)
    runner.compiler_safe_families = mean_reversion_families
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
