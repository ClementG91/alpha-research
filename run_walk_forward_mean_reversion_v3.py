from __future__ import annotations

from typing import Any

import run_walk_forward_mean_reversion_v2 as calibrated


def conditional_families() -> list[dict[str, Any]]:
    """Mean reversion with bounded GARCH regime sizing."""
    common_risk = {
        "conditional_vol": "garch(close, 0.000001, 0.10, 0.85)",
        "vol_anchor": "ema(conditional_vol, 72)",
        "vol_ratio": "conditional_vol / (vol_anchor + 0.000000001)",
        "risk_multiplier": "when(vol_ratio > param('risk_cut', default=1.25), 0.50, 1.0)",
        "risk_size": "param('size', default=0.08) * risk_multiplier",
    }
    return [
        {
            "name": "garch_shock_fade",
            "signals": {
                **common_risk,
                "shock": "roc(close, param('horizon', default=3))",
                "range_vol": "natr(14)",
            },
            "size": "when((shock < (0.0 - param('entry', default=0.04))) & (range_vol > param('min_natr', default=0.0)), risk_size, when((shock > param('entry', default=0.04)) & (range_vol > param('min_natr', default=0.0)), 0.0 - risk_size, 0.0))",
            "grid": {
                "horizon": [2, 4, 6], "entry": [0.015, 0.03, 0.06],
                "min_natr": [0.0, 2.0], "risk_cut": [1.0, 1.5],
                "size": [0.08],
            },
        },
        {
            "name": "garch_kalman_fade",
            "signals": {
                **common_risk,
                "log_price": "log(close)",
                "state": "kalman(log_price)",
                "innovation": "log_price - state",
            },
            "size": "when(innovation < (0.0 - param('entry', default=0.015)), risk_size, when(innovation > param('entry', default=0.015), 0.0 - risk_size, 0.0))",
            "grid": {
                "entry": [0.003, 0.008, 0.015, 0.03],
                "risk_cut": [1.0, 1.5], "size": [0.08, 0.12],
            },
        },
        {
            "name": "garch_median_fade",
            "signals": {
                **common_risk,
                "median": "rolling_median(close, param('window', default=48))",
                "deviation": "close / (median + 0.000000001) - 1.0",
            },
            "size": "when(deviation < (0.0 - param('entry', default=0.03)), risk_size, when(deviation > param('entry', default=0.03), 0.0 - risk_size, 0.0))",
            "grid": {
                "window": [24, 48, 96], "entry": [0.008, 0.02, 0.05],
                "risk_cut": [1.0, 1.5], "size": [0.08, 0.12],
            },
        },
        {
            "name": "garch_kama_fade",
            "signals": {
                **common_risk,
                "anchor": "kama(close, param('anchor_period', default=30))",
                "deviation": "close / (anchor + 0.000000001) - 1.0",
            },
            "size": "when(deviation < (0.0 - param('entry', default=0.03)), risk_size, when(deviation > param('entry', default=0.03), 0.0 - risk_size, 0.0))",
            "grid": {
                "anchor_period": [15, 30, 60], "entry": [0.008, 0.02, 0.05],
                "risk_cut": [1.0, 1.5], "size": [0.08, 0.12],
            },
        },
        {
            "name": "dual_garch_exhaustion",
            "signals": {
                "fast_vol": "garch(close, 0.000001, 0.18, 0.75)",
                "slow_vol": "garch(close, 0.000001, 0.05, 0.93)",
                "vol_ratio": "fast_vol / (slow_vol + 0.000000001)",
                "risk_multiplier": "when(vol_ratio > param('risk_cut', default=1.25), 0.50, 1.0)",
                "risk_size": "param('size', default=0.08) * risk_multiplier",
                "shock": "roc(close, param('horizon', default=3))",
            },
            "size": "when((shock < (0.0 - param('entry', default=0.04))) & (vol_ratio > param('min_vol_ratio', default=0.0)), risk_size, when((shock > param('entry', default=0.04)) & (vol_ratio > param('min_vol_ratio', default=0.0)), 0.0 - risk_size, 0.0))",
            "grid": {
                "horizon": [2, 4, 6], "entry": [0.015, 0.03, 0.06],
                "min_vol_ratio": [0.0, 0.8, 1.1], "risk_cut": [1.0, 1.5],
                "size": [0.08],
            },
        },
    ]


def main() -> int:
    calibrated.conditional_families = conditional_families
    return calibrated.main()


if __name__ == "__main__":
    raise SystemExit(main())
