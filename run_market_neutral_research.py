from __future__ import annotations

from typing import Any

import manifold_research as engine
import run_manifold_research as runner


_ORIGINAL_CFG = engine.cfg
_ORIGINAL_FAMILIES = runner.compiler_safe_families
_SYMBOL_IDS = {
    "BTCUSDT": 1,
    "ETHUSDT": 2,
    "SOLUSDT": 3,
    "BNBUSDT": 4,
    "XRPUSDT": 5,
    "ADAUSDT": 6,
    "DOGEUSDT": 7,
    "LINKUSDT": 10,
    "LTCUSDT": 11,
    "TRXUSDT": 12,
}


def manifold_cross_asset_cfg(
    start: str,
    end: str,
    interval: str,
    *,
    slippage_bps: float = 3.0,
    delay: int = 1,
) -> dict[str, Any]:
    config = _ORIGINAL_CFG(
        start,
        end,
        interval,
        slippage_bps=slippage_bps,
        delay=delay,
    )
    config["universe"] = list(_SYMBOL_IDS.values())
    config["symbol_names"] = dict(_SYMBOL_IDS)
    return config


def compact_families() -> list[dict[str, Any]]:
    """A disciplined first pass: 12 structural combinations per family."""
    docs = _ORIGINAL_FAMILIES()
    grids = {
        "btc_ratio_deviation_reversion": {
            "ratio_window": [72, 144],
            "resid_window": [96],
            "horizon": [6, 18],
            "entry_dev": [0.035, 0.065, 0.10],
            "confirm_dev": [0.025],
            "max_natr": [8.0],
        },
        "dual_anchor_kalman_reversion": {
            "horizon": [6, 18],
            "btc_weight": [0.45, 0.70],
            "entry_dev": [0.025, 0.05, 0.085],
            "slope_limit": [0.002],
            "max_natr": [9.0],
        },
        "liquidity_shock_snapback": {
            "shock_horizon": [2, 6],
            "shock_window": [72, 144],
            "confirm_horizon": [2],
            "entry_dev": [0.04, 0.075, 0.12],
            "min_natr": [3.0],
            "rebound_floor": [0.025],
        },
        "relative_rsi_reversion": {
            "rsi_period": [7, 14],
            "ratio_window": [72, 168],
            "rsi_low": [25.0],
            "rsi_high": [75.0],
            "entry_dev": [0.025, 0.05, 0.085],
            "max_natr": [9.0],
        },
        "multi_horizon_residual_fade": {
            "fast_horizon": [3, 9],
            "mid_horizon": [18, 48],
            "fast_window": [96],
            "mid_window": [168],
            "fast_entry": [0.03, 0.065, 0.10],
            "mid_limit": [0.08],
            "max_natr": [10.0],
        },
    }
    for doc in docs:
        doc["grid"] = grids[doc["name"]]
    return docs


def main() -> int:
    engine.cfg = manifold_cross_asset_cfg
    engine.INTERVALS = ("4h",)
    runner.compiler_safe_families = compact_families
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
