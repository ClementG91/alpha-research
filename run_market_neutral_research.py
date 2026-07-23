from __future__ import annotations

from typing import Any

import manifold_research as engine
import run_manifold_research as runner


_ORIGINAL_CFG = engine.cfg
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


def main() -> int:
    engine.cfg = manifold_cross_asset_cfg
    return runner.main()


if __name__ == "__main__":
    raise SystemExit(main())
