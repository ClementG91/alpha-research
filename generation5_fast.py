from __future__ import annotations

import asyncio
import traceback

import generation5

# Every validated leader so far came from MAJORS5 on 4h bars. Keep the final
# search focused enough to finish quickly while retaining meaningful parameter
# neighbourhoods and a frozen out-of-sample period.
generation5.INTERVALS = ("4h",)
generation5.UNIVERSES = [
    {"name": "MAJORS5", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]},
]
generation5.TOP_TRAIN = 6
generation5.MAX_FINALISTS = 10

for name, spec in generation5.STRATEGIES.items():
    spec["grid"]["mom_period"] = [7, 14, 28]
    spec["grid"]["smooth"] = [3, 6]
    spec["grid"]["vol_window"] = [10, 14]
    spec["grid"]["entry"] = [0.75, 1.0, 1.5]
    if "regime" in spec["grid"]:
        spec["grid"]["regime"] = [72, 168]


if __name__ == "__main__":
    try:
        asyncio.run(generation5.main())
    except Exception:
        generation5.OUT.mkdir(exist_ok=True)
        (generation5.OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
