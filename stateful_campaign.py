from __future__ import annotations

import asyncio
import traceback

import campaign

campaign.INTERVALS = ("4h", "12h", "1d")
campaign.TOP_TRAIN = 6
campaign.MAX_FINALISTS = 15
campaign.CALL_INTERVAL_SECONDS = 1.6
campaign.UNIVERSES = [
    {"name": "BTCUSDT", "tickers": ["BTCUSDT"]},
    {"name": "ETHUSDT", "tickers": ["ETHUSDT"]},
    {"name": "SOLUSDT", "tickers": ["SOLUSDT"]},
    {"name": "MAJORS5", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]},
    {"name": "BROAD10", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "LTCUSDT", "TRXUSDT"]},
]
campaign.STRATEGIES = {
    "ema_flip_hold": {
        "signals": {
            "fast": "ema(close, param('fast', default=24))",
            "slow": "ema(close, param('slow', default=168))",
        },
        "size": "when(crossover(fast, slow), 0.40, when(crossunder(fast, slow), -0.40, hold()))",
        "grid": {
            "fast": [8, 12, 24, 36, 48],
            "slow": [72, 96, 168, 240, 360],
        },
    },
    "ema_long_cash_hold": {
        "signals": {
            "fast": "ema(close, param('fast', default=24))",
            "slow": "ema(close, param('slow', default=168))",
        },
        "size": "when(crossover(fast, slow), 0.45, when(crossunder(fast, slow), 0.0, hold()))",
        "grid": {
            "fast": [8, 12, 24, 36, 48],
            "slow": [72, 96, 168, 240, 360],
        },
    },
    "donchian_flip_hold": {
        "signals": {
            "upper": "highest(close, param('breakout', default=96))",
            "lower": "lowest(close, param('breakout', default=96))",
            "regime": "ema(close, param('regime', default=168))",
        },
        "size": "when((close >= upper) & (close > regime), 0.40, when((close <= lower) & (close < regime), -0.40, hold()))",
        "grid": {
            "breakout": [24, 48, 72, 96, 120, 168],
            "regime": [72, 96, 168, 240],
        },
    },
    "regime_reversal_hold": {
        "signals": {
            "osc": "rsi(close, param('rsi_period', default=10))",
            "regime": "ema(close, param('regime', default=168))",
        },
        "size": "when((close > regime) & (osc < 28), 0.32, when((close < regime) & (osc > 72), -0.32, when((close > regime) & (osc > 58), 0.0, when((close < regime) & (osc < 42), 0.0, hold()))))",
        "grid": {
            "rsi_period": [5, 7, 10, 14, 21],
            "regime": [72, 96, 168, 240, 360],
        },
    },
}


if __name__ == "__main__":
    try:
        asyncio.run(campaign.main())
    except Exception:
        campaign.OUT.mkdir(exist_ok=True)
        (campaign.OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
