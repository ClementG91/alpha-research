from __future__ import annotations

from typing import Any

import pandas as pd

import lse_institutional_campaign as campaign


def cot_relative_value(
    cot: dict[str, pd.DataFrame],
    futures: dict[str, pd.DataFrame],
    cost_bps: float = 2.0,
    delay: int = 1,
) -> campaign.StrategyResult | None:
    """Corrected COT adapter that never evaluates a DataFrame as a boolean."""
    signals: dict[str, pd.Series] = {}
    closes: dict[str, pd.Series] = {}
    for root, frame in cot.items():
        signal = campaign._cot_signal(frame)
        price = futures.get(root)
        if price is None:
            price = next((value for symbol, value in futures.items() if symbol.startswith(root)), None)
        if signal is not None and price is not None:
            signals[root] = signal
            closes[root] = price["close"]
    if len(signals) < 4:
        return None
    close = pd.concat(closes, axis=1).sort_index().ffill(limit=3)
    returns = close.pct_change(fill_method=None)
    signal = pd.concat(signals, axis=1).resample("W-FRI").last().reindex(close.index).ffill(limit=7)
    positions = campaign._normalise(signal.rank(axis=1, pct=True) - 0.5).shift(delay).fillna(0.0)
    turnover = positions.diff().fillna(positions).abs().sum(axis=1)
    gross = (positions * returns).sum(axis=1, min_count=1).fillna(0.0)
    return campaign.StrategyResult(
        "cot_commercial_pressure",
        gross - turnover * cost_bps / 10_000.0,
        positions,
        turnover,
    )


campaign.cot_relative_value = cot_relative_value

if __name__ == "__main__":
    raise SystemExit(campaign.main())
