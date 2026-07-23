from __future__ import annotations

from typing import Any

import pandas as pd

import lse_institutional_campaign as campaign


def paged_candles(client: Any, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    """Page LSE candles using the endpoint's exact date contract.

    Daily and slower endpoints accept only YYYY-MM-DD. Intraday endpoints retain
    full ISO timestamps. Pagination advances strictly beyond the last returned bar.
    """
    pages: list[pd.DataFrame] = []
    cursor = pd.Timestamp(start, tz="UTC")
    end_timestamp = pd.Timestamp(end, tz="UTC")
    daily = timeframe in {"1d", "1w", "1mo"}
    step = pd.Timedelta(days=1) if daily else pd.Timedelta(seconds=1)
    for _ in range(30):
        start_arg = cursor.date().isoformat() if daily else cursor.isoformat()
        end_arg = end_timestamp.date().isoformat() if daily else end_timestamp.isoformat()
        rows = campaign.safe_call(
            client.candles,
            symbol,
            timeframe,
            start=start_arg,
            end=end_arg,
            limit=5000,
            order="asc",
        )
        page = campaign.standardise_candles(rows)
        if page.empty:
            break
        pages.append(page)
        last = pd.Timestamp(page.index.max())
        last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
        if last >= end_timestamp or len(page) < 5000:
            break
        cursor = last.normalize() + step if daily else last + step
    if not pages:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(pages).sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]


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


campaign.paged_candles = paged_candles
campaign.cot_relative_value = cot_relative_value

if __name__ == "__main__":
    raise SystemExit(campaign.main())
