from __future__ import annotations

from typing import Any

import pandas as pd

import lse_intraday_macro_alpha as campaign
from lse_institutional_campaign import standardise_candles
from lse_vault import safe_call


def date_window_candles(
    client: Any,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    max_pages: int = 80,
) -> pd.DataFrame:
    """Page intraday history with date-only windows accepted by the LSE API.

    A 45-calendar-day window contains fewer than 5,000 15-minute bars for
    exchange-traded and weekday FX markets, so no timestamp cursor is needed.
    """
    pages: list[pd.DataFrame] = []
    cursor = pd.Timestamp(start).normalize()
    end_timestamp = pd.Timestamp(end).normalize()
    for _ in range(max_pages):
        if cursor > end_timestamp:
            break
        window_end = min(cursor + pd.Timedelta(days=44), end_timestamp)
        rows = safe_call(
            client.candles,
            symbol,
            timeframe,
            start=cursor.date().isoformat(),
            end=window_end.date().isoformat(),
        )
        page = campaign.ensure_utc_index(standardise_candles(rows))
        if len(page) >= 5000:
            raise RuntimeError(
                f"{symbol} {timeframe}: a 45-day window reached the 5,000-row cap; "
                "reduce the date window before trusting the history"
            )
        if not page.empty:
            pages.append(page)
        cursor = window_end + pd.Timedelta(days=1)
    if cursor <= end_timestamp:
        raise RuntimeError(f"{symbol} {timeframe}: max_pages exhausted before {end_timestamp.date()}")
    if not pages:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(pages).sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]


campaign.paged_candles = date_window_candles

if __name__ == "__main__":
    raise SystemExit(campaign.main())
