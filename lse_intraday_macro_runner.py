from __future__ import annotations

from typing import Any

import pandas as pd

import lse_intraday_macro_alpha as campaign
from lse_institutional_campaign import standardise_candles
from lse_vault import safe_call

BASE_STANDARDISE_CALENDAR = campaign.standardise_calendar


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


def release_timestamp(index_value: Any, date_value: Any, time_value: Any) -> pd.Timestamp:
    """Combine LSE's separate release date and UTC time fields exactly."""
    date_source = date_value if pd.notna(date_value) else index_value
    day = pd.to_datetime(date_source, errors="coerce", utc=True)
    if pd.isna(day):
        return pd.NaT
    if time_value is None or pd.isna(time_value) or not str(time_value).strip():
        return pd.Timestamp(day)
    time_text = str(time_value).strip()
    upper = time_text.upper()
    if "AM" not in upper and "PM" not in upper and time_text.count(":") == 1:
        time_text += ":00"
    combined = pd.to_datetime(
        f"{pd.Timestamp(day).date()} {time_text}",
        errors="coerce",
        utc=True,
    )
    return pd.Timestamp(combined) if not pd.isna(combined) else pd.Timestamp(day)


def standardise_calendar_exact(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build exact UTC release timestamps before computing event surprises."""
    corrected: dict[str, pd.DataFrame] = {}
    for region, frame in frames.items():
        if frame.empty:
            corrected[region] = frame
            continue
        date_column = campaign._find_column(frame, ("datetime", "date", "day"))
        time_column = campaign._find_column(frame, ("time", "release_time"))
        timestamps = [
            release_timestamp(
                index_value,
                row[date_column] if date_column else index_value,
                row[time_column] if time_column else None,
            )
            for index_value, row in frame.iterrows()
        ]
        updated = frame.copy()
        updated.index = pd.DatetimeIndex(timestamps)
        updated = updated.loc[updated.index.notna()].sort_index()
        corrected[region] = updated
    return BASE_STANDARDISE_CALENDAR(corrected)


campaign.paged_candles = date_window_candles
campaign.standardise_calendar = standardise_calendar_exact

if __name__ == "__main__":
    raise SystemExit(campaign.main())
