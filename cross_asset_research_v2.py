from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

import cross_asset_research as engine


def download_yahoo(symbol: str, cache_dir: Path, start: str, retries: int = 5) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}.yahoo.csv"
    if not path.exists() or path.stat().st_size < 100:
        start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
        end_ts = int((pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            "period1": start_ts,
            "period2": end_ts,
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true",
        }
        headers = {
            "User-Agent": "Mozilla/5.0 alpha-research/1.0",
            "Accept": "application/json,text/plain,*/*",
        }
        last: Exception | None = None
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=45)
                response.raise_for_status()
                chart = response.json().get("chart") or {}
                if chart.get("error"):
                    raise RuntimeError(str(chart["error"]))
                result = (chart.get("result") or [None])[0]
                if not result:
                    raise RuntimeError("empty Yahoo chart result")
                timestamps = result.get("timestamp") or []
                quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
                adjusted = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
                frame = pd.DataFrame({
                    "Date": pd.to_datetime(timestamps, unit="s", utc=True),
                    "Open": quote.get("open"),
                    "High": quote.get("high"),
                    "Low": quote.get("low"),
                    "Close": quote.get("close"),
                    "Volume": quote.get("volume"),
                    "Adjusted": adjusted,
                })
                ratio = frame["Adjusted"].div(frame["Close"]).replace([np.inf, -np.inf], np.nan)
                for column in ("Open", "High", "Low", "Close"):
                    frame[column] = frame[column] * ratio
                frame = frame.drop(columns=["Adjusted"])
                normalized = engine.normalize_ohlcv(frame, symbol)
                if len(normalized) < 500:
                    raise RuntimeError(f"only {len(normalized)} Yahoo rows")
                normalized.reset_index().to_csv(path, index=False)
                break
            except Exception as exc:  # pragma: no cover - network path
                last = exc
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Yahoo download failed for {symbol}: {last}")
    return engine.normalize_ohlcv(pd.read_csv(path), symbol).loc[pd.Timestamp(start):]


def load_market_data(
    symbols: Iterable[str],
    start: str,
    cache_dir: Path,
    lse_dir: Path | None = None,
    prefer_lse: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    api_key = os.getenv("LSE_API_KEY", "").strip()
    data: dict[str, pd.DataFrame] = {}
    provenance: dict[str, Any] = {"requested": list(symbols), "loaded": {}, "failed": {}}
    for symbol in symbols:
        errors: list[str] = []
        frame: pd.DataFrame | None = None
        source = ""
        try:
            if lse_dir:
                frame = engine.load_lse_export(lse_dir, symbol)
                if frame is not None:
                    source = "london-strategic-edge-export"
        except Exception as exc:
            errors.append(f"LSE export: {exc}")
        if frame is None and prefer_lse and api_key:
            try:
                frame = engine.load_lse_client(symbol, start, api_key)
                source = "london-strategic-edge-api"
            except Exception as exc:
                errors.append(f"LSE API: {exc}")
        if frame is None:
            try:
                frame = download_yahoo(symbol, cache_dir, start)
                source = "yahoo-public-adjusted"
            except Exception as exc:
                errors.append(str(exc))
        if frame is None:
            try:
                frame = engine.download_stooq(symbol, cache_dir, start)
                source = "stooq-keyless-fallback"
            except Exception as exc:
                errors.append(str(exc))
        if frame is None or len(frame) < 500:
            provenance["failed"][symbol] = errors or ["insufficient rows"]
            continue
        data[symbol] = frame
        cache_candidates = [cache_dir / f"{symbol}.yahoo.csv", cache_dir / f"{symbol}.csv"]
        cache_path = next((path for path in cache_candidates if path.exists()), None)
        provenance["loaded"][symbol] = {
            "source": source,
            "rows": len(frame),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "sha256": engine.stable_hash(cache_path) if cache_path else None,
        }
    if "SPY" not in data:
        raise RuntimeError(f"SPY unavailable: {provenance['failed'].get('SPY')}")
    if len(data) < 24:
        raise RuntimeError(f"insufficient universe breadth: {len(data)} loaded; failures={provenance['failed']}")
    return data, provenance


engine.load_market_data = load_market_data

if __name__ == "__main__":
    raise SystemExit(engine.main())
