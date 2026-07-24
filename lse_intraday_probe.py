from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from lse_intraday_macro_alpha import (
    FX_CORE,
    FX_EXTERNAL,
    REGIONS,
    choose_future_symbols,
    ensure_utc_index,
)
from lse_intraday_macro_runner import date_window_candles
from lse_vault import assert_secret_absent, official_client, rows_to_frame, safe_call, sanitise_payload, write_json


def describe_frame(frame: Any) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
        "first": str(frame.index.min()) if len(frame) else None,
        "last": str(frame.index.max()) if len(frame) else None,
    }


def sample_records(frame: Any, columns: tuple[str, ...], rows: int = 5) -> list[dict[str, Any]]:
    selected = [column for column in columns if column in frame.columns]
    if frame.empty or not selected:
        return []
    sample = frame[selected].reset_index().head(rows)
    return json.loads(sample.to_json(orient="records", date_format="iso"))


def run(output: Path) -> dict[str, Any]:
    client = official_client()
    report: dict[str, Any] = {
        "candles": {},
        "calendars": {},
        "calendar_samples": {},
        "yields": {},
        "yield_samples": {},
        "errors": [],
    }
    try:
        catalog_payload = safe_call(client.catalog)
        catalog = (
            list(catalog_payload)
            if not isinstance(catalog_payload, dict)
            else list(catalog_payload.get("data", catalog_payload.get("rows", [])))
        )
        futures = choose_future_symbols(catalog)
        report["futures_mapping"] = futures
        symbols = list(FX_CORE + FX_EXTERNAL) + list(futures.values()) + ["SPY"]
        for symbol in symbols:
            try:
                frame = date_window_candles(
                    client,
                    symbol,
                    "15m",
                    "2026-01-01",
                    "2026-02-14",
                    max_pages=2,
                )
                report["candles"][symbol] = describe_frame(frame)
            except Exception as exc:
                report["errors"].append({"dataset": f"candles:{symbol}", "error": str(exc)})
        for region in REGIONS:
            try:
                frame = ensure_utc_index(
                    rows_to_frame(
                        safe_call(
                            client.economic_calendar,
                            region=region,
                            start="2025-01-01",
                            end="2026-07-24",
                        )
                    )
                )
                report["calendars"][region] = describe_frame(frame)
                report["calendar_samples"][region] = sample_records(
                    frame,
                    ("time", "datetime", "region_code", "event", "actual", "forecast", "consensus", "created_at", "updated_at"),
                )
            except Exception as exc:
                report["errors"].append({"dataset": f"calendar:{region}", "error": str(exc)})
        for series in ("US2Y", "US10Y", "DE2Y", "GB2Y", "JP2Y"):
            try:
                frame = ensure_utc_index(
                    rows_to_frame(
                        safe_call(client.bond_yields, series, start="2020-01-01", end="2026-07-24")
                    )
                )
                report["yields"][series] = describe_frame(frame)
                report["yield_samples"][series] = sample_records(frame, ("value", "yield", "rate", "close"), rows=3)
            except Exception as exc:
                report["errors"].append({"dataset": f"yield:{series}", "error": str(exc)})
    finally:
        close = getattr(client, "close", None) or getattr(client, "disconnect", None)
        if callable(close):
            close()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "probe.json", sanitise_payload(report))
    assert_secret_absent(output)
    return report


def main() -> int:
    output = Path("results/intraday_macro/probe")
    try:
        report = run(output)
        print(json.dumps(report, indent=2, default=str))
        return 0
    except Exception:
        output.mkdir(parents=True, exist_ok=True)
        (output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        assert_secret_absent(output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
