from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import lse_vault
from lse_campaign_runner import paged_candles
from lse_institutional_campaign import standardise_candles


class FakeCandleClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def candles(self, symbol: str, timeframe: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"symbol": symbol, "timeframe": timeframe, **kwargs})
        return [
            {
                "ts": "2026-01-02 00:00:00.000000",
                "open": 10,
                "high": 12,
                "low": 9,
                "close": 11,
                "volume": 100,
            }
        ]


def test_secret_scanner_rejects_key_like_material(tmp_path: Path) -> None:
    (tmp_path / "bad.txt").write_text("lse_" + "live_example_not_a_real_key", encoding="utf-8")
    with pytest.raises(RuntimeError):
        lse_vault.assert_secret_absent(tmp_path)


def test_redaction_masks_key() -> None:
    value = lse_vault.redact("request failed for " + "lse_" + "live_example_123")
    assert "lse_live_" not in value
    assert "REDACTED" in value


def test_candle_payload_is_standardised_without_future_fill() -> None:
    rows = [
        {"ts": "2026-01-01 00:00:00.000000", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 100},
        {"ts": "2026-01-02 00:00:00.000000", "open": 11, "high": 13, "low": 10, "close": 12, "volume": 120},
    ]
    frame = standardise_candles(rows)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert frame.index.is_monotonic_increasing
    assert frame.iloc[-1]["close"] == 12


def test_daily_candle_request_uses_plain_dates() -> None:
    client = FakeCandleClient()
    frame = paged_candles(client, "ES.F", "1d", "2004-01-01", "2026-07-24")
    assert len(frame) == 1
    assert client.calls == [
        {
            "symbol": "ES.F",
            "timeframe": "1d",
            "start": "2004-01-01",
            "end": "2026-07-24",
            "limit": 5000,
            "order": "asc",
        }
    ]
    assert "T" not in client.calls[0]["start"]
    assert "T" not in client.calls[0]["end"]


def test_frame_hash_is_deterministic() -> None:
    frame = pd.DataFrame({"value": [1.0, 2.0]}, index=pd.date_range("2020-01-01", periods=2))
    assert lse_vault.frame_sha256(frame) == lse_vault.frame_sha256(frame.copy())
