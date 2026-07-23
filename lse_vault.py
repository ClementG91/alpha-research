from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

_KEY_PATTERN = re.compile(r"lse_live_[A-Za-z0-9_-]+")


class LSEConfigurationError(RuntimeError):
    pass


class LSEDataError(RuntimeError):
    pass


def redact(value: Any) -> str:
    return _KEY_PATTERN.sub("[REDACTED_LSE_KEY]", str(value))


def require_api_key() -> str:
    key = os.environ.get("LSE_API_KEY", "").strip()
    if not key:
        raise LSEConfigurationError("LSE_API_KEY is required; strict mode has no public-data fallback.")
    if not key.startswith("lse_live_"):
        raise LSEConfigurationError("LSE_API_KEY has an unexpected format.")
    return key


def official_client() -> Any:
    try:
        from lse import LSE  # type: ignore
    except ImportError as exc:
        raise LSEConfigurationError("Install the pinned lse-data package before running the LSE campaign.") from exc
    return LSE(api_key=require_api_key())


def safe_call(function: Callable[..., Any], *args: Any, retries: int = 3, **kwargs: Any) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            status = getattr(exc, "status", None)
            if status in {401, 403, 404}:
                break
            if attempt + 1 < retries:
                time.sleep(min(8.0, 2.0**attempt))
    raise LSEDataError(redact(last_error or "Unknown London Strategic Edge error")) from last_error


def rows_to_frame(rows: Any, timestamp_candidates: Iterable[str] = ("ts", "timestamp", "date", "datetime")) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame()
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    elif isinstance(rows, dict):
        for key in ("data", "rows", "results", "items"):
            if isinstance(rows.get(key), list):
                frame = pd.DataFrame(rows[key])
                break
        else:
            frame = pd.DataFrame([rows])
    else:
        frame = pd.DataFrame(list(rows))
    for candidate in timestamp_candidates:
        if candidate in frame.columns:
            parsed = pd.to_datetime(frame[candidate], errors="coerce", utc=True)
            if parsed.notna().any():
                frame[candidate] = parsed
                frame = frame.sort_values(candidate).drop_duplicates(candidate, keep="last")
                frame = frame.set_index(candidate)
                break
    return frame


def frame_sha256(frame: pd.DataFrame) -> str:
    normalised = frame.reset_index().to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S.%fZ")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def sanitise_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        cleaned = {}
        for key, value in payload.items():
            if str(key).lower() in {"api_key", "apikey", "x-api-key", "authorization", "token"}:
                cleaned[key] = "[REDACTED]"
            else:
                cleaned[key] = sanitise_payload(value)
        return cleaned
    if isinstance(payload, list):
        return [sanitise_payload(value) for value in payload]
    if isinstance(payload, str):
        return redact(payload)
    return payload


def assert_secret_absent(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size > 100_000_000:
            continue
        try:
            content = path.read_bytes()
        except OSError:
            continue
        if b"lse_live_" in content:
            raise RuntimeError(f"Secret-like material detected in artifact: {path}")


@dataclass
class DatasetRecord:
    name: str
    category: str
    rows: int
    first: str | None
    last: str | None
    sha256: str
    path: str


class VaultSnapshot:
    def __init__(self, output: Path):
        self.output = output
        self.records: list[DatasetRecord] = []

    def save_frame(self, name: str, category: str, frame: pd.DataFrame) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
        path = self.output / "data" / category / f"{safe_name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
        first = str(frame.index.min()) if len(frame) else None
        last = str(frame.index.max()) if len(frame) else None
        self.records.append(DatasetRecord(name, category, int(len(frame)), first, last, frame_sha256(frame), str(path.relative_to(self.output))))
        return path

    def save_json(self, name: str, category: str, payload: Any) -> Path:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
        path = self.output / "data" / category / f"{safe_name}.json"
        cleaned = sanitise_payload(payload)
        write_json(path, cleaned)
        rows = len(cleaned) if isinstance(cleaned, list) else 1
        self.records.append(DatasetRecord(name, category, rows, None, None, json_sha256(cleaned), str(path.relative_to(self.output))))
        return path

    def finish(self) -> dict[str, Any]:
        manifest = {
            "strict_lse_only": True,
            "datasets": [record.__dict__ for record in self.records],
            "dataset_count": len(self.records),
            "total_rows": sum(record.rows for record in self.records),
        }
        write_json(self.output / "manifest.json", manifest)
        assert_secret_absent(self.output)
        return manifest
