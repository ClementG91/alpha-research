from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lse_vault import (
    VaultSnapshot,
    assert_secret_absent,
    official_client,
    rows_to_frame,
    safe_call,
    sanitise_payload,
    write_json,
)
from quant_validation import annualised_sharpe, evaluate_candidate_set

FUTURES_ROOTS = ("ES", "NQ", "RTY", "ZN", "ZB", "6E", "6J", "CL", "NG", "GC", "SI", "HG", "ZC", "ZS")
FX_PAIRS = ("EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD")
YIELD_SERIES = {
    "USD": "US2Y", "EUR": "DE2Y", "GBP": "GB2Y", "JPY": "JP2Y",
    "AUD": "AU2Y", "CAD": "CA2Y", "CHF": "CH2Y", "NZD": "NZ2Y",
}
ANNUAL_DAYS = 252


@dataclass(frozen=True)
class StrategyResult:
    name: str
    returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    return next((lowered[alias.lower()] for alias in aliases if alias.lower() in lowered), None)


def standardise_candles(rows: Any) -> pd.DataFrame:
    frame = rows_to_frame(rows)
    aliases = {
        "open": ("open", "o"), "high": ("high", "h"), "low": ("low", "l"),
        "close": ("close", "c", "price"), "volume": ("volume", "v", "size"),
    }
    result = pd.DataFrame(index=frame.index)
    for target, options in aliases.items():
        source = _find_column(frame, options)
        if source is not None:
            result[target] = pd.to_numeric(frame[source], errors="coerce")
    if "close" not in result:
        raise ValueError("Candle payload contains no recognisable close column")
    return result.dropna(subset=["close"]).sort_index()


def paged_candles(client: Any, symbol: str, timeframe: str, start: str, end: str) -> pd.DataFrame:
    pages: list[pd.DataFrame] = []
    cursor = pd.Timestamp(start, tz="UTC")
    end_timestamp = pd.Timestamp(end, tz="UTC")
    step = pd.Timedelta(days=1) if timeframe in {"1d", "1w", "1mo"} else pd.Timedelta(seconds=1)
    for _ in range(30):
        rows = safe_call(
            client.candles, symbol, timeframe, start=cursor.isoformat(), end=end_timestamp.isoformat(),
            limit=5000, order="asc",
        )
        page = standardise_candles(rows)
        if page.empty:
            break
        pages.append(page)
        last = pd.Timestamp(page.index.max())
        last = last.tz_localize("UTC") if last.tzinfo is None else last
        if last >= end_timestamp or len(page) < 5000:
            break
        cursor = last + step
    if not pages:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(pages).sort_index().loc[lambda frame: ~frame.index.duplicated(keep="last")]


def _catalog_symbols(catalog: list[dict[str, Any]], category: str) -> set[str]:
    result = set()
    for row in catalog:
        row_category = str(row.get("category", row.get("asset_class", ""))).lower()
        symbol = row.get("symbol") or row.get("ticker") or row.get("code")
        if symbol and category.lower() in row_category:
            result.add(str(symbol))
    return result


def collect(client: Any, output: Path, start: str, end: str) -> dict[str, Any]:
    snapshot = VaultSnapshot(output)
    usage = safe_call(client.usage) if hasattr(client, "usage") else {"status": "usage method unavailable"}
    snapshot.save_json("usage", "discovery", usage)
    catalog = safe_call(client.catalog)
    catalog_rows = list(catalog) if not isinstance(catalog, dict) else list(catalog.get("data", catalog.get("rows", [])))
    snapshot.save_json("catalog", "discovery", catalog_rows)
    available = _catalog_symbols(catalog_rows, "future")
    futures = [root for root in FUTURES_ROOTS if root in available]
    if not futures:
        for root in FUTURES_ROOTS:
            matches = sorted(symbol for symbol in available if re.match(rf"^{re.escape(root)}(?:[._-]|$)", symbol))
            if matches:
                futures.append(matches[0])
    errors: list[dict[str, str]] = []

    def save_candles(symbol: str, category: str) -> None:
        try:
            frame = paged_candles(client, symbol, "1d", start, end)
            if len(frame) >= 1000:
                snapshot.save_frame(symbol, category, frame)
        except Exception as exc:
            errors.append({"dataset": f"{category}:{symbol}", "error": str(exc)})

    for symbol in futures:
        save_candles(symbol, "futures_candles")
    save_candles("SPY", "benchmark_candles")
    for symbol in FX_PAIRS:
        save_candles(symbol, "fx_candles")

    for root in FUTURES_ROOTS:
        try:
            frame = rows_to_frame(safe_call(client.cot, root, start=start, end=end))
            if len(frame) >= 100:
                snapshot.save_frame(root, "cot", frame)
        except Exception as exc:
            errors.append({"dataset": f"cot:{root}", "error": str(exc)})

    for currency, series in YIELD_SERIES.items():
        try:
            frame = rows_to_frame(safe_call(client.bond_yields, series, start=start, end=end))
            if len(frame) >= 500:
                snapshot.save_frame(currency, "two_year_yields", frame)
        except Exception as exc:
            errors.append({"dataset": f"yield:{series}", "error": str(exc)})

    for region in ("US", "EU", "GB"):
        try:
            snapshot.save_frame(region, "economic_calendar", rows_to_frame(
                safe_call(client.economic_calendar, region=region, start=start, end=end)
            ))
        except Exception as exc:
            errors.append({"dataset": f"calendar:{region}", "error": str(exc)})

    for underlying in ("SPY", "QQQ"):
        try:
            snapshot.save_frame(underlying, "options_flow", rows_to_frame(
                safe_call(client.options_flow, underlying, start=start, end=end)
            ))
        except Exception as exc:
            errors.append({"dataset": f"options:{underlying}", "error": str(exc)})

    manifest = snapshot.finish()
    write_json(output / "collection_errors.json", sanitise_payload(errors))
    assert_secret_absent(output)
    return {"manifest": manifest, "errors": errors, "selected_futures": futures}


def _load(output: Path, category: str) -> dict[str, pd.DataFrame]:
    root = output / "data" / category
    return {path.stem: pd.read_parquet(path).sort_index() for path in sorted(root.glob("*.parquet"))} if root.exists() else {}


def _normalise(weights: pd.DataFrame) -> pd.DataFrame:
    return weights.div(weights.abs().sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)


def futures_trend(frames: dict[str, pd.DataFrame], cost_bps: float = 2.0, delay: int = 1) -> StrategyResult | None:
    closes = pd.concat({name: frame["close"] for name, frame in frames.items()}, axis=1).sort_index().ffill(limit=3)
    if closes.shape[1] < 6 or len(closes) < 1500:
        return None
    returns = closes.pct_change(fill_method=None)
    signal = sum(np.sign(closes.pct_change(h, fill_method=None)) for h in (21, 63, 126, 252)) / 4.0
    volatility = returns.rolling(126, min_periods=63).std() * math.sqrt(ANNUAL_DAYS)
    positions = _normalise(signal.div(volatility.replace(0.0, np.nan)).clip(-10.0, 10.0)).shift(delay).fillna(0.0)
    turnover = positions.diff().fillna(positions).abs().sum(axis=1)
    gross = (positions * returns).sum(axis=1, min_count=1).fillna(0.0)
    return StrategyResult("futures_trend", gross - turnover * cost_bps / 10_000.0, positions, turnover)


def _cot_signal(frame: pd.DataFrame) -> pd.Series | None:
    aliases = {
        "cl": ("commercial_long", "prod_merc_long", "producer_long"),
        "cs": ("commercial_short", "prod_merc_short", "producer_short"),
        "ml": ("managed_money_long", "noncommercial_long", "money_manager_long"),
        "ms": ("managed_money_short", "noncommercial_short", "money_manager_short"),
    }
    columns = {key: _find_column(frame, values) for key, values in aliases.items()}
    if any(value is None for value in columns.values()):
        return None
    pressure = (
        pd.to_numeric(frame[columns["cl"]], errors="coerce")
        - pd.to_numeric(frame[columns["cs"]], errors="coerce")
        - pd.to_numeric(frame[columns["ml"]], errors="coerce")
        + pd.to_numeric(frame[columns["ms"]], errors="coerce")
    )
    low, high = pressure.rolling(156, min_periods=78).min(), pressure.rolling(156, min_periods=78).max()
    return ((pressure - low).div((high - low).replace(0.0, np.nan)) - 0.5).shift(1)


def cot_relative_value(cot: dict[str, pd.DataFrame], futures: dict[str, pd.DataFrame], cost_bps: float = 2.0, delay: int = 1) -> StrategyResult | None:
    signals, closes = {}, {}
    for root, frame in cot.items():
        signal = _cot_signal(frame)
        price = futures.get(root) or next((value for symbol, value in futures.items() if symbol.startswith(root)), None)
        if signal is not None and price is not None:
            signals[root], closes[root] = signal, price["close"]
    if len(signals) < 4:
        return None
    close = pd.concat(closes, axis=1).sort_index().ffill(limit=3)
    returns = close.pct_change(fill_method=None)
    signal = pd.concat(signals, axis=1).resample("W-FRI").last().reindex(close.index).ffill(limit=7)
    positions = _normalise(signal.rank(axis=1, pct=True) - 0.5).shift(delay).fillna(0.0)
    turnover = positions.diff().fillna(positions).abs().sum(axis=1)
    gross = (positions * returns).sum(axis=1, min_count=1).fillna(0.0)
    return StrategyResult("cot_commercial_pressure", gross - turnover * cost_bps / 10_000.0, positions, turnover)


def _yield(frame: pd.DataFrame) -> pd.Series | None:
    column = _find_column(frame, ("value", "yield", "rate", "close"))
    if column is None:
        numeric = frame.select_dtypes(include=[np.number]).columns
        column = str(numeric[0]) if len(numeric) else None
    return pd.to_numeric(frame[column], errors="coerce") if column else None


def fx_carry(fx: dict[str, pd.DataFrame], yields: dict[str, pd.DataFrame], cost_bps: float = 1.5, delay: int = 1) -> StrategyResult | None:
    yield_values = {currency: _yield(frame) for currency, frame in yields.items()}
    if sum(value is not None for value in yield_values.values()) < 5:
        return None
    closes = pd.concat({pair: frame["close"] for pair, frame in fx.items()}, axis=1).sort_index().ffill(limit=3)
    returns = closes.pct_change(fill_method=None)
    monthly = pd.DataFrame(index=closes.resample("ME").last().index, columns=closes.columns, dtype=float)
    for pair in closes:
        base, quote = pair.split("/")
        if yield_values.get(base) is not None and yield_values.get(quote) is not None:
            monthly[pair] = (
                yield_values[base].resample("ME").last().reindex(monthly.index).ffill()
                - yield_values[quote].resample("ME").last().reindex(monthly.index).ffill()
            )
    signal = (monthly.rank(axis=1, pct=True) - 0.5).reindex(closes.index).ffill()
    positions = _normalise(signal).shift(delay).fillna(0.0)
    turnover = positions.diff().fillna(positions).abs().sum(axis=1)
    gross = (positions * returns).sum(axis=1, min_count=1).fillna(0.0)
    return StrategyResult("fx_yield_carry", gross - turnover * cost_bps / 10_000.0, positions, turnover)


def ensemble(results: dict[str, StrategyResult], name: str = "fixed_ensemble") -> StrategyResult | None:
    if len(results) < 2:
        return None
    returns = pd.concat({key: value.returns for key, value in results.items()}, axis=1).fillna(0.0)
    inverse = 1.0 / (returns.rolling(126, min_periods=63).std() * math.sqrt(ANNUAL_DAYS)).replace(0.0, np.nan)
    allocation = inverse.div(inverse.sum(axis=1), axis=0).clip(upper=0.50)
    allocation = allocation.div(allocation.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    portfolio = (allocation.shift(1).fillna(0.0) * returns).sum(axis=1)
    return StrategyResult(name, portfolio, allocation, allocation.diff().abs().sum(axis=1).fillna(0.0))


def render_plots(results: dict[str, StrategyResult], output: Path) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    returns = pd.concat({name: result.returns for name, result in results.items()}, axis=1).fillna(0.0)
    figure, axis = plt.subplots(figsize=(11, 5.2))
    for name in returns:
        axis.plot(returns.index, (1.0 + returns[name]).cumprod(), label=name)
    axis.set_title("LSE-only strategy equity curves")
    axis.set_ylabel("Growth of 1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "equity_curves.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "equity_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(11, 4.6))
    for name in returns:
        equity = (1.0 + returns[name]).cumprod()
        axis.plot(equity.index, (equity.div(equity.cummax()) - 1.0) * 100.0, label=name)
    axis.set_title("LSE-only strategy drawdowns")
    axis.set_ylabel("Drawdown (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "drawdowns.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "drawdowns.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def research(output: Path) -> dict[str, Any]:
    futures, cot = _load(output, "futures_candles"), _load(output, "cot")
    fx, yields = _load(output, "fx_candles"), _load(output, "two_year_yields")
    benchmark_frames = _load(output, "benchmark_candles")
    benchmark = benchmark_frames.get("SPY")
    benchmark_returns = benchmark["close"].pct_change(fill_method=None) if benchmark is not None else None
    base = [futures_trend(futures), cot_relative_value(cot, futures), fx_carry(fx, yields)]
    results = {item.name: item for item in base if item is not None}
    fixed = ensemble(results)
    if fixed is not None:
        results[fixed.name] = fixed
    doubled = ensemble({item.name: item for item in [
        futures_trend(futures, 4.0), cot_relative_value(cot, futures, 4.0), fx_carry(fx, yields, 3.0)
    ] if item is not None}, "double_costs")
    delayed = ensemble({item.name: item for item in [
        futures_trend(futures, delay=2), cot_relative_value(cot, futures, delay=2), fx_carry(fx, yields, delay=2)
    ] if item is not None}, "extra_delay")
    validation = None
    if fixed is not None and benchmark_returns is not None:
        candidate_frame = pd.concat({name: item.returns for name, item in results.items()}, axis=1).fillna(0.0)
        validation = evaluate_candidate_set(candidate_frame, benchmark_returns, "fixed_ensemble", trials=7)
        validation["stress"] = {
            "double_cost_sharpe": annualised_sharpe(doubled.returns) if doubled is not None else None,
            "extra_delay_sharpe": annualised_sharpe(delayed.returns) if delayed is not None else None,
        }
        validation["accepted"] = bool(
            validation["accepted_before_cost_delay_stress"]
            and validation["stress"]["double_cost_sharpe"] is not None
            and validation["stress"]["double_cost_sharpe"] > 0.0
            and validation["stress"]["extra_delay_sharpe"] is not None
            and validation["stress"]["extra_delay_sharpe"] > 0.0
        )
    report = {
        "strict_lse_only": True,
        "strategies_completed": list(results),
        "strategies_skipped": [name for name in ("futures_trend", "cot_commercial_pressure", "fx_yield_carry") if name not in results],
        "validation": validation,
        "warning": "No strategy is accepted unless every frozen validation and stress gate passes.",
    }
    if results:
        render_plots(results, output)
        pd.concat({name: item.returns for name, item in results.items()}, axis=1).to_csv(output / "lse_strategy_returns.csv", index_label="date")
        pd.concat({name: item.turnover for name, item in results.items()}, axis=1).to_csv(output / "lse_strategy_turnover.csv", index_label="date")
    write_json(output / "research_summary.json", report)
    assert_secret_absent(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict London Strategic Edge institutional-alpha campaign")
    parser.add_argument("--output", type=Path, default=Path("results/lse_institutional"))
    parser.add_argument("--start", default="2004-01-01")
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--phase", choices=("discover", "collect", "research", "full"), default="full")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    client = official_client()
    try:
        if args.phase in {"discover", "collect", "full"}:
            print(json.dumps({"collection": sanitise_payload(collect(client, args.output, args.start, args.end))}, indent=2, default=str))
        if args.phase in {"research", "full"}:
            print(json.dumps({"research": research(args.output)}, indent=2, default=str))
    finally:
        close = getattr(client, "close", None) or getattr(client, "disconnect", None)
        if callable(close):
            close()
    assert_secret_absent(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
