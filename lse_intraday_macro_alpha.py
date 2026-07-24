from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lse_institutional_campaign import _catalog_symbols, _find_column, standardise_candles
from lse_vault import (
    VaultSnapshot,
    assert_secret_absent,
    official_client,
    rows_to_frame,
    safe_call,
    sanitise_payload,
    write_json,
)
from quant_validation import (
    annualised_sharpe,
    deflated_sharpe_probability,
    hac_alpha,
    probability_of_backtest_overfitting,
    white_reality_check,
)

TIMEFRAME = "15m"
BAR_DELTA = pd.Timedelta(minutes=15)
START = "2020-01-01"
DIAGNOSTIC_START = "2026-01-01"
FX_CORE = ("EUR/USD", "GBP/USD", "USD/JPY")
FX_EXTERNAL = ("AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD")
FUTURES_CORE = ("ES", "NQ", "ZN", "ZB")
FUTURES_EXTERNAL = ("RTY", "GC", "CL", "SI")
REGIONS = ("US", "EU", "GB", "JP", "CA", "AU", "NZ")
ONE_WAY_COST_BPS = {"fx": 1.5, "equity": 2.0, "rates": 1.5, "commodity": 2.5, "benchmark": 0.0}

FOLDS = (
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2025", "2024-01-01", "2025-12-31"),
)


@dataclass(frozen=True)
class Candidate:
    family: str
    hold_bars: int
    threshold: float

    @property
    def key(self) -> str:
        return f"{self.family}[hold={self.hold_bars},threshold={self.threshold:g}]"


@dataclass
class Backtest:
    candidate: Candidate
    bar_returns: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    trades: int


def candidates() -> list[Candidate]:
    return [
        Candidate("macro_surprise_drift", 4, 0.75),
        Candidate("macro_surprise_drift", 16, 0.75),
        Candidate("macro_overreaction_reversal", 4, 1.25),
        Candidate("macro_overreaction_reversal", 8, 1.25),
        Candidate("liquidity_shock_reversal", 2, 2.00),
        Candidate("liquidity_shock_reversal", 4, 2.00),
        Candidate("liquidity_shock_reversal", 8, 2.25),
    ]


def ensure_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    index = pd.to_datetime(output.index, errors="coerce", utc=True)
    output.index = index
    output = output.loc[output.index.notna()]
    return output.sort_index().loc[lambda value: ~value.index.duplicated(keep="last")]


def paged_candles(
    client: Any,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    max_pages: int = 80,
) -> pd.DataFrame:
    pages: list[pd.DataFrame] = []
    cursor = pd.Timestamp(start, tz="UTC")
    end_timestamp = pd.Timestamp(end, tz="UTC")
    for _ in range(max_pages):
        rows = safe_call(
            client.candles,
            symbol,
            timeframe,
            start=cursor.isoformat(),
            end=end_timestamp.isoformat(),
            limit=5000,
            order="asc",
        )
        page = ensure_utc_index(standardise_candles(rows))
        if page.empty:
            break
        pages.append(page)
        last = page.index.max()
        if last >= end_timestamp or len(page) < 5000:
            break
        cursor = last + pd.Timedelta(seconds=1)
    if not pages:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(pages).sort_index().loc[lambda value: ~value.index.duplicated(keep="last")]


def choose_future_symbols(catalog: list[dict[str, Any]]) -> dict[str, str]:
    available = _catalog_symbols(catalog, "future")
    selected: dict[str, str] = {}
    for root in FUTURES_CORE + FUTURES_EXTERNAL:
        exact = next((symbol for symbol in available if symbol.upper() == root), None)
        matches = sorted(
            symbol
            for symbol in available
            if re.match(rf"^{re.escape(root)}(?:[._=!-]|$)", symbol.upper())
        )
        symbol = exact or (matches[0] if matches else None)
        if symbol:
            selected[root] = symbol
    return selected


def collect(client: Any, output: Path, start: str, end: str) -> dict[str, Any]:
    snapshot = VaultSnapshot(output)
    catalog_payload = safe_call(client.catalog)
    catalog = (
        list(catalog_payload)
        if not isinstance(catalog_payload, dict)
        else list(catalog_payload.get("data", catalog_payload.get("rows", [])))
    )
    snapshot.save_json("catalog", "discovery", catalog)
    futures = choose_future_symbols(catalog)
    errors: list[dict[str, str]] = []

    def save(symbol: str, name: str, category: str) -> None:
        try:
            frame = paged_candles(client, symbol, TIMEFRAME, start, end)
            if len(frame) < 20_000:
                raise RuntimeError(f"insufficient {TIMEFRAME} history: {len(frame)} rows")
            snapshot.save_frame(name, category, frame)
        except Exception as exc:
            errors.append({"dataset": f"{category}:{name}", "error": str(exc)})

    for symbol in FX_CORE + FX_EXTERNAL:
        save(symbol, symbol, "intraday_fx")
    for root, symbol in futures.items():
        save(symbol, root, "intraday_futures")
    save("SPY", "SPY", "intraday_benchmark")

    for region in REGIONS:
        try:
            calendar = rows_to_frame(
                safe_call(client.economic_calendar, region=region, start=start, end=end)
            )
            snapshot.save_frame(region, "economic_calendar", ensure_utc_index(calendar))
        except Exception as exc:
            errors.append({"dataset": f"calendar:{region}", "error": str(exc)})

    for series in ("US2Y", "US10Y", "DE2Y", "GB2Y", "JP2Y"):
        try:
            yields = rows_to_frame(safe_call(client.bond_yields, series, start=start, end=end))
            snapshot.save_frame(series, "bond_yields", ensure_utc_index(yields))
        except Exception as exc:
            errors.append({"dataset": f"yield:{series}", "error": str(exc)})

    manifest = snapshot.finish()
    write_json(output / "collection_errors.json", sanitise_payload(errors))
    assert_secret_absent(output)
    return {"manifest": manifest, "errors": errors, "futures": futures}


def load_frames(output: Path, category: str) -> dict[str, pd.DataFrame]:
    root = output / "data" / category
    if not root.exists():
        return {}
    return {
        path.stem.replace("_", "/") if category == "intraday_fx" else path.stem: ensure_utc_index(pd.read_parquet(path))
        for path in sorted(root.glob("*.parquet"))
    }


def parse_numeric(value: Any) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "n/a", "-"}:
        return float("nan")
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1]
        multiplier = 0.01
    elif text[-1:].upper() in {"K", "M", "B", "T"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[text[-1].upper()]
        text = text[:-1]
    try:
        return float(text) * multiplier
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+", text)
        return float(match.group(0)) * multiplier if match else float("nan")


def event_family(name: str) -> tuple[str, float] | None:
    lowered = name.lower()
    if any(term in lowered for term in ("unemployment rate", "jobless claims", "unemployment claims")):
        return "labour_slack", -1.0
    if any(term in lowered for term in ("nonfarm", "payroll", "employment change", "job creation")):
        return "labour_growth", 1.0
    if any(term in lowered for term in ("cpi", "consumer price", "pce", "ppi", "inflation", "wage", "earnings")):
        return "inflation", 1.0
    if any(term in lowered for term in ("interest rate", "rate decision", "fed funds", "deposit facility", "bank rate")):
        return "policy_rate", 1.0
    if any(term in lowered for term in ("gdp", "retail sales", "pmi", "industrial production", "durable goods")):
        return "growth", 1.0
    return None


def standardise_calendar(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    region_currency = {"US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY", "CA": "CAD", "AU": "AUD", "NZ": "NZD"}
    for region, frame in frames.items():
        if frame.empty:
            continue
        event_col = _find_column(frame, ("event", "name", "title", "indicator"))
        actual_col = _find_column(frame, ("actual", "value", "released"))
        forecast_col = _find_column(frame, ("forecast", "consensus", "expected"))
        currency_col = _find_column(frame, ("currency", "ccy"))
        impact_col = _find_column(frame, ("impact", "importance", "priority"))
        if not event_col or not actual_col or not forecast_col:
            continue
        for timestamp, row in frame.iterrows():
            name = str(row[event_col])
            family = event_family(name)
            if family is None:
                continue
            impact = str(row[impact_col]).lower() if impact_col else "high"
            if impact_col and not any(token in impact for token in ("high", "3", "red")):
                continue
            actual = parse_numeric(row[actual_col])
            forecast = parse_numeric(row[forecast_col])
            if not np.isfinite(actual) or not np.isfinite(forecast):
                continue
            currency = str(row[currency_col]).upper() if currency_col else region_currency.get(region, "")
            if len(currency) != 3:
                currency = region_currency.get(region, "")
            records.append(
                {
                    "timestamp": pd.Timestamp(timestamp),
                    "region": region,
                    "currency": currency,
                    "event": name,
                    "family": family[0],
                    "direction": family[1],
                    "actual": actual,
                    "forecast": forecast,
                    "raw_surprise": family[1] * (actual - forecast),
                }
            )
    if not records:
        return pd.DataFrame()
    calendar = pd.DataFrame(records).sort_values("timestamp")
    calendar["surprise_z"] = np.nan
    for _, indices in calendar.groupby(["currency", "family"]).groups.items():
        values = calendar.loc[indices, "raw_surprise"].sort_index()
        scale = values.rolling(40, min_periods=12).std().shift(1)
        calendar.loc[values.index, "surprise_z"] = values.div(scale.replace(0.0, np.nan)).clip(-5.0, 5.0)
    return calendar.dropna(subset=["surprise_z"]).reset_index(drop=True)


def combine_market(frames: dict[str, pd.DataFrame], symbols: Iterable[str]) -> dict[str, pd.DataFrame]:
    selected = {symbol: frames[symbol] for symbol in symbols if symbol in frames}
    if not selected:
        return {}
    index = sorted(set().union(*(frame.index for frame in selected.values())))
    output: dict[str, pd.DataFrame] = {}
    for field in ("open", "high", "low", "close", "volume"):
        output[field] = pd.DataFrame(
            {symbol: selected[symbol].get(field, pd.Series(index=selected[symbol].index, dtype=float)) for symbol in selected},
            index=pd.DatetimeIndex(index),
        ).sort_index()
    return output


def asset_class(symbol: str) -> str:
    if "/" in symbol:
        return "fx"
    if symbol in {"ES", "NQ", "RTY"}:
        return "equity"
    if symbol in {"ZN", "ZB"}:
        return "rates"
    if symbol in {"GC", "CL", "SI"}:
        return "commodity"
    return "benchmark"


def expected_event_sign(currency: str, family: str, symbol: str) -> float:
    if "/" in symbol:
        base, quote = symbol.split("/")
        if currency == base:
            return 1.0
        if currency == quote:
            return -1.0
        return 0.0
    if currency != "USD":
        return 0.0
    if symbol in {"ZN", "ZB"}:
        return -1.0
    if symbol in {"ES", "NQ", "RTY"}:
        return -1.0 if family in {"inflation", "policy_rate"} else 1.0
    if symbol in {"GC", "SI"}:
        return -1.0 if family in {"inflation", "policy_rate"} else 0.0
    if symbol == "CL":
        return 1.0 if family == "growth" else 0.0
    return 0.0


def normalise_targets(raw: pd.DataFrame, classes: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    for timestamp in raw.index:
        row = raw.loc[timestamp].dropna()
        row = row[row.abs() > 0.0]
        if row.empty:
            continue
        weighted = pd.Series(0.0, index=row.index)
        active_classes = [name for name in sorted(classes.loc[row.index].unique()) if name != "benchmark"]
        for class_name in active_classes:
            members = row.index[classes.loc[row.index] == class_name]
            values = row.loc[members].clip(-5.0, 5.0)
            gross = values.abs().sum()
            if gross > 0.0:
                weighted.loc[members] = values / gross / len(active_classes)
        weighted = weighted.clip(-0.25, 0.25)
        gross = weighted.abs().sum()
        if gross > 0.0:
            result.loc[timestamp, weighted.index] = weighted / gross
    return result


def event_impulses(
    market: dict[str, pd.DataFrame],
    calendar: pd.DataFrame,
    family: str,
    threshold: float,
) -> pd.DataFrame:
    close = market["close"]
    impulses = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    returns = close.pct_change(fill_method=None)
    return_scale = returns.rolling(20 * 24 * 4, min_periods=500).std().shift(1)
    for event in calendar.itertuples(index=False):
        surprise = float(event.surprise_z)
        if abs(surprise) < threshold:
            continue
        location = close.index.searchsorted(pd.Timestamp(event.timestamp) + BAR_DELTA, side="left")
        if location >= len(close.index):
            continue
        timestamp = close.index[location]
        for symbol in close.columns:
            direction = expected_event_sign(str(event.currency), str(event.family), symbol)
            if direction == 0.0:
                continue
            expected = direction * surprise
            if family == "macro_surprise_drift":
                impulses.loc[timestamp, symbol] += expected
            else:
                observed = returns.loc[timestamp, symbol]
                scale = return_scale.loc[timestamp, symbol]
                if np.isfinite(observed) and np.isfinite(scale) and scale > 0.0:
                    shock = observed / scale
                    if np.sign(shock) == np.sign(expected) and abs(shock) >= threshold:
                        impulses.loc[timestamp, symbol] += -shock
    return impulses


def macro_blackout(index: pd.DatetimeIndex, calendar: pd.DataFrame, bars: int = 4) -> pd.Series:
    blackout = pd.Series(False, index=index)
    for timestamp in calendar["timestamp"]:
        center = index.searchsorted(pd.Timestamp(timestamp), side="left")
        start, end = max(0, center - bars), min(len(index), center + bars + 1)
        blackout.iloc[start:end] = True
    return blackout


def liquidity_impulses(
    market: dict[str, pd.DataFrame],
    calendar: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    high, low, close, volume = market["high"], market["low"], market["close"], market["volume"]
    returns = close.pct_change(fill_method=None)
    rolling = 20 * 24 * 4
    mean = returns.rolling(rolling, min_periods=500).mean().shift(1)
    scale = returns.rolling(rolling, min_periods=500).std().shift(1)
    shock = returns.sub(mean).div(scale.replace(0.0, np.nan))
    range_fraction = high.sub(low).div(close.replace(0.0, np.nan))
    range_mean = range_fraction.rolling(rolling, min_periods=500).mean().shift(1)
    range_scale = range_fraction.rolling(rolling, min_periods=500).std().shift(1)
    range_z = range_fraction.sub(range_mean).div(range_scale.replace(0.0, np.nan))
    log_volume = np.log(volume.where(volume > 0.0))
    volume_z = log_volume.sub(log_volume.rolling(rolling, min_periods=500).median().shift(1)).div(
        log_volume.rolling(rolling, min_periods=500).std().shift(1).replace(0.0, np.nan)
    )
    condition = shock.abs().ge(threshold) & range_z.ge(0.5)
    condition &= volume_z.ge(0.0) | volume_z.isna()
    condition &= ~macro_blackout(close.index, calendar).to_numpy()[:, None]
    return (-shock * (1.0 + range_z.clip(lower=0.0, upper=3.0))).where(condition, 0.0)


def build_backtest(
    candidate: Candidate,
    market: dict[str, pd.DataFrame],
    calendar: pd.DataFrame,
    extra_delay: int = 0,
    cost_multiplier: float = 1.0,
) -> Backtest:
    classes = pd.Series({symbol: asset_class(symbol) for symbol in market["close"].columns})
    if candidate.family == "liquidity_shock_reversal":
        raw = liquidity_impulses(market, calendar, candidate.threshold)
    else:
        raw = event_impulses(market, calendar, candidate.family, candidate.threshold)
    targets = normalise_targets(raw, classes)
    positions = pd.DataFrame(0.0, index=targets.index, columns=targets.columns)
    for offset in range(candidate.hold_bars):
        positions = positions.add(targets.shift(1 + extra_delay + offset).fillna(0.0), fill_value=0.0)
    gross_exposure = positions.abs().sum(axis=1).replace(0.0, np.nan)
    positions = positions.div(gross_exposure.clip(lower=1.0), axis=0).fillna(0.0)
    open_returns = market["open"].shift(-1).div(market["open"]).sub(1.0)
    gross = (positions * open_returns).sum(axis=1, min_count=1).fillna(0.0)
    changes = positions.sub(positions.shift(1).fillna(0.0)).abs()
    costs = pd.Series(0.0, index=positions.index)
    for symbol in positions:
        costs += changes[symbol] * ONE_WAY_COST_BPS[classes[symbol]] * cost_multiplier / 10_000.0
    net = gross - costs
    daily = net.groupby(net.index.floor("D")).sum(min_count=1).fillna(0.0)
    trades = int((changes > 1e-10).sum().sum())
    return Backtest(candidate, net, daily, positions, changes.sum(axis=1), costs, trades)


def benchmark_daily(benchmark: pd.DataFrame) -> pd.Series:
    returns = benchmark["open"].shift(-1).div(benchmark["open"]).sub(1.0)
    return returns.groupby(returns.index.floor("D")).sum(min_count=1).fillna(0.0)


def metrics(backtest: Backtest, benchmark: pd.Series, start: str, end: str) -> dict[str, Any]:
    returns = backtest.daily_returns.loc[start:end]
    aligned_benchmark = benchmark.reindex(returns.index).fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity.div(equity.cummax()) - 1.0
    alpha = hac_alpha(returns, aligned_benchmark, maxlags=10)
    return {
        "observations": int(len(returns)),
        "trades": int(backtest.trades),
        "sharpe": annualised_sharpe(returns),
        "annual_return": float(returns.mean() * 252.0),
        "annual_volatility": float(returns.std(ddof=1) * math.sqrt(252.0)),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else float("nan"),
        "hit_rate": float((returns > 0.0).mean()) if len(returns) else float("nan"),
        "turnover": float(backtest.turnover.loc[start:end].sum()),
        "cost_drag": float(backtest.costs.loc[start:end].sum()),
        **alpha,
    }


def rank_candidates(
    backtests: dict[str, Backtest], benchmark: pd.Series
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    details: dict[str, list[dict[str, Any]]] = {}
    ranking: list[dict[str, Any]] = []
    for key, backtest in backtests.items():
        folds = [{"name": name, "metrics": metrics(backtest, benchmark, start, end)} for name, start, end in FOLDS]
        details[key] = folds
        values = [row["metrics"] for row in folds]
        if any(int(value["observations"]) < 450 for value in values):
            continue
        sharpes = np.asarray([float(value["sharpe"]) for value in values])
        alpha_ts = np.asarray([float(value["alpha_t"]) for value in values])
        positive_alpha = sum(float(value["alpha"]) > 0.0 for value in values)
        score = float(np.nanmedian(sharpes) + 0.20 * positive_alpha + 0.10 * np.nanmedian(alpha_ts))
        ranking.append(
            {
                "key": key,
                "score": score,
                "median_sharpe": float(np.nanmedian(sharpes)),
                "median_alpha_t": float(np.nanmedian(alpha_ts)),
                "positive_alpha_folds": int(positive_alpha),
            }
        )
    ranking.sort(key=lambda row: row["score"], reverse=True)
    return ranking, details


def remove_best_year_sharpe(returns: pd.Series) -> float:
    yearly = returns.groupby(returns.index.year).sum()
    if len(yearly) < 3:
        return float("nan")
    best = int(yearly.idxmax())
    return annualised_sharpe(returns[returns.index.year != best])


def render_plots(
    output: Path,
    selected: str,
    core: Backtest,
    external: Backtest,
    folds: list[dict[str, Any]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 5.2))
    for label, returns in {"core": core.daily_returns, "external": external.daily_returns}.items():
        axis.plot(returns.index, (1.0 + returns).cumprod(), label=label)
    axis.set_title(f"Selected intraday macro candidate — {selected}")
    axis.set_ylabel("Growth of 1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "equity_curves.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "equity_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4.8))
    for label, returns in {"core": core.daily_returns, "external": external.daily_returns}.items():
        equity = (1.0 + returns).cumprod()
        axis.plot(equity.index, (equity.div(equity.cummax()) - 1.0) * 100.0, label=label)
    axis.set_title("Intraday macro drawdowns")
    axis.set_ylabel("Drawdown (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "drawdowns.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "drawdowns.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    labels = [row["name"] for row in folds]
    sharpes = [float(row["metrics"]["sharpe"]) for row in folds]
    axis.bar(labels, sharpes)
    axis.axhline(0.0, linewidth=1)
    axis.set_title("Core walk-forward fold Sharpe")
    axis.set_ylabel("Sharpe")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots / "fold_sharpes.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "fold_sharpes.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    yearly = pd.concat(
        {
            "core": core.daily_returns.groupby(core.daily_returns.index.year).sum(),
            "external": external.daily_returns.groupby(external.daily_returns.index.year).sum(),
        },
        axis=1,
    ) * 100.0
    yearly.plot(kind="bar", ax=axis)
    axis.set_title("Annual net return")
    axis.set_ylabel("Return (%)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots / "annual_returns.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "annual_returns.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(output: Path) -> dict[str, Any]:
    fx = load_frames(output, "intraday_fx")
    futures = load_frames(output, "intraday_futures")
    benchmark_frames = load_frames(output, "intraday_benchmark")
    calendars = load_frames(output, "economic_calendar")
    calendar = standardise_calendar(calendars)
    if calendar.empty:
        raise RuntimeError("No usable actual-versus-forecast macro events were collected")
    benchmark_frame = benchmark_frames.get("SPY")
    if benchmark_frame is None:
        raise RuntimeError("SPY intraday benchmark is required")

    combined = {**fx, **futures}
    core_symbols = [symbol for symbol in FX_CORE + FUTURES_CORE if symbol in combined]
    external_symbols = [symbol for symbol in FX_EXTERNAL + FUTURES_EXTERNAL if symbol in combined]
    if len(core_symbols) < 5 or len(external_symbols) < 5:
        raise RuntimeError(f"insufficient independent universes: core={core_symbols}, external={external_symbols}")
    core_market = combine_market(combined, core_symbols)
    external_market = combine_market(combined, external_symbols)
    benchmark = benchmark_daily(benchmark_frame)

    core_backtests = {candidate.key: build_backtest(candidate, core_market, calendar) for candidate in candidates()}
    external_backtests = {candidate.key: build_backtest(candidate, external_market, calendar) for candidate in candidates()}
    ranking, fold_details = rank_candidates(core_backtests, benchmark)
    if not ranking:
        raise RuntimeError("No predeclared candidate met the minimum fold history")
    selected = str(ranking[0]["key"])
    core = core_backtests[selected]
    external = external_backtests[selected]
    development_end = "2025-12-31"
    core_frame = pd.concat({key: value.daily_returns.loc[START:development_end] for key, value in core_backtests.items()}, axis=1).fillna(0.0)
    external_frame = pd.concat({key: value.daily_returns.loc[START:development_end] for key, value in external_backtests.items()}, axis=1).fillna(0.0)
    core_metrics = metrics(core, benchmark, START, development_end)
    external_metrics = metrics(external, benchmark, START, development_end)
    diagnostic = metrics(external, benchmark, DIAGNOSTIC_START, str(external.daily_returns.index.max().date()))
    double_cost = build_backtest(core.candidate, external_market, calendar, cost_multiplier=2.0)
    extra_delay = build_backtest(core.candidate, external_market, calendar, extra_delay=1)
    double_cost_metrics = metrics(double_cost, benchmark, START, development_end)
    extra_delay_metrics = metrics(extra_delay, benchmark, START, development_end)
    reality = white_reality_check(core_frame, selected)
    dsr = deflated_sharpe_probability(core.daily_returns.loc[START:development_end], len(core_backtests))
    pbo = probability_of_backtest_overfitting(core_frame)
    external_reality = white_reality_check(external_frame, selected)
    best_year_removed = remove_best_year_sharpe(external.daily_returns.loc[START:development_end])
    selected_folds = fold_details[selected]
    positive_folds = sum(float(row["metrics"]["alpha"]) > 0.0 for row in selected_folds)

    gates = {
        "three positive-alpha core folds": positive_folds == 3,
        "median core fold Sharpe at least 0.50": float(ranking[0]["median_sharpe"]) >= 0.50,
        "external Sharpe at least 0.50": float(external_metrics["sharpe"]) >= 0.50,
        "external alpha t-stat at least 1.80": float(external_metrics["alpha_t"]) >= 1.80,
        "absolute external beta at most 0.10": abs(float(external_metrics["beta"])) <= 0.10,
        "at least 1,000 external trades": int(external_metrics["trades"]) >= 1_000,
        "White Reality Check at most 0.05": float(reality) <= 0.05,
        "external Reality Check at most 0.10": float(external_reality) <= 0.10,
        "Deflated Sharpe probability at least 0.95": float(dsr) >= 0.95,
        "PBO at most 0.20": float(pbo["pbo"]) <= 0.20,
        "double-cost Sharpe positive": float(double_cost_metrics["sharpe"]) > 0.0,
        "extra-delay Sharpe positive": float(extra_delay_metrics["sharpe"]) > 0.0,
        "Sharpe remains positive without best year": float(best_year_removed) > 0.0,
        "external drawdown below 15%": float(external_metrics["max_drawdown"]) > -0.15,
    }
    accepted = all(gates.values())
    report = {
        "status": "ACCEPTED_FOR_FORWARD_PAPER_ONLY" if accepted else "REJECTED",
        "deployment_allowed": False,
        "selected": selected,
        "candidate": asdict(core.candidate),
        "ranking": ranking,
        "folds": selected_folds,
        "core": core_metrics,
        "external_replication": external_metrics,
        "diagnostic_2026_consumed": diagnostic,
        "validation": {
            "white_reality_check": reality,
            "external_white_reality_check": external_reality,
            "deflated_sharpe_probability": dsr,
            "pbo": pbo,
            "remove_best_year_sharpe": best_year_removed,
        },
        "stress": {"double_cost": double_cost_metrics, "extra_delay": extra_delay_metrics},
        "gates": gates,
        "protocol": {
            "timeframe": TIMEFRAME,
            "execution": "signal after completed bar; enter next bar open; one extra bar in stress",
            "statistics": "intraday PnL aggregated to daily before Sharpe, HAC, DSR, PBO and Reality Check",
            "core_universe": core_symbols,
            "external_universe": external_symbols,
            "candidate_count": len(core_backtests),
            "cost_bps_one_way": ONE_WAY_COST_BPS,
            "one_minute_excluded": "No bid/ask or order-book simulation; 1-minute OHLC would understate implementation costs.",
            "holdout_warning": "2026 is diagnostic only. A passing result requires a new forward paper window.",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    core_frame.to_csv(output / "core_candidate_daily_returns.csv", index_label="date")
    external_frame.to_csv(output / "external_candidate_daily_returns.csv", index_label="date")
    pd.concat({"core": core.daily_returns, "external": external.daily_returns}, axis=1).to_csv(
        output / "selected_daily_returns.csv", index_label="date"
    )
    render_plots(output, selected, core, external, selected_folds)
    lines = [
        "# LSE 15-minute macro and microstructure alpha campaign",
        "",
        f"**Verdict: {report['status']}.**",
        "",
        f"Selected: `{selected}`.",
        "",
        "## Independent external replication",
        "",
        f"- Sharpe: **{external_metrics['sharpe']:.3f}**; annual return: **{external_metrics['annual_return']:.2%}**;",
        f"- alpha t-stat: **{external_metrics['alpha_t']:.2f}**; beta: **{external_metrics['beta']:.3f}**;",
        f"- drawdown: **{external_metrics['max_drawdown']:.2%}**; trades: **{external_metrics['trades']}**.",
        "",
        "## Statistical controls",
        "",
        f"- core White Reality Check: **{reality:.4f}**; external: **{external_reality:.4f}**;",
        f"- Deflated Sharpe probability: **{dsr:.4f}**; PBO: **{pbo['pbo']:.3f}**;",
        f"- Sharpe without best year: **{best_year_removed:.3f}**.",
        "",
        "## Gates",
        "",
        *[f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in gates.items()],
        "",
        "No 1-minute result is reported because OHLC-only data cannot support credible spread, queue and adverse-selection modelling.",
        "A pass authorises forward paper monitoring only, never immediate deployment.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert_secret_absent(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Causal LSE 15-minute macro and microstructure alpha research")
    parser.add_argument("--output", type=Path, default=Path("results/intraday_macro/latest"))
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=str(date.today()))
    parser.add_argument("--phase", choices=("collect", "research", "full"), default="full")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase in {"collect", "full"}:
            client = official_client()
            try:
                collect(client, args.output, args.start, args.end)
            finally:
                close = getattr(client, "close", None) or getattr(client, "disconnect", None)
                if callable(close):
                    close()
        if args.phase in {"research", "full"}:
            result = run(args.output)
            print(json.dumps({"status": result["status"], "selected": result["selected"]}, indent=2))
        return 0
    except Exception:
        (args.output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        assert_secret_absent(args.output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
