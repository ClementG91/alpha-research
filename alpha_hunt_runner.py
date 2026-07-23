from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

import alpha_hunt


_ORIGINAL_LOAD_MARKET_DATA = alpha_hunt.load_market_data

ANCHOR_PREFERENCES: dict[str, tuple[str, ...]] = {
    "us_equity": ("QQQ", "VTI", "IWM", "RSP", "MDY", "IJR", "VOO", "VB"),
    "international_equity": ("EFA", "EEM", "VWO", "EWJ", "EWG"),
    "international": ("ACWI", "VEA", "VGK", "VPL", "MCHI"),
    "rates_credit": ("AGG", "GOVT", "BND", "IEF", "VGIT", "LQD"),
    "real_assets": ("DBC", "GSG", "PDBC", "GLD", "CPER"),
    "us_sectors": ("XLE", "XLF", "XLK", "XLV", "XLI"),
    "themes": ("SOXX", "IGV", "XRT", "ITB", "IHI"),
    "equity_factors": ("QUAL", "MTUM", "USMV", "VLUE"),
    "equity_styles": ("IWF", "IWD", "VTV", "VUG", "VBR"),
}


def normalize_session_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse provider-specific daily timestamps onto the UTC session date.

    London Strategic Edge may label a daily bar at the market open while Stooq
    labels the same session at midnight. Normalising only the index prevents a
    false two-calendar panel without changing any observed OHLCV value.
    """
    work = frame.copy().sort_index()
    parsed = pd.to_datetime(work.index, utc=True, errors="coerce")
    valid = ~parsed.isna()
    work = work.loc[valid].copy()
    work.index = parsed[valid].tz_convert(None).normalize()
    aggregations: dict[str, Any] = {}
    for column in work.columns:
        lower = str(column).lower()
        if lower == "open":
            aggregations[column] = "first"
        elif lower == "high":
            aggregations[column] = "max"
        elif lower == "low":
            aggregations[column] = "min"
        elif lower in {"close", "adj close", "adjusted_close"}:
            aggregations[column] = "last"
        elif lower == "volume":
            aggregations[column] = "sum"
        else:
            aggregations[column] = "last"
    daily = work.groupby(level=0, sort=True).agg(aggregations)
    return daily.loc[~daily.index.duplicated(keep="last")]


def normalised_load_market_data(*args: Any, **kwargs: Any) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    data, provenance = _ORIGINAL_LOAD_MARKET_DATA(*args, **kwargs)
    normalised = {symbol: normalize_session_frame(frame) for symbol, frame in data.items()}
    provenance = dict(provenance)
    provenance["session_index_normalised"] = True
    provenance["session_timezone"] = "UTC date"
    return normalised, provenance


def _first_available(preferences: Iterable[str], available: set[str]) -> str | None:
    return next((symbol for symbol in preferences if symbol in available), None)


def hedge_universe(columns: pd.Index, classes: pd.Series) -> list[str]:
    """Choose fixed liquid hedge instruments from the available universe.

    One anchor is reserved per risk class, SPY covers the broad market factor, and
    a second US-equity anchor gives the hedge system enough rank to neutralise beta
    without reintroducing class or dollar exposure.
    """
    available = set(map(str, columns))
    hedges: list[str] = []
    if "SPY" in available:
        hedges.append("SPY")
    non_benchmark_classes = sorted(
        class_name for class_name in classes.dropna().unique() if str(class_name) != "benchmark"
    )
    for class_name in non_benchmark_classes:
        members = set(classes.index[classes == class_name]).intersection(available)
        preferred = ANCHOR_PREFERENCES.get(str(class_name), tuple(sorted(members)))
        anchor = _first_available(preferred, members)
        if anchor is None and members:
            anchor = sorted(members)[0]
        if anchor is not None and anchor not in hedges:
            hedges.append(anchor)
    us_members = set(classes.index[classes == "us_equity"]).intersection(available)
    second_us = _first_available(
        (symbol for symbol in ANCHOR_PREFERENCES["us_equity"] if symbol not in hedges),
        us_members,
    )
    if second_us is not None:
        hedges.append(second_us)
    return hedges


def _factor_matrix(
    symbols: list[str],
    beta: pd.Series,
    classes: pd.Series,
    class_names: list[str],
) -> np.ndarray:
    beta_values = beta.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
    rows = [np.ones(len(symbols), dtype=float), beta_values]
    rows.extend(
        (classes.reindex(symbols).to_numpy() == class_name).astype(float)
        for class_name in class_names
    )
    return np.vstack(rows)


def hedged_sparse_weights(
    score: pd.Series,
    beta: pd.Series,
    classes: pd.Series,
    quantile: float,
) -> pd.Series:
    """Build sparse alpha legs, then neutralise their factor exposures with anchors.

    Signal assets never need an opposite shock inside the same class. Instead, the
    alpha book is selected globally and a separate minimum-norm hedge overlay
    removes dollar, class, and rolling-SPY-beta exposures.
    """
    result = pd.Series(0.0, index=score.index, dtype=float)
    valid_beta = beta.reindex(score.index).replace([np.inf, -np.inf], np.nan)
    valid_classes = classes.reindex(score.index)
    hedges = hedge_universe(score.index, valid_classes)
    if len(hedges) < 4 or "SPY" not in hedges:
        return result

    alpha_symbols = [
        symbol
        for symbol in score.index
        if symbol not in hedges
        and pd.notna(score.loc[symbol])
        and pd.notna(valid_beta.loc[symbol])
        and pd.notna(valid_classes.loc[symbol])
        and str(valid_classes.loc[symbol]) != "benchmark"
    ]
    values = score.reindex(alpha_symbols).dropna()
    positive = values[values > 0.0].sort_values(ascending=False)
    negative = values[values < 0.0].sort_values()
    if positive.empty or negative.empty:
        return result

    target_count = max(1, int(math.floor(len(values) * quantile)))
    long_index = list(positive.index[: min(target_count, len(positive))])
    short_index = list(negative.index[: min(target_count, len(negative))])
    if not long_index or not short_index:
        return result

    raw = pd.Series(0.0, index=score.index, dtype=float)
    raw.loc[long_index] = 0.5 / len(long_index)
    raw.loc[short_index] = -0.5 / len(short_index)

    class_names = sorted(
        class_name
        for class_name in valid_classes.dropna().unique()
        if str(class_name) != "benchmark"
    )
    required_hedges = 2 + len(class_names)
    hedge_symbols = [
        symbol
        for symbol in hedges
        if pd.notna(valid_beta.get(symbol)) and pd.notna(valid_classes.get(symbol))
    ]
    if len(hedge_symbols) < required_hedges:
        return result

    factor_symbols = list(score.index)
    full_factors = _factor_matrix(factor_symbols, valid_beta, valid_classes, class_names)
    hedge_factors = _factor_matrix(hedge_symbols, valid_beta, valid_classes, class_names)
    exposure = full_factors @ raw.reindex(factor_symbols).to_numpy(dtype=float)
    hedge_weights, _, rank, _ = np.linalg.lstsq(hedge_factors, -exposure, rcond=1e-10)
    if rank < required_hedges or not np.isfinite(hedge_weights).all():
        return result
    result = raw.copy()
    result.loc[hedge_symbols] += hedge_weights

    residual = full_factors @ result.reindex(factor_symbols).to_numpy(dtype=float)
    if float(np.max(np.abs(residual))) > 1e-7:
        return pd.Series(0.0, index=score.index, dtype=float)
    gross = float(result.abs().sum())
    if not np.isfinite(gross) or gross <= 0.0 or gross > 8.0:
        return pd.Series(0.0, index=score.index, dtype=float)
    return result / gross


def hedged_target_weights(
    score: pd.DataFrame,
    beta: pd.DataFrame,
    classes: pd.Series,
    quantile: float,
    rebalance: int,
) -> pd.DataFrame:
    common = score.index.intersection(beta.index)
    score = score.loc[common]
    beta = beta.loc[common]
    weights = pd.DataFrame(0.0, index=common, columns=score.columns)
    for offset, timestamp in enumerate(common):
        if offset % rebalance:
            weights.loc[timestamp] = pd.NA
            continue
        weights.loc[timestamp] = hedged_sparse_weights(
            score.loc[timestamp], beta.loc[timestamp], classes, quantile
        )
    if rebalance > 1:
        weights = weights.ffill(limit=rebalance - 1).fillna(0.0)
    return weights.astype(float)


alpha_hunt.load_market_data = normalised_load_market_data
alpha_hunt.target_weights = hedged_target_weights

if __name__ == "__main__":
    raise SystemExit(alpha_hunt.main())
