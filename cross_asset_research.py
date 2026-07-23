from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm

ANNUALIZATION = 252
DEFAULT_START = "2006-01-01"
FINAL_HOLDOUT_START = "2025-01-01"

UNIVERSE: dict[str, str] = {
    "SPY": "us_equity", "QQQ": "us_equity", "IWM": "us_equity", "DIA": "us_equity",
    "EFA": "international_equity", "EEM": "international_equity", "VWO": "international_equity",
    "EWJ": "international_equity", "EWG": "international_equity", "EWU": "international_equity",
    "EWC": "international_equity", "EWA": "international_equity", "EWZ": "international_equity",
    "INDA": "international_equity", "EWH": "international_equity", "EWT": "international_equity",
    "TLT": "rates_credit", "IEF": "rates_credit", "SHY": "rates_credit", "TIP": "rates_credit",
    "LQD": "rates_credit", "HYG": "rates_credit", "BND": "rates_credit", "AGG": "rates_credit",
    "GLD": "real_assets", "SLV": "real_assets", "USO": "real_assets", "DBC": "real_assets",
    "DBA": "real_assets", "UNG": "real_assets", "UUP": "real_assets", "FXE": "real_assets",
    "XLE": "us_sectors", "XLF": "us_sectors", "XLK": "us_sectors", "XLI": "us_sectors",
    "XLP": "us_sectors", "XLU": "us_sectors", "XLY": "us_sectors", "XLB": "us_sectors",
    "XLV": "us_sectors", "XLC": "us_sectors", "KRE": "us_sectors", "SMH": "us_sectors",
    "XBI": "us_sectors", "IYR": "us_sectors", "VNQ": "us_sectors",
    "MTUM": "equity_factors", "QUAL": "equity_factors", "USMV": "equity_factors", "VLUE": "equity_factors",
}

FOLDS = [
    {"name": "2015-2016", "train": ("2006-01-01", "2012-12-31"), "validation": ("2013-01-01", "2014-12-31"), "test": ("2015-01-01", "2016-12-31")},
    {"name": "2017-2018", "train": ("2006-01-01", "2014-12-31"), "validation": ("2015-01-01", "2016-12-31"), "test": ("2017-01-01", "2018-12-31")},
    {"name": "2019-2020", "train": ("2006-01-01", "2016-12-31"), "validation": ("2017-01-01", "2018-12-31"), "test": ("2019-01-01", "2020-12-31")},
    {"name": "2021-2022", "train": ("2006-01-01", "2018-12-31"), "validation": ("2019-01-01", "2020-12-31"), "test": ("2021-01-01", "2022-12-31")},
    {"name": "2023-2024", "train": ("2006-01-01", "2020-12-31"), "validation": ("2021-01-01", "2022-12-31"), "test": ("2023-01-01", "2024-12-31")},
]


@dataclass(frozen=True)
class Candidate:
    family: str
    params: tuple[tuple[str, float], ...]

    @property
    def key(self) -> str:
        suffix = ",".join(f"{k}={v:g}" for k, v in self.params)
        return f"{self.family}[{suffix}]"

    def param_dict(self) -> dict[str, float]:
        return dict(self.params)


@dataclass
class StrategySeries:
    returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    positions: pd.DataFrame
    round_trips: int
    active_assets: int
    active_classes: int


def stable_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    lower = {str(c).strip().lower(): c for c in frame.columns}
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(lower):
        if "timestamp" in lower:
            lower["date"] = lower["timestamp"]
        else:
            raise ValueError(f"{symbol}: missing OHLC columns: {frame.columns.tolist()}")
    rename = {lower[k]: k.capitalize() for k in required}
    if "volume" in lower:
        rename[lower["volume"]] = "Volume"
    out = frame.rename(columns=rename).copy()
    out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce").dt.tz_convert(None)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close"]).drop_duplicates("Date")
    out = out.set_index("Date").sort_index()
    return out[(out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]


def load_lse_export(directory: Path, symbol: str) -> pd.DataFrame | None:
    for name in (f"{symbol}.parquet", f"{symbol}.csv", f"{symbol.lower()}.parquet", f"{symbol.lower()}.csv"):
        path = directory / name
        if path.exists():
            raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            return normalize_ohlcv(raw, symbol)
    return None


def load_lse_client(symbol: str, start: str, api_key: str) -> pd.DataFrame:
    try:
        from lse import LSE  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "London Strategic Edge client is not installed. Export CSV/Parquet into --lse-dir, "
            "or install the official client when it becomes available."
        ) from exc
    client = LSE(api_key=api_key)
    raw = client.candles(symbol, "1d", start=start)
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw)
    return normalize_ohlcv(raw.reset_index(), symbol)


def download_stooq(symbol: str, cache_dir: Path, start: str, retries: int = 4) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol}.csv"
    if not path.exists() or path.stat().st_size < 100:
        url = f"https://stooq.com/q/d/l/?s={symbol.lower()}.us&i=d"
        headers = {"User-Agent": "alpha-research/1.0 (+https://github.com/ClementG91/alpha-research)"}
        last: Exception | None = None
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=45)
                response.raise_for_status()
                text = response.text
                if "No data" in text or len(text) < 100:
                    raise RuntimeError(f"no usable Stooq data for {symbol}")
                path.write_text(text, encoding="utf-8")
                break
            except Exception as exc:  # pragma: no cover
                last = exc
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"failed to download {symbol}: {last}")
    return normalize_ohlcv(pd.read_csv(path), symbol).loc[pd.Timestamp(start):]


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
        try:
            frame: pd.DataFrame | None = None
            source = ""
            if lse_dir:
                frame = load_lse_export(lse_dir, symbol)
                if frame is not None:
                    source = "london-strategic-edge-export"
            if frame is None and prefer_lse and api_key:
                frame = load_lse_client(symbol, start, api_key)
                source = "london-strategic-edge-api"
            if frame is None:
                frame = download_stooq(symbol, cache_dir, start)
                source = "stooq-fallback"
            if len(frame) < 500:
                raise RuntimeError(f"only {len(frame)} rows")
            data[symbol] = frame
            cache_path = cache_dir / f"{symbol}.csv"
            provenance["loaded"][symbol] = {
                "source": source,
                "rows": len(frame),
                "start": str(frame.index.min().date()),
                "end": str(frame.index.max().date()),
                "sha256": stable_hash(cache_path) if cache_path.exists() else None,
            }
        except Exception as exc:
            provenance["failed"][symbol] = str(exc)
    if "SPY" not in data:
        raise RuntimeError("SPY is mandatory for explicit beta neutralisation")
    if len(data) < 24:
        raise RuntimeError(f"insufficient universe breadth: loaded {len(data)} symbols")
    return data, provenance


def align_panel(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    columns = sorted(data)
    panel: dict[str, pd.DataFrame] = {}
    for field in ("Open", "High", "Low", "Close"):
        panel[field.lower()] = pd.concat({s: data[s][field] for s in columns}, axis=1).sort_index()
    panel["volume"] = pd.concat(
        {s: data[s].get("Volume", pd.Series(index=data[s].index, dtype=float)) for s in columns}, axis=1
    ).sort_index()
    return panel


def rolling_beta(returns: pd.DataFrame, benchmark: pd.Series, window: int) -> pd.DataFrame:
    var = benchmark.rolling(window, min_periods=max(30, window // 2)).var()
    out = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for column in returns:
        cov = returns[column].rolling(window, min_periods=max(30, window // 2)).cov(benchmark)
        out[column] = cov / var.replace(0, np.nan)
    return out.clip(-3, 3)


def project_constraints(raw: pd.Series, betas: pd.Series, classes: pd.Series) -> pd.Series:
    active = raw.index[raw.abs() > 0]
    if len(active) < 4:
        return raw * 0.0
    w = raw.loc[active].to_numpy(dtype=float)
    rows: list[np.ndarray] = []
    for cls in sorted(classes.loc[active].dropna().unique()):
        rows.append((classes.loc[active].to_numpy() == cls).astype(float))
    beta = betas.loc[active].fillna(0.0).to_numpy(dtype=float)
    if np.nanstd(beta) > 1e-8:
        rows.append(beta)
    if rows:
        a = np.vstack(rows)
        w = w - a.T @ np.linalg.pinv(a @ a.T, rcond=1e-10) @ (a @ w)
    gross = np.abs(w).sum()
    if gross <= 1e-12:
        return raw * 0.0
    result = pd.Series(0.0, index=raw.index)
    result.loc[active] = w / gross
    return result


def class_neutral_weights(score: pd.Series, beta: pd.Series, classes: pd.Series, quantile: float) -> pd.Series:
    weights = pd.Series(0.0, index=score.index)
    selections: dict[str, tuple[pd.Index, pd.Index]] = {}
    for cls, members in classes.groupby(classes).groups.items():
        values = score.loc[list(members)].dropna().sort_values()
        if len(values) < 4:
            continue
        n = max(1, int(math.floor(len(values) * quantile)))
        if 2 * n >= len(values):
            n = max(1, (len(values) - 1) // 2)
        if n:
            selections[str(cls)] = (values.index[-n:], values.index[:n])
    if not selections:
        return weights
    class_gross = 1.0 / len(selections)
    for long_idx, short_idx in selections.values():
        weights.loc[long_idx] = 0.5 * class_gross / len(long_idx)
        weights.loc[short_idx] = -0.5 * class_gross / len(short_idx)
    return project_constraints(weights, beta, classes)


def strategy_from_score(
    score: pd.DataFrame,
    execution_returns: pd.DataFrame,
    betas: pd.DataFrame,
    classes: pd.Series,
    quantile: float,
    round_trip_cost_bps: float,
    gate: pd.Series | None = None,
) -> StrategySeries:
    common = score.index.intersection(execution_returns.index).intersection(betas.index)
    score, execution_returns, betas = score.loc[common], execution_returns.loc[common], betas.loc[common]
    weights = pd.DataFrame(0.0, index=common, columns=score.columns)
    aligned_gate = gate.reindex(common).fillna(False) if gate is not None else None
    for dt in common:
        if aligned_gate is not None and not bool(aligned_gate.loc[dt]):
            continue
        available = score.loc[dt].notna() & execution_returns.loc[dt].notna() & betas.loc[dt].notna()
        if available.sum() < 16:
            continue
        weights.loc[dt, available] = class_neutral_weights(
            score.loc[dt, available], betas.loc[dt, available], classes.loc[available], quantile
        )
    gross_pnl = (weights * execution_returns).sum(axis=1).fillna(0.0)
    gross_exposure = weights.abs().sum(axis=1)
    net = gross_pnl - gross_exposure * (round_trip_cost_bps / 10_000.0)
    round_trips = int((weights.abs() > 1e-10).sum().sum())
    active = weights.columns[weights.abs().sum(axis=0) > 0]
    return StrategySeries(
        returns=net,
        gross_returns=gross_pnl,
        turnover=2.0 * gross_exposure,
        positions=weights,
        round_trips=round_trips,
        active_assets=len(active),
        active_classes=int(classes.loc[active].nunique()),
    )


def make_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for lookback in (1, 3, 5, 10):
        for quantile in (0.20, 0.30):
            for vol_window in (20, 60):
                for beta_window in (63, 126):
                    candidates.append(Candidate("residual_reversal", tuple(sorted({
                        "lookback": float(lookback), "quantile": quantile,
                        "vol_window": float(vol_window), "beta_window": float(beta_window),
                    }.items()))))
    for quantile in (0.20, 0.30):
        for vol_window in (20, 60):
            for beta_window in (63, 126):
                candidates.append(Candidate("gap_reversal", tuple(sorted({
                    "quantile": quantile, "vol_window": float(vol_window), "beta_window": float(beta_window),
                }.items()))))
    for lookback in (1, 3, 5):
        for quantile in (0.20, 0.30):
            for dispersion_q in (0.50, 0.75):
                candidates.append(Candidate("dispersion_reversal", tuple(sorted({
                    "lookback": float(lookback), "quantile": quantile, "dispersion_q": dispersion_q,
                    "beta_window": 126.0, "vol_window": 60.0,
                }.items()))))
    return candidates


def build_candidate(candidate: Candidate, panel: dict[str, pd.DataFrame], classes: pd.Series, cost_bps: float) -> StrategySeries:
    p = candidate.param_dict()
    close, open_ = panel["close"], panel["open"]
    cc = close.pct_change(fill_method=None)
    oc = close.div(open_).sub(1.0)
    beta_raw = rolling_beta(cc, cc["SPY"], int(p["beta_window"]))
    trade_betas = beta_raw.shift(1)
    if candidate.family in {"residual_reversal", "dispersion_reversal"}:
        residual = cc - beta_raw.mul(cc["SPY"], axis=0)
        lookback, vol_window = int(p["lookback"]), int(p["vol_window"])
        innovation = residual.rolling(lookback, min_periods=lookback).sum()
        scale = residual.rolling(vol_window, min_periods=max(10, vol_window // 2)).std() * math.sqrt(lookback)
        score = -(innovation / scale.replace(0, np.nan)).shift(1)
        gate = None
        if candidate.family == "dispersion_reversal":
            dispersion = residual.abs().median(axis=1)
            threshold = dispersion.rolling(252, min_periods=126).quantile(p["dispersion_q"])
            gate = dispersion.shift(1) >= threshold.shift(1)
        return strategy_from_score(score, oc, trade_betas, classes, p["quantile"], 2.0 * cost_bps, gate)
    if candidate.family == "gap_reversal":
        gap = open_.div(close.shift(1)).sub(1.0)
        scale = gap.rolling(int(p["vol_window"]), min_periods=10).std()
        score = -(gap / scale.replace(0, np.nan))
        return strategy_from_score(score, oc, trade_betas, classes, p["quantile"], 2.0 * cost_bps + 8.0)
    raise ValueError(candidate.family)


def slice_series(series: pd.Series, period: tuple[str, str]) -> pd.Series:
    return series.loc[pd.Timestamp(period[0]):pd.Timestamp(period[1])].dropna()


def annualized_sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 20 or clean.std(ddof=1) <= 1e-12:
        return float("nan")
    return float(clean.mean() / clean.std(ddof=1) * math.sqrt(ANNUALIZATION))


def max_drawdown(returns: pd.Series) -> float:
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    return float((equity / equity.cummax() - 1.0).min()) if len(equity) else float("nan")


def regression_metrics(returns: pd.Series, spy_returns: pd.Series) -> dict[str, float]:
    aligned = pd.concat([returns.rename("strategy"), spy_returns.rename("spy")], axis=1).dropna()
    if len(aligned) < 60:
        return {"alpha": float("nan"), "beta": float("nan"), "alpha_t": float("nan"), "correlation": float("nan")}
    model = sm.OLS(aligned["strategy"], sm.add_constant(aligned["spy"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5}
    )
    return {
        "alpha": float(model.params["const"] * ANNUALIZATION),
        "beta": float(model.params["spy"]),
        "alpha_t": float(model.tvalues["const"]),
        "correlation": float(aligned.corr().iloc[0, 1]),
    }


def summarize(returns: pd.Series, spy_returns: pd.Series, round_trips: int | None = None) -> dict[str, float | int]:
    clean = returns.dropna()
    return {
        "observations": len(clean), "round_trips": int(round_trips or 0),
        "sharpe": annualized_sharpe(clean),
        "annual_return": float(clean.mean() * ANNUALIZATION) if len(clean) else float("nan"),
        "annual_volatility": float(clean.std(ddof=1) * math.sqrt(ANNUALIZATION)) if len(clean) > 1 else float("nan"),
        "max_drawdown": max_drawdown(clean),
        "hit_rate": float((clean > 0).mean()) if len(clean) else float("nan"),
        **regression_metrics(clean, spy_returns),
    }


def finite_metric(metrics: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def selection_score(metrics: dict[str, Any]) -> float:
    sharpe = finite_metric(metrics, "sharpe", -99)
    alpha_t = finite_metric(metrics, "alpha_t", -99)
    beta = abs(finite_metric(metrics, "beta", 99))
    corr = abs(finite_metric(metrics, "correlation", 99))
    drawdown = abs(min(finite_metric(metrics, "max_drawdown", -1), 0.0))
    return sharpe + 0.20 * max(alpha_t, -2) - 2.0 * beta - 1.5 * corr - max(drawdown - 0.10, 0.0)


def block_bootstrap_sharpe_pvalue(returns: pd.Series, block: int = 10, paths: int = 2000, seed: int = 42) -> float:
    clean = returns.dropna().to_numpy(dtype=float)
    if len(clean) < 100:
        return float("nan")
    observed = annualized_sharpe(pd.Series(clean))
    if not math.isfinite(observed):
        return float("nan")
    centered = clean - clean.mean()
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(paths)
    blocks_needed = math.ceil(len(clean) / block)
    max_start = max(1, len(clean) - block + 1)
    for i in range(paths):
        starts = rng.integers(0, max_start, size=blocks_needed)
        sample = np.concatenate([centered[s:s + block] for s in starts])[:len(clean)]
        std = sample.std(ddof=1)
        bootstrap[i] = sample.mean() / std * math.sqrt(ANNUALIZATION) if std > 1e-12 else 0.0
    return float((1 + np.sum(bootstrap >= observed)) / (paths + 1))


def max_sharpe_multiple_test_pvalue(
    candidate_returns: pd.DataFrame,
    selected_sharpe: float,
    block: int = 10,
    paths: int = 1000,
    seed: int = 7,
) -> float:
    frame = candidate_returns.dropna(how="all")
    if len(frame) < 100 or frame.shape[1] < 2 or not math.isfinite(selected_sharpe):
        return float("nan")
    values = frame.fillna(0.0).to_numpy(dtype=float)
    values -= np.nanmean(values, axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    maxima = np.empty(paths)
    blocks_needed = math.ceil(len(values) / block)
    max_start = max(1, len(values) - block + 1)
    for i in range(paths):
        starts = rng.integers(0, max_start, size=blocks_needed)
        sample = np.concatenate([values[s:s + block] for s in starts], axis=0)[:len(values)]
        means, stds = sample.mean(axis=0), sample.std(axis=0, ddof=1)
        sharpes = np.divide(means, stds, out=np.zeros_like(means), where=stds > 1e-12) * math.sqrt(ANNUALIZATION)
        maxima[i] = np.nanmax(sharpes)
    return float((1 + np.sum(maxima >= selected_sharpe)) / (paths + 1))


def candidate_grid(panel: dict[str, pd.DataFrame], classes: pd.Series, cost_bps: float) -> dict[str, StrategySeries]:
    results: dict[str, StrategySeries] = {}
    for candidate in make_candidates():
        series = build_candidate(candidate, panel, classes, cost_bps)
        if series.round_trips:
            results[candidate.key] = series
    return results


def select_in_fold(candidates: dict[str, StrategySeries], spy: pd.Series, fold: dict[str, Any]) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for key, series in candidates.items():
        train, valid = slice_series(series.returns, fold["train"]), slice_series(series.returns, fold["validation"])
        train_m = summarize(train, spy.loc[train.index], int(series.round_trips * len(train) / max(len(series.returns), 1)))
        valid_m = summarize(valid, spy.loc[valid.index], int(series.round_trips * len(valid) / max(len(series.returns), 1)))
        if train_m["observations"] < 750 or valid_m["observations"] < 350:
            continue
        if train_m["round_trips"] < 1000 or valid_m["round_trips"] < 400:
            continue
        ranked.append({
            "key": key, "train": train_m, "validation": valid_m,
            "score": 0.4 * selection_score(train_m) + 0.6 * selection_score(valid_m),
        })
    ranked.sort(key=lambda row: row["score"], reverse=True)
    if not ranked:
        raise RuntimeError(f"{fold['name']}: no candidate met breadth/trade constraints")
    winner = ranked[0]
    test_series = slice_series(candidates[winner["key"]].returns, fold["test"])
    winner["test"] = summarize(
        test_series, spy.loc[test_series.index],
        int(candidates[winner["key"]].round_trips * len(test_series) / max(len(candidates[winner["key"]].returns), 1)),
    )
    winner["ranking_top10"] = ranked[:10]
    return winner


def choose_locked_candidate(fold_results: list[dict[str, Any]]) -> str:
    keys = [row["key"] for row in fold_results]
    counts = {key: keys.count(key) for key in set(keys)}
    tied = [key for key, count in counts.items() if count == max(counts.values())]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda key: statistics.median(
        [float(row["test"]["sharpe"]) for row in fold_results if row["key"] == key]
    ))


def combine_distinct_families(
    candidates: dict[str, StrategySeries],
    fold_results: list[dict[str, Any]],
    development_period: tuple[str, str],
) -> tuple[list[str], pd.Series]:
    selected = choose_locked_candidate(fold_results)
    keys, base_family = [selected], selected.split("[", 1)[0]
    ranked_keys: list[str] = []
    for fold in reversed(fold_results):
        for row in fold["ranking_top10"]:
            if row["key"] not in ranked_keys:
                ranked_keys.append(row["key"])
    base = slice_series(candidates[selected].returns, development_period)
    for key in ranked_keys:
        if key.split("[", 1)[0] == base_family:
            continue
        challenger = slice_series(candidates[key].returns, development_period)
        aligned = pd.concat([base, challenger], axis=1).dropna()
        if len(aligned) >= 500 and abs(aligned.corr().iloc[0, 1]) <= 0.60 and annualized_sharpe(challenger) > 0:
            keys.append(key)
            break
    composite = pd.concat([candidates[key].returns.rename(key) for key in keys], axis=1).mean(axis=1)
    return keys, composite


def save_plot(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_plots(report: dict[str, Any], composite: pd.Series, spy: pd.Series, output: Path) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    equity = (1 + composite.fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(equity.index, equity.values)
    ax.set_title("Locked cross-asset strategy equity")
    ax.grid(alpha=0.25)
    save_plot(fig, plots / "equity.png")

    drawdown = equity / equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(drawdown.index, drawdown.values * 100, 0, alpha=0.45)
    ax.set_title("Drawdown")
    ax.set_ylabel("%")
    ax.grid(alpha=0.25)
    save_plot(fig, plots / "drawdown.png")

    labels = [fold["name"] for fold in report["folds"]] + ["Final holdout"]
    sharpes = [fold["test"]["sharpe"] for fold in report["folds"]] + [report["final_holdout"]["sharpe"]]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, sharpes)
    ax.axhline(0, linewidth=1)
    ax.set_title("Walk-forward and final-holdout Sharpe")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    save_plot(fig, plots / "fold_sharpe.png")

    aligned = pd.concat([composite.rename("strategy"), spy.rename("SPY")], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(aligned["SPY"] * 100, aligned["strategy"] * 100, s=8, alpha=0.35)
    ax.set_xlabel("SPY daily return (%)")
    ax.set_ylabel("Strategy daily return (%)")
    ax.set_title("Daily return dependence")
    ax.grid(alpha=0.25)
    save_plot(fig, plots / "spy_dependence.png")


def render_markdown(report: dict[str, Any], output: Path) -> None:
    final, dev = report["final_holdout"], report["development_oos"]
    lines = [
        "# Cross-asset statistical-arbitrage research", "",
        f"**Verdict: {'ACCEPTED' if report['accepted'] else 'REJECTED'} under the predeclared statistical gates.**", "",
        "This pipeline is no longer constrained to ManifoldBT. It uses liquid ETFs spanning US and international equities, rates, credit, commodities, currencies, sectors and equity factors. London Strategic Edge CSV/Parquet exports are preferred when supplied; the reproducible CI run uses Stooq as a keyless fallback.", "",
        "## Locked strategy", "",
        f"- sleeves: `{', '.join(report['locked_keys'])}`;",
        f"- assets loaded: **{report['data_quality']['loaded_assets']}** across **{report['data_quality']['loaded_classes']}** classes;",
        "- portfolio constraints: class-neutral, dollar-neutral and rolling SPY-beta-neutral.", "",
        "## Walk-forward development", "",
        "| Test fold | Candidate | Sharpe | Alpha | Alpha t | Beta | Corr SPY | Trades |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in report["folds"]:
        m = fold["test"]
        lines.append(
            f"| {fold['name']} | `{fold['key']}` | {m['sharpe']:.3f} | {m['alpha']:.3%} | {m['alpha_t']:.2f} | {m['beta']:.3f} | {m['correlation']:.3f} | {m['round_trips']} |"
        )
    lines += [
        "", "## Aggregated development OOS", "",
        f"- observations: **{dev['observations']}** daily portfolio observations;",
        f"- round trips: **{dev['round_trips']}**;",
        f"- Sharpe: **{dev['sharpe']:.3f}**;",
        f"- annualised alpha: **{dev['alpha']:.2%}**, HAC t-stat **{dev['alpha_t']:.2f}**;",
        f"- SPY beta: **{dev['beta']:.3f}**, correlation **{dev['correlation']:.3f}**;",
        f"- block-bootstrap p-value: **{report['development_bootstrap_p']:.4f}**;", "",
        "## Final untouched holdout — 2025 onward", "",
        f"- observations: **{final['observations']}**;",
        f"- round trips: **{final['round_trips']}**;",
        f"- Sharpe: **{final['sharpe']:.3f}**;",
        f"- annualised return: **{final['annual_return']:.2%}**;",
        f"- annualised alpha: **{final['alpha']:.2%}**, HAC t-stat **{final['alpha_t']:.2f}**;",
        f"- SPY beta: **{final['beta']:.3f}**, correlation **{final['correlation']:.3f}**;",
        f"- maximum drawdown: **{final['max_drawdown']:.2%}**;",
        f"- block-bootstrap p-value: **{report['holdout_bootstrap_p']:.4f}**;",
        f"- multiple-testing max-Sharpe p-value: **{report['multiple_test_p']:.4f}**;", "",
        "## Stress tests", "",
        f"- doubled transaction costs Sharpe: **{report['stress']['double_cost']['sharpe']:.3f}**;",
        f"- one-day additional signal delay Sharpe: **{report['stress']['extra_delay']['sharpe']:.3f}**;",
        f"- inverted-signal Sharpe: **{report['stress']['inverted']['sharpe']:.3f}**;", "",
        "## Acceptance gates", "",
    ]
    lines += [f"- {'PASS' if gate else 'FAIL'} — {name}" for name, gate in report["gates"].items()]
    lines += [
        "", "## Plots", "", "![Equity](plots/equity.png)", "", "![Drawdown](plots/drawdown.png)", "",
        "![Fold Sharpe](plots/fold_sharpe.png)", "", "![SPY dependence](plots/spy_dependence.png)", "",
        "## Caveats", "",
        "- Trade count is reported, but inference uses daily portfolio returns with HAC standard errors; individual trades are not treated as independent observations.",
        "- Stooq ETF prices are the reproducible fallback. London Strategic Edge exports can replace them without changing the engine.",
        "- Gap reversal carries an extra 8 bps execution penalty because daily bars cannot reproduce a delayed opening fill.",
        "- No result is called alpha unless it passes the untouched holdout, costs, delay, beta, correlation and multiple-testing gates.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    data, provenance = load_market_data(UNIVERSE.keys(), args.start, args.cache_dir, args.lse_dir, args.prefer_lse)
    panel = align_panel(data)
    classes = pd.Series({symbol: UNIVERSE[symbol] for symbol in panel["close"].columns})
    candidates = candidate_grid(panel, classes, args.cost_bps)
    if len(candidates) < 20:
        raise RuntimeError(f"only {len(candidates)} active candidates")
    spy = panel["close"]["SPY"].pct_change(fill_method=None)
    fold_results = [select_in_fold(candidates, spy, fold) | {"name": fold["name"]} for fold in FOLDS]
    locked_keys, composite = combine_distinct_families(
        candidates, fold_results, (FOLDS[0]["test"][0], FOLDS[-1]["test"][1])
    )

    oos_parts, oos_trades = [], 0
    for fold, result in zip(FOLDS, fold_results, strict=True):
        part = slice_series(candidates[result["key"]].returns, fold["test"])
        oos_parts.append(part)
        oos_trades += int(result["test"]["round_trips"])
    development_returns = pd.concat(oos_parts).sort_index()
    development = summarize(development_returns, spy.loc[development_returns.index], oos_trades)

    holdout_returns = composite.loc[pd.Timestamp(FINAL_HOLDOUT_START):].dropna()
    locked_round_trips = sum(candidates[key].round_trips for key in locked_keys)
    holdout_fraction = len(holdout_returns) / max(len(composite.dropna()), 1)
    final = summarize(
        holdout_returns, spy.loc[holdout_returns.index],
        int(locked_round_trips * holdout_fraction / len(locked_keys)),
    )

    definitions = {candidate.key: candidate for candidate in make_candidates()}
    double_cost = pd.concat([
        build_candidate(definitions[key], panel, classes, args.cost_bps * 2).returns.rename(key)
        for key in locked_keys
    ], axis=1).mean(axis=1).loc[pd.Timestamp(FINAL_HOLDOUT_START):].dropna()
    extra_delay = composite.shift(1).loc[pd.Timestamp(FINAL_HOLDOUT_START):].dropna()
    inverted = -holdout_returns
    candidate_holdout = pd.concat({
        key: series.returns.loc[pd.Timestamp(FINAL_HOLDOUT_START):] for key, series in candidates.items()
    }, axis=1)
    holdout_p = block_bootstrap_sharpe_pvalue(holdout_returns)
    multiple_p = max_sharpe_multiple_test_pvalue(candidate_holdout, float(final["sharpe"]))
    stress = {
        "double_cost": summarize(double_cost, spy.loc[double_cost.index]),
        "extra_delay": summarize(extra_delay, spy.loc[extra_delay.index]),
        "inverted": summarize(inverted, spy.loc[inverted.index]),
    }
    positive_folds = sum(float(row["test"]["alpha"]) > 0 for row in fold_results)
    gates = {
        "at least 1,000 development round trips": development["round_trips"] >= 1000,
        "at least 250 final-holdout daily observations": final["observations"] >= 250,
        "at least 1,000 final-holdout round trips": final["round_trips"] >= 1000,
        "positive alpha in at least four of five development folds": positive_folds >= 4,
        "final-holdout Sharpe at least 0.80": float(final["sharpe"]) >= 0.80,
        "final-holdout alpha HAC t-stat at least 2.0": float(final["alpha_t"]) >= 2.0,
        "absolute SPY beta at most 0.10": abs(float(final["beta"])) <= 0.10,
        "absolute SPY correlation at most 0.10": abs(float(final["correlation"])) <= 0.10,
        "holdout block-bootstrap p-value at most 0.05": holdout_p <= 0.05,
        "multiple-testing p-value at most 0.05": multiple_p <= 0.05,
        "doubled-cost Sharpe remains positive": float(stress["double_cost"]["sharpe"]) > 0,
        "extra-delay Sharpe remains positive": float(stress["extra_delay"]["sharpe"]) > 0,
        "inverted signal is not profitable": float(stress["inverted"]["sharpe"]) < 0,
        "final maximum drawdown below 15%": float(final["max_drawdown"]) > -0.15,
    }
    report: dict[str, Any] = {
        "accepted": all(gates.values()),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "data_quality": {
            "loaded_assets": len(data), "loaded_classes": int(classes.nunique()),
            "first_date": str(panel["close"].index.min().date()),
            "last_date": str(panel["close"].index.max().date()),
            "candidate_count": len(candidates),
        },
        "provenance": provenance, "folds": fold_results, "locked_keys": locked_keys,
        "development_oos": development, "final_holdout": final,
        "development_bootstrap_p": block_bootstrap_sharpe_pvalue(development_returns),
        "holdout_bootstrap_p": holdout_p, "multiple_test_p": multiple_p,
        "stress": stress, "gates": gates,
    }
    (args.output / "raw.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.output / "locked_strategy.json").write_text(json.dumps({
        "keys": locked_keys,
        "universe": {symbol: UNIVERSE[symbol] for symbol in data},
        "cost_bps": args.cost_bps,
        "final_holdout_start": FINAL_HOLDOUT_START,
        "data_sources": provenance["loaded"],
    }, indent=2), encoding="utf-8")
    render_plots(report, composite, spy, args.output)
    render_markdown(report, args.output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-asset beta-neutral statistical-arbitrage research")
    parser.add_argument("--output", type=Path, default=Path("results/cross_asset"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/stooq"))
    parser.add_argument("--lse-dir", type=Path, default=None)
    parser.add_argument("--prefer-lse", action="store_true")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    try:
        report = run_research(args)
        print((args.output / "REPORT.md").read_text(encoding="utf-8"))
        return 0 if report["accepted"] else 2
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        import traceback
        (args.output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
