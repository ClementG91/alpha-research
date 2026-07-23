from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.covariance import LedoitWolf

from .core import (
    DEFAULT_GROUPS,
    ResearchConfig,
    ResearchResult,
    _cap_and_redistribute,
    _fit_predict,
    _regime_scalar,
    allocate_weights,
    block_bootstrap,
    build_panel,
    compute_strategy_returns,
    performance_metrics,
    permutation_pvalue,
)


HORIZONS = (1, 3, 6)


def _periodic_psr(returns: pd.Series, benchmark_periodic_sharpe: float = 0.0) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 3 or r.std(ddof=1) <= 0:
        return np.nan
    sr = float(r.mean() / r.std(ddof=1))
    sk = float(skew(r, bias=False))
    ku = float(kurtosis(r, fisher=False, bias=False))
    denom = np.sqrt(max(1e-12, 1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr * sr))
    return float(norm.cdf((sr - benchmark_periodic_sharpe) * np.sqrt(len(r) - 1) / denom))


def _periodic_dsr(returns: pd.Series, trial_count: int) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 3 or r.std(ddof=1) <= 0 or trial_count < 2:
        return np.nan
    sr = float(r.mean() / r.std(ddof=1))
    sk = float(skew(r, bias=False))
    ku = float(kurtosis(r, fisher=False, bias=False))
    sr_std = np.sqrt(
        max(
            1e-12,
            (1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr * sr) / (len(r) - 1),
        )
    )
    euler_gamma = 0.5772156649
    expected_max = sr_std * (
        (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trial_count)
        + euler_gamma * norm.ppf(1.0 - 1.0 / (trial_count * np.e))
    )
    return float(norm.cdf((sr - expected_max) / sr_std))


def _future_target(monthly: pd.DataFrame, horizon: int) -> pd.Series:
    future = monthly.shift(-horizon) / monthly - 1.0
    future = future / np.sqrt(float(horizon))
    return future.rename_axis(index="date", columns="symbol").stack(dropna=False)


def _run_ml_sleeve(
    base_panel: pd.DataFrame,
    monthly: pd.DataFrame,
    asset_returns: pd.DataFrame,
    regime_features: pd.DataFrame,
    benchmark_forward: pd.Series,
    groups: Mapping[str, str],
    config: ResearchConfig,
    horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    panel = base_panel.copy()
    panel["target"] = _future_target(monthly, horizon).reindex(panel.index)
    feature_cols = [c for c in panel.columns if c != "target"]
    dates = list(monthly.index)
    weight_rows: dict[pd.Timestamp, pd.Series] = {}
    audit_rows: list[dict[str, Any]] = []

    for i in range(config.min_train_months, len(dates) - 1):
        signal_date = dates[i]
        last_train_pos = i - horizon - config.embargo_months
        if last_train_pos < 0:
            continue
        first_train_pos = max(0, last_train_pos - config.lookback_months + 1)
        train_dates = dates[first_train_pos : last_train_pos + 1]
        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train = panel.loc[train_mask].dropna(subset=["target"])
        current = panel.loc[panel.index.get_level_values("date") == signal_date]
        current = current[
            current.index.get_level_values("symbol").isin(
                monthly.loc[signal_date].dropna().index
            )
        ]
        if len(train) < 100 or current.empty:
            continue

        scores = _fit_predict(train, current, feature_cols, config)
        scalar = _regime_scalar(
            regime_features,
            benchmark_forward,
            train_dates,
            signal_date,
            config.random_seed + horizon,
        )
        trailing = asset_returns.loc[:signal_date].tail(config.covariance_months)
        target = allocate_weights(scores, trailing, groups, scalar, config)
        weight_rows[signal_date] = target.reindex(monthly.columns).fillna(0.0)

        max_train_date = max(train_dates)
        max_label_pos = dates.index(max_train_date) + horizon
        max_label_date = dates[max_label_pos]
        audit_rows.append(
            {
                "signal_date": signal_date,
                "sleeve": f"ml_{horizon}m",
                "train_start": min(train_dates),
                "train_feature_end": max_train_date,
                "train_label_end": max_label_date,
                "anti_lookahead_pass": bool(max_label_date < signal_date),
                "horizon_months": horizon,
            }
        )

    weights = pd.DataFrame.from_dict(weight_rows, orient="index").sort_index().fillna(0.0)
    weights.index.name = "signal_date"
    sleeve_returns, _, _ = compute_strategy_returns(
        weights, asset_returns, groups, config
    )
    audit = pd.DataFrame(audit_rows).set_index("signal_date")
    return weights, sleeve_returns, audit


def _run_statistical_trend_sleeve(
    panel: pd.DataFrame,
    monthly: pd.DataFrame,
    asset_returns: pd.DataFrame,
    groups: Mapping[str, str],
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    dates = list(monthly.index)
    rows: dict[pd.Timestamp, pd.Series] = {}
    audits: list[dict[str, Any]] = []
    for i in range(config.min_train_months, len(dates) - 1):
        date = dates[i]
        current = panel.loc[panel.index.get_level_values("date") == date].copy()
        if current.empty:
            continue
        features = current.reset_index().set_index("symbol")
        raw = pd.DataFrame(index=features.index)
        raw["mom_3m"] = features["mom_3m"]
        raw["mom_6m"] = features["mom_6m"]
        raw["mom_12m"] = features["mom_12m"]
        raw["slope_t_6m"] = features["slope_t_6m"]
        raw["slope_t_12m"] = features["slope_t_12m"]
        ranked = raw.rank(pct=True, axis=0) - 0.5
        score = ranked.mean(axis=1) - 0.15 * features["vol_6m"].rank(pct=True)
        score = score.where(monthly.loc[date].notna()).dropna()
        breadth = float(features["breadth_12m"].dropna().median())
        scalar = float(np.clip(0.35 + breadth, 0.35, 1.0))
        trailing = asset_returns.loc[:date].tail(config.covariance_months)
        target = allocate_weights(score, trailing, groups, scalar, config)
        rows[date] = target.reindex(monthly.columns).fillna(0.0)
        audits.append(
            {
                "signal_date": date,
                "sleeve": "statistical_trend",
                "train_start": dates[max(0, i - 12)],
                "train_feature_end": date,
                "train_label_end": date,
                "anti_lookahead_pass": True,
                "horizon_months": 0,
            }
        )
    weights = pd.DataFrame.from_dict(rows, orient="index").sort_index().fillna(0.0)
    weights.index.name = "signal_date"
    sleeve_returns, _, _ = compute_strategy_returns(
        weights, asset_returns, groups, config
    )
    audit = pd.DataFrame(audits).set_index("signal_date")
    return weights, sleeve_returns, audit


def _sleeve_risk_weights(history: pd.DataFrame) -> pd.Series:
    cols = history.columns.tolist()
    if len(history.dropna(how="all")) < 18:
        return pd.Series(1.0 / len(cols), index=cols)
    clean = history.fillna(0.0)
    cov = LedoitWolf().fit(clean.to_numpy()).covariance_ * 12.0
    min_var = np.linalg.pinv(cov) @ np.ones(len(cols))
    min_var = np.clip(min_var, 0.0, None)
    if min_var.sum() <= 0:
        min_var = np.ones(len(cols))
    min_var /= min_var.sum()
    equal = np.ones(len(cols)) / len(cols)
    weights = pd.Series(0.65 * min_var + 0.35 * equal, index=cols)
    weights = weights.clip(lower=0.10, upper=0.45)
    return weights / weights.sum()


def _combine_sleeves(
    sleeve_weights: dict[str, pd.DataFrame],
    sleeve_returns: dict[str, pd.Series],
    asset_returns: pd.DataFrame,
    groups: Mapping[str, str],
    config: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_dates = sorted(set().union(*(frame.index for frame in sleeve_weights.values())))
    sleeve_return_frame = pd.DataFrame(sleeve_returns).sort_index()
    combined_rows: dict[pd.Timestamp, pd.Series] = {}
    meta_rows: dict[pd.Timestamp, pd.Series] = {}
    previous = pd.Series(0.0, index=asset_returns.columns)

    for date in all_dates:
        available = [name for name, frame in sleeve_weights.items() if date in frame.index]
        if not available:
            continue
        history = sleeve_return_frame.loc[:date, available].tail(36)
        meta = _sleeve_risk_weights(history)
        target = pd.Series(0.0, index=asset_returns.columns)
        for name in available:
            target = target.add(
                sleeve_weights[name].loc[date].reindex(target.index).fillna(0.0)
                * meta[name],
                fill_value=0.0,
            )
        target = 0.65 * target + 0.35 * previous
        target = _cap_and_redistribute(target, config.per_asset_cap)
        crypto = [s for s in target.index if groups.get(s) == "crypto"]
        crypto_total = float(target.loc[crypto].sum()) if crypto else 0.0
        if crypto_total > config.crypto_cap:
            target.loc[crypto] *= config.crypto_cap / crypto_total
        if target.sum() > 1.0:
            target /= target.sum()
        combined_rows[date] = target
        meta_rows[date] = meta
        previous = target

    combined = pd.DataFrame.from_dict(combined_rows, orient="index").sort_index()
    combined.index.name = "signal_date"
    meta_frame = pd.DataFrame.from_dict(meta_rows, orient="index").sort_index()
    meta_frame.index.name = "signal_date"
    return combined.fillna(0.0), meta_frame.fillna(0.0)


def run_advanced_walk_forward(
    prices: pd.DataFrame,
    groups: Mapping[str, str] | None = None,
    config: ResearchConfig | None = None,
) -> ResearchResult:
    config = config or ResearchConfig()
    groups = dict(DEFAULT_GROUPS if groups is None else groups)
    panel, monthly, asset_returns = build_panel(prices, groups)
    regime_cols = [
        c
        for c in panel.columns
        if c.startswith(
            (
                "breadth_",
                "benchmark_",
                "bond_",
                "gold_",
                "commodity_",
                "usd_",
                "equity_bond_",
            )
        )
    ]
    regime_features = (
        panel.reset_index().drop_duplicates("date").set_index("date")[regime_cols]
    )
    benchmark = "SPY" if "SPY" in asset_returns.columns else asset_returns.columns[0]
    benchmark_forward = asset_returns[benchmark].shift(-1)

    sleeve_weights: dict[str, pd.DataFrame] = {}
    sleeve_returns: dict[str, pd.Series] = {}
    audits: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        weights, returns, audit = _run_ml_sleeve(
            panel,
            monthly,
            asset_returns,
            regime_features,
            benchmark_forward,
            groups,
            config,
            horizon,
        )
        name = f"ml_{horizon}m"
        sleeve_weights[name] = weights
        sleeve_returns[name] = returns
        audits.append(audit)

    trend_weights, trend_returns, trend_audit = _run_statistical_trend_sleeve(
        panel, monthly, asset_returns, groups, config
    )
    sleeve_weights["statistical_trend"] = trend_weights
    sleeve_returns["statistical_trend"] = trend_returns
    audits.append(trend_audit)

    weights, meta_weights = _combine_sleeves(
        sleeve_weights, sleeve_returns, asset_returns, groups, config
    )
    net, gross, costs = compute_strategy_returns(weights, asset_returns, groups, config)
    audit = pd.concat(audits).sort_index()
    if not bool(audit["anti_lookahead_pass"].all()):
        raise AssertionError("advanced anti-lookahead audit failed")

    holdout = net.tail(config.holdout_months)
    metrics = performance_metrics(net)
    holdout_metrics = performance_metrics(holdout)
    metrics["probabilistic_sharpe_ratio"] = _periodic_psr(net)
    metrics["deflated_sharpe_ratio"] = _periodic_dsr(net, config.trial_count)
    holdout_metrics["probabilistic_sharpe_ratio"] = _periodic_psr(holdout)
    holdout_metrics["deflated_sharpe_ratio"] = _periodic_dsr(
        holdout, config.trial_count
    )

    double_cost, _, _ = compute_strategy_returns(
        weights, asset_returns, groups, config, cost_multiplier=2.0
    )
    extra_delay, _, _ = compute_strategy_returns(
        weights, asset_returns, groups, config, extra_delay_months=1
    )
    no_crypto_weights = weights.copy()
    crypto_cols = [s for s in no_crypto_weights if groups.get(s) == "crypto"]
    no_crypto_weights.loc[:, crypto_cols] = 0.0
    no_crypto, _, _ = compute_strategy_returns(
        no_crypto_weights, asset_returns, groups, config
    )
    stress = {
        "double_cost": performance_metrics(double_cost),
        "extra_month_delay": performance_metrics(extra_delay),
        "no_crypto": performance_metrics(no_crypto),
    }
    mc = block_bootstrap(
        holdout if len(holdout) >= 12 else net,
        paths=config.monte_carlo_paths,
        block_months=config.monte_carlo_block_months,
        seed=config.random_seed,
    )
    diagnostics = {
        "config": asdict(config),
        "data_start": str(monthly.index.min().date()),
        "data_end": str(monthly.index.max().date()),
        "asset_count": int(monthly.shape[1]),
        "assets": monthly.columns.tolist(),
        "anti_lookahead_pass": True,
        "average_monthly_turnover": float(weights.diff().abs().sum(axis=1).mean()),
        "permutation_pvalue": permutation_pvalue(
            weights,
            asset_returns,
            observed_mean=float(gross.mean()),
            trials=config.permutation_trials,
            seed=config.random_seed,
        ),
        "sleeve_metrics": {
            name: performance_metrics(values) for name, values in sleeve_returns.items()
        },
        "average_sleeve_weights": meta_weights.mean().to_dict(),
    }
    return ResearchResult(
        weights=weights,
        returns=net,
        gross_returns=gross,
        costs=costs,
        audit=audit,
        metrics=metrics,
        holdout_metrics=holdout_metrics,
        monte_carlo=mc,
        stress=stress,
        diagnostics=diagnostics,
    )
