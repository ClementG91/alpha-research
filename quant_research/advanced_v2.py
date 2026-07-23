from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import numpy as np
import pandas as pd

from .advanced import (
    HORIZONS,
    _periodic_dsr,
    _periodic_psr,
    _run_ml_sleeve,
    _run_statistical_trend_sleeve,
    _sleeve_risk_weights,
)
from .core import (
    DEFAULT_GROUPS,
    ResearchConfig,
    ResearchResult,
    block_bootstrap,
    build_panel,
    compute_strategy_returns,
    performance_metrics,
    permutation_pvalue,
)


def _cap_preserving_gross(weights: pd.Series, cap: float) -> pd.Series:
    """Enforce an asset cap without converting cash into risky exposure."""
    original_gross = min(float(weights.clip(lower=0.0).sum()), 1.0)
    capped = weights.clip(lower=0.0, upper=cap).copy()
    missing = original_gross - float(capped.sum())
    for _ in range(12):
        if missing <= 1e-12:
            break
        room = (cap - capped).clip(lower=0.0)
        if room.sum() <= 1e-12:
            break
        addition = missing * room / room.sum()
        addition = np.minimum(addition, room)
        capped += addition
        missing = original_gross - float(capped.sum())
    return capped


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
        target = _cap_preserving_gross(target, config.per_asset_cap)

        crypto = [symbol for symbol in target.index if groups.get(symbol) == "crypto"]
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
        column
        for column in panel.columns
        if column.startswith(
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
    crypto_cols = [symbol for symbol in weights if groups.get(symbol) == "crypto"]
    no_crypto_weights.loc[:, crypto_cols] = 0.0
    no_crypto, _, _ = compute_strategy_returns(
        no_crypto_weights, asset_returns, groups, config
    )
    stress = {
        "double_cost": performance_metrics(double_cost),
        "extra_month_delay": performance_metrics(extra_delay),
        "no_crypto": performance_metrics(no_crypto),
    }
    monte_carlo = block_bootstrap(
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
        "average_gross_exposure": float(weights.sum(axis=1).mean()),
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
        monte_carlo=monte_carlo,
        stress=stress,
        diagnostics=diagnostics,
    )
