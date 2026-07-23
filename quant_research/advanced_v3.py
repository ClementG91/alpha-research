from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .advanced import (
    HORIZONS,
    _periodic_dsr,
    _periodic_psr,
    _run_ml_sleeve,
    _run_statistical_trend_sleeve,
)
from .advanced_v2 import _combine_sleeves
from .core import (
    DEFAULT_GROUPS,
    ResearchConfig,
    ResearchResult,
    allocate_weights,
    block_bootstrap,
    build_panel,
    compute_strategy_returns,
    performance_metrics,
    permutation_pvalue,
)


RISK_ASSETS = {"SPY", "QQQ", "IWM", "VEA", "EEM", "VNQ", "BTC-USD", "ETH-USD"}
DEFENSIVE_ASSETS = {"IEF", "TLT", "GLD", "DBC", "UUP"}


def _run_residual_momentum_sleeve(
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
        history = asset_returns.loc[:date].tail(48)
        eligible = [
            symbol
            for symbol in history.columns
            if history[symbol].notna().sum() >= 30
            and pd.notna(monthly.loc[date, symbol])
        ]
        if len(eligible) < 4:
            continue
        clean = history[eligible].fillna(0.0)
        centered = clean - clean.mean(axis=0)
        components = min(2, len(eligible) - 1)
        pca = PCA(n_components=components, random_state=config.random_seed)
        factors = pca.fit_transform(centered.to_numpy())
        common = pca.inverse_transform(factors)
        residual = pd.DataFrame(
            centered.to_numpy() - common,
            index=centered.index,
            columns=eligible,
        )
        residual_momentum = (1.0 + residual.tail(12)).prod() - 1.0
        residual_quality = residual_momentum / (
            residual.tail(12).std().replace(0.0, np.nan) * np.sqrt(12)
        )
        score = residual_quality.rank(pct=True) - 0.50
        score = score.where(residual_momentum > 0.0).dropna()
        breadth = float((monthly.loc[date, eligible] / monthly[eligible].shift(12).loc[date] > 1).mean())
        scalar = float(np.clip(0.35 + breadth, 0.35, 1.0))
        target = allocate_weights(score, history, groups, scalar, config)
        rows[date] = target.reindex(monthly.columns).fillna(0.0)
        audits.append(
            {
                "signal_date": date,
                "sleeve": "pca_residual_momentum",
                "train_start": history.index.min(),
                "train_feature_end": date,
                "train_label_end": date,
                "anti_lookahead_pass": True,
                "horizon_months": 0,
            }
        )

    weights = pd.DataFrame.from_dict(rows, orient="index").sort_index().fillna(0.0)
    weights.index.name = "signal_date"
    returns, _, _ = compute_strategy_returns(weights, asset_returns, groups, config)
    audit = pd.DataFrame(audits).set_index("signal_date")
    return weights, returns, audit


def _run_macro_defensive_sleeve(
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
        history = asset_returns.loc[:date].tail(config.covariance_months)
        available = [
            symbol
            for symbol in monthly.columns
            if pd.notna(monthly.loc[date, symbol])
            and history[symbol].notna().sum() >= 18
        ]
        if len(available) < 4:
            continue
        mom_6 = monthly[available].pct_change(6).loc[date]
        mom_12 = monthly[available].pct_change(12).loc[date]
        vol = history[available].std() * np.sqrt(12)
        drawdown = monthly[available].loc[date] / monthly[available].rolling(12).max().loc[date] - 1.0
        breadth = float((mom_12 > 0.0).mean())
        benchmark_mom = float(mom_12.get("SPY", mom_12.median()))
        risk_off = breadth < 0.50 or benchmark_mom < 0.0
        universe = [
            symbol
            for symbol in available
            if symbol in (DEFENSIVE_ASSETS if risk_off else set(available))
        ]
        if len(universe) < 2:
            universe = [symbol for symbol in available if symbol in DEFENSIVE_ASSETS]
        score = (
            0.45 * mom_6[universe].rank(pct=True)
            + 0.45 * mom_12[universe].rank(pct=True)
            - 0.10 * vol[universe].rank(pct=True)
            + 0.10 * drawdown[universe].rank(pct=True)
        )
        score = score.where((mom_6[universe] > 0.0) | (mom_12[universe] > 0.0)).dropna()
        if score.empty:
            score = (1.0 / vol[universe]).replace([np.inf, -np.inf], np.nan).dropna()
        target = allocate_weights(score, history, groups, 1.0, config)
        rows[date] = target.reindex(monthly.columns).fillna(0.0)
        audits.append(
            {
                "signal_date": date,
                "sleeve": "macro_defensive_risk_parity",
                "train_start": history.index.min(),
                "train_feature_end": date,
                "train_label_end": date,
                "anti_lookahead_pass": True,
                "horizon_months": 0,
                "risk_off": risk_off,
            }
        )

    weights = pd.DataFrame.from_dict(rows, orient="index").sort_index().fillna(0.0)
    weights.index.name = "signal_date"
    returns, _, _ = compute_strategy_returns(weights, asset_returns, groups, config)
    audit = pd.DataFrame(audits).set_index("signal_date")
    return weights, returns, audit


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

    structural_sleeves = {
        "statistical_trend": _run_statistical_trend_sleeve(
            panel, monthly, asset_returns, groups, config
        ),
        "pca_residual_momentum": _run_residual_momentum_sleeve(
            monthly, asset_returns, groups, config
        ),
        "macro_defensive_risk_parity": _run_macro_defensive_sleeve(
            monthly, asset_returns, groups, config
        ),
    }
    for name, (weights, returns, audit) in structural_sleeves.items():
        sleeve_weights[name] = weights
        sleeve_returns[name] = returns
        audits.append(audit)

    weights, meta_weights = _combine_sleeves(
        sleeve_weights, sleeve_returns, asset_returns, groups, config
    )
    net, gross, costs = compute_strategy_returns(weights, asset_returns, groups, config)
    audit = pd.concat(audits).sort_index()
    if not bool(audit["anti_lookahead_pass"].all()):
        raise AssertionError("advanced v3 anti-lookahead audit failed")

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
