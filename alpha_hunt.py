from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cross_asset_research as engine
from cross_asset_momentum import CORE_UNIVERSE, EXTERNAL_UNIVERSE
from cross_asset_research_v2 import load_market_data
from quant_validation import (
    annualised_sharpe,
    deflated_sharpe_probability,
    hac_alpha,
    probability_of_backtest_overfitting,
    white_reality_check,
)

START = "2006-01-01"
RESEARCH_START = "2015-01-01"
RESEARCH_END = "2024-12-31"
DIAGNOSTIC_START = "2025-01-01"
ONE_WAY_COST_BPS = 4.0


@dataclass(frozen=True)
class Candidate:
    family: str
    hold: int
    quantile: float
    threshold: float
    rebalance: int = 1

    @property
    def key(self) -> str:
        return (
            f"{self.family}[hold={self.hold},q={self.quantile:g},"
            f"threshold={self.threshold:g},rebalance={self.rebalance}]"
        )


@dataclass
class Backtest:
    candidate: Candidate
    returns: pd.Series
    gross_returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


def make_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for hold in (1, 2, 3):
        candidates.append(Candidate("auction_flow_reversal", hold, 0.15, 1.25))
    for hold in (2, 3, 5):
        candidates.append(Candidate("close_flow_reversal", hold, 0.15, 1.50))
    for hold in (1, 2):
        candidates.append(Candidate("gap_absorption_reversal", hold, 0.15, 1.25))
    for hold in (2, 3):
        candidates.append(Candidate("dispersion_reversal", hold, 0.20, 1.25))
    candidates.append(Candidate("idiosyncratic_volatility_carry", 20, 0.20, 0.0, 5))
    candidates.append(Candidate("residual_trend", 20, 0.20, 0.0, 5))
    return candidates


def prior_zscore(frame: pd.DataFrame, window: int = 252, minimum: int = 126) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=minimum).mean().shift(1)
    standard_deviation = frame.rolling(window, min_periods=minimum).std().shift(1)
    return frame.sub(mean).div(standard_deviation.replace(0.0, np.nan))


def cross_sectional_gate(score: pd.DataFrame, condition: pd.DataFrame) -> pd.DataFrame:
    return score.where(condition)


def build_features(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    open_, high, low, close, volume = (
        panel["open"], panel["high"], panel["low"], panel["close"], panel["volume"]
    )
    close_returns = close.pct_change(fill_method=None)
    intraday = close.div(open_).sub(1.0)
    gap = open_.div(close.shift(1)).sub(1.0)

    beta_close = engine.rolling_beta(close_returns, close_returns["SPY"], 252).shift(1)
    beta_intraday = engine.rolling_beta(intraday, intraday["SPY"], 252).shift(1)
    beta_gap = engine.rolling_beta(gap, gap["SPY"], 252).shift(1)

    residual_close = close_returns - beta_close.mul(close_returns["SPY"], axis=0)
    residual_intraday = intraday - beta_intraday.mul(intraday["SPY"], axis=0)
    residual_gap = gap - beta_gap.mul(gap["SPY"], axis=0)

    z_close = prior_zscore(residual_close)
    z_intraday = prior_zscore(residual_intraday)
    z_gap = prior_zscore(residual_gap)

    dollar_volume = (volume * close).where((volume > 0.0) & (close > 0.0))
    log_dollar_volume = np.log(dollar_volume)
    volume_median = log_dollar_volume.rolling(252, min_periods=126).median().shift(1)
    volume_scale = log_dollar_volume.rolling(252, min_periods=126).std().shift(1)
    abnormal_volume = log_dollar_volume.sub(volume_median).div(volume_scale.replace(0.0, np.nan))

    range_ = (high - low).replace(0.0, np.nan)
    close_location = (2.0 * close.sub(low).div(range_) - 1.0).clip(-1.0, 1.0)

    auction_condition = (
        z_intraday.abs().ge(1.25)
        & abnormal_volume.gt(0.0)
        & np.sign(residual_intraday).eq(np.sign(close_location))
    )
    auction_score = -z_intraday * abnormal_volume.clip(lower=0.0, upper=3.0) * (
        0.5 + close_location.abs()
    )

    close_condition = z_close.abs().ge(1.50) & abnormal_volume.gt(0.50)
    close_score = -z_close * (1.0 + abnormal_volume.clip(lower=0.0, upper=3.0))

    absorption_ratio = intraday.abs().div(gap.abs().replace(0.0, np.nan)).clip(upper=3.0)
    absorption_condition = (
        z_gap.abs().ge(1.25)
        & gap.mul(intraday).lt(0.0)
        & absorption_ratio.ge(0.50)
    )
    gap_score = -z_gap * (1.0 + absorption_ratio)

    dispersion = residual_close.abs().median(axis=1)
    dispersion_threshold = dispersion.rolling(504, min_periods=252).quantile(0.75).shift(1)
    high_dispersion = dispersion.gt(dispersion_threshold)
    dispersion_condition = z_close.abs().ge(1.25) & pd.DataFrame(
        np.repeat(high_dispersion.to_numpy()[:, None], z_close.shape[1], axis=1),
        index=z_close.index,
        columns=z_close.columns,
    )
    dispersion_score = -z_close

    idiosyncratic_volatility = residual_close.rolling(60, min_periods=40).std()
    idio_score = -np.log(idiosyncratic_volatility.replace(0.0, np.nan))

    residual_trend_raw = residual_close.rolling(20, min_periods=20).sum()
    residual_trend_scale = residual_close.rolling(60, min_periods=40).std() * math.sqrt(20)
    trend_score = residual_trend_raw.div(residual_trend_scale.replace(0.0, np.nan))

    features = {
        "auction_flow_reversal": cross_sectional_gate(auction_score, auction_condition),
        "close_flow_reversal": cross_sectional_gate(close_score, close_condition),
        "gap_absorption_reversal": cross_sectional_gate(gap_score, absorption_condition),
        "dispersion_reversal": cross_sectional_gate(dispersion_score, dispersion_condition),
        "idiosyncratic_volatility_carry": idio_score,
        "residual_trend": trend_score,
        "trade_beta": engine.rolling_beta(
            open_.shift(-1).div(open_).sub(1.0),
            open_["SPY"].shift(-1).div(open_["SPY"]).sub(1.0),
            252,
        ).shift(1),
        "execution_returns": open_.shift(-1).div(open_).sub(1.0),
    }
    for name, frame in features.items():
        if isinstance(frame, pd.DataFrame) and "SPY" in frame.columns and name not in {
            "trade_beta", "execution_returns"
        }:
            frame.loc[:, "SPY"] = np.nan
    return features


def target_weights(
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
            weights.loc[timestamp] = np.nan
            continue
        available = score.loc[timestamp].notna() & beta.loc[timestamp].notna()
        available &= classes.reindex(score.columns).notna()
        if int(available.sum()) < 16:
            continue
        weights.loc[timestamp, available] = engine.class_neutral_weights(
            score.loc[timestamp, available],
            beta.loc[timestamp, available],
            classes.loc[available],
            quantile,
        )
    if rebalance > 1:
        weights = weights.ffill(limit=rebalance - 1).fillna(0.0)
    return weights


def backtest_candidate(
    candidate: Candidate,
    features: dict[str, pd.DataFrame],
    classes: pd.Series,
    cost_bps: float = ONE_WAY_COST_BPS,
    extra_delay: int = 0,
) -> Backtest:
    score = features[candidate.family]
    beta = features["trade_beta"]
    execution_returns = features["execution_returns"]
    targets = target_weights(score, beta, classes, candidate.quantile, candidate.rebalance)
    delayed_targets = targets.shift(1 + extra_delay)
    positions = delayed_targets.rolling(candidate.hold, min_periods=1).mean().fillna(0.0)
    positions = positions.reindex(execution_returns.index).fillna(0.0)
    gross = (positions * execution_returns).sum(axis=1, min_count=1).fillna(0.0)
    turnover = positions.sub(positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    costs = turnover * cost_bps / 10_000.0
    return Backtest(candidate, gross - costs, gross, positions, turnover, costs)


def period_metrics(backtest: Backtest, benchmark: pd.Series, start: str, end: str) -> dict[str, Any]:
    returns = backtest.returns.loc[start:end].dropna()
    positions = backtest.positions.loc[start:end]
    trades = int((positions.sub(positions.shift(1).fillna(0.0)).abs() > 1e-10).sum().sum())
    return engine.summarize(returns, benchmark.loc[returns.index], trades)


def selection_table(
    candidates: dict[str, Backtest], benchmark: pd.Series
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    details: dict[str, list[dict[str, Any]]] = {}
    ranking: list[dict[str, Any]] = []
    for key, backtest in candidates.items():
        folds: list[dict[str, Any]] = []
        for fold in engine.FOLDS:
            metrics = period_metrics(backtest, benchmark, *fold["test"])
            folds.append({"name": fold["name"], "metrics": metrics})
        details[key] = folds
        valid = [row["metrics"] for row in folds if int(row["metrics"]["observations"]) >= 400]
        if len(valid) < 5:
            continue
        sharpes = np.asarray([float(row["sharpe"]) for row in valid], dtype=float)
        alpha_ts = np.asarray([float(row["alpha_t"]) for row in valid], dtype=float)
        betas = np.asarray([abs(float(row["beta"])) for row in valid], dtype=float)
        correlations = np.asarray([abs(float(row["correlation"])) for row in valid], dtype=float)
        positive_alpha = int(sum(float(row["alpha"]) > 0.0 for row in valid))
        score = (
            float(np.nanmedian(sharpes))
            + 0.15 * positive_alpha
            + 0.10 * float(np.nanmedian(alpha_ts))
            - 1.5 * float(np.nanmedian(betas))
            - 1.0 * float(np.nanmedian(correlations))
        )
        ranking.append(
            {
                "key": key,
                "score": score,
                "median_sharpe": float(np.nanmedian(sharpes)),
                "median_alpha_t": float(np.nanmedian(alpha_ts)),
                "positive_alpha_folds": positive_alpha,
            }
        )
    ranking.sort(key=lambda row: row["score"], reverse=True)
    return ranking, details


def remove_best_year_sharpe(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return float("nan")
    annual = clean.groupby(clean.index.year).sum()
    if len(annual) < 3:
        return float("nan")
    best_year = int(annual.idxmax())
    return annualised_sharpe(clean[clean.index.year != best_year])


def render_plots(
    output: Path,
    key: str,
    core: Backtest,
    external: Backtest,
    core_benchmark: pd.Series,
    external_benchmark: pd.Series,
    fold_details: list[dict[str, Any]],
    ranking_metrics: dict[str, dict[str, Any]],
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 5.2))
    for label, series in {
        "core": core.returns.loc[RESEARCH_START:],
        "external": external.returns.loc[RESEARCH_START:],
    }.items():
        axis.plot(series.index, (1.0 + series.fillna(0.0)).cumprod(), label=label)
    axis.set_title(f"Selected alpha candidate — {key}")
    axis.set_ylabel("Growth of 1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "equity_curves.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "equity_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4.8))
    for label, series in {
        "core": core.returns.loc[RESEARCH_START:],
        "external": external.returns.loc[RESEARCH_START:],
    }.items():
        equity = (1.0 + series.fillna(0.0)).cumprod()
        drawdown = equity.div(equity.cummax()).sub(1.0)
        axis.plot(drawdown.index, drawdown * 100.0, label=label)
    axis.set_title("Drawdowns")
    axis.set_ylabel("Drawdown (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "drawdowns.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "drawdowns.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    names = [row["name"] for row in fold_details]
    values = [float(row["metrics"]["sharpe"]) for row in fold_details]
    axis.bar(names, values)
    axis.axhline(0.0, linewidth=1)
    axis.set_title("Core walk-forward fold Sharpe")
    axis.set_ylabel("Sharpe")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots / "fold_sharpe.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "fold_sharpe.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5.2))
    for name, metrics in ranking_metrics.items():
        axis.scatter(float(metrics["beta"]), float(metrics["alpha"]) * 100.0, s=30)
        if name == key:
            axis.annotate(name.split("[")[0], (float(metrics["beta"]), float(metrics["alpha"]) * 100.0))
    axis.axhline(0.0, linewidth=1)
    axis.axvline(0.0, linewidth=1)
    axis.set_title("Candidate alpha versus SPY beta — core 2015–2024")
    axis.set_xlabel("SPY beta")
    axis.set_ylabel("Annual alpha (%)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots / "alpha_beta.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "alpha_beta.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4.5))
    axis.plot(core.turnover.loc[RESEARCH_START:].rolling(21).mean() * 100.0, label="core 21d average")
    axis.plot(external.turnover.loc[RESEARCH_START:].rolling(21).mean() * 100.0, label="external 21d average")
    axis.set_title("One-way turnover")
    axis.set_ylabel("Turnover (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "turnover.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "turnover.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    core_data, core_provenance = load_market_data(
        CORE_UNIVERSE, args.start, args.cache_dir / "core", args.lse_dir, args.prefer_lse
    )
    external_data, external_provenance = load_market_data(
        EXTERNAL_UNIVERSE, args.start, args.cache_dir / "external", args.lse_dir, args.prefer_lse
    )
    core_panel = engine.align_panel(core_data)
    external_panel = engine.align_panel(external_data)
    core_classes = pd.Series({symbol: CORE_UNIVERSE[symbol] for symbol in core_panel["close"].columns})
    external_classes = pd.Series({symbol: EXTERNAL_UNIVERSE[symbol] for symbol in external_panel["close"].columns})
    core_classes.loc["SPY"] = "benchmark"
    external_classes.loc["SPY"] = "benchmark"

    core_features = build_features(core_panel)
    external_features = build_features(external_panel)
    candidates = make_candidates()
    core_results = {
        candidate.key: backtest_candidate(candidate, core_features, core_classes)
        for candidate in candidates
    }
    external_results = {
        candidate.key: backtest_candidate(candidate, external_features, external_classes)
        for candidate in candidates
    }

    core_benchmark = core_features["execution_returns"]["SPY"]
    external_benchmark = external_features["execution_returns"]["SPY"]
    ranking, fold_details = selection_table(core_results, core_benchmark)
    if not ranking:
        raise RuntimeError("No candidate had five complete chronological folds")
    selected_key = ranking[0]["key"]
    selected_candidate = core_results[selected_key].candidate
    selected_core = core_results[selected_key]
    selected_external = external_results[selected_key]

    core_period_returns = selected_core.returns.loc[RESEARCH_START:RESEARCH_END]
    external_period_returns = selected_external.returns.loc[RESEARCH_START:RESEARCH_END]
    core_candidate_frame = pd.concat(
        {key: value.returns.loc[RESEARCH_START:RESEARCH_END] for key, value in core_results.items()},
        axis=1,
    ).fillna(0.0)
    external_candidate_frame = pd.concat(
        {key: value.returns.loc[RESEARCH_START:RESEARCH_END] for key, value in external_results.items()},
        axis=1,
    ).fillna(0.0)

    core_metrics = engine.summarize(
        core_period_returns,
        core_benchmark.loc[core_period_returns.index],
        int((selected_core.positions.loc[RESEARCH_START:RESEARCH_END].diff().abs() > 1e-10).sum().sum()),
    )
    external_metrics = engine.summarize(
        external_period_returns,
        external_benchmark.loc[external_period_returns.index],
        int((selected_external.positions.loc[RESEARCH_START:RESEARCH_END].diff().abs() > 1e-10).sum().sum()),
    )
    core_validation = {
        "white_reality_check_pvalue": white_reality_check(core_candidate_frame, selected_key),
        "deflated_sharpe_probability": deflated_sharpe_probability(
            core_period_returns, len(candidates)
        ),
        "pbo": probability_of_backtest_overfitting(core_candidate_frame),
    }
    external_validation = {
        "white_reality_check_pvalue": white_reality_check(external_candidate_frame, selected_key),
        "deflated_sharpe_probability": deflated_sharpe_probability(
            external_period_returns, len(candidates)
        ),
        "pbo": probability_of_backtest_overfitting(external_candidate_frame),
    }

    doubled_core = backtest_candidate(
        selected_candidate, core_features, core_classes, 2.0 * ONE_WAY_COST_BPS
    )
    doubled_external = backtest_candidate(
        selected_candidate, external_features, external_classes, 2.0 * ONE_WAY_COST_BPS
    )
    delayed_core = backtest_candidate(
        selected_candidate, core_features, core_classes, ONE_WAY_COST_BPS, extra_delay=1
    )
    delayed_external = backtest_candidate(
        selected_candidate, external_features, external_classes, ONE_WAY_COST_BPS, extra_delay=1
    )
    stress = {
        "double_cost_core_sharpe": annualised_sharpe(
            doubled_core.returns.loc[RESEARCH_START:RESEARCH_END]
        ),
        "double_cost_external_sharpe": annualised_sharpe(
            doubled_external.returns.loc[RESEARCH_START:RESEARCH_END]
        ),
        "extra_delay_core_sharpe": annualised_sharpe(
            delayed_core.returns.loc[RESEARCH_START:RESEARCH_END]
        ),
        "extra_delay_external_sharpe": annualised_sharpe(
            delayed_external.returns.loc[RESEARCH_START:RESEARCH_END]
        ),
        "remove_best_year_core_sharpe": remove_best_year_sharpe(core_period_returns),
        "remove_best_year_external_sharpe": remove_best_year_sharpe(external_period_returns),
    }

    diagnostic_core = period_metrics(
        selected_core,
        core_benchmark,
        DIAGNOSTIC_START,
        str(core_panel["close"].index.max().date()),
    )
    diagnostic_external = period_metrics(
        selected_external,
        external_benchmark,
        DIAGNOSTIC_START,
        str(external_panel["close"].index.max().date()),
    )
    positive_alpha_folds = sum(
        float(row["metrics"]["alpha"]) > 0.0 for row in fold_details[selected_key]
    )
    gates = {
        "core observations >= 2000": int(core_metrics["observations"]) >= 2000,
        "core trades >= 3000": int(core_metrics["round_trips"]) >= 3000,
        "core Sharpe >= 0.75": float(core_metrics["sharpe"]) >= 0.75,
        "core alpha t-stat >= 2.0": float(core_metrics["alpha_t"]) >= 2.0,
        "core absolute beta <= 0.10": abs(float(core_metrics["beta"])) <= 0.10,
        "core absolute correlation <= 0.15": abs(float(core_metrics["correlation"])) <= 0.15,
        "positive alpha in >= 4 folds": positive_alpha_folds >= 4,
        "external Sharpe >= 0.50": float(external_metrics["sharpe"]) >= 0.50,
        "external alpha t-stat >= 1.50": float(external_metrics["alpha_t"]) >= 1.50,
        "external absolute beta <= 0.10": abs(float(external_metrics["beta"])) <= 0.10,
        "external absolute correlation <= 0.20": abs(float(external_metrics["correlation"])) <= 0.20,
        "core White p <= 0.05": float(core_validation["white_reality_check_pvalue"]) <= 0.05,
        "core DSR >= 0.95": float(core_validation["deflated_sharpe_probability"]) >= 0.95,
        "core PBO <= 0.10": float(core_validation["pbo"]["pbo"]) <= 0.10,
        "external White p <= 0.05": float(external_validation["white_reality_check_pvalue"]) <= 0.05,
        "external DSR >= 0.95": float(external_validation["deflated_sharpe_probability"]) >= 0.95,
        "external PBO <= 0.10": float(external_validation["pbo"]["pbo"]) <= 0.10,
        "double-cost core Sharpe > 0.30": float(stress["double_cost_core_sharpe"]) > 0.30,
        "double-cost external Sharpe > 0": float(stress["double_cost_external_sharpe"]) > 0.0,
        "extra-delay core Sharpe > 0": float(stress["extra_delay_core_sharpe"]) > 0.0,
        "extra-delay external Sharpe > 0": float(stress["extra_delay_external_sharpe"]) > 0.0,
        "remove-best-year core Sharpe > 0.30": float(stress["remove_best_year_core_sharpe"]) > 0.30,
        "remove-best-year external Sharpe > 0": float(stress["remove_best_year_external_sharpe"]) > 0.0,
    }
    accepted = all(gates.values())

    ranking_metrics = {
        key: engine.summarize(
            result.returns.loc[RESEARCH_START:RESEARCH_END],
            core_benchmark.loc[RESEARCH_START:RESEARCH_END],
        )
        for key, result in core_results.items()
    }
    render_plots(
        args.output,
        selected_key,
        selected_core,
        selected_external,
        core_benchmark,
        external_benchmark,
        fold_details[selected_key],
        ranking_metrics,
    )

    pd.concat(
        {
            "core": selected_core.returns,
            "external": selected_external.returns,
            "core_spy": core_benchmark,
            "external_spy": external_benchmark,
        },
        axis=1,
    ).to_csv(args.output / "selected_returns.csv", index_label="date")
    core_candidate_frame.to_csv(args.output / "core_candidate_returns.csv", index_label="date")
    external_candidate_frame.to_csv(
        args.output / "external_candidate_returns.csv", index_label="date"
    )

    report = {
        "status": "validated_alpha" if accepted else "rejected_best_candidate",
        "accepted": accepted,
        "deployment_allowed": False,
        "selected": selected_key,
        "selected_definition": asdict(selected_candidate),
        "candidate_count": len(candidates),
        "selection_ranking": ranking,
        "core_folds": fold_details[selected_key],
        "core_2015_2024": core_metrics,
        "external_2015_2024": external_metrics,
        "core_validation": core_validation,
        "external_validation": external_validation,
        "stress": stress,
        "diagnostic_2025_onward": {
            "warning": "This interval was already inspected and is not an untouched holdout.",
            "core": diagnostic_core,
            "external": diagnostic_external,
        },
        "gates": gates,
        "data": {
            "core": core_provenance,
            "external": external_provenance,
            "prefer_lse": bool(args.prefer_lse),
        },
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    lines = [
        "# Causal cross-asset alpha hunt",
        "",
        f"**Verdict: {'VALIDATED ALPHA CANDIDATE' if accepted else 'REJECTED'}.**",
        "",
        f"Selected: `{selected_key}` from {len(candidates)} predeclared candidates.",
        "",
        "The strategy uses close-known flow information and enters at the next open. "
        "Returns are measured open-to-open, costs are charged to actual turnover, and "
        "weights are class- and beta-neutralised.",
        "",
        "## Core universe — 2015–2024",
        "",
        f"- Sharpe: **{float(core_metrics['sharpe']):.3f}**",
        f"- annual alpha: **{float(core_metrics['alpha']):.2%}**",
        f"- HAC alpha t-stat: **{float(core_metrics['alpha_t']):.2f}**",
        f"- SPY beta: **{float(core_metrics['beta']):.3f}**",
        f"- correlation: **{float(core_metrics['correlation']):.3f}**",
        f"- max drawdown: **{float(core_metrics['max_drawdown']):.2%}**",
        f"- observations: **{int(core_metrics['observations'])}**",
        f"- position changes: **{int(core_metrics['round_trips'])}**",
        "",
        "## Separate ETF replication — 2015–2024",
        "",
        f"- Sharpe: **{float(external_metrics['sharpe']):.3f}**",
        f"- annual alpha: **{float(external_metrics['alpha']):.2%}**",
        f"- HAC alpha t-stat: **{float(external_metrics['alpha_t']):.2f}**",
        f"- SPY beta: **{float(external_metrics['beta']):.3f}**",
        f"- correlation: **{float(external_metrics['correlation']):.3f}**",
        f"- max drawdown: **{float(external_metrics['max_drawdown']):.2%}**",
        "",
        "## Multiple-testing and overfit controls",
        "",
        f"- core White Reality Check p: **{float(core_validation['white_reality_check_pvalue']):.4f}**",
        f"- core deflated-Sharpe probability: **{float(core_validation['deflated_sharpe_probability']):.4f}**",
        f"- core PBO: **{float(core_validation['pbo']['pbo']):.4f}**",
        f"- external White Reality Check p: **{float(external_validation['white_reality_check_pvalue']):.4f}**",
        f"- external deflated-Sharpe probability: **{float(external_validation['deflated_sharpe_probability']):.4f}**",
        f"- external PBO: **{float(external_validation['pbo']['pbo']):.4f}**",
        "",
        "## Stress",
        "",
        f"- doubled-cost core Sharpe: **{float(stress['double_cost_core_sharpe']):.3f}**",
        f"- doubled-cost external Sharpe: **{float(stress['double_cost_external_sharpe']):.3f}**",
        f"- extra-delay core Sharpe: **{float(stress['extra_delay_core_sharpe']):.3f}**",
        f"- extra-delay external Sharpe: **{float(stress['extra_delay_external_sharpe']):.3f}**",
        f"- remove-best-year core Sharpe: **{float(stress['remove_best_year_core_sharpe']):.3f}**",
        f"- remove-best-year external Sharpe: **{float(stress['remove_best_year_external_sharpe']):.3f}**",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'} — {name}" for name, value in gates.items())
    lines.extend(
        [
            "",
            "The 2025+ interval is diagnostic only because it has already been inspected. "
            "Even a passing result remains paper-trading only until genuinely new data arrive.",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Causal, low-beta cross-asset alpha hunt")
    parser.add_argument("--output", type=Path, default=Path("results/alpha_hunt"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/alpha_hunt"))
    parser.add_argument("--lse-dir", type=Path, default=None)
    parser.add_argument("--prefer-lse", action="store_true")
    parser.add_argument("--start", default=START)
    args = parser.parse_args()
    try:
        report = run(args)
        print((args.output / "REPORT.md").read_text(encoding="utf-8"))
        return 0
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
