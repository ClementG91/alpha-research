from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import cross_asset_research as engine
from cross_asset_research_v2 import load_market_data
from cross_asset_research_v3 import vectorized_strategy_from_score
from cross_asset_momentum import CORE_UNIVERSE, EXTERNAL_UNIVERSE

VALIDATION_START = "2025-01-01"


@dataclass(frozen=True)
class PersistentCandidate:
    family: str
    params: tuple[tuple[str, float], ...]

    @property
    def key(self) -> str:
        params = ",".join(f"{key}={value:g}" for key, value in self.params)
        return f"{self.family}[{params}]"

    def values(self) -> dict[str, float]:
        return dict(self.params)


def make_candidates() -> list[PersistentCandidate]:
    output: list[PersistentCandidate] = []
    for lookback in (20, 60, 120, 252):
        for skip in (0, 5, 20):
            if skip >= lookback:
                continue
            for quantile in (0.20, 0.30):
                for rebalance in (5, 10, 20):
                    for beta_window in (126, 252):
                        output.append(PersistentCandidate("residual_trend", tuple(sorted({
                            "lookback": float(lookback), "skip": float(skip),
                            "quantile": quantile, "rebalance": float(rebalance),
                            "beta_window": float(beta_window), "vol_window": 60.0,
                        }.items()))))
    for lookback in (1, 5, 10):
        for quantile in (0.20, 0.30):
            for rebalance in (5, 10):
                for vol_window in (20, 60):
                    output.append(PersistentCandidate("residual_reversion", tuple(sorted({
                        "lookback": float(lookback), "skip": 0.0,
                        "quantile": quantile, "rebalance": float(rebalance),
                        "beta_window": 126.0, "vol_window": float(vol_window),
                    }.items()))))
    for short_window in (20, 60):
        for long_window in (120, 252):
            for quantile in (0.20, 0.30):
                for rebalance in (5, 10, 20):
                    output.append(PersistentCandidate("dual_horizon_trend", tuple(sorted({
                        "short_window": float(short_window), "long_window": float(long_window),
                        "quantile": quantile, "rebalance": float(rebalance),
                        "beta_window": 252.0, "vol_window": 60.0,
                    }.items()))))
    return output


def target_weights(
    candidate: PersistentCandidate,
    panel: dict[str, pd.DataFrame],
    classes: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = candidate.values()
    close = panel["close"]
    returns = close.pct_change(fill_method=None)
    beta = engine.rolling_beta(returns, returns["SPY"], int(params["beta_window"]))
    residual = returns - beta.mul(returns["SPY"], axis=0)
    scale = residual.rolling(
        int(params["vol_window"]),
        min_periods=max(20, int(params["vol_window"]) // 2),
    ).std()

    if candidate.family in {"residual_trend", "residual_reversion"}:
        lookback = int(params["lookback"])
        skip = int(params["skip"])
        source = residual.shift(skip).rolling(lookback, min_periods=lookback).sum()
        score = source.div((scale * math.sqrt(lookback)).replace(0, np.nan))
        if candidate.family == "residual_reversion":
            score = -score
    elif candidate.family == "dual_horizon_trend":
        short_window = int(params["short_window"])
        long_window = int(params["long_window"])
        short_score = residual.rolling(short_window, min_periods=short_window).sum().div(
            (scale * math.sqrt(short_window)).replace(0, np.nan)
        )
        long_score = residual.shift(20).rolling(long_window, min_periods=long_window).sum().div(
            (scale * math.sqrt(long_window)).replace(0, np.nan)
        )
        agreement = np.sign(short_score).eq(np.sign(long_score))
        score = (0.5 * short_score + 0.5 * long_score).where(agreement)
    else:
        raise ValueError(candidate.family)

    daily_targets = vectorized_strategy_from_score(
        score,
        returns,
        beta,
        classes,
        params["quantile"],
        round_trip_cost_bps=0.0,
    ).positions
    rebalance = int(params["rebalance"])
    rebalance_mask = pd.Series(np.arange(len(daily_targets)) % rebalance == 0, index=daily_targets.index)
    scheduled = daily_targets.where(rebalance_mask, np.nan).ffill().fillna(0.0)
    positions = scheduled.shift(1).fillna(0.0)
    return positions, returns


def evaluate(
    candidate: PersistentCandidate,
    panel: dict[str, pd.DataFrame],
    classes: pd.Series,
    cost_bps: float,
) -> engine.StrategySeries:
    positions, returns = target_weights(candidate, panel, classes)
    gross = (positions * returns).sum(axis=1).fillna(0.0)
    turnover = positions.sub(positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = gross - turnover * (cost_bps / 10_000.0)
    active = positions.columns[positions.abs().sum(axis=0) > 0]
    return engine.StrategySeries(
        returns=net,
        gross_returns=gross,
        turnover=turnover,
        positions=positions,
        round_trips=int((positions.diff().abs().fillna(positions.abs()) > 1e-10).sum().sum()),
        active_assets=len(active),
        active_classes=int(classes.loc[active].nunique()),
    )


def build_grid(
    panel: dict[str, pd.DataFrame],
    classes: pd.Series,
    cost_bps: float,
) -> tuple[dict[str, engine.StrategySeries], dict[str, PersistentCandidate]]:
    series: dict[str, engine.StrategySeries] = {}
    definitions: dict[str, PersistentCandidate] = {}
    for candidate in make_candidates():
        result = evaluate(candidate, panel, classes, cost_bps)
        if result.round_trips:
            series[candidate.key] = result
            definitions[candidate.key] = candidate
    return series, definitions


def period_metrics(
    series: engine.StrategySeries,
    spy: pd.Series,
    period: tuple[str, str],
) -> dict[str, Any]:
    returns = engine.slice_series(series.returns, period)
    positions = series.positions.loc[pd.Timestamp(period[0]):pd.Timestamp(period[1])]
    trades = int((positions.diff().abs().fillna(positions.abs()) > 1e-10).sum().sum())
    return engine.summarize(returns, spy.loc[returns.index], trades)


def select_fold(
    candidates: dict[str, engine.StrategySeries],
    spy: pd.Series,
    fold: dict[str, Any],
) -> dict[str, Any]:
    ranking: list[dict[str, Any]] = []
    for key, series in candidates.items():
        train = period_metrics(series, spy, fold["train"])
        validation = period_metrics(series, spy, fold["validation"])
        if train["observations"] < 750 or validation["observations"] < 350:
            continue
        if train["round_trips"] < 300 or validation["round_trips"] < 100:
            continue
        score = 0.35 * engine.selection_score(train) + 0.65 * engine.selection_score(validation)
        stability = min(float(train["sharpe"]), float(validation["sharpe"]))
        ranking.append({
            "key": key, "train": train, "validation": validation,
            "score": score + 0.25 * stability,
        })
    ranking.sort(key=lambda row: row["score"], reverse=True)
    if not ranking:
        raise RuntimeError(f"{fold['name']}: no persistent candidate passed activity constraints")
    winner = dict(ranking[0])
    winner["test"] = period_metrics(candidates[winner["key"]], spy, fold["test"])
    winner["ranking_top10"] = [dict(row) for row in ranking[:10]]
    return winner


def choose_locked(folds: list[dict[str, Any]]) -> str:
    keys = [fold["key"] for fold in folds]
    counts = {key: keys.count(key) for key in set(keys)}
    maximum = max(counts.values())
    tied = [key for key, count in counts.items() if count == maximum]
    return max(tied, key=lambda key: np.median([
        fold["test"]["sharpe"] for fold in folds if fold["key"] == key
    ]))


def stress_metrics(
    candidate: PersistentCandidate,
    panel: dict[str, pd.DataFrame],
    classes: pd.Series,
    spy: pd.Series,
    period: tuple[str, str],
    cost_bps: float,
) -> dict[str, dict[str, Any]]:
    positions, returns = target_weights(candidate, panel, classes)

    def calculate(pos: pd.DataFrame, cost: float) -> dict[str, Any]:
        gross = (pos * returns).sum(axis=1).fillna(0.0)
        turnover = pos.sub(pos.shift(1).fillna(0.0)).abs().sum(axis=1)
        net = gross - turnover * (cost / 10_000.0)
        net = engine.slice_series(net, period)
        trades = int((pos.loc[net.index].diff().abs().fillna(pos.loc[net.index].abs()) > 1e-10).sum().sum())
        return engine.summarize(net, spy.loc[net.index], trades)

    return {
        "double_cost": calculate(positions, 2.0 * cost_bps),
        "extra_delay": calculate(positions.shift(1).fillna(0.0), cost_bps),
        "inverted": calculate(-positions, cost_bps),
    }


def class_metrics(
    candidate: PersistentCandidate,
    panel: dict[str, pd.DataFrame],
    classes: pd.Series,
    spy: pd.Series,
    period: tuple[str, str],
    cost_bps: float,
) -> dict[str, dict[str, Any]]:
    positions, returns = target_weights(candidate, panel, classes)
    output: dict[str, dict[str, Any]] = {}
    for class_name in sorted(value for value in classes.unique() if value != "benchmark"):
        columns = classes.index[classes == class_name].intersection(positions.columns)
        if len(columns) < 4:
            continue
        pos = positions[columns]
        gross = (pos * returns[columns]).sum(axis=1).fillna(0.0)
        turnover = pos.sub(pos.shift(1).fillna(0.0)).abs().sum(axis=1)
        net = engine.slice_series(gross - turnover * (cost_bps / 10_000.0), period)
        trades = int((pos.loc[net.index].diff().abs().fillna(pos.loc[net.index].abs()) > 1e-10).sum().sum())
        output[str(class_name)] = engine.summarize(net, spy.loc[net.index], trades)
    return output


def render(report: dict[str, Any], output: Path) -> None:
    validation = report["external_validation"]
    lines = [
        "# Persistent factor-neutral cross-asset research", "",
        f"**Historical replication verdict: {'PASS' if report['historical_replication_passed'] else 'FAIL'}.**", "",
        "This generation replaces daily round trips with positions held across deterministic 5–20 day rebalance intervals. Transaction costs are charged only on actual notional turnover.", "",
        "The 2025+ ETF window has been inspected by earlier experiments, so it is replication evidence rather than a pristine holdout. Live deployment remains disabled regardless of the historical result.", "",
        "## Protocol", "",
        f"- candidates: **{report['candidate_count']}**;",
        f"- core assets: **{report['core_assets']}**; external ETF assets: **{report['external_assets']}**;",
        f"- locked candidate: `{report['locked_key']}`;",
        "- execution: next close-to-close return after a one-day signal lag;",
        "- constraints: class-neutral, dollar-neutral, rolling SPY-beta-neutral;",
        "- costs: 5 bps per unit of one-way turnover.", "",
        "## Five unseen development folds", "",
        "| Fold | Candidate | Sharpe | Alpha | Alpha t | Beta | Corr | Trades |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in report["folds"]:
        metrics = fold["test"]
        lines.append(
            f"| {fold['name']} | `{fold['key']}` | {metrics['sharpe']:.3f} | {metrics['alpha']:.2%} | {metrics['alpha_t']:.2f} | {metrics['beta']:.3f} | {metrics['correlation']:.3f} | {metrics['round_trips']} |"
        )
    lines += [
        "", "## External ETF replication — 2025 onward", "",
        f"- observations: **{validation['observations']}**; trades: **{validation['round_trips']}**;",
        f"- Sharpe: **{validation['sharpe']:.3f}**; annualised return: **{validation['annual_return']:.2%}**;",
        f"- alpha: **{validation['alpha']:.2%}**, HAC t-stat **{validation['alpha_t']:.2f}**;",
        f"- SPY beta: **{validation['beta']:.3f}**, correlation **{validation['correlation']:.3f}**;",
        f"- max drawdown: **{validation['max_drawdown']:.2%}**;",
        f"- block-bootstrap p: **{report['bootstrap_p']:.4f}**; max-candidate p: **{report['multiple_test_p']:.4f}**;", "",
        "## External class replication", "",
        "| Class | Sharpe | Alpha t | Return | Trades |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["class_metrics"].items():
        lines.append(
            f"| {name} | {metrics['sharpe']:.3f} | {metrics['alpha_t']:.2f} | {metrics['annual_return']:.2%} | {metrics['round_trips']} |"
        )
    lines += [
        "", "## Stress", "",
        f"- doubled-cost Sharpe: **{report['stress']['double_cost']['sharpe']:.3f}**;",
        f"- extra-delay Sharpe: **{report['stress']['extra_delay']['sharpe']:.3f}**;",
        f"- inverted-position Sharpe: **{report['stress']['inverted']['sharpe']:.3f}**;", "",
        "## Gates", "",
    ]
    lines += [f"- {'PASS' if value else 'FAIL'} — {name}" for name, value in report["gates"].items()]
    lines += [
        "", "## Status", "",
        "A historical pass promotes the strategy to frozen paper monitoring only. It cannot be called pristine temporal alpha because the current date range has already been inspected.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    core_data, core_provenance = load_market_data(
        CORE_UNIVERSE, args.start, args.cache_dir, args.lse_dir, args.prefer_lse,
    )
    external_data, external_provenance = load_market_data(
        EXTERNAL_UNIVERSE, args.start, args.cache_dir, args.lse_dir, args.prefer_lse,
    )
    core_panel = engine.align_panel(core_data)
    external_panel = engine.align_panel(external_data)
    core_classes = pd.Series({symbol: CORE_UNIVERSE[symbol] for symbol in core_panel["close"].columns})
    external_classes = pd.Series({symbol: EXTERNAL_UNIVERSE[symbol] for symbol in external_panel["close"].columns})
    core_spy = core_panel["close"]["SPY"].pct_change(fill_method=None)
    external_spy = external_panel["close"]["SPY"].pct_change(fill_method=None)

    core_grid, definitions = build_grid(core_panel, core_classes, args.cost_bps)
    folds = [select_fold(core_grid, core_spy, fold) | {"name": fold["name"]} for fold in engine.FOLDS]
    locked_key = choose_locked(folds)
    locked = definitions[locked_key]

    external_series = evaluate(locked, external_panel, external_classes, args.cost_bps)
    external_period = (VALIDATION_START, str(external_panel["close"].index.max().date()))
    external_metrics = period_metrics(external_series, external_spy, external_period)
    external_returns = engine.slice_series(external_series.returns, external_period)
    external_grid, _ = build_grid(external_panel, external_classes, args.cost_bps)
    candidate_returns = pd.concat({
        key: engine.slice_series(value.returns, external_period)
        for key, value in external_grid.items()
    }, axis=1)
    bootstrap_p = engine.block_bootstrap_sharpe_pvalue(external_returns, paths=3000)
    multiple_test_p = engine.max_sharpe_multiple_test_pvalue(
        candidate_returns, float(external_metrics["sharpe"]), paths=2000,
    )
    stresses = stress_metrics(
        locked, external_panel, external_classes, external_spy,
        external_period, args.cost_bps,
    )
    classes = class_metrics(
        locked, external_panel, external_classes, external_spy,
        external_period, args.cost_bps,
    )
    positive_folds = sum(float(fold["test"]["alpha"]) > 0 for fold in folds)
    positive_classes = sum(float(metrics["annual_return"]) > 0 for metrics in classes.values())
    gates = {
        "positive alpha in at least four of five unseen folds": positive_folds >= 4,
        "external replication has at least 250 days": external_metrics["observations"] >= 250,
        "external replication has at least 1,000 instrument trades": external_metrics["round_trips"] >= 1000,
        "external Sharpe at least 0.80": float(external_metrics["sharpe"]) >= 0.80,
        "external alpha HAC t-stat at least 2.0": float(external_metrics["alpha_t"]) >= 2.0,
        "absolute SPY beta at most 0.10": abs(float(external_metrics["beta"])) <= 0.10,
        "absolute SPY correlation at most 0.10": abs(float(external_metrics["correlation"])) <= 0.10,
        "block-bootstrap p-value at most 0.05": bootstrap_p <= 0.05,
        "max-candidate p-value at most 0.05": multiple_test_p <= 0.05,
        "at least four external classes are profitable": positive_classes >= 4,
        "doubled-cost Sharpe remains above 0.40": float(stresses["double_cost"]["sharpe"]) >= 0.40,
        "extra-delay Sharpe remains positive": float(stresses["extra_delay"]["sharpe"]) > 0,
        "inverted positions are unprofitable": float(stresses["inverted"]["sharpe"]) < 0,
        "maximum drawdown below 15%": float(external_metrics["max_drawdown"]) > -0.15,
    }
    report: dict[str, Any] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "candidate_count": len(make_candidates()),
        "core_assets": len(core_data), "external_assets": len(external_data),
        "locked_key": locked_key,
        "locked_definition": {"family": locked.family, "params": locked.values()},
        "folds": folds, "external_validation": external_metrics,
        "class_metrics": classes, "stress": stresses,
        "bootstrap_p": bootstrap_p, "multiple_test_p": multiple_test_p,
        "gates": gates, "historical_replication_passed": all(gates.values()),
        "deployment_allowed": False,
        "core_provenance": core_provenance, "external_provenance": external_provenance,
    }
    (args.output / "raw.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.output / "locked_persistent.json").write_text(json.dumps({
        "candidate": report["locked_definition"],
        "cost_bps_per_turnover_unit": args.cost_bps,
        "paper_start_after": str(external_panel["close"].index.max().date()),
        "deployment_allowed": False,
    }, indent=2), encoding="utf-8")
    render(report, args.output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent cross-asset factor-neutral research")
    parser.add_argument("--output", type=Path, default=Path("results/persistent"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/stooq"))
    parser.add_argument("--lse-dir", type=Path, default=None)
    parser.add_argument("--prefer-lse", action="store_true")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    try:
        report = run(args)
        print((args.output / "REPORT.md").read_text(encoding="utf-8"))
        return 0 if report["historical_replication_passed"] else 2
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
