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

CORE_UNIVERSE = dict(engine.UNIVERSE)
EXTERNAL_UNIVERSE: dict[str, str] = {
    "SPY": "benchmark",
    "VTI": "us_equity", "RSP": "us_equity", "MDY": "us_equity", "IJR": "us_equity",
    "VB": "us_equity", "VOO": "us_equity", "SPLV": "us_equity", "SCHD": "us_equity",
    "ACWI": "international", "VEA": "international", "VGK": "international", "VPL": "international",
    "MCHI": "international", "EWY": "international", "EWS": "international", "EWW": "international",
    "EWL": "international", "EWP": "international", "EWI": "international", "EZA": "international",
    "TUR": "international", "THD": "international", "EPOL": "international",
    "GOVT": "rates_credit", "VGIT": "rates_credit", "VGLT": "rates_credit", "SPTL": "rates_credit",
    "EMB": "rates_credit", "MUB": "rates_credit", "BKLN": "rates_credit", "JNK": "rates_credit",
    "VCIT": "rates_credit", "VCSH": "rates_credit",
    "GSG": "real_assets", "PDBC": "real_assets", "CPER": "real_assets", "CORN": "real_assets",
    "WEAT": "real_assets", "FXY": "real_assets", "FXB": "real_assets", "FXA": "real_assets",
    "FXC": "real_assets", "CYB": "real_assets", "REMX": "real_assets",
    "SOXX": "themes", "IGV": "themes", "XRT": "themes", "ITB": "themes", "XHB": "themes",
    "IHI": "themes", "IHF": "themes", "KIE": "themes", "KBE": "themes", "TAN": "themes",
    "ICLN": "themes", "PBW": "themes", "XME": "themes",
    "IWF": "equity_styles", "IWD": "equity_styles", "VTV": "equity_styles", "VUG": "equity_styles",
    "VBR": "equity_styles", "VOE": "equity_styles", "DGRO": "equity_styles", "SPHQ": "equity_styles",
    "SDY": "equity_styles", "SPYD": "equity_styles",
}
EXTERNAL_START = "2025-01-01"
COST_BPS_PER_SIDE = 5.0


@dataclass(frozen=True)
class MomentumCandidate:
    family: str
    params: tuple[tuple[str, float], ...]

    @property
    def key(self) -> str:
        return f"{self.family}[{','.join(f'{key}={value:g}' for key, value in self.params)}]"

    def values(self) -> dict[str, float]:
        return dict(self.params)


def make_candidates() -> list[MomentumCandidate]:
    candidates: list[MomentumCandidate] = []
    for lookback in (1, 3, 5, 10, 20):
        for quantile in (0.20, 0.30):
            for vol_window in (20, 60):
                for beta_window in (63, 126):
                    candidates.append(MomentumCandidate("residual_momentum", tuple(sorted({
                        "lookback": float(lookback), "quantile": quantile,
                        "vol_window": float(vol_window), "beta_window": float(beta_window),
                    }.items()))))
    for lookback in (1, 3, 5, 10):
        for quantile in (0.20, 0.30):
            for vol_window in (20, 60):
                for beta_window in (63, 126):
                    candidates.append(MomentumCandidate("intraday_momentum", tuple(sorted({
                        "lookback": float(lookback), "quantile": quantile,
                        "vol_window": float(vol_window), "beta_window": float(beta_window),
                    }.items()))))
    for lookback in (1, 3, 5):
        for quantile in (0.20, 0.30):
            for vol_window in (20, 60):
                for beta_window in (63, 126):
                    candidates.append(MomentumCandidate("overnight_momentum", tuple(sorted({
                        "lookback": float(lookback), "quantile": quantile,
                        "vol_window": float(vol_window), "beta_window": float(beta_window),
                    }.items()))))
    for lookback in (1, 3, 5, 10):
        for quantile in (0.20, 0.30):
            for dispersion_q in (0.50, 0.75):
                candidates.append(MomentumCandidate("dispersion_momentum", tuple(sorted({
                    "lookback": float(lookback), "quantile": quantile,
                    "dispersion_q": dispersion_q, "vol_window": 60.0, "beta_window": 126.0,
                }.items()))))
    return candidates


def build_candidate(
    candidate: MomentumCandidate,
    panel: dict[str, pd.DataFrame],
    classes: pd.Series,
    cost_bps: float = COST_BPS_PER_SIDE,
) -> engine.StrategySeries:
    params = candidate.values()
    close, open_ = panel["close"], panel["open"]
    close_returns = close.pct_change(fill_method=None)
    intraday = close.div(open_).sub(1.0)
    overnight = open_.div(close.shift(1)).sub(1.0)
    beta = engine.rolling_beta(close_returns, close_returns["SPY"], int(params["beta_window"]))
    trade_beta = beta.shift(1)
    residual = close_returns - beta.mul(close_returns["SPY"], axis=0)

    if candidate.family in {"residual_momentum", "dispersion_momentum"}:
        source = residual
    elif candidate.family == "intraday_momentum":
        source = intraday
    elif candidate.family == "overnight_momentum":
        source = overnight
    else:
        raise ValueError(candidate.family)

    lookback = int(params["lookback"])
    innovation = source.rolling(lookback, min_periods=lookback).sum()
    scale = source.rolling(
        int(params["vol_window"]),
        min_periods=max(10, int(params["vol_window"]) // 2),
    ).std() * math.sqrt(lookback)
    score = innovation.div(scale.replace(0, np.nan)).shift(1)
    gate: pd.Series | None = None
    if candidate.family == "dispersion_momentum":
        dispersion = residual.abs().median(axis=1)
        threshold = dispersion.rolling(252, min_periods=126).quantile(params["dispersion_q"])
        gate = dispersion.shift(1).ge(threshold.shift(1))

    return vectorized_strategy_from_score(
        score,
        intraday,
        trade_beta,
        classes,
        params["quantile"],
        round_trip_cost_bps=2.0 * cost_bps,
        gate=gate,
    )


def build_grid(
    panel: dict[str, pd.DataFrame],
    classes: pd.Series,
    cost_bps: float = COST_BPS_PER_SIDE,
) -> tuple[dict[str, engine.StrategySeries], dict[str, MomentumCandidate]]:
    series: dict[str, engine.StrategySeries] = {}
    definitions: dict[str, MomentumCandidate] = {}
    for candidate in make_candidates():
        result = build_candidate(candidate, panel, classes, cost_bps)
        if result.round_trips:
            series[candidate.key] = result
            definitions[candidate.key] = candidate
    return series, definitions


def actual_round_trips(series: engine.StrategySeries, period: tuple[str, str]) -> int:
    return int((series.positions.loc[pd.Timestamp(period[0]):pd.Timestamp(period[1])].abs() > 1e-10).sum().sum())


def metrics_for_period(
    series: engine.StrategySeries,
    spy: pd.Series,
    period: tuple[str, str],
) -> dict[str, Any]:
    returns = engine.slice_series(series.returns, period)
    return engine.summarize(returns, spy.loc[returns.index], actual_round_trips(series, period))


def select_fold(
    candidates: dict[str, engine.StrategySeries],
    spy: pd.Series,
    fold: dict[str, Any],
) -> dict[str, Any]:
    ranking: list[dict[str, Any]] = []
    for key, series in candidates.items():
        train = metrics_for_period(series, spy, fold["train"])
        validation = metrics_for_period(series, spy, fold["validation"])
        if train["observations"] < 750 or validation["observations"] < 350:
            continue
        if train["round_trips"] < 1000 or validation["round_trips"] < 400:
            continue
        ranking.append({
            "key": key,
            "train": train,
            "validation": validation,
            "score": 0.40 * engine.selection_score(train) + 0.60 * engine.selection_score(validation),
        })
    ranking.sort(key=lambda row: row["score"], reverse=True)
    if not ranking:
        raise RuntimeError(f"{fold['name']}: no momentum candidate passed activity constraints")
    winner = dict(ranking[0])
    winner["test"] = metrics_for_period(candidates[winner["key"]], spy, fold["test"])
    winner["ranking_top10"] = [dict(row) for row in ranking[:10]]
    return winner


def choose_locked_key(folds: list[dict[str, Any]]) -> str:
    keys = [row["key"] for row in folds]
    counts = {key: keys.count(key) for key in set(keys)}
    best_count = max(counts.values())
    tied = [key for key, count in counts.items() if count == best_count]
    if len(tied) == 1:
        return tied[0]
    return max(tied, key=lambda key: np.median([
        row["test"]["sharpe"] for row in folds if row["key"] == key
    ]))


def evaluate_positions(
    positions: pd.DataFrame,
    execution_returns: pd.DataFrame,
    spy: pd.Series,
    period: tuple[str, str],
    round_trip_cost_bps: float,
) -> tuple[pd.Series, dict[str, Any]]:
    dates = positions.index.intersection(execution_returns.index)
    positions = positions.loc[dates]
    execution_returns = execution_returns.loc[dates]
    gross = (positions * execution_returns).sum(axis=1).fillna(0.0)
    exposure = positions.abs().sum(axis=1)
    net = gross - exposure * (round_trip_cost_bps / 10_000.0)
    net = engine.slice_series(net, period)
    metrics = engine.summarize(
        net,
        spy.loc[net.index],
        int((positions.loc[net.index].abs() > 1e-10).sum().sum()),
    )
    return net, metrics


def per_class_metrics(
    series: engine.StrategySeries,
    panel: dict[str, pd.DataFrame],
    classes: pd.Series,
    spy: pd.Series,
    period: tuple[str, str],
    cost_bps: float,
) -> dict[str, dict[str, Any]]:
    intraday = panel["close"].div(panel["open"]).sub(1.0)
    output: dict[str, dict[str, Any]] = {}
    for class_name in sorted(value for value in classes.unique() if value != "benchmark"):
        columns = classes.index[classes == class_name].intersection(series.positions.columns)
        if len(columns) < 4:
            continue
        _, metrics = evaluate_positions(
            series.positions[columns], intraday[columns], spy, period, 2.0 * cost_bps,
        )
        output[str(class_name)] = metrics
    return output


def render_report(report: dict[str, Any], output: Path) -> None:
    external = report["external_asset_validation"]
    contaminated = report["contaminated_core_audit"]
    lines = [
        "# Cross-asset residual-momentum validation", "",
        f"**External-asset verdict: {'PASS' if report['external_validation_passed'] else 'FAIL'}.**", "",
        "The mean-reversion direction was rejected before this experiment. Momentum parameters were selected only on the original 2006–2024 universe. The 2025+ validation below uses a separate ETF universe that was not part of that parameter search.", "",
        "This is an external cross-sectional validation, not a pristine future-time validation: the 2025–2026 market regime had already been observed on the original asset set. Deployment remains prohibited until forward paper data accumulates.", "",
        "## Design", "",
        f"- predeclared momentum candidates: **{report['candidate_count']}**;",
        f"- development assets: **{report['core_assets']}**;",
        f"- external validation assets: **{report['external_assets']}** across **{report['external_classes']}** tradable classes;",
        f"- locked candidate: `{report['locked_key']}`;",
        "- execution: next-day open to close, 5 bps per side;",
        "- constraints: class-neutral, dollar-neutral and rolling SPY-beta-neutral.", "",
        "## Development folds on original assets", "",
        "| Fold | Candidate | Sharpe | Alpha | Alpha t | Beta | Corr SPY | Trades |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in report["folds"]:
        metrics = fold["test"]
        lines.append(
            f"| {fold['name']} | `{fold['key']}` | {metrics['sharpe']:.3f} | {metrics['alpha']:.2%} | {metrics['alpha_t']:.2f} | {metrics['beta']:.3f} | {metrics['correlation']:.3f} | {metrics['round_trips']} |"
        )
    lines += [
        "", "## External asset validation — 2025 onward", "",
        f"- daily observations: **{external['observations']}**;",
        f"- instrument-day round trips: **{external['round_trips']}**;",
        f"- Sharpe: **{external['sharpe']:.3f}**;",
        f"- annualised return: **{external['annual_return']:.2%}**;",
        f"- annualised alpha: **{external['alpha']:.2%}**, HAC t-stat **{external['alpha_t']:.2f}**;",
        f"- SPY beta: **{external['beta']:.3f}**, correlation **{external['correlation']:.3f}**;",
        f"- maximum drawdown: **{external['max_drawdown']:.2%}**;",
        f"- block-bootstrap p-value: **{report['external_bootstrap_p']:.4f}**;",
        f"- max-candidate multiple-testing p-value: **{report['external_multiple_test_p']:.4f}**;", "",
        "## Class breadth", "",
        "| Class | Sharpe | Alpha t | Return | Trades |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_name, metrics in report["class_metrics"].items():
        lines.append(
            f"| {class_name} | {metrics['sharpe']:.3f} | {metrics['alpha_t']:.2f} | {metrics['annual_return']:.2%} | {metrics['round_trips']} |"
        )
    lines += [
        "", "## Stress and placebo", "",
        f"- doubled costs Sharpe: **{report['stress']['double_cost']['sharpe']:.3f}**;",
        f"- one-day additional position delay Sharpe: **{report['stress']['extra_delay']['sharpe']:.3f}**;",
        f"- correctly costed inverted-position Sharpe: **{report['stress']['inverted']['sharpe']:.3f}**;", "",
        "## Previously consumed core-universe audit — diagnostic only", "",
        f"- Sharpe: **{contaminated['sharpe']:.3f}**; alpha t-stat: **{contaminated['alpha_t']:.2f}**; trades: **{contaminated['round_trips']}**.", "",
        "## Gates", "",
    ]
    lines += [f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in report["gates"].items()]
    lines += [
        "", "## Statistical interpretation", "",
        "Trade count measures breadth and implementability. Significance is calculated on daily portfolio returns with HAC standard errors and block bootstrap; simultaneous positions are not falsely treated as independent samples.",
        "", "A passing external-asset result creates a research candidate only. It does not restore a pristine temporal holdout and does not authorise live capital.",
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
    tradable_external_classes = [value for value in external_classes.unique() if value != "benchmark"]
    if len(external_data) < 40 or len(tradable_external_classes) < 5:
        raise RuntimeError("external asset universe lacks required breadth")

    core_grid, definitions = build_grid(core_panel, core_classes, args.cost_bps)
    core_spy = core_panel["close"]["SPY"].pct_change(fill_method=None)
    folds = [select_fold(core_grid, core_spy, fold) | {"name": fold["name"]} for fold in engine.FOLDS]
    locked_key = choose_locked_key(folds)
    locked_definition = definitions[locked_key]

    external_series = build_candidate(locked_definition, external_panel, external_classes, args.cost_bps)
    external_spy = external_panel["close"]["SPY"].pct_change(fill_method=None)
    external_period = (EXTERNAL_START, str(external_panel["close"].index.max().date()))
    external_metrics = metrics_for_period(external_series, external_spy, external_period)
    external_returns = engine.slice_series(external_series.returns, external_period)

    core_locked = core_grid[locked_key]
    contaminated_period = (EXTERNAL_START, str(core_panel["close"].index.max().date()))
    contaminated_metrics = metrics_for_period(core_locked, core_spy, contaminated_period)

    external_all, _ = build_grid(external_panel, external_classes, args.cost_bps)
    external_candidate_returns = pd.concat({
        key: engine.slice_series(series.returns, external_period)
        for key, series in external_all.items()
    }, axis=1)
    external_bootstrap_p = engine.block_bootstrap_sharpe_pvalue(external_returns, paths=3000)
    external_multiple_test_p = engine.max_sharpe_multiple_test_pvalue(
        external_candidate_returns,
        float(external_metrics["sharpe"]),
        paths=2000,
    )

    intraday = external_panel["close"].div(external_panel["open"]).sub(1.0)
    _, double_cost = evaluate_positions(
        external_series.positions, intraday, external_spy, external_period, 4.0 * args.cost_bps,
    )
    _, extra_delay = evaluate_positions(
        external_series.positions.shift(1).fillna(0.0), intraday, external_spy,
        external_period, 2.0 * args.cost_bps,
    )
    _, inverted = evaluate_positions(
        -external_series.positions, intraday, external_spy,
        external_period, 2.0 * args.cost_bps,
    )
    class_metrics = per_class_metrics(
        external_series, external_panel, external_classes, external_spy,
        external_period, args.cost_bps,
    )
    positive_classes = sum(
        math.isfinite(float(metrics["annual_return"])) and float(metrics["annual_return"]) > 0
        for metrics in class_metrics.values()
    )
    positive_alpha_folds = sum(float(fold["test"]["alpha"]) > 0 for fold in folds)
    gates = {
        "positive alpha in at least four of five development folds": positive_alpha_folds >= 4,
        "external validation has at least 250 daily observations": external_metrics["observations"] >= 250,
        "external validation has at least 1,000 instrument-day round trips": external_metrics["round_trips"] >= 1000,
        "external validation Sharpe at least 0.80": float(external_metrics["sharpe"]) >= 0.80,
        "external alpha HAC t-stat at least 2.0": float(external_metrics["alpha_t"]) >= 2.0,
        "absolute external SPY beta at most 0.10": abs(float(external_metrics["beta"])) <= 0.10,
        "absolute external SPY correlation at most 0.10": abs(float(external_metrics["correlation"])) <= 0.10,
        "external block-bootstrap p-value at most 0.05": external_bootstrap_p <= 0.05,
        "external max-candidate p-value at most 0.05": external_multiple_test_p <= 0.05,
        "at least four external asset classes are profitable": positive_classes >= 4,
        "doubled-cost Sharpe remains above 0.40": float(double_cost["sharpe"]) >= 0.40,
        "additional-delay Sharpe remains positive": float(extra_delay["sharpe"]) > 0,
        "inverted positions are unprofitable": float(inverted["sharpe"]) < 0,
        "external maximum drawdown below 15%": float(external_metrics["max_drawdown"]) > -0.15,
    }
    report: dict[str, Any] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "candidate_count": len(make_candidates()),
        "core_assets": len(core_data),
        "external_assets": len(external_data),
        "external_classes": len(tradable_external_classes),
        "locked_key": locked_key,
        "locked_definition": {"family": locked_definition.family, "params": locked_definition.values()},
        "folds": folds,
        "external_asset_validation": external_metrics,
        "contaminated_core_audit": contaminated_metrics,
        "external_bootstrap_p": external_bootstrap_p,
        "external_multiple_test_p": external_multiple_test_p,
        "class_metrics": class_metrics,
        "stress": {"double_cost": double_cost, "extra_delay": extra_delay, "inverted": inverted},
        "gates": gates,
        "external_validation_passed": all(gates.values()),
        "deployment_allowed": False,
        "core_provenance": core_provenance,
        "external_provenance": external_provenance,
    }
    (args.output / "raw.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.output / "locked_momentum.json").write_text(json.dumps({
        "candidate": report["locked_definition"],
        "selected_without": ["external asset universe", "2025+ external returns"],
        "cost_bps_per_side": args.cost_bps,
        "external_start": EXTERNAL_START,
        "deployment_allowed": False,
    }, indent=2), encoding="utf-8")
    render_report(report, args.output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fresh-universe residual-momentum validation")
    parser.add_argument("--output", type=Path, default=Path("results/momentum_external"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/stooq"))
    parser.add_argument("--lse-dir", type=Path, default=None)
    parser.add_argument("--prefer-lse", action="store_true")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--cost-bps", type=float, default=COST_BPS_PER_SIDE)
    args = parser.parse_args()
    try:
        report = run(args)
        print((args.output / "REPORT.md").read_text(encoding="utf-8"))
        return 0 if report["external_validation_passed"] else 2
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
