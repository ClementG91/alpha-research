from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import cross_asset_research as engine
from cross_asset_research_v2 import load_market_data
from cross_asset_research_v3 import vectorized_strategy_from_score
from cross_asset_momentum import CORE_UNIVERSE, EXTERNAL_UNIVERSE
from cross_asset_persistent import PersistentCandidate, evaluate as persistent_evaluate

START = "2025-01-01"
COST_BPS = 5.0


def hold(targets: pd.DataFrame, days: int) -> pd.DataFrame:
    mask = pd.Series(np.arange(len(targets)) % days == 0, index=targets.index)
    return targets.where(mask, np.nan).ffill().fillna(0.0).shift(1).fillna(0.0)


def from_positions(pos: pd.DataFrame, returns: pd.DataFrame, cost: float) -> engine.StrategySeries:
    returns = returns.reindex(pos.index)
    gross = (pos * returns).sum(axis=1).fillna(0.0)
    turnover = pos.sub(pos.shift(1).fillna(0.0)).abs().sum(axis=1)
    net = gross - turnover * cost / 10_000.0
    return engine.StrategySeries(
        net, gross, turnover, pos,
        int((pos.diff().abs().fillna(pos.abs()) > 1e-10).sum().sum()),
        int((pos.abs().sum() > 0).sum()), 0,
    )


def trend(panel: dict[str, pd.DataFrame], classes: pd.Series, cost: float) -> engine.StrategySeries:
    close = panel["close"]
    returns = close.pct_change(fill_method=None)
    vol = returns.rolling(60, min_periods=40).std() * math.sqrt(252)
    signal = (
        np.sign(close.pct_change(60, fill_method=None))
        + np.sign(close.pct_change(120, fill_method=None))
        + np.sign(close.shift(20).pct_change(252, fill_method=None))
    ) / 3.0
    raw = signal.div(vol.replace(0, np.nan)).fillna(0.0)
    raw["SPY"] = 0.0
    names = [name for name in sorted(classes.unique()) if name != "benchmark"]
    weights = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    for name in names:
        cols = classes.index[classes == name].intersection(raw.columns)
        if len(cols) < 4:
            continue
        weights[cols] = raw[cols].div(raw[cols].abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) / len(names)
    beta = engine.rolling_beta(returns, returns["SPY"], 252)
    weights["SPY"] = -(weights * beta).sum(axis=1).fillna(0.0)
    weights = weights.div(weights.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return from_positions(hold(weights, 20), returns, cost)


def low_vol(panel: dict[str, pd.DataFrame], classes: pd.Series, cost: float) -> engine.StrategySeries:
    returns = panel["close"].pct_change(fill_method=None)
    vol = returns.rolling(60, min_periods=40).std()
    beta = engine.rolling_beta(returns, returns["SPY"], 252)
    targets = vectorized_strategy_from_score(-vol, returns, beta, classes, 0.30, 0.0).positions
    return from_positions(hold(targets, 20), returns, cost)


def residual(panel: dict[str, pd.DataFrame], classes: pd.Series, cost: float) -> engine.StrategySeries:
    candidate = PersistentCandidate("residual_trend", tuple(sorted({
        "beta_window": 126.0, "lookback": 60.0, "quantile": 0.20,
        "rebalance": 10.0, "skip": 0.0, "vol_window": 60.0,
    }.items())))
    return persistent_evaluate(candidate, panel, classes, cost)


def build(panel: dict[str, pd.DataFrame], classes: pd.Series, cost: float = COST_BPS) -> tuple[engine.StrategySeries, dict[str, engine.StrategySeries]]:
    sleeves = {"trend": trend(panel, classes, cost), "low_vol": low_vol(panel, classes, cost), "residual": residual(panel, classes, cost)}
    ret = pd.concat({k: v.returns for k, v in sleeves.items()}, axis=1).mean(axis=1)
    gross = pd.concat({k: v.gross_returns for k, v in sleeves.items()}, axis=1).mean(axis=1)
    turnover = pd.concat({k: v.turnover for k, v in sleeves.items()}, axis=1).mean(axis=1)
    pos = sum((v.positions for v in sleeves.values())) / 3.0
    return engine.StrategySeries(ret, gross, turnover, pos, int((pos.diff().abs().fillna(pos.abs()) > 1e-10).sum().sum()), int((pos.abs().sum() > 0).sum()), 0), sleeves


def metrics(series: engine.StrategySeries, spy: pd.Series, period: tuple[str, str]) -> dict[str, Any]:
    ret = engine.slice_series(series.returns, period)
    pos = series.positions.loc[pd.Timestamp(period[0]):pd.Timestamp(period[1])]
    trades = int((pos.diff().abs().fillna(pos.abs()) > 1e-10).sum().sum())
    return engine.summarize(ret, spy.loc[ret.index], trades)


def delayed_or_inverted(sleeves: dict[str, engine.StrategySeries], panel: dict[str, pd.DataFrame], invert: bool, delay: int) -> pd.Series:
    returns = panel["close"].pct_change(fill_method=None)
    output = []
    for sleeve in sleeves.values():
        pos = sleeve.positions.shift(delay).fillna(0.0)
        if invert:
            pos = -pos
        output.append(from_positions(pos, returns, COST_BPS).returns)
    return pd.concat(output, axis=1).mean(axis=1)


def render(report: dict[str, Any], out: Path) -> None:
    value = report["external"]
    lines = ["# Fixed multi-premia cross-asset ensemble", "", f"**Verdict: {'PASS' if report['passed'] else 'FAIL'}.**", "",
             "No parameter search: equal weights in multi-horizon trend, cross-sectional low volatility and 60-day residual trend. Costs apply only to actual turnover.", "",
             "## Development folds", "", "| Fold | Sharpe | Alpha t | Beta | Corr | Trades |", "|---|---:|---:|---:|---:|---:|"]
    for fold in report["folds"]:
        m = fold["metrics"]
        lines.append(f"| {fold['name']} | {m['sharpe']:.3f} | {m['alpha_t']:.2f} | {m['beta']:.3f} | {m['correlation']:.3f} | {m['round_trips']} |")
    lines += ["", "## External ETF replication — 2025 onward", "",
              f"- Sharpe **{value['sharpe']:.3f}**, return **{value['annual_return']:.2%}**, alpha t **{value['alpha_t']:.2f}**;",
              f"- beta **{value['beta']:.3f}**, correlation **{value['correlation']:.3f}**, drawdown **{value['max_drawdown']:.2%}**;",
              f"- observations **{value['observations']}**, trades **{value['round_trips']}**, bootstrap p **{report['bootstrap_p']:.4f}**.", "",
              "## Sleeves", "", "| Sleeve | Sharpe | Alpha t | Return |", "|---|---:|---:|---:|"]
    for name, m in report["sleeves"].items():
        lines.append(f"| {name} | {m['sharpe']:.3f} | {m['alpha_t']:.2f} | {m['annual_return']:.2%} |")
    lines += ["", "## Stress", "",
              f"- doubled costs: **{report['stress']['double_cost']['sharpe']:.3f}** Sharpe;",
              f"- extra delay: **{report['stress']['delay']['sharpe']:.3f}**;",
              f"- inverted: **{report['stress']['inverted']['sharpe']:.3f}**.", "", "## Gates", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'} — {name}" for name, ok in report["gates"].items()]
    lines += ["", "A pass freezes the ensemble for forward paper monitoring only; the 2025+ range is already inspected."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    core, core_prov = load_market_data(CORE_UNIVERSE, args.start, args.cache_dir, args.lse_dir, args.prefer_lse)
    ext, ext_prov = load_market_data(EXTERNAL_UNIVERSE, args.start, args.cache_dir, args.lse_dir, args.prefer_lse)
    core_panel, ext_panel = engine.align_panel(core), engine.align_panel(ext)
    core_classes = pd.Series({s: CORE_UNIVERSE[s] for s in core_panel["close"].columns})
    ext_classes = pd.Series({s: EXTERNAL_UNIVERSE[s] for s in ext_panel["close"].columns})
    core_spy = core_panel["close"]["SPY"].pct_change(fill_method=None)
    ext_spy = ext_panel["close"]["SPY"].pct_change(fill_method=None)
    core_result, _ = build(core_panel, core_classes)
    folds = [{"name": f["name"], "metrics": metrics(core_result, core_spy, f["test"])} for f in engine.FOLDS]
    ext_result, sleeves = build(ext_panel, ext_classes)
    period = (START, str(ext_panel["close"].index.max().date()))
    external = metrics(ext_result, ext_spy, period)
    sleeve_metrics = {k: metrics(v, ext_spy, period) for k, v in sleeves.items()}
    doubled, _ = build(ext_panel, ext_classes, 2 * COST_BPS)
    delayed = engine.slice_series(delayed_or_inverted(sleeves, ext_panel, False, 1), period)
    inverted = engine.slice_series(delayed_or_inverted(sleeves, ext_panel, True, 0), period)
    stress = {
        "double_cost": metrics(doubled, ext_spy, period),
        "delay": engine.summarize(delayed, ext_spy.loc[delayed.index]),
        "inverted": engine.summarize(inverted, ext_spy.loc[inverted.index]),
    }
    bootstrap = engine.block_bootstrap_sharpe_pvalue(engine.slice_series(ext_result.returns, period), paths=5000)
    positive_folds = sum(float(f["metrics"]["alpha"]) > 0 for f in folds)
    gates = {
        "positive alpha in four of five folds": positive_folds >= 4,
        "at least 250 external days": external["observations"] >= 250,
        "at least 1,000 external trades": external["round_trips"] >= 1000,
        "external Sharpe at least 0.80": float(external["sharpe"]) >= 0.80,
        "external alpha t-stat at least 2.0": float(external["alpha_t"]) >= 2.0,
        "absolute beta at most 0.10": abs(float(external["beta"])) <= 0.10,
        "absolute correlation at most 0.10": abs(float(external["correlation"])) <= 0.10,
        "bootstrap p-value at most 0.05": bootstrap <= 0.05,
        "all sleeves profitable": all(float(m["annual_return"]) > 0 for m in sleeve_metrics.values()),
        "doubled-cost Sharpe above 0.40": float(stress["double_cost"]["sharpe"]) >= 0.40,
        "delay Sharpe positive": float(stress["delay"]["sharpe"]) > 0,
        "inverted ensemble unprofitable": float(stress["inverted"]["sharpe"]) < 0,
        "drawdown below 15%": float(external["max_drawdown"]) > -0.15,
    }
    report = {"folds": folds, "external": external, "sleeves": sleeve_metrics, "stress": stress,
              "bootstrap_p": bootstrap, "gates": gates, "passed": all(gates.values()),
              "deployment_allowed": False, "core_provenance": core_prov, "external_provenance": ext_prov}
    (args.output / "raw.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.output / "locked_ensemble.json").write_text(json.dumps({"weights": {"trend": 1/3, "low_vol": 1/3, "residual": 1/3}, "deployment_allowed": False}, indent=2), encoding="utf-8")
    render(report, args.output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/ensemble"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/stooq"))
    parser.add_argument("--lse-dir", type=Path, default=None)
    parser.add_argument("--prefer-lse", action="store_true")
    parser.add_argument("--start", default="2006-01-01")
    args = parser.parse_args()
    try:
        report = run(args)
        print((args.output / "REPORT.md").read_text(encoding="utf-8"))
        return 0 if report["passed"] else 2
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"fatal: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
