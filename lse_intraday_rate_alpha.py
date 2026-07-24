from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lse_intraday_macro_alpha as base
import lse_intraday_macro_runner as runner  # noqa: F401  # installs strict API and calendar adapters
from lse_vault import assert_secret_absent, write_json
from quant_validation import (
    annualised_sharpe,
    deflated_sharpe_probability,
    hac_alpha,
    probability_of_backtest_overfitting,
    white_reality_check,
)

START = "2020-01-01"
DEVELOPMENT_END = "2025-12-31"
DIAGNOSTIC_START = "2026-01-01"


@dataclass(frozen=True)
class RateCandidate:
    family: str
    hold_bars: int
    threshold: float = 2.0

    @property
    def key(self) -> str:
        return f"{self.family}[hold={self.hold_bars},threshold={self.threshold:g}]"


@dataclass
class RateBacktest:
    candidate: RateCandidate
    bar_returns: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series


RATE_FOLDS = (
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2025", "2024-01-01", "2025-12-31"),
)


def candidates() -> list[RateCandidate]:
    """Small predeclared family set; no post-result parameter expansion."""
    return [
        RateCandidate("unconditional_reversal", 2),
        RateCandidate("unconditional_reversal", 4),
        RateCandidate("rate_calm_reversal", 2),
        RateCandidate("rate_calm_reversal", 4),
        RateCandidate("rate_stress_continuation", 2),
        RateCandidate("rate_stress_continuation", 4),
        RateCandidate("us_open_rate_calm_reversal", 2),
        RateCandidate("us_open_rate_stress_continuation", 4),
    ]


def yield_close(frame: pd.DataFrame) -> pd.Series:
    if "close" not in frame:
        raise ValueError(f"yield frame contains no close column: {frame.columns.tolist()}")
    return pd.to_numeric(frame["close"], errors="coerce").dropna().sort_index()


def rate_regimes(
    intraday_index: pd.DatetimeIndex,
    yield_frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Map strictly lagged daily rates to intraday bars.

    The regime observed on business day t becomes tradable only on the next
    available yield observation. Weekend bars inherit Friday's already-lagged state.
    """
    if "US2Y" not in yield_frames or "US10Y" not in yield_frames:
        raise RuntimeError("US2Y and US10Y are required for lagged rate regimes")
    us2 = yield_close(yield_frames["US2Y"])
    us10 = yield_close(yield_frames["US10Y"])
    daily = pd.concat({"us2": us2, "us10": us10}, axis=1).sort_index().ffill(limit=5).dropna()
    change_2y = daily["us2"].diff()
    change_scale = change_2y.rolling(60, min_periods=30).std()
    rate_shock_z = change_2y.div(change_scale.replace(0.0, np.nan))
    rate_volatility = change_2y.rolling(20, min_periods=12).std()
    volatility_baseline = rate_volatility.rolling(252, min_periods=80).median()
    stress_raw = rate_volatility.gt(volatility_baseline) | rate_shock_z.abs().ge(1.50)
    curve = daily["us10"] - daily["us2"]

    observed = pd.DataFrame(
        {
            "stress": stress_raw,
            "calm": ~stress_raw,
            "curve_inverted": curve.lt(0.0),
            "curve": curve,
            "rate_shock_z": rate_shock_z,
            "rate_volatility": rate_volatility,
        },
        index=daily.index,
    ).shift(1)
    day_index = intraday_index.floor("D")
    mapped = observed.reindex(day_index, method="ffill")
    mapped.index = intraday_index
    mapped[["stress", "calm", "curve_inverted"]] = mapped[
        ["stress", "calm", "curve_inverted"]
    ].fillna(False).astype(bool)
    return mapped


def market_shock_features(market: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    volume_center = log_volume.rolling(rolling, min_periods=500).median().shift(1)
    volume_scale = log_volume.rolling(rolling, min_periods=500).std().shift(1)
    volume_z = log_volume.sub(volume_center).div(volume_scale.replace(0.0, np.nan))
    quality = range_z.ge(0.50) & (volume_z.ge(0.0) | volume_z.isna())
    magnitude = shock * (1.0 + range_z.clip(lower=0.0, upper=3.0))
    return magnitude, quality


def us_open_mask(index: pd.DatetimeIndex) -> pd.Series:
    local = index.tz_convert("America/New_York")
    minute = local.hour * 60 + local.minute
    return pd.Series((minute >= 9 * 60 + 30) & (minute < 11 * 60), index=index)


def raw_signal(
    candidate: RateCandidate,
    market: dict[str, pd.DataFrame],
    regimes: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    magnitude, quality = market_shock_features(market)
    condition = magnitude.abs().ge(candidate.threshold) & quality
    if not calendar.empty:
        condition &= ~base.macro_blackout(magnitude.index, calendar, bars=4).to_numpy()[:, None]

    if candidate.family == "unconditional_reversal":
        direction = -magnitude
    elif candidate.family == "rate_calm_reversal":
        condition &= regimes["calm"].to_numpy()[:, None]
        direction = -magnitude
    elif candidate.family == "rate_stress_continuation":
        condition &= regimes["stress"].to_numpy()[:, None]
        direction = magnitude
    elif candidate.family == "us_open_rate_calm_reversal":
        condition &= regimes["calm"].to_numpy()[:, None]
        condition &= us_open_mask(magnitude.index).to_numpy()[:, None]
        direction = -magnitude
    elif candidate.family == "us_open_rate_stress_continuation":
        condition &= regimes["stress"].to_numpy()[:, None]
        condition &= us_open_mask(magnitude.index).to_numpy()[:, None]
        direction = magnitude
    else:
        raise ValueError(candidate.family)
    return direction.where(condition, 0.0)


def build_backtest(
    candidate: RateCandidate,
    market: dict[str, pd.DataFrame],
    regimes: pd.DataFrame,
    calendar: pd.DataFrame,
    cost_multiplier: float = 1.0,
    extra_delay: int = 0,
) -> RateBacktest:
    classes = pd.Series({symbol: base.asset_class(symbol) for symbol in market["close"].columns})
    signal = raw_signal(candidate, market, regimes, calendar)
    targets = base.normalise_targets(signal, classes)
    positions = pd.DataFrame(0.0, index=targets.index, columns=targets.columns)
    for offset in range(candidate.hold_bars):
        positions = positions.add(targets.shift(1 + extra_delay + offset).fillna(0.0), fill_value=0.0)
    gross_exposure = positions.abs().sum(axis=1).replace(0.0, np.nan)
    positions = positions.div(gross_exposure.clip(lower=1.0), axis=0).fillna(0.0)

    execution_returns = market["open"].shift(-1).div(market["open"]).sub(1.0)
    gross = (positions * execution_returns).sum(axis=1, min_count=1).fillna(0.0)
    changes = positions.sub(positions.shift(1).fillna(0.0)).abs()
    costs = pd.Series(0.0, index=positions.index)
    for symbol in positions:
        costs += (
            changes[symbol]
            * base.ONE_WAY_COST_BPS[classes[symbol]]
            * cost_multiplier
            / 10_000.0
        )
    net = gross - costs
    daily = net.groupby(net.index.floor("D")).sum(min_count=1).fillna(0.0)
    return RateBacktest(candidate, net, daily, positions, changes.sum(axis=1), costs)


def period_metrics(
    backtest: RateBacktest,
    benchmark: pd.Series,
    start: str,
    end: str,
) -> dict[str, Any]:
    returns = backtest.daily_returns.loc[start:end]
    benchmark_aligned = benchmark.reindex(returns.index).fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity.div(equity.cummax()) - 1.0
    position_changes = backtest.positions.loc[start:end].sub(
        backtest.positions.loc[start:end].shift(1).fillna(0.0)
    ).abs()
    return {
        "observations": int(len(returns)),
        "trades": int((position_changes > 1e-10).sum().sum()),
        "sharpe": annualised_sharpe(returns),
        "annual_return": float(returns.mean() * 252.0),
        "annual_volatility": float(returns.std(ddof=1) * math.sqrt(252.0)),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else float("nan"),
        "hit_rate": float((returns > 0.0).mean()) if len(returns) else float("nan"),
        "turnover": float(backtest.turnover.loc[start:end].sum()),
        "cost_drag": float(backtest.costs.loc[start:end].sum()),
        **hac_alpha(returns, benchmark_aligned, maxlags=10),
    }


def rank_candidates(
    backtests: dict[str, RateBacktest],
    benchmark: pd.Series,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    details: dict[str, list[dict[str, Any]]] = {}
    ranking: list[dict[str, Any]] = []
    for key, backtest in backtests.items():
        folds = [
            {"name": name, "metrics": period_metrics(backtest, benchmark, start, end)}
            for name, start, end in RATE_FOLDS
        ]
        details[key] = folds
        values = [row["metrics"] for row in folds]
        if any(int(value["observations"]) < 450 for value in values):
            continue
        sharpes = np.asarray([float(value["sharpe"]) for value in values])
        alpha_ts = np.asarray([float(value["alpha_t"]) for value in values])
        positive_alpha = int(sum(float(value["alpha"]) > 0.0 for value in values))
        median_sharpe = float(np.nanmedian(sharpes))
        if not np.isfinite(median_sharpe):
            continue
        score = median_sharpe + 0.20 * positive_alpha + 0.10 * float(np.nanmedian(alpha_ts))
        ranking.append(
            {
                "key": key,
                "score": score,
                "median_sharpe": median_sharpe,
                "median_alpha_t": float(np.nanmedian(alpha_ts)),
                "positive_alpha_folds": positive_alpha,
            }
        )
    ranking.sort(key=lambda row: row["score"], reverse=True)
    return ranking, details


def remove_best_year_sharpe(returns: pd.Series) -> float:
    yearly = returns.groupby(returns.index.year).sum()
    if len(yearly) < 3:
        return float("nan")
    best_year = int(yearly.idxmax())
    return annualised_sharpe(returns[returns.index.year != best_year])


def event_diagnostics(
    market: dict[str, pd.DataFrame],
    calendar: pd.DataFrame,
    benchmark: pd.Series,
) -> dict[str, Any]:
    """Report 2025+ event behavior without using it for candidate selection."""
    if calendar.empty:
        return {"status": "unavailable", "reason": "no usable actual-versus-forecast events"}
    result: dict[str, Any] = {
        "status": "diagnostic_only",
        "warning": "Calendar actual/forecast coverage starts in 2025 and is not a historical validation sample.",
        "event_count": int(len(calendar)),
        "strategies": {},
    }
    for candidate in [item for item in base.candidates() if item.family.startswith("macro_")]:
        backtest = base.build_backtest(candidate, market, calendar)
        result["strategies"][candidate.key] = base.metrics(
            backtest,
            benchmark,
            "2025-01-01",
            str(backtest.daily_returns.index.max().date()),
        )
    return result


def render_plots(
    output: Path,
    selected: str,
    core: RateBacktest,
    external: RateBacktest,
    folds: list[dict[str, Any]],
    regimes: pd.DataFrame,
) -> None:
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(11, 5.2))
    for label, returns in {"core": core.daily_returns, "external": external.daily_returns}.items():
        axis.plot(returns.index, (1.0 + returns).cumprod(), label=label)
    axis.set_title(f"Lagged-rate-regime intraday candidate — {selected}")
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
    axis.set_title("Drawdowns after costs")
    axis.set_ylabel("Drawdown (%)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "drawdowns.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "drawdowns.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(
        [row["name"] for row in folds],
        [float(row["metrics"]["sharpe"]) for row in folds],
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_title("Core chronological fold Sharpe")
    axis.set_ylabel("Sharpe")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots / "fold_sharpes.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "fold_sharpes.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    daily_regime = regimes.resample("D").last()
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.plot(daily_regime.index, daily_regime["curve"], label="US 10Y-2Y curve")
    stress = daily_regime["stress"].fillna(False)
    axis.fill_between(
        daily_regime.index,
        daily_regime["curve"].min(),
        daily_regime["curve"].max(),
        where=stress,
        alpha=0.15,
        label="lagged rate-stress regime",
    )
    axis.axhline(0.0, linewidth=1)
    axis.set_title("Lagged US rate regime used by the strategy")
    axis.set_ylabel("Percentage points")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots / "rate_regimes.svg", format="svg", bbox_inches="tight")
    figure.savefig(plots / "rate_regimes.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(input_path: Path, output: Path) -> dict[str, Any]:
    fx = base.load_frames(input_path, "intraday_fx")
    futures = base.load_frames(input_path, "intraday_futures")
    benchmark_frames = base.load_frames(input_path, "intraday_benchmark")
    yield_frames = base.load_frames(input_path, "bond_yields")
    calendars = base.load_frames(input_path, "economic_calendar")
    calendar = base.standardise_calendar(calendars)
    benchmark_frame = benchmark_frames.get("SPY")
    if benchmark_frame is None:
        raise RuntimeError("SPY intraday benchmark is required")

    combined = {**fx, **futures}
    core_symbols = [symbol for symbol in base.FX_CORE + base.FUTURES_CORE if symbol in combined]
    external_symbols = [
        symbol for symbol in base.FX_EXTERNAL + base.FUTURES_EXTERNAL if symbol in combined
    ]
    if len(core_symbols) < 5 or len(external_symbols) < 5:
        raise RuntimeError(
            f"insufficient independent universes: core={core_symbols}, external={external_symbols}"
        )
    core_market = base.combine_market(combined, core_symbols)
    external_market = base.combine_market(combined, external_symbols)
    core_regimes = rate_regimes(core_market["close"].index, yield_frames)
    external_regimes = rate_regimes(external_market["close"].index, yield_frames)
    benchmark = base.benchmark_daily(benchmark_frame)

    core_backtests = {
        candidate.key: build_backtest(candidate, core_market, core_regimes, calendar)
        for candidate in candidates()
    }
    external_backtests = {
        candidate.key: build_backtest(candidate, external_market, external_regimes, calendar)
        for candidate in candidates()
    }
    ranking, fold_details = rank_candidates(core_backtests, benchmark)
    if not ranking:
        raise RuntimeError("No predeclared rate-regime candidate met the fold requirements")
    selected = str(ranking[0]["key"])
    core = core_backtests[selected]
    external = external_backtests[selected]
    core_frame = pd.concat(
        {key: value.daily_returns.loc[START:DEVELOPMENT_END] for key, value in core_backtests.items()},
        axis=1,
    ).fillna(0.0)
    external_frame = pd.concat(
        {
            key: value.daily_returns.loc[START:DEVELOPMENT_END]
            for key, value in external_backtests.items()
        },
        axis=1,
    ).fillna(0.0)

    core_metrics = period_metrics(core, benchmark, START, DEVELOPMENT_END)
    external_metrics = period_metrics(external, benchmark, START, DEVELOPMENT_END)
    diagnostic = period_metrics(
        external,
        benchmark,
        DIAGNOSTIC_START,
        str(external.daily_returns.index.max().date()),
    )
    double_cost = build_backtest(
        core.candidate, external_market, external_regimes, calendar, cost_multiplier=2.0
    )
    delayed = build_backtest(
        core.candidate, external_market, external_regimes, calendar, extra_delay=1
    )
    double_cost_metrics = period_metrics(double_cost, benchmark, START, DEVELOPMENT_END)
    delay_metrics = period_metrics(delayed, benchmark, START, DEVELOPMENT_END)
    reality = white_reality_check(core_frame, selected)
    external_reality = white_reality_check(external_frame, selected)
    dsr = deflated_sharpe_probability(core.daily_returns.loc[START:DEVELOPMENT_END], len(core_backtests))
    pbo = probability_of_backtest_overfitting(core_frame)
    without_best_year = remove_best_year_sharpe(
        external.daily_returns.loc[START:DEVELOPMENT_END]
    )
    selected_folds = fold_details[selected]
    positive_alpha_folds = sum(
        float(row["metrics"]["alpha"]) > 0.0 for row in selected_folds
    )

    gates = {
        "three positive-alpha core folds": positive_alpha_folds == 3,
        "median core fold Sharpe at least 0.50": float(ranking[0]["median_sharpe"]) >= 0.50,
        "external Sharpe at least 0.50": float(external_metrics["sharpe"]) >= 0.50,
        "external alpha t-stat at least 1.80": float(external_metrics["alpha_t"]) >= 1.80,
        "absolute external beta at most 0.10": abs(float(external_metrics["beta"])) <= 0.10,
        "at least 1,000 external trades": int(external_metrics["trades"]) >= 1_000,
        "core White Reality Check at most 0.05": float(reality) <= 0.05,
        "external Reality Check at most 0.10": float(external_reality) <= 0.10,
        "Deflated Sharpe probability at least 0.95": float(dsr) >= 0.95,
        "PBO at most 0.20": float(pbo["pbo"]) <= 0.20,
        "double-cost external Sharpe positive": float(double_cost_metrics["sharpe"]) > 0.0,
        "extra-delay external Sharpe positive": float(delay_metrics["sharpe"]) > 0.0,
        "external Sharpe positive without best year": float(without_best_year) > 0.0,
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
        "event_diagnostics_2025_onward": event_diagnostics(core_market, calendar, benchmark),
        "validation": {
            "core_white_reality_check": reality,
            "external_white_reality_check": external_reality,
            "deflated_sharpe_probability": dsr,
            "pbo": pbo,
            "external_remove_best_year_sharpe": without_best_year,
        },
        "stress": {"double_cost": double_cost_metrics, "extra_delay": delay_metrics},
        "gates": gates,
        "protocol": {
            "timeframe": "15m",
            "execution": "signal after completed bar; enter next bar open",
            "rate_information_lag": "one full daily yield observation",
            "rate_inputs": ["US2Y", "US10Y"],
            "core_universe": core_symbols,
            "external_universe": external_symbols,
            "candidate_count": len(core_backtests),
            "event_calendar_role": "2025+ diagnostic only; never used for historical candidate selection",
            "one_minute_excluded": "OHLC-only data cannot model spread, queue priority or adverse selection credibly.",
            "holdout_warning": "2026 is consumed diagnostic data; a passing result still requires new forward paper data.",
        },
    }

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", report)
    core_frame.to_csv(output / "core_candidate_daily_returns.csv", index_label="date")
    external_frame.to_csv(output / "external_candidate_daily_returns.csv", index_label="date")
    pd.concat({"core": core.daily_returns, "external": external.daily_returns}, axis=1).to_csv(
        output / "selected_daily_returns.csv", index_label="date"
    )
    render_plots(output, selected, core, external, selected_folds, core_regimes)
    lines = [
        "# LSE 15-minute alpha with lagged US rate regimes",
        "",
        f"**Verdict: {report['status']}.**",
        "",
        f"Selected: `{selected}` from {len(core_backtests)} predeclared candidates.",
        "",
        "The economic hypothesis is fixed: liquidity shocks should revert in calm rate regimes, while shocks may continue when lagged US two-year volatility is elevated.",
        "",
        "## Independent external replication",
        "",
        f"- Sharpe **{external_metrics['sharpe']:.3f}**, annual return **{external_metrics['annual_return']:.2%}**;",
        f"- HAC alpha t-stat **{external_metrics['alpha_t']:.2f}**, beta **{external_metrics['beta']:.3f}**;",
        f"- drawdown **{external_metrics['max_drawdown']:.2%}**, trades **{external_metrics['trades']}**.",
        "",
        "## Statistical controls",
        "",
        f"- core White Reality Check **{reality:.4f}**, external **{external_reality:.4f}**;",
        f"- Deflated Sharpe probability **{dsr:.4f}**, PBO **{pbo['pbo']:.3f}**;",
        f"- external Sharpe without best year **{without_best_year:.3f}**.",
        "",
        "## Gates",
        "",
        *[f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in gates.items()],
        "",
        "Actual-versus-forecast announcements begin only in 2025 and are reported separately as diagnostics, never as a 2020–2024 backtest.",
        "A passing result authorises forward paper monitoring only, not deployment.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert_secret_absent(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="LSE 15-minute alpha using strictly lagged US rate regimes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args.input, args.output)
        print(json.dumps({"status": report["status"], "selected": report["selected"]}, indent=2))
        return 0
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FATAL.txt").write_text(traceback.format_exc(), encoding="utf-8")
        assert_secret_absent(args.output)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
