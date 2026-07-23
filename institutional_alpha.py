from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

ANNUALIZATION = 12
DEFAULT_COST_BPS = 5.0
CRYPTO_COST_BPS = 10.0

ASSET_CLASS = {
    "SPY": "equity", "QQQ": "equity", "IWM": "equity", "VEA": "equity", "EEM": "equity",
    "VNQ": "real_asset", "IEF": "rates", "TLT": "rates", "GLD": "commodity", "DBC": "commodity",
    "UUP": "fx", "BTC-USD": "crypto", "ETH-USD": "crypto",
}
CRYPTO = {"BTC-USD", "ETH-USD"}

@dataclass(frozen=True)
class Backtest:
    name: str
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame | None
    turnover: pd.Series
    costs: pd.Series


def _rolling_compound(returns: pd.DataFrame, months: int) -> pd.DataFrame:
    return (1.0 + returns).rolling(months, min_periods=months).apply(np.prod, raw=True) - 1.0


def _normalise_gross(weights: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    denominator = weights.abs().sum(axis=1).replace(0.0, np.nan)
    return weights.div(denominator, axis=0).fillna(0.0) * gross


def _class_risk_budget(raw: pd.DataFrame, classes: dict[str, str]) -> pd.DataFrame:
    result = pd.DataFrame(0.0, index=raw.index, columns=raw.columns)
    class_names = sorted(set(classes.values()))
    for class_name in class_names:
        columns = [column for column in raw if classes.get(column) == class_name]
        if not columns:
            continue
        result.loc[:, columns] = _normalise_gross(raw[columns], 1.0) / len(class_names)
    return result


def tsmom_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """Fixed multi-horizon time-series momentum; no parameter selection."""
    signal = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for horizon in (1, 3, 6, 12):
        momentum = _rolling_compound(returns, horizon)
        scale = returns.rolling(36, min_periods=18).std() * math.sqrt(horizon)
        signal += momentum.div(scale.replace(0.0, np.nan)).clip(-2.0, 2.0)
    signal /= 4.0
    annual_vol = returns.rolling(36, min_periods=18).std() * math.sqrt(ANNUALIZATION)
    return _class_risk_budget(signal.div(annual_vol.replace(0.0, np.nan)), ASSET_CLASS)


def defensive_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """Long/cash inverse-volatility allocation gated by medium-term trend."""
    momentum_12 = _rolling_compound(returns, 12)
    momentum_3 = _rolling_compound(returns, 3)
    eligible = (momentum_12 > 0.0) & (momentum_3 > -0.05)
    annual_vol = returns.rolling(36, min_periods=18).std() * math.sqrt(ANNUALIZATION)
    raw = (1.0 / annual_vol.replace(0.0, np.nan)).where(eligible, 0.0)
    weights = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    for _ in range(8):
        weights = weights.clip(upper=0.25)
        residual = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
        open_slots = (weights > 0.0) & (weights < 0.25)
        count = open_slots.sum(axis=1).replace(0, np.nan)
        weights = weights.add(open_slots.div(count, axis=0).mul(residual, axis=0)).fillna(weights)
    return weights


def cross_sectional_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """Class-neutral cross-sectional momentum with a small short-horizon reversal term."""
    score = (
        0.45 * _rolling_compound(returns, 6).rank(axis=1, pct=True)
        + 0.45 * _rolling_compound(returns, 12).rank(axis=1, pct=True)
        + 0.10 * (-returns).rank(axis=1, pct=True)
        - 0.50
    )
    annual_vol = returns.rolling(36, min_periods=18).std() * math.sqrt(ANNUALIZATION)
    raw = score.div(annual_vol.replace(0.0, np.nan))
    result = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for class_name in sorted(set(ASSET_CLASS.values())):
        columns = [column for column in returns if ASSET_CLASS.get(column) == class_name]
        if len(columns) < 2:
            continue
        demeaned = raw[columns].sub(raw[columns].mean(axis=1), axis=0)
        result.loc[:, columns] = _normalise_gross(demeaned, 1.0) / len(set(ASSET_CLASS.values()))
    return result


def pca_residual_weights(returns: pd.DataFrame) -> pd.DataFrame:
    """Rolling PCA residual trend, refitted using strictly prior observations."""
    result = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    for index in range(60, len(returns)):
        history = returns.iloc[index - 60:index]
        columns = list(history.columns[history.count() >= 36])
        if len(columns) < 6:
            continue
        sample = history[columns].fillna(0.0)
        scale = sample.std().replace(0.0, np.nan)
        standardised = sample.sub(sample.mean()).div(scale).fillna(0.0)
        components = min(3, len(columns) - 1)
        pca = PCA(n_components=components, random_state=0).fit(standardised)
        residual = pd.DataFrame(
            standardised.to_numpy() - pca.inverse_transform(pca.transform(standardised)),
            index=standardised.index,
            columns=columns,
        )
        score = 0.5 * residual.tail(6).sum() + 0.5 * residual.tail(12).sum()
        raw = score.div(residual.tail(36).std().replace(0.0, np.nan))
        row = pd.Series(0.0, index=returns.columns)
        for class_name in sorted(set(ASSET_CLASS.values())):
            members = [column for column in columns if ASSET_CLASS.get(column) == class_name]
            if len(members) < 2:
                continue
            values = raw[members] - raw[members].mean()
            gross = values.abs().sum()
            if gross > 0.0:
                row.loc[members] = values / gross / len(set(ASSET_CLASS.values()))
        if row.abs().sum() > 0.0:
            row /= row.abs().sum()
        result.iloc[index] = row
    return result


def crisis_convexity_weights(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Rules-based crisis overlay using only lagged volatility and drawdown state."""
    spy = returns["SPY"]
    annual_vol = spy.rolling(12, min_periods=12).std() * math.sqrt(ANNUALIZATION)
    vol_z = (annual_vol - annual_vol.rolling(60, min_periods=24).mean()).div(
        annual_vol.rolling(60, min_periods=24).std().replace(0.0, np.nan)
    )
    drawdown = prices["SPY"].div(prices["SPY"].rolling(12, min_periods=12).max()) - 1.0
    stress = ((vol_z > 0.5) | (drawdown < -0.10)).astype(float)
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    weights.loc[:, ["IEF", "TLT", "GLD", "UUP"]] = stress.to_numpy()[:, None] * np.array([0.20, 0.20, 0.20, 0.15])
    weights.loc[:, ["SPY", "QQQ", "IWM", "EEM"]] = stress.to_numpy()[:, None] * np.array([-0.07, -0.07, -0.06, -0.05])
    weights += defensive_weights(returns).mul(1.0 - stress, axis=0)
    return weights


def apply_execution(weights: pd.DataFrame, returns: pd.DataFrame, name: str, cost_bps: float) -> Backtest:
    """Signals are shifted one full month before returns are earned."""
    executed = weights.shift(1).fillna(0.0)
    eligible = returns.notna().rolling(24, min_periods=24).sum().eq(24).shift(1, fill_value=False)
    executed = executed.where(eligible, 0.0)
    crypto_columns = [column for column in executed if column in CRYPTO]
    if crypto_columns:
        crypto_gross = executed[crypto_columns].abs().sum(axis=1)
        scale = (0.15 / crypto_gross).clip(upper=1.0).fillna(1.0)
        executed.loc[:, crypto_columns] = executed[crypto_columns].mul(scale, axis=0)
    executed = executed.clip(-0.25, 0.25)
    gross_exposure = executed.abs().sum(axis=1)
    executed = executed.mul((1.5 / gross_exposure).clip(upper=1.0).fillna(1.0), axis=0)
    changes = executed.diff().fillna(executed)
    non_crypto = [column for column in executed if column not in CRYPTO]
    costs = changes[non_crypto].abs().sum(axis=1) * cost_bps / 10_000.0
    if crypto_columns:
        costs += changes[crypto_columns].abs().sum(axis=1) * CRYPTO_COST_BPS / 10_000.0
    gross = (executed * returns).sum(axis=1, min_count=1).fillna(0.0)
    return Backtest(name, gross - costs, gross, executed, changes.abs().sum(axis=1), costs)


def fixed_ensemble(backtests: dict[str, Backtest]) -> Backtest:
    sleeve_returns = pd.concat({name: backtest.returns for name, backtest in backtests.items()}, axis=1)
    rolling_vol = sleeve_returns.rolling(36, min_periods=18).std() * math.sqrt(ANNUALIZATION)
    allocation = (1.0 / rolling_vol.replace(0.0, np.nan)).div(
        (1.0 / rolling_vol.replace(0.0, np.nan)).sum(axis=1), axis=0
    )
    allocation = allocation.clip(upper=0.35)
    allocation = allocation.div(allocation.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    returns = (allocation.shift(1).fillna(0.0) * sleeve_returns).sum(axis=1)
    return Backtest("ensemble", returns, returns, None, pd.Series(0.0, index=returns.index), pd.Series(0.0, index=returns.index))


def metrics(returns: pd.Series, benchmark: pd.Series) -> dict[str, float | int]:
    frame = pd.concat([returns.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    strategy = frame["strategy"]
    mean = strategy.mean() * ANNUALIZATION
    volatility = strategy.std(ddof=1) * math.sqrt(ANNUALIZATION)
    equity = (1.0 + strategy).cumprod()
    drawdown = equity.div(equity.cummax()) - 1.0
    result: dict[str, float | int] = {
        "observations": int(len(strategy)),
        "annual_return": float(mean),
        "annual_volatility": float(volatility),
        "sharpe": float(mean / volatility) if volatility > 0.0 else float("nan"),
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else float("nan"),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else float("nan"),
    }
    if len(frame) >= 24:
        model = sm.OLS(strategy.to_numpy(), sm.add_constant(frame["benchmark"].to_numpy())).fit(
            cov_type="HAC", cov_kwds={"maxlags": 6}
        )
        result.update({
            "alpha": float(model.params[0] * ANNUALIZATION),
            "alpha_t": float(model.tvalues[0]),
            "beta": float(model.params[1]),
            "correlation": float(frame.corr().iloc[0, 1]),
        })
    return result


def circular_block_bootstrap_pvalue(returns: pd.Series, paths: int = 4000, block: int = 6, seed: int = 20260723) -> float:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) < 60:
        return float("nan")
    observed = values.mean() / values.std(ddof=1) * math.sqrt(ANNUALIZATION)
    centred = values - values.mean()
    rng = np.random.default_rng(seed)
    exceed = 0
    blocks = math.ceil(len(values) / block)
    for _ in range(paths):
        starts = rng.integers(0, len(values), size=blocks)
        indices = np.concatenate([(np.arange(start, start + block) % len(values)) for start in starts])[:len(values)]
        sample = centred[indices]
        standard_deviation = sample.std(ddof=1)
        statistic = sample.mean() / standard_deviation * math.sqrt(ANNUALIZATION) if standard_deviation > 0.0 else 0.0
        exceed += statistic >= observed
    return float((exceed + 1) / (paths + 1))


def max_sharpe_reality_check(candidate_returns: pd.DataFrame, selected: str, paths: int = 3000, block: int = 6) -> float:
    frame = candidate_returns.dropna(how="all").fillna(0.0)
    selected_values = frame[selected]
    observed = selected_values.mean() / selected_values.std(ddof=1) * math.sqrt(ANNUALIZATION)
    values = frame.to_numpy(dtype=float)
    values -= values.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(7357)
    exceed = 0
    blocks = math.ceil(len(values) / block)
    for _ in range(paths):
        starts = rng.integers(0, len(values), size=blocks)
        indices = np.concatenate([(np.arange(start, start + block) % len(values)) for start in starts])[:len(values)]
        sample = values[indices]
        standard_deviation = sample.std(axis=0, ddof=1)
        sharpes = np.divide(sample.mean(axis=0), standard_deviation, out=np.zeros_like(standard_deviation), where=standard_deviation > 0.0)
        exceed += float(np.nanmax(sharpes) * math.sqrt(ANNUALIZATION)) >= observed
    return float((exceed + 1) / (paths + 1))


def probabilistic_sharpe_ratio(sharpe: float, observations: int, skewness: float, kurtosis: float, benchmark: float = 0.0) -> float:
    if observations < 3 or not np.isfinite(sharpe):
        return float("nan")
    denominator = math.sqrt(max(1e-12, 1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2))
    z = (sharpe - benchmark) * math.sqrt(observations - 1.0) / denominator
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def deflated_sharpe_probability(returns: pd.Series, trials: int) -> float:
    sample = returns.dropna()
    if len(sample) < 24:
        return float("nan")
    sharpe = sample.mean() / sample.std(ddof=1) * math.sqrt(ANNUALIZATION)
    expected_max = math.sqrt(max(0.0, 2.0 * math.log(max(2, trials)))) / math.sqrt(ANNUALIZATION)
    return probabilistic_sharpe_ratio(sharpe, len(sample), float(sample.skew()), float(sample.kurtosis() + 3.0), expected_max)


def fold_windows(index: pd.DatetimeIndex) -> list[tuple[str, str, str]]:
    return [
        ("2007–2010", "2007-01-01", "2010-12-31"),
        ("2011–2014", "2011-01-01", "2014-12-31"),
        ("2015–2018", "2015-01-01", "2018-12-31"),
        ("2019–2022", "2019-01-01", "2022-12-31"),
        ("2023–2026 consumed", "2023-01-01", str(index.max().date())),
    ]


def _save_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, format="svg", bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_plots(backtests: dict[str, Backtest], benchmark: pd.Series, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    returns = pd.concat({name: result.returns for name, result in backtests.items()}, axis=1).loc["2007-01-01":]
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for column in returns:
        axis.plot(returns.index, (1.0 + returns[column]).cumprod(), label=column)
    axis.set_title("Institutional sleeves — cumulative net return")
    axis.set_ylabel("Growth of 1")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    _save_figure(figure, output / "equity_curves.svg")

    figure, axis = plt.subplots(figsize=(11, 4.8))
    equity = (1.0 + returns["ensemble"]).cumprod()
    drawdown = equity.div(equity.cummax()) - 1.0
    axis.fill_between(drawdown.index, drawdown * 100.0, 0.0, alpha=0.45)
    axis.set_title("Fixed ensemble drawdown")
    axis.set_ylabel("Drawdown (%)")
    axis.grid(alpha=0.25)
    _save_figure(figure, output / "ensemble_drawdown.svg")

    figure, axis = plt.subplots(figsize=(11, 4.8))
    rolling = returns.rolling(36).mean().div(returns.rolling(36).std()).mul(math.sqrt(ANNUALIZATION))
    for column in ("tsmom", "crisis_convexity", "ensemble"):
        axis.plot(rolling.index, rolling[column], label=column)
    axis.axhline(0.0, linewidth=1)
    axis.set_title("Rolling 36-month Sharpe")
    axis.set_ylabel("Sharpe")
    axis.grid(alpha=0.25)
    axis.legend()
    _save_figure(figure, output / "rolling_sharpe.svg")

    metrics_rows = []
    for name, result in backtests.items():
        row = metrics(result.returns.loc["2007-01-01":], benchmark.loc["2007-01-01":])
        metrics_rows.append((name, float(row.get("beta", np.nan)), float(row.get("alpha", np.nan)) * 100.0, float(row.get("alpha_t", np.nan))))
    figure, axis = plt.subplots(figsize=(8, 5.5))
    for name, beta, alpha, alpha_t in metrics_rows:
        axis.scatter(beta, alpha, s=max(25.0, abs(alpha_t) * 35.0))
        axis.annotate(name, (beta, alpha), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.axvline(0.0, linewidth=1)
    axis.axhline(0.0, linewidth=1)
    axis.set_title("Annual alpha versus SPY beta")
    axis.set_xlabel("SPY beta")
    axis.set_ylabel("Annualised alpha (%)")
    axis.grid(alpha=0.25)
    _save_figure(figure, output / "alpha_beta.svg")

    ensemble = returns["ensemble"].dropna()
    table = pd.DataFrame({"year": ensemble.index.year, "month": ensemble.index.month, "return": ensemble.to_numpy()})
    heatmap = table.pivot(index="year", columns="month", values="return") * 100.0
    figure, axis = plt.subplots(figsize=(11, 6.5))
    image = axis.imshow(heatmap.to_numpy(), aspect="auto")
    axis.set_title("Fixed ensemble monthly returns (%)")
    axis.set_xticks(range(12), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    axis.set_yticks(range(len(heatmap.index)), heatmap.index.astype(str))
    figure.colorbar(image, ax=axis, label="Return (%)")
    _save_figure(figure, output / "monthly_heatmap.svg")

    turnover = pd.concat({name: result.turnover for name, result in backtests.items() if result.weights is not None}, axis=1)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(turnover.columns, turnover.mean() * 100.0)
    axis.set_title("Average monthly one-way turnover")
    axis.set_ylabel("Turnover (%)")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output / "turnover.svg")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(prices_path: Path, output: Path, cost_bps: float = DEFAULT_COST_BPS) -> dict[str, Any]:
    daily = pd.read_csv(prices_path, parse_dates=["Date"]).set_index("Date").sort_index()
    missing = sorted(set(ASSET_CLASS).difference(daily.columns))
    if missing:
        raise ValueError(f"Missing required assets: {missing}")
    prices = daily.resample("ME").last()
    returns = prices.pct_change(fill_method=None)
    sleeve_weights = {
        "tsmom": tsmom_weights(returns),
        "defensive": defensive_weights(returns),
        "cross_sectional": cross_sectional_weights(returns),
        "pca_residual": pca_residual_weights(returns),
        "crisis_convexity": crisis_convexity_weights(returns, prices),
    }
    backtests = {name: apply_execution(weights, returns, name, cost_bps) for name, weights in sleeve_weights.items()}
    backtests["ensemble"] = fixed_ensemble(backtests)
    start = "2007-01-01"
    candidate_frame = pd.concat({name: result.returns.loc[start:] for name, result in backtests.items()}, axis=1)
    report: dict[str, Any] = {
        "generated_from": str(prices_path.name),
        "source_sha256": sha256_file(prices_path),
        "protocol": {
            "frequency": "monthly",
            "execution_lag": "one full month",
            "parameter_search": False,
            "asset_cap": 0.25,
            "crypto_gross_cap": 0.15,
            "portfolio_gross_cap": 1.5,
            "one_way_cost_bps": cost_bps,
            "crypto_one_way_cost_bps": CRYPTO_COST_BPS,
            "note": "2023–2026 is already inspected and is not a pristine holdout.",
        },
        "strategies": {},
        "folds": {},
    }
    for name, result in backtests.items():
        full = result.returns.loc[start:]
        strategy_report: dict[str, Any] = {
            "development_2007_2022": metrics(full.loc[:"2022-12-31"], returns["SPY"].loc[:"2022-12-31"]),
            "consumed_2023_onward": metrics(full.loc["2023-01-01":], returns["SPY"].loc["2023-01-01":]),
            "full_history": metrics(full, returns["SPY"].loc[start:]),
            "bootstrap_pvalue_full": circular_block_bootstrap_pvalue(full),
            "deflated_sharpe_probability_full": deflated_sharpe_probability(full, len(backtests)),
        }
        if result.weights is not None:
            strategy_report["average_monthly_turnover"] = float(result.turnover.loc[start:].mean())
            strategy_report["total_cost_drag"] = float(result.costs.loc[start:].sum())
        report["strategies"][name] = strategy_report
    report["multiple_testing_pvalue_ensemble"] = max_sharpe_reality_check(candidate_frame, "ensemble")
    for label, first, last in fold_windows(returns.index):
        report["folds"][label] = {
            name: metrics(result.returns.loc[first:last], returns["SPY"].loc[first:last])
            for name, result in backtests.items()
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    render_plots(backtests, returns["SPY"], output / "plots")
    candidate_frame.to_csv(output / "strategy_returns_monthly.csv", index_label="date")
    pd.concat({name: result.turnover for name, result in backtests.items()}, axis=1).to_csv(output / "turnover_monthly.csv", index_label="date")
    (output / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Fixed institutional multi-asset research with no parameter search.")
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/institutional_alpha"))
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    args = parser.parse_args()
    report = run(args.prices, args.output, args.cost_bps)
    ensemble = report["strategies"]["ensemble"]
    print(json.dumps({"ensemble": ensemble, "multiple_testing_pvalue": report["multiple_testing_pvalue_ensemble"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
