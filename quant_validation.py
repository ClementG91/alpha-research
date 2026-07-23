from __future__ import annotations

import itertools
import math
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def infer_annualisation(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
        return 252
    median_days = float(np.median(np.diff(index.values).astype("timedelta64[D]").astype(float)))
    if median_days <= 2.0:
        return 252
    if median_days <= 10.0:
        return 52
    return 12


def annualised_sharpe(returns: pd.Series, annualisation: int | None = None) -> float:
    sample = returns.dropna()
    if len(sample) < 2:
        return float("nan")
    annualisation = annualisation or infer_annualisation(sample.index)
    standard_deviation = sample.std(ddof=1)
    return float(sample.mean() / standard_deviation * math.sqrt(annualisation)) if standard_deviation > 0.0 else float("nan")


def hac_alpha(returns: pd.Series, benchmark: pd.Series, maxlags: int = 10) -> dict[str, float]:
    aligned = pd.concat([returns.rename("strategy"), benchmark.rename("benchmark")], axis=1).dropna()
    if len(aligned) < 100:
        return {"alpha": float("nan"), "alpha_t": float("nan"), "beta": float("nan"), "correlation": float("nan")}
    annualisation = infer_annualisation(aligned.index)
    model = sm.OLS(aligned["strategy"], sm.add_constant(aligned["benchmark"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return {
        "alpha": float(model.params.iloc[0] * annualisation),
        "alpha_t": float(model.tvalues.iloc[0]),
        "beta": float(model.params.iloc[1]),
        "correlation": float(aligned.corr().iloc[0, 1]),
    }


def circular_block_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    blocks = math.ceil(length / block)
    starts = rng.integers(0, length, size=blocks)
    return np.concatenate([(np.arange(start, start + block) % length) for start in starts])[:length]


def block_bootstrap_sharpe_pvalue(
    returns: pd.Series,
    paths: int = 5000,
    block: int = 10,
    seed: int = 20260723,
) -> float:
    sample = returns.dropna()
    values = sample.to_numpy(dtype=float)
    if len(values) < 100:
        return float("nan")
    annualisation = infer_annualisation(sample.index)
    observed = annualised_sharpe(sample, annualisation)
    centred = values - values.mean()
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(paths):
        draw = centred[circular_block_indices(len(centred), block, rng)]
        standard_deviation = draw.std(ddof=1)
        statistic = draw.mean() / standard_deviation * math.sqrt(annualisation) if standard_deviation > 0.0 else 0.0
        exceed += statistic >= observed
    return float((exceed + 1) / (paths + 1))


def white_reality_check(
    candidate_returns: pd.DataFrame,
    selected: str,
    paths: int = 4000,
    block: int = 10,
    seed: int = 9743,
) -> float:
    frame = candidate_returns.dropna(how="all").fillna(0.0)
    if selected not in frame or len(frame) < 100 or frame.shape[1] < 2:
        return float("nan")
    annualisation = infer_annualisation(frame.index)
    observed = annualised_sharpe(frame[selected], annualisation)
    values = frame.to_numpy(dtype=float)
    values -= values.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(paths):
        draw = values[circular_block_indices(len(values), block, rng)]
        standard_deviation = draw.std(axis=0, ddof=1)
        statistics = np.divide(draw.mean(axis=0), standard_deviation, out=np.zeros_like(standard_deviation), where=standard_deviation > 0.0)
        exceed += float(np.nanmax(statistics) * math.sqrt(annualisation)) >= observed
    return float((exceed + 1) / (paths + 1))


def expected_maximum_sharpe(trials: int, independent_trials: int | None = None) -> float:
    effective = max(2, independent_trials or trials)
    normal = NormalDist()
    gamma = 0.5772156649015329
    first = normal.inv_cdf(1.0 - 1.0 / effective)
    second = normal.inv_cdf(1.0 - 1.0 / (effective * math.e))
    return float((1.0 - gamma) * first + gamma * second)


def deflated_sharpe_probability(
    returns: pd.Series,
    trials: int,
    independent_trials: int | None = None,
) -> float:
    sample = returns.dropna()
    if len(sample) < 100:
        return float("nan")
    annualisation = infer_annualisation(sample.index)
    sharpe = annualised_sharpe(sample, annualisation)
    skewness = float(sample.skew())
    kurtosis = float(sample.kurtosis() + 3.0)
    benchmark = expected_maximum_sharpe(trials, independent_trials)
    denominator = math.sqrt(max(1e-12, 1.0 - skewness * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe**2))
    z = (sharpe - benchmark) * math.sqrt(len(sample) - 1.0) / denominator
    return float(NormalDist().cdf(z))


def probability_of_backtest_overfitting(candidate_returns: pd.DataFrame, partitions: int = 10) -> dict[str, Any]:
    """CSCV-style PBO estimate across predeclared candidates.

    Degenerate folds are treated conservatively as an out-of-sample failure rather
    than dropped or allowed to crash the validation pipeline.
    """
    frame = candidate_returns.dropna(how="all").fillna(0.0)
    if frame.shape[1] < 2 or len(frame) < partitions * 20 or partitions % 2:
        return {"pbo": float("nan"), "combinations": 0, "logits": []}
    groups = [indices for indices in np.array_split(np.arange(len(frame)), partitions) if len(indices)]
    half = partitions // 2
    logits: list[float] = []
    all_groups = set(range(partitions))
    worst_percentile = 1e-6
    for train_groups in itertools.combinations(range(partitions), half):
        test_groups = sorted(all_groups.difference(train_groups))
        train_indices = np.concatenate([groups[index] for index in train_groups])
        test_indices = np.concatenate([groups[index] for index in test_groups])
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        train_sharpes = train.mean().div(train.std(ddof=1).replace(0.0, np.nan))
        train_sharpes = train_sharpes.replace([np.inf, -np.inf], np.nan).dropna()
        if train_sharpes.empty:
            percentile = worst_percentile
        else:
            selected = train_sharpes.idxmax()
            test_sharpes = test.mean().div(test.std(ddof=1).replace(0.0, np.nan))
            test_sharpes = test_sharpes.replace([np.inf, -np.inf], np.nan).dropna().sort_values()
            if test_sharpes.empty or selected not in test_sharpes.index:
                percentile = worst_percentile
            else:
                rank = int(test_sharpes.index.get_loc(selected)) + 1
                percentile = (rank - 0.5) / len(test_sharpes)
                percentile = min(max(percentile, worst_percentile), 1.0 - worst_percentile)
        logits.append(float(math.log(percentile / (1.0 - percentile))))
    pbo = float(np.mean(np.asarray(logits) <= 0.0)) if logits else float("nan")
    return {"pbo": pbo, "combinations": len(logits), "logits": logits}


def chronological_subperiod_stability(returns: pd.Series, periods: int = 10) -> dict[str, Any]:
    sample = returns.dropna()
    chunks = [chunk for chunk in np.array_split(sample, periods) if len(chunk) >= 10]
    sharpes = [annualised_sharpe(chunk) for chunk in chunks]
    positive = float(np.mean(np.asarray(sharpes) > 0.0)) if sharpes else float("nan")
    return {"positive_fraction": positive, "sharpes": sharpes, "periods": len(sharpes)}


def evaluate_candidate_set(
    candidate_returns: pd.DataFrame,
    benchmark: pd.Series,
    selected: str,
    trials: int,
) -> dict[str, Any]:
    frame = candidate_returns.dropna(how="all").fillna(0.0)
    selected_returns = frame[selected]
    alpha = hac_alpha(selected_returns, benchmark)
    stability = chronological_subperiod_stability(selected_returns)
    metrics = {
        "observations": int(len(selected_returns)),
        "sharpe": annualised_sharpe(selected_returns),
        "block_bootstrap_pvalue": block_bootstrap_sharpe_pvalue(selected_returns),
        "white_reality_check_pvalue": white_reality_check(frame, selected),
        "deflated_sharpe_probability": deflated_sharpe_probability(selected_returns, trials),
        "pbo": probability_of_backtest_overfitting(frame),
        "subperiod_stability": stability,
        **alpha,
    }
    gates = {
        "observations": metrics["observations"] >= 1500,
        "sharpe": metrics["sharpe"] >= 0.8,
        "alpha_t": metrics["alpha_t"] >= 2.0,
        "deflated_sharpe": metrics["deflated_sharpe_probability"] >= 0.95,
        "pbo": metrics["pbo"]["pbo"] <= 0.10,
        "reality_check": metrics["white_reality_check_pvalue"] <= 0.05,
        "beta": abs(metrics["beta"]) <= 0.10,
        "correlation": abs(metrics["correlation"]) <= 0.20,
        "subperiod_stability": stability["positive_fraction"] >= 0.70,
    }
    return {"metrics": metrics, "gates": gates, "accepted_before_cost_delay_stress": all(gates.values())}
