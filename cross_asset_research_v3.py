from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import cross_asset_research as engine
from cross_asset_research_v2 import load_market_data


def vectorized_strategy_from_score(
    score: pd.DataFrame,
    execution_returns: pd.DataFrame,
    betas: pd.DataFrame,
    classes: pd.Series,
    quantile: float,
    round_trip_cost_bps: float,
    gate: pd.Series | None = None,
) -> engine.StrategySeries:
    common = score.index.intersection(execution_returns.index).intersection(betas.index)
    score = score.loc[common]
    execution_returns = execution_returns.loc[common]
    betas = betas.loc[common]
    class_names = sorted(str(value) for value in classes.dropna().unique())
    raw = pd.DataFrame(0.0, index=common, columns=score.columns)
    valid_classes = pd.DataFrame(False, index=common, columns=class_names)

    for class_name in class_names:
        columns = [column for column in score if str(classes.get(column)) == class_name]
        if len(columns) < 4:
            continue
        available = score[columns].notna() & execution_returns[columns].notna() & betas[columns].notna()
        available_count = available.sum(axis=1)
        selected_count = np.floor(available_count * quantile).clip(lower=1)
        rank = score[columns].where(available).rank(axis=1, method="first", ascending=True)
        valid = available_count.ge(4)
        short_mask = rank.le(selected_count, axis=0) & valid.to_numpy()[:, None]
        long_mask = rank.gt(available_count - selected_count, axis=0) & valid.to_numpy()[:, None]
        long_count = long_mask.sum(axis=1).replace(0, np.nan)
        short_count = short_mask.sum(axis=1).replace(0, np.nan)
        class_weights = long_mask.div(long_count, axis=0) * 0.5 - short_mask.div(short_count, axis=0) * 0.5
        valid = valid & long_count.notna() & short_count.notna()
        raw.loc[:, columns] = class_weights.where(valid, 0.0)
        valid_classes.loc[:, class_name] = valid

    active_class_count = valid_classes.sum(axis=1).replace(0, np.nan)
    raw = raw.div(active_class_count, axis=0).fillna(0.0)

    # Closed-form projection onto the intersection of class-neutral and beta-neutral constraints.
    # Centering beta inside each class preserves every class sum at zero while removing total beta.
    centered_beta = pd.DataFrame(0.0, index=common, columns=score.columns)
    for class_name in class_names:
        columns = [column for column in score if str(classes.get(column)) == class_name]
        if not columns:
            continue
        active = raw[columns].abs() > 0
        count = active.sum(axis=1).replace(0, np.nan)
        class_beta_mean = betas[columns].where(active).sum(axis=1).div(count)
        centered_beta.loc[:, columns] = betas[columns].sub(class_beta_mean, axis=0).where(active, 0.0)

    beta_exposure = (raw * betas).sum(axis=1)
    denominator = (centered_beta * centered_beta).sum(axis=1)
    multiplier = beta_exposure.div(denominator.where(denominator > 1e-10, np.nan)).fillna(0.0)
    weights = raw - centered_beta.mul(multiplier, axis=0)
    gross = weights.abs().sum(axis=1).replace(0, np.nan)
    weights = weights.div(gross, axis=0).fillna(0.0)
    if gate is not None:
        weights = weights.where(gate.reindex(common).fillna(False), 0.0)

    gross_returns = (weights * execution_returns).sum(axis=1).fillna(0.0)
    gross_exposure = weights.abs().sum(axis=1)
    returns = gross_returns - gross_exposure * (round_trip_cost_bps / 10_000.0)
    active_columns = weights.columns[weights.abs().sum(axis=0) > 0]
    return engine.StrategySeries(
        returns=returns,
        gross_returns=gross_returns,
        turnover=2.0 * gross_exposure,
        positions=weights,
        round_trips=int((weights.abs() > 1e-10).sum().sum()),
        active_assets=len(active_columns),
        active_classes=int(classes.loc[active_columns].nunique()),
    )


engine.load_market_data = load_market_data
engine.strategy_from_score = vectorized_strategy_from_score

if __name__ == "__main__":
    raise SystemExit(engine.main())
