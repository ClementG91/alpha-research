from __future__ import annotations

import math

import pandas as pd

import alpha_hunt
import cross_asset_research as engine


def sparse_class_neutral_weights(
    score: pd.Series,
    beta: pd.Series,
    classes: pd.Series,
    quantile: float,
) -> pd.Series:
    """Build balanced long/short books without requiring four simultaneous shocks per class."""
    raw = pd.Series(0.0, index=score.index)
    selections: list[tuple[pd.Index, pd.Index]] = []
    for class_name, members in classes.groupby(classes).groups.items():
        if str(class_name) == "benchmark":
            continue
        values = score.loc[list(members)].dropna()
        positive = values[values > 0.0].sort_values(ascending=False)
        negative = values[values < 0.0].sort_values()
        if positive.empty or negative.empty:
            continue
        target_count = max(1, int(math.floor(len(values) * quantile)))
        long_index = positive.index[: min(target_count, len(positive))]
        short_index = negative.index[: min(target_count, len(negative))]
        if len(long_index) and len(short_index):
            selections.append((long_index, short_index))
    if not selections:
        return raw
    class_gross = 1.0 / len(selections)
    for long_index, short_index in selections:
        raw.loc[long_index] = 0.5 * class_gross / len(long_index)
        raw.loc[short_index] = -0.5 * class_gross / len(short_index)
    return engine.project_constraints(raw, beta.reindex(raw.index), classes.reindex(raw.index))


def sparse_target_weights(
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
            weights.loc[timestamp] = pd.NA
            continue
        available = score.loc[timestamp].notna() & beta.loc[timestamp].notna()
        available &= classes.reindex(score.columns).notna()
        if int(available.sum()) < 4:
            continue
        weights.loc[timestamp, available] = sparse_class_neutral_weights(
            score.loc[timestamp, available],
            beta.loc[timestamp, available],
            classes.loc[available],
            quantile,
        )
    if rebalance > 1:
        weights = weights.ffill(limit=rebalance - 1).fillna(0.0)
    return weights.astype(float)


alpha_hunt.target_weights = sparse_target_weights

if __name__ == "__main__":
    raise SystemExit(alpha_hunt.main())
