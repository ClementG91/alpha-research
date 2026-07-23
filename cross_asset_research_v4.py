from __future__ import annotations

from typing import Any

import cross_asset_research as engine
import cross_asset_research_v3  # noqa: F401  # installs data and vectorized portfolio hooks


def serializable_select_in_fold(
    candidates: dict[str, engine.StrategySeries],
    spy: engine.pd.Series,
    fold: dict[str, Any],
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for key, series in candidates.items():
        train = engine.slice_series(series.returns, fold["train"])
        valid = engine.slice_series(series.returns, fold["validation"])
        total = max(len(series.returns), 1)
        train_metrics = engine.summarize(
            train,
            spy.loc[train.index],
            int(series.round_trips * len(train) / total),
        )
        valid_metrics = engine.summarize(
            valid,
            spy.loc[valid.index],
            int(series.round_trips * len(valid) / total),
        )
        if train_metrics["observations"] < 750 or valid_metrics["observations"] < 350:
            continue
        if train_metrics["round_trips"] < 1000 or valid_metrics["round_trips"] < 400:
            continue
        ranked.append({
            "key": key,
            "train": train_metrics,
            "validation": valid_metrics,
            "score": 0.4 * engine.selection_score(train_metrics) + 0.6 * engine.selection_score(valid_metrics),
        })
    ranked.sort(key=lambda row: row["score"], reverse=True)
    if not ranked:
        raise RuntimeError(f"{fold['name']}: no candidate met breadth/trade constraints")

    winner = dict(ranked[0])
    test_series = engine.slice_series(candidates[winner["key"]].returns, fold["test"])
    winner["test"] = engine.summarize(
        test_series,
        spy.loc[test_series.index],
        int(candidates[winner["key"]].round_trips * len(test_series) / max(len(candidates[winner["key"]].returns), 1)),
    )
    winner["ranking_top10"] = [dict(row) for row in ranked[:10]]
    return winner


engine.select_in_fold = serializable_select_in_fold

if __name__ == "__main__":
    raise SystemExit(engine.main())
