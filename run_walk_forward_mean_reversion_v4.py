from __future__ import annotations

import itertools
from typing import Any

import manifold_research as engine
import run_walk_forward_mean_reversion as walk
import run_walk_forward_mean_reversion_v2 as calibrated
import run_walk_forward_mean_reversion_v3 as bounded

TOP_ACTIVE_TRAIN = 12


def combinations(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[key] for key in keys))]


async def evaluate_fold(
    session: Any,
    family: dict[str, Any],
    strategy: dict[str, Any],
    fold: dict[str, Any],
) -> dict[str, Any]:
    parameter_sets = combinations(family["grid"])
    train_docs = [
        walk.freeze(strategy, params, f"{family['name']}_{fold['name']}_train_{rank}")
        for rank, params in enumerate(parameter_sets)
    ]
    train_rows = engine.rows(await engine.call(session, "run_batch", {
        "strategies": [{"strategy_json": doc} for doc in train_docs],
        "config": engine.cfg(*fold["train"], walk.INTERVAL),
        "lite": False,
        "max_parallelism": 0,
    }))

    active_train = [
        {"params": params, "train": row, "strategy_json": doc}
        for params, row, doc in zip(parameter_sets, train_rows, train_docs, strict=False)
        if engine.trades(row) > 0
    ]
    active_train.sort(
        key=lambda item: (
            engine.metric(item["train"], "sharpe", -99),
            engine.metric(item["train"], "alpha", -99),
            -abs(engine.metric(item["train"], "beta", 99)),
        ),
        reverse=True,
    )
    finalists = active_train[:TOP_ACTIVE_TRAIN]
    if not finalists:
        raise RuntimeError(f"No active train candidate for {family['name']} {fold['name']}")

    test_rows = engine.rows(await engine.call(session, "run_batch", {
        "strategies": [{"strategy_json": item["strategy_json"]} for item in finalists],
        "config": engine.cfg(*fold["test"], walk.INTERVAL),
        "lite": False,
        "max_parallelism": 0,
    }))
    candidates = []
    for item, test_row in zip(finalists, test_rows, strict=False):
        if engine.trades(test_row) == 0:
            continue
        candidates.append({
            **item,
            "test": test_row,
            "score": walk.fold_score(item["train"], test_row, 0.0),
        })
    candidates.sort(key=lambda row: row["score"], reverse=True)
    if not candidates:
        raise RuntimeError(f"No active unseen candidate for {family['name']} {fold['name']}")
    return {
        "name": fold["name"],
        "train_period": fold["train"],
        "test_period": fold["test"],
        "enumerated_parameters": len(parameter_sets),
        "active_train_candidates": len(active_train),
        **candidates[0],
    }


def main() -> int:
    calibrated.conditional_families = bounded.conditional_families
    calibrated.evaluate_fold = evaluate_fold
    return calibrated.main()


if __name__ == "__main__":
    raise SystemExit(main())
