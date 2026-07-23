from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import manifold_research as engine
import run_walk_forward_mean_reversion_v2 as calibrated
import run_walk_forward_mean_reversion_v3 as bounded
import run_walk_forward_mean_reversion_v4 as enumerated

_original_call = engine.call


def log_record(filename: str, record: dict[str, Any]) -> None:
    path = Path("results/walk_forward") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


async def logging_call(session: Any, name: str, args: dict[str, Any]) -> Any:
    try:
        return await _original_call(session, name, args)
    except Exception as exc:
        strategy = args.get("strategy_json")
        log_record("call-errors.jsonl", {
            "tool": name,
            "strategy": strategy.get("name") if isinstance(strategy, dict) else None,
            "strategies_count": len(args.get("strategies") or []),
            "error": str(exc),
        })
        raise


async def evaluate_fold(session: Any, family: dict[str, Any], strategy: dict[str, Any], fold: dict[str, Any]) -> dict[str, Any]:
    try:
        result = await enumerated.evaluate_fold(session, family, strategy, fold)
        log_record("fold-diagnostics.jsonl", {
            "family": family["name"],
            "fold": fold["name"],
            "status": "success",
            "enumerated": result.get("enumerated_parameters"),
            "active_train": result.get("active_train_candidates"),
            "train_trades": engine.trades(result.get("train")),
            "test_trades": engine.trades(result.get("test")),
            "test_sharpe": engine.metric(result.get("test"), "sharpe"),
        })
        return result
    except Exception as exc:
        log_record("fold-diagnostics.jsonl", {
            "family": family["name"],
            "fold": fold["name"],
            "status": "failure",
            "error": str(exc),
        })
        raise


def main() -> int:
    engine.call = logging_call
    calibrated.conditional_families = bounded.conditional_families
    calibrated.evaluate_fold = evaluate_fold
    return calibrated.main()


if __name__ == "__main__":
    raise SystemExit(main())
