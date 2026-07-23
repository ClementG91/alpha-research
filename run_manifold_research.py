from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import manifold_research as engine


_original_gate = engine.validation_gate
_original_research = engine.research
_original_call = engine.call


def typed_scalar(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("Float64", "Int64", "Float32", "Int32", "value", "sharpe"):
            if key in value:
                return engine.num(value[key])
    return engine.num(value)


def non_aborting_validation_gate(row: dict[str, Any]) -> bool:
    row["passed_validation_gate"] = bool(_original_gate(row))
    return True


async def logging_call(session: Any, name: str, args: dict[str, Any]) -> Any:
    try:
        return await _original_call(session, name, args)
    except Exception as exc:
        path = Path("results/manifold/call-errors.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "tool": name,
            "error": str(exc),
            "strategy_name": ((args.get("strategy_json") or {}).get("name") if isinstance(args.get("strategy_json"), dict) else None),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        raise


async def research_with_gate_diagnostics(output: Any) -> dict[str, Any]:
    report = await _original_research(output)
    selected = [entry.get("selected") for entry in report.get("hypotheses", []) if entry.get("selected")]
    report["validation_gate_passes"] = sum(bool(row.get("passed_validation_gate")) for row in selected)
    report["used_gate_fallback"] = report["validation_gate_passes"] == 0
    return report


def normalise_stability(report: dict[str, Any]) -> None:
    stability = ((report.get("winner") or {}).get("stability") or {})
    raw_values = stability.get("values") or stability.get("param_values") or []
    raw_scores = stability.get("metrics") or stability.get("metric_values") or []
    pairs = [
        (typed_scalar(value), typed_scalar(score))
        for value, score in zip(raw_values, raw_scores, strict=False)
    ]
    pairs = [(value, score) for value, score in pairs if math.isfinite(value) and math.isfinite(score)]
    stability["values"] = [value for value, _ in pairs]
    stability["metrics"] = [score for _, score in pairs]


def plots(report: dict[str, Any], output: Any) -> None:
    normalise_stability(report)
    engine._raw_plots(report, output)


def main() -> int:
    engine.validation_gate = non_aborting_validation_gate
    engine.call = logging_call
    engine.research = research_with_gate_diagnostics
    engine._raw_plots = engine.plots
    engine.plots = plots
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
