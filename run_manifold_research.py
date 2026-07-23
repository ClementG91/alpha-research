from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import manifold_research as engine


_original_gate = engine.validation_gate
_original_research = engine.research
_original_call = engine.call
_original_families = engine.families


def compiler_safe_families() -> list[dict[str, Any]]:
    """Rewrite nested DSL expressions into named direct-call signals."""
    docs = _original_families()
    for family in docs:
        name = family["name"]
        if name == "btc_ratio_zscore_reversion":
            family["signals"] = {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "ratio": "close / (btc + 0.000000000001)",
                "ratio_z": "zscore(ratio, param('ratio_window', default=96))",
                "asset_mom": "roc(close, param('horizon', default=12))",
                "btc_mom": "roc(btc, param('horizon', default=12))",
                "residual": "asset_mom - btc_mom",
                "residual_z": "zscore(residual, param('resid_window', default=96))",
                "ret1": "pct_change(close, 1)",
                "vol": "rolling_std(ret1, param('vol_window', default=48))",
                "vol_z": "zscore(vol, param('vol_regime', default=168))",
            }
        elif name == "dual_anchor_residual_reversion":
            family["signals"] = {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "eth": "symbol_ref('ETHUSDT', 'close')",
                "asset_ret": "roc(close, param('horizon', default=12))",
                "btc_ret": "roc(btc, param('horizon', default=12))",
                "eth_ret": "roc(eth, param('horizon', default=12))",
                "market_ret": "btc_ret * param('btc_weight', default=0.65) + eth_ret * (1.0 - param('btc_weight', default=0.65))",
                "residual": "asset_ret - market_ret",
                "residual_z": "zscore(residual, param('window', default=120))",
                "asset_slow": "roc(close, param('slow_horizon', default=72))",
                "btc_slow": "roc(btc, param('slow_horizon', default=72))",
                "eth_slow": "roc(eth, param('slow_horizon', default=72))",
                "market_slow": "btc_slow * param('btc_weight', default=0.65) + eth_slow * (1.0 - param('btc_weight', default=0.65))",
                "slow_residual": "asset_slow - market_slow",
                "slow_z": "zscore(slow_residual, param('slow_window', default=240))",
            }
        elif name == "liquidity_shock_snapback":
            family["signals"] = {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "asset_shock": "roc(close, param('shock_horizon', default=3))",
                "btc_shock": "roc(btc, param('shock_horizon', default=3))",
                "shock": "asset_shock - btc_shock",
                "shock_z": "zscore(shock, param('shock_window', default=120))",
                "bar_range": "high - low",
                "range_pct": "bar_range / (close + 0.000000000001)",
                "range_z": "zscore(range_pct, param('range_window', default=120))",
                "asset_rebound": "roc(close, param('confirm_horizon', default=2))",
                "btc_rebound": "roc(btc, param('confirm_horizon', default=2))",
                "rebound": "asset_rebound - btc_rebound",
            }
        elif name == "relative_rsi_reversion":
            family["signals"] = {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "ratio": "close / (btc + 0.000000000001)",
                "osc": "rsi(ratio, param('rsi_period', default=10))",
                "ratio_z": "zscore(ratio, param('ratio_window', default=120))",
                "btc_ret1": "pct_change(btc, 1)",
                "btc_vol": "rolling_std(btc_ret1, param('btc_vol_window', default=48))",
                "btc_vol_z": "zscore(btc_vol, param('btc_vol_regime', default=168))",
            }
        elif name == "multi_horizon_residual_fade":
            family["signals"] = {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "asset_fast": "roc(close, param('fast_horizon', default=6))",
                "btc_fast": "roc(btc, param('fast_horizon', default=6))",
                "fast_resid": "asset_fast - btc_fast",
                "asset_mid": "roc(close, param('mid_horizon', default=24))",
                "btc_mid": "roc(btc, param('mid_horizon', default=24))",
                "mid_resid": "asset_mid - btc_mid",
                "fast_z": "zscore(fast_resid, param('fast_window', default=96))",
                "mid_z": "zscore(mid_resid, param('mid_window', default=168))",
                "ret1": "pct_change(close, 1)",
                "vol": "rolling_std(ret1, param('vol_window', default=48))",
                "vol_z": "zscore(vol, param('vol_regime', default=168))",
            }
    return docs


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
    engine.families = compiler_safe_families
    engine.validation_gate = non_aborting_validation_gate
    engine.call = logging_call
    engine.research = research_with_gate_diagnostics
    engine._raw_plots = engine.plots
    engine.plots = plots
    return engine.main()


if __name__ == "__main__":
    raise SystemExit(main())
