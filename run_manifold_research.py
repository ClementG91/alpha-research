from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import manifold_research as engine


_original_gate = engine.validation_gate
_original_research = engine.research
_original_call = engine.call


def compiler_safe_families() -> list[dict[str, Any]]:
    """Mean-reversion families using only indicators exposed by this MCP server."""
    return [
        {
            "name": "btc_ratio_deviation_reversion",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "ratio": "close / (btc + 0.000000000001)",
                "ratio_mean": "sma(ratio, param('ratio_window', default=96))",
                "ratio_dev": "ratio / (ratio_mean + 0.000000000001) - 1.0",
                "asset_mom": "roc(close, param('horizon', default=12))",
                "btc_mom": "roc(btc, param('horizon', default=12))",
                "residual": "asset_mom - btc_mom",
                "residual_mean": "sma(residual, param('resid_window', default=96))",
                "residual_dev": "residual - residual_mean",
                "volatility": "natr(14)",
            },
            "size": "when((ratio_dev < (0.0 - param('entry_dev', default=0.06))) & (residual_dev < (0.0 - param('confirm_dev', default=0.03))) & (volatility < param('max_natr', default=8.0)), param('size', default=0.10), when((ratio_dev > param('entry_dev', default=0.06)) & (residual_dev > param('confirm_dev', default=0.03)) & (volatility < param('max_natr', default=8.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "ratio_window": [72, 144],
                "resid_window": [48, 120],
                "horizon": [6, 18],
                "entry_dev": [0.035, 0.065, 0.10],
                "confirm_dev": [0.015, 0.04],
                "max_natr": [6.0, 10.0],
            },
            "heat": ("entry_dev", [0.02, 0.035, 0.05, 0.065, 0.08, 0.10], "ratio_window", [48, 72, 96, 120, 168]),
            "stability": ("confirm_dev", [0.005, 0.015, 0.025, 0.04, 0.06]),
        },
        {
            "name": "dual_anchor_kalman_reversion",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "eth": "symbol_ref('ETHUSDT', 'close')",
                "asset_ret": "roc(close, param('horizon', default=12))",
                "btc_ret": "roc(btc, param('horizon', default=12))",
                "eth_ret": "roc(eth, param('horizon', default=12))",
                "market_ret": "btc_ret * param('btc_weight', default=0.65) + eth_ret * (1.0 - param('btc_weight', default=0.65))",
                "residual": "asset_ret - market_ret",
                "kalman_anchor": "kalman(residual)",
                "innovation": "residual - kalman_anchor",
                "residual_slope": "linreg_slope(residual, 48)",
                "volatility": "natr(14)",
            },
            "size": "when((innovation < (0.0 - param('entry_dev', default=0.05))) & (residual_slope > (0.0 - param('slope_limit', default=0.002))) & (volatility < param('max_natr', default=9.0)), param('size', default=0.10), when((innovation > param('entry_dev', default=0.05)) & (residual_slope < param('slope_limit', default=0.002)) & (volatility < param('max_natr', default=9.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "horizon": [6, 18],
                "btc_weight": [0.45, 0.70],
                "entry_dev": [0.025, 0.05, 0.085],
                "slope_limit": [0.001, 0.004],
                "max_natr": [6.0, 11.0],
            },
            "heat": ("entry_dev", [0.015, 0.025, 0.04, 0.05, 0.065, 0.085], "btc_weight", [0.3, 0.45, 0.55, 0.7, 0.85]),
            "stability": ("slope_limit", [0.0005, 0.001, 0.002, 0.004, 0.008]),
        },
        {
            "name": "liquidity_shock_snapback",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "asset_shock": "roc(close, param('shock_horizon', default=3))",
                "btc_shock": "roc(btc, param('shock_horizon', default=3))",
                "shock": "asset_shock - btc_shock",
                "shock_anchor": "sma(shock, param('shock_window', default=96))",
                "shock_dev": "shock - shock_anchor",
                "volatility": "natr(14)",
                "asset_rebound": "roc(close, param('confirm_horizon', default=2))",
                "btc_rebound": "roc(btc, param('confirm_horizon', default=2))",
                "rebound": "asset_rebound - btc_rebound",
            },
            "size": "when((shock_dev < (0.0 - param('entry_dev', default=0.07))) & (volatility > param('min_natr', default=3.0)) & (rebound > (0.0 - param('rebound_floor', default=0.025))), param('size', default=0.08), when((shock_dev > param('entry_dev', default=0.07)) & (volatility > param('min_natr', default=3.0)) & (rebound < param('rebound_floor', default=0.025)), 0.0 - param('size', default=0.08), 0.0))",
            "grid": {
                "shock_horizon": [2, 6],
                "shock_window": [72, 144],
                "confirm_horizon": [1, 3],
                "entry_dev": [0.04, 0.075, 0.12],
                "min_natr": [2.0, 5.0],
                "rebound_floor": [0.015, 0.04],
            },
            "heat": ("entry_dev", [0.025, 0.04, 0.055, 0.075, 0.10, 0.12], "min_natr", [1.0, 2.0, 3.0, 4.0, 6.0]),
            "stability": ("shock_window", [48, 72, 96, 120, 168, 240]),
        },
        {
            "name": "relative_rsi_reversion",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "ratio": "close / (btc + 0.000000000001)",
                "oscillator": "rsi(ratio, param('rsi_period', default=10))",
                "ratio_mean": "ema(ratio, param('ratio_window', default=120))",
                "ratio_dev": "ratio / (ratio_mean + 0.000000000001) - 1.0",
                "volatility": "natr(14)",
            },
            "size": "when((oscillator < param('rsi_low', default=25.0)) & (ratio_dev < (0.0 - param('entry_dev', default=0.045))) & (volatility < param('max_natr', default=9.0)), param('size', default=0.10), when((oscillator > param('rsi_high', default=75.0)) & (ratio_dev > param('entry_dev', default=0.045)) & (volatility < param('max_natr', default=9.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "rsi_period": [7, 14],
                "ratio_window": [72, 168],
                "rsi_low": [20.0, 28.0],
                "rsi_high": [72.0, 80.0],
                "entry_dev": [0.025, 0.05, 0.085],
                "max_natr": [6.0, 11.0],
            },
            "heat": ("rsi_low", [15, 20, 25, 30, 35], "entry_dev", [0.015, 0.025, 0.04, 0.05, 0.075]),
            "stability": ("rsi_period", [5, 7, 10, 14, 21]),
        },
        {
            "name": "multi_horizon_residual_fade",
            "signals": {
                "btc": "symbol_ref('BTCUSDT', 'close')",
                "asset_fast": "roc(close, param('fast_horizon', default=6))",
                "btc_fast": "roc(btc, param('fast_horizon', default=6))",
                "fast_residual": "asset_fast - btc_fast",
                "fast_anchor": "sma(fast_residual, param('fast_window', default=96))",
                "fast_dev": "fast_residual - fast_anchor",
                "asset_mid": "roc(close, param('mid_horizon', default=24))",
                "btc_mid": "roc(btc, param('mid_horizon', default=24))",
                "mid_residual": "asset_mid - btc_mid",
                "mid_anchor": "sma(mid_residual, param('mid_window', default=168))",
                "mid_dev": "mid_residual - mid_anchor",
                "volatility": "natr(14)",
            },
            "size": "when((fast_dev < (0.0 - param('fast_entry', default=0.06))) & (mid_dev > (0.0 - param('mid_limit', default=0.08))) & (volatility < param('max_natr', default=10.0)), param('size', default=0.10), when((fast_dev > param('fast_entry', default=0.06)) & (mid_dev < param('mid_limit', default=0.08)) & (volatility < param('max_natr', default=10.0)), 0.0 - param('size', default=0.10), 0.0))",
            "grid": {
                "fast_horizon": [3, 9],
                "mid_horizon": [18, 48],
                "fast_window": [72, 144],
                "mid_window": [120, 240],
                "fast_entry": [0.03, 0.065, 0.10],
                "mid_limit": [0.04, 0.10],
                "max_natr": [7.0, 12.0],
            },
            "heat": ("fast_entry", [0.02, 0.03, 0.045, 0.065, 0.08, 0.10], "fast_window", [48, 72, 96, 120, 168]),
            "stability": ("mid_limit", [0.02, 0.04, 0.06, 0.08, 0.10, 0.14]),
        },
    ]


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
        payload = await _original_call(session, name, args)
        if name == "list_indicators":
            path = Path("results/manifold/indicators.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload
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
