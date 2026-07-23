from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MCP_URL = os.getenv("MANIFOLDBT_MCP_URL", "https://mcp.manifoldbt.com/mcp")
TRAIN = ("2021-01-01", "2023-12-31")
VALID = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-07-01")
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
TARGET = 1.5


def lit(x: float) -> dict[str, Any]: return {"Literal": {"Float64": float(x)}}
def col(x: str) -> dict[str, Any]: return {"Column": x}
def param(x: str) -> dict[str, Any]: return {"Parameter": x}
def op(name: str, *xs: Any) -> dict[str, Any]: return {name: list(xs)}
def spec(name: str, value: Any) -> dict[str, Any]:
    typed = {"Int64": value} if isinstance(value, int) else {"Float64": float(value)}
    return {"name": name, "default": typed, "description": "", "range": None}


def strategy(name: str, *, long_only: bool, range_vol: bool) -> dict[str, Any]:
    close = col("close")
    fast, slow = {"EwmMean": [close, "fast"]}, {"EwmMean": [close, "slow"]}
    if range_vol:
        vol = op("Div", {"RollingMean": [op("Sub", col("high"), col("low")), 14]}, op("Add", close, lit(1e-12)))
    else:
        vol = {"RollingStd": [{"PctChange": [close, 1]}, 14]}
    vol_z = {"ZScore": [vol, "vol_window"]}
    high_vol = op("Gt", vol_z, param("risk_z"))
    long_size = op("IfElse", high_vol, param("risk_long"), param("base_long"))
    short_size = lit(0.0) if long_only else op(
        "IfElse", high_vol, op("Mul", lit(-1), param("risk_short")), op("Mul", lit(-1), param("base_short"))
    )
    position = op("IfElse", op("Gt", fast, slow), long_size, op("IfElse", op("Lt", fast, slow), short_size, lit(0)))
    return {
        "name": name,
        "signals": {"fast": fast, "slow": slow, "volatility": vol, "vol_z": vol_z},
        "position_sizing": position,
        "parameters": {
            "fast": spec("fast", 8), "slow": spec("slow", 168), "vol_window": spec("vol_window", 72),
            "risk_z": spec("risk_z", 1.0), "base_long": spec("base_long", 0.15),
            "base_short": spec("base_short", 0.15), "risk_long": spec("risk_long", 0.15),
            "risk_short": spec("risk_short", 0.0),
        },
        "constraints": [],
        "metadata": {"description": "ManifoldBT-native EMA trend with volatility-state sizing"},
    }


def families() -> list[dict[str, Any]]:
    grid = {"fast": [8, 12, 24], "slow": [96, 168, 240], "vol_window": [48, 72, 120], "risk_z": [0.75, 1.25], "risk_long": [0.05, 0.15]}
    return [
        {"name": "return_vol_long_short", "strategy": strategy("return_vol_long_short", long_only=False, range_vol=False),
         "grid": {**grid, "base_long": [0.15], "base_short": [0.15], "risk_short": [0.0, 0.05]},
         "heat": ("fast", [8, 12, 18, 24], "slow", [96, 144, 168, 240]), "stability": ("vol_window", [36, 48, 72, 96, 120, 168])},
        {"name": "return_vol_long_cash", "strategy": strategy("return_vol_long_cash", long_only=True, range_vol=False),
         "grid": {**grid, "base_long": [0.15], "base_short": [0.0], "risk_short": [0.0]},
         "heat": ("fast", [8, 12, 18, 24], "slow", [96, 144, 168, 240]), "stability": ("risk_z", [0.5, 0.75, 1, 1.25, 1.5, 2])},
        {"name": "range_vol_long_cash", "strategy": strategy("range_vol_long_cash", long_only=True, range_vol=True),
         "grid": {**grid, "base_long": [0.15], "base_short": [0.0], "risk_short": [0.0]},
         "heat": ("vol_window", [36, 48, 72, 120], "risk_z", [0.5, 0.75, 1, 1.5]), "stability": ("risk_long", [0, 0.05, 0.1, 0.15, 0.2])},
    ]


def cfg(start: str, end: str) -> dict[str, Any]:
    return {"universe": UNIVERSE, "start": start, "end": end, "bar_interval": "4h", "initial_capital": 10000,
            "fees": "binance_perps", "slippage": {"kind": "fixed_bps", "bps": 2.0}, "warmup_bars": 500,
            "execution": {"signal_delay": 1, "execution_price": "AtOpen", "max_position_pct": 0.15,
                          "allow_short": True, "allow_fractional": True, "position_sizing_mode": "FractionOfEquity", "pyramiding": False}}


def unpack(result: Any) -> Any:
    if getattr(result, "structuredContent", None) is not None: return result.structuredContent
    text = "\n".join(getattr(x, "text", "") for x in getattr(result, "content", []))
    try: return json.loads(text)
    except Exception: return {"text": text}


async def call(session: Any, name: str, args: dict[str, Any]) -> Any:
    result = await session.call_tool(name, args)
    payload = unpack(result)
    if getattr(result, "isError", False): raise RuntimeError(f"{name} failed: {payload}")
    return payload


def rows(x: Any) -> list[dict[str, Any]]:
    if isinstance(x, list): return x
    if isinstance(x, dict):
        for key in ("result", "results", "rows"):
            if isinstance(x.get(key), list): return x[key]
    raise TypeError(f"expected list, got {type(x).__name__}")


def num(x: Any, default: float = float("nan")) -> float:
    try: value = float(x)
    except (TypeError, ValueError): return default
    return value if math.isfinite(value) else default


def metric(x: dict[str, Any] | None, key: str, default: float = float("nan")) -> float:
    return num(((x or {}).get("metrics") or {}).get(key), default)


def trades(x: dict[str, Any] | None) -> int:
    x = x or {}; value = x.get("trade_count") or (((x.get("metrics") or {}).get("trade_stats") or {}).get("total_trades")) or 0
    try: return int(value)
    except (TypeError, ValueError): return 0


def freeze(doc: dict[str, Any], params: dict[str, Any], name: str) -> dict[str, Any]:
    out = copy.deepcopy(doc); out["name"] = name
    for key, value in params.items():
        if key in out["parameters"]: out["parameters"][key]["default"] = {"Int64": value} if isinstance(value, int) else {"Float64": float(value)}
    return out


def score(train: dict[str, Any], valid: dict[str, Any], probability: float) -> float:
    ts, vs = metric(train, "sharpe", -99), metric(valid, "sharpe", -99)
    return min(ts, vs) + .2 * (ts + vs) + .25 * probability - max(0, 25 - trades(valid)) * .03 - max(0, abs(metric(valid, "max_drawdown", -1)) - .3) * 2


def curve(x: dict[str, Any]) -> list[float]:
    raw = x.get("equity_curve") or ((x.get("result") or {}).get("equity_curve") if isinstance(x.get("result"), dict) else None)
    if not isinstance(raw, list): return []
    out = []
    for item in raw:
        value = num(item.get("equity") or item.get("value") or item.get("capital")) if isinstance(item, dict) else num(item)
        if math.isfinite(value): out.append(value)
    return out


def daily_returns(x: dict[str, Any]) -> list[float]:
    raw = x.get("daily_returns") or []
    return [num(i.get("return") or i.get("value")) if isinstance(i, dict) else num(i) for i in raw if math.isfinite(num(i.get("return") or i.get("value")) if isinstance(i, dict) else num(i))]


def drawdown(values: list[float]) -> list[float]:
    peak = float("-inf"); out = []
    for value in values:
        peak = max(peak, value); out.append(value / peak - 1 if peak > 0 else 0)
    return out


def mcp_imports() -> tuple[Any, Any]:
    from mcp import ClientSession
    try: from mcp.client.streamable_http import streamablehttp_client
    except ImportError: from mcp.client.streamable_http import streamable_http_client as streamablehttp_client
    return ClientSession, streamablehttp_client


async def research(out: Path) -> dict[str, Any]:
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "mcp_url": MCP_URL, "engine_only": True,
              "train": TRAIN, "validation": VALID, "holdout": TEST, "target": TARGET, "families": [], "errors": [], "blockers": []}
    ClientSession, streamable = mcp_imports()
    async with streamable(MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools(); tool_docs = [t.model_dump(mode="json") for t in listed.tools]
            (out / "mcp-tools.json").write_text(json.dumps(tool_docs, indent=2))
            tool_map = {t["name"]: t for t in tool_docs}
            report["version"] = await call(session, "get_version", {})
            report["symbols"] = rows(await call(session, "list_symbols", {}))
            if "ingest_data" not in tool_map: report["blockers"].append("No ingest_data tool: only the server's preloaded datastore can be used.")
            if str((report["version"] or {}).get("license_tier", "")).lower() == "community": report["blockers"].append("Native run_walk_forward is Pro-only; train/validation/holdout are separate MCP backtests.")
            report["blockers"] += ["The datastore exposes Binance CryptoSpot proxies, not complete perpetual funding/basis/OI history.", "Community Monte Carlo is capped at 1,000 paths."]
            finalists = []
            for family in families():
                entry = {"name": family["name"]}
                try:
                    entry["compile"] = await call(session, "validate_strategy", {"strategy_json": json.dumps(family["strategy"])})
                    sweep = await call(session, "run_sweep", {"strategy_json": family["strategy"], "param_grid": family["grid"],
                        "config": cfg(*TRAIN), "lite": True, "top_k": 8, "rank_metric": "sharpe", "device": "auto", "precision": "fp64"})
                    entry["sweep"] = sweep; correction = sweep.get("overfitting_correction") or {}; probability = num(correction.get("probability_edge_is_real"), 0)
                    train_rows = [x for x in sweep.get("top", []) if x.get("params")]
                    docs = [freeze(family["strategy"], x["params"], f"{family['name']}_{i}") for i, x in enumerate(train_rows)]
                    valid_rows = rows(await call(session, "run_batch", {"strategies": [{"strategy_json": x} for x in docs], "config": cfg(*VALID), "lite": False, "max_parallelism": 0})) if docs else []
                    candidates = [{"params": a["params"], "train": a, "validation": b, "strategy_json": c, "score": score(a, b, probability)} for a, b, c in zip(train_rows, valid_rows, docs, strict=False)]
                    candidates.sort(key=lambda x: x["score"], reverse=True); entry["selected"] = candidates[0] if candidates else None
                    if candidates: finalists.append({"family": family, **candidates[0], "overfitting_correction": correction})
                except Exception as exc:
                    entry["error"] = str(exc); report["errors"].append({"family": family["name"], "error": str(exc)})
                report["families"].append(entry)
            finalists.sort(key=lambda x: x["score"], reverse=True)
            if not finalists: raise RuntimeError("No ManifoldBT candidate survived train/validation")
            winner = finalists[0]; family = winner.pop("family")
            holdout = await call(session, "run_backtest", {"strategy_json": winner["strategy_json"], "config": cfg(*TEST), "include_equity_curve": True, "equity_points": 500, "include_daily_returns": True})
            x_name, x_values, y_name, y_values = family["heat"]
            winner.update({
                "family": family["name"], "holdout": holdout,
                "test_2025": await call(session, "run_backtest", {"strategy_json": winner["strategy_json"], "config": cfg("2025-01-01", "2025-12-31")}),
                "test_2026_ytd": await call(session, "run_backtest", {"strategy_json": winner["strategy_json"], "config": cfg("2026-01-01", TEST[1])}),
                "heatmap": await call(session, "run_sweep_2d", {"strategy_json": winner["strategy_json"], "sweep_config": {"x_param": x_name, "x_values": x_values, "y_param": y_name, "y_values": y_values, "metric": "sharpe", "max_parallelism": 0}, "config": cfg(*TRAIN)}),
                "stability": await call(session, "run_stability", {"strategy_json": winner["strategy_json"], "stability_config": {"param_name": family["stability"][0], "values": family["stability"][1], "metric": "sharpe", "max_parallelism": 0}, "config": cfg(*VALID)}),
                "monte_carlo": await call(session, "run_monte_carlo", {"strategy_json": winner["strategy_json"], "config": cfg(*TEST), "mc_config": {"n_paths": 1000, "method": {"type": "block_bootstrap", "block_size": 24}, "rng_seed": 42, "confidence_levels": [.9, .95, .99], "dd_thresholds": [-.2, -.3, -.4]}}),
            })
            report["winner"] = winner
            report["accepted"] = metric(holdout, "sharpe", -99) >= TARGET and metric(winner["validation"], "sharpe", -99) >= .7 and metric(winner["train"], "sharpe", -99) > 0 and trades(holdout) >= 30
    return report


def plt_module() -> Any:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save(fig: Any, path: Path) -> None:
    plt = plt_module(); fig.tight_layout(); fig.savefig(path, format="svg", bbox_inches="tight"); plt.close(fig)


def plots(report: dict[str, Any], out: Path) -> None:
    plt = plt_module(); path = out / "plots"; path.mkdir(parents=True, exist_ok=True); w = report["winner"]; eq = curve(w["holdout"])
    fig, ax = plt.subplots(figsize=(10, 4.5)); ax.plot(eq) if eq else ax.text(.5, .5, "No MCP equity curve returned", ha="center", transform=ax.transAxes); ax.set_title("Frozen holdout equity — ManifoldBT"); ax.grid(alpha=.25); save(fig, path / "equity_curve.svg")
    fig, ax = plt.subplots(figsize=(10, 4.2)); dd = [x * 100 for x in drawdown(eq)]; ax.fill_between(range(len(dd)), dd, 0, alpha=.45) if dd else ax.text(.5, .5, "No MCP equity curve returned", ha="center", transform=ax.transAxes); ax.set_title("Drawdown from MCP equity"); ax.set_ylabel("%"); ax.grid(alpha=.25); save(fig, path / "drawdown.svg")
    periods = [("Train", w["train"]), ("Validation", w["validation"]), ("Holdout", w["holdout"]), ("2025", w["test_2025"]), ("2026 YTD", w["test_2026_ytd"])]
    fig, ax = plt.subplots(figsize=(9, 4.5)); ax.bar(range(len(periods)), [metric(x, "sharpe", 0) for _, x in periods]); ax.axhline(TARGET, ls="--", label="Target 1.5"); ax.set_xticks(range(len(periods)), [x for x, _ in periods]); ax.set_title("Sharpe by chronological segment — ManifoldBT"); ax.legend(); ax.grid(axis="y", alpha=.25); save(fig, path / "period_metrics.svg")
    h = w.get("heatmap") or {}; grid = h.get("metric_grid") or (h.get("result") or {}).get("metric_grid"); xv = h.get("x_values") or (h.get("result") or {}).get("x_values") or []; yv = h.get("y_values") or (h.get("result") or {}).get("y_values") or []
    fig, ax = plt.subplots(figsize=(7.5, 5));
    if grid: image = ax.imshow(grid, aspect="auto", origin="lower"); ax.set_xticks(range(len(yv)), yv); ax.set_yticks(range(len(xv)), xv); fig.colorbar(image, ax=ax, label="Sharpe")
    else: ax.text(.5, .5, "No run_sweep_2d grid returned", ha="center", transform=ax.transAxes)
    ax.set_title("Training parameter surface — ManifoldBT"); save(fig, path / "parameter_heatmap.svg")
    s = w.get("stability") or {}; values = s.get("values") or s.get("param_values") or (s.get("result") or {}).get("values") or []; scores = s.get("metrics") or s.get("metric_values") or (s.get("result") or {}).get("metrics") or []
    if scores and isinstance(scores[0], dict): scores = [num(x.get("value") or x.get("sharpe")) for x in scores]
    fig, ax = plt.subplots(figsize=(8.5, 4.5)); ax.plot(values[:len(scores)], scores[:len(values)], marker="o") if values and scores else ax.text(.5, .5, "No plottable run_stability series", ha="center", transform=ax.transAxes); ax.set_title("Validation parameter stability — ManifoldBT"); ax.grid(alpha=.25); save(fig, path / "stability.svg")
    mc = w.get("monte_carlo") or {}; rp = dict((mc.get("final_return") or {}).get("percentiles") or []); dp = dict((mc.get("max_drawdown") or {}).get("percentiles") or []); labels = ["Return p90", "Return p95", "Return p99", "DD p90", "DD p95", "DD p99"]; data = [num(rp.get(x), 0)*100 for x in (.9,.95,.99)] + [-num(dp.get(x), 0)*100 for x in (.9,.95,.99)]
    fig, ax = plt.subplots(figsize=(9.5, 4.5)); ax.bar(labels, data); ax.axhline(0, lw=.8); ax.tick_params(axis="x", rotation=20); ax.set_title(f"Monte Carlo risk — ManifoldBT ({mc.get('n_paths', 1000)} paths)"); ax.set_ylabel("%"); ax.grid(axis="y", alpha=.25); save(fig, path / "monte_carlo.svg")


def markdown(report: dict[str, Any], out: Path) -> None:
    w = report["winner"]
    lines = ["# ManifoldBT MCP research results", "", f"Generated: `{report['generated_at']}`", "", "All performance, equity, sweep, stability and Monte Carlo data come from ManifoldBT MCP. Python only orchestrates and plots its payloads.", "", f"- Selected: `{w['family']}`", f"- Target: `{TARGET}`", f"- Accepted: `{'YES' if report['accepted'] else 'NO'}`", "", "| Period | Sharpe | Return | Max DD | Trades |", "|---|---:|---:|---:|---:|"]
    for name, row in [("Train", w["train"]), ("Validation", w["validation"]), ("Frozen holdout", w["holdout"])]: lines.append(f"| {name} | {metric(row,'sharpe'):.3f} | {metric(row,'total_return'):.2%} | {metric(row,'max_drawdown'):.2%} | {trades(row)} |")
    lines += ["", "## Blocking limitations", ""] + [f"- {x}" for x in report["blockers"]] + ["", "Raw MCP responses: `raw.json`. Tool schemas: `mcp-tools.json`. Plots: `plots/*.svg`."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("results/manifold")); args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    try:
        report = asyncio.run(research(args.output)); (args.output / "raw.json").write_text(json.dumps(report, indent=2, default=str)); markdown(report, args.output); plots(report, args.output); return 0
    except Exception:
        (args.output / "FATAL.txt").write_text(traceback.format_exc()); return 1


if __name__ == "__main__": raise SystemExit(main())