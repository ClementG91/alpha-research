from __future__ import annotations

import asyncio
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

import campaign

OUT = Path("results")
TRAIN = ("2021-01-01", "2023-12-31")
VALID = ("2024-01-01", "2024-12-31")
TEST = ("2025-01-01", "2026-07-01")
INTERVALS = ("4h", "12h")

MAJORS5 = {"name": "MAJORS5", "tickers": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]}
ALTS4 = {"name": "ALTS4", "tickers": ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]}
SYMBOL_NAMES = {"BTCUSDT": 1, "ETHUSDT": 2, "SOLUSDT": 3, "BNBUSDT": 4, "XRPUSDT": 5}


def cfg(universe: dict[str, Any], start: str, end: str, interval: str) -> dict[str, Any]:
    tickers = universe["tickers"]
    return {
        "universe": tickers[0] if len(tickers) == 1 else tickers,
        "symbol_names": SYMBOL_NAMES,
        "start": start,
        "end": end,
        "bar_interval": interval,
        "initial_capital": 10_000,
        "fees": "binance_perps",
        "slippage": {"kind": "fixed_bps", "bps": 2.0},
        "warmup_bars": 500,
        "execution": {
            "signal_delay": 1,
            "execution_price": "AtOpen",
            "max_position_pct": 0.15,
            "allow_short": True,
            "allow_fractional": True,
            "position_sizing_mode": "FractionOfEquity",
            "pyramiding": False,
        },
    }


def metric(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    return campaign.metric(row, key, default)


def trades(row: dict[str, Any]) -> int:
    return campaign.trades(row)


def score(train: dict[str, Any], valid: dict[str, Any]) -> float:
    ts = metric(train, "sharpe", -99)
    vs = metric(valid, "sharpe", -99)
    vdd = abs(metric(valid, "max_drawdown", -1))
    return min(ts, vs) + 0.20 * (ts + vs) - max(0.0, vdd - 0.35) * 1.5


def checkpoint(report: dict[str, Any]) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "generation3-checkpoint.json").write_text(json.dumps(report, indent=2, default=str))


SINGLE_SPECS: dict[str, dict[str, Any]] = {
    "consensus_fast_core": {
        "signals": {
            "f1": "ema(close, 8)",
            "s1": "ema(close, 72)",
            "f2": "ema(close, 8)",
            "s2": "ema(close, 168)",
        },
        "size": "when((f1 > s1) & (f2 > s2), 0.42, when((f1 < s1) & (f2 < s2), -0.42, 0.0))",
        "universe": MAJORS5,
    },
    "consensus_core_slow": {
        "signals": {
            "f1": "ema(close, 8)",
            "s1": "ema(close, 168)",
            "f2": "ema(close, 12)",
            "s2": "ema(close, 240)",
        },
        "size": "when((f1 > s1) & (f2 > s2), 0.42, when((f1 < s1) & (f2 < s2), -0.42, 0.0))",
        "universe": MAJORS5,
    },
    "consensus_triple": {
        "signals": {
            "f1": "ema(close, 8)",
            "s1": "ema(close, 72)",
            "f2": "ema(close, 8)",
            "s2": "ema(close, 168)",
            "f3": "ema(close, 12)",
            "s3": "ema(close, 240)",
        },
        "size": "when((f1 > s1) & (f2 > s2) & (f3 > s3), 0.45, when((f1 < s1) & (f2 < s2) & (f3 < s3), -0.45, 0.0))",
        "universe": MAJORS5,
    },
    "btc_regime_flip": {
        "signals": {
            "btc": "symbol_ref('BTCUSDT', 'close')",
            "btcf": "ema(btc, 24)",
            "btcs": "ema(btc, 168)",
            "fast": "ema(close, 8)",
            "slow": "ema(close, 168)",
        },
        "size": "when((btcf > btcs) & crossover(fast, slow), 0.42, when((btcf < btcs) & crossunder(fast, slow), -0.42, when((btcf < btcs) & (fast > slow), 0.0, when((btcf > btcs) & (fast < slow), 0.0, hold()))))",
        "universe": ALTS4,
    },
    "btc_regime_long_cash": {
        "signals": {
            "btc": "symbol_ref('BTCUSDT', 'close')",
            "btcf": "ema(btc, 24)",
            "btcs": "ema(btc, 168)",
            "fast": "ema(close, 8)",
            "slow": "ema(close, 168)",
        },
        "size": "when((btcf > btcs) & crossover(fast, slow), 0.45, when((btcf < btcs) | crossunder(fast, slow), 0.0, hold()))",
        "universe": ALTS4,
    },
}

COMPONENT_SPECS: dict[str, dict[str, Any]] = {
    "fast": {
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 72)"},
        "size": "when(crossover(fast, slow), 0.40, when(crossunder(fast, slow), -0.40, hold()))",
    },
    "core": {
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 168)"},
        "size": "when(crossover(fast, slow), 0.40, when(crossunder(fast, slow), -0.40, hold()))",
    },
    "slow": {
        "signals": {"fast": "ema(close, 12)", "slow": "ema(close, 240)"},
        "size": "when(crossover(fast, slow), 0.40, when(crossunder(fast, slow), -0.40, hold()))",
    },
    "very_slow": {
        "signals": {"fast": "ema(close, 24)", "slow": "ema(close, 360)"},
        "size": "when(crossover(fast, slow), 0.40, when(crossunder(fast, slow), -0.40, hold()))",
    },
    "donchian": {
        "signals": {"upper": "highest(close, 24)", "lower": "lowest(close, 24)", "regime": "ema(close, 240)"},
        "size": "when((close >= upper) & (close > regime), 0.40, when((close <= lower) & (close < regime), -0.40, hold()))",
    },
    "long_core": {
        "signals": {"fast": "ema(close, 8)", "slow": "ema(close, 168)"},
        "size": "when(crossover(fast, slow), 0.45, when(crossunder(fast, slow), 0.0, hold()))",
    },
}

PORTFOLIOS = [
    {"name": "multi_horizon_equal", "weights": {"fast": 0.34, "core": 0.33, "slow": 0.33}},
    {"name": "core_fast", "weights": {"core": 0.60, "fast": 0.40}},
    {"name": "core_slow", "weights": {"core": 0.60, "slow": 0.40}},
    {"name": "barbell", "weights": {"fast": 0.50, "slow": 0.50}},
    {"name": "core_donchian", "weights": {"core": 0.65, "donchian": 0.35}},
    {"name": "trend_diversified", "weights": {"fast": 0.25, "core": 0.35, "slow": 0.25, "donchian": 0.15}},
    {"name": "asymmetric_bull", "weights": {"core": 0.45, "long_core": 0.35, "donchian": 0.20}},
    {"name": "slow_defensive", "weights": {"core": 0.35, "slow": 0.35, "very_slow": 0.30}},
]


def portfolio_payload(weights: dict[str, float], docs: dict[str, dict[str, Any]], guarded: bool) -> dict[str, Any]:
    risk_rules: list[dict[str, Any]] = []
    if guarded:
        risk_rules = [
            {"type": "MaxDrawdown", "threshold_pct": 22.0},
            {"type": "MaxGrossExposure", "max_pct": 100.0},
            {"type": "MaxNetExposure", "max_pct": 70.0},
        ]
    return {
        "strategies": [
            {"strategy_json": docs[name], "weight": weight}
            for name, weight in weights.items()
        ],
        "risk_rules": risk_rules,
        "rebalance": {"type": "Periodic", "every_n_bars": 30},
    }


def weight_neighbours(weights: dict[str, float]) -> list[dict[str, float]]:
    names = list(weights)
    out = [dict(weights)]
    if len(names) < 2:
        return out
    a, b = names[0], names[1]
    for delta in (-0.10, 0.10):
        changed = dict(weights)
        changed[a] = max(0.05, changed[a] + delta)
        changed[b] = max(0.05, changed[b] - delta)
        total = sum(changed.values())
        out.append({k: v / total for k, v in changed.items()})
    return out


async def main() -> None:
    OUT.mkdir(exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "train": TRAIN,
        "validation": VALID,
        "test": TEST,
        "singles": [],
        "portfolios": [],
        "finalists": [],
        "errors": [],
    }

    async with streamablehttp_client(campaign.MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            report["version"] = await campaign.call(session, "get_version", {})

            single_docs: dict[str, dict[str, Any]] = {}
            for name, spec in SINGLE_SPECS.items():
                try:
                    payload = await campaign.call(session, "compose_strategy", {"name": name, "signals": spec["signals"], "size": spec["size"]})
                    single_docs[name] = payload["strategy_json"]
                except Exception as exc:
                    report["errors"].append({"stage": "compose_single", "name": name, "error": str(exc)})

            component_docs: dict[str, dict[str, Any]] = {}
            for name, spec in COMPONENT_SPECS.items():
                try:
                    payload = await campaign.call(session, "compose_strategy", {"name": name, "signals": spec["signals"], "size": spec["size"]})
                    component_docs[name] = payload["strategy_json"]
                except Exception as exc:
                    report["errors"].append({"stage": "compose_component", "name": name, "error": str(exc)})
            checkpoint(report)

            for name, doc in single_docs.items():
                spec = SINGLE_SPECS[name]
                universe = spec["universe"]
                for interval in INTERVALS:
                    rid = f"single:{name}:{interval}"
                    try:
                        train = await campaign.call(session, "run_backtest", {"strategy_json": doc, "config": cfg(universe, TRAIN[0], TRAIN[1], interval)})
                        valid = await campaign.call(session, "run_backtest", {"strategy_json": doc, "config": cfg(universe, VALID[0], VALID[1], interval)})
                        report["singles"].append({"id": rid, "kind": "single", "name": name, "universe": universe, "interval": interval, "strategy_json": doc, "train": train, "validation": valid, "score": score(train, valid)})
                    except Exception as exc:
                        report["errors"].append({"stage": "single", "id": rid, "error": str(exc)})
                    checkpoint(report)

            for spec in PORTFOLIOS:
                if any(name not in component_docs for name in spec["weights"]):
                    continue
                for guarded in (False, True):
                    for interval in INTERVALS:
                        name = f"{spec['name']}{'_guarded' if guarded else ''}"
                        rid = f"portfolio:{name}:{interval}"
                        payload = portfolio_payload(spec["weights"], component_docs, guarded)
                        try:
                            train = await campaign.call(session, "run_portfolio", {"portfolio": payload, "config": cfg(MAJORS5, TRAIN[0], TRAIN[1], interval)})
                            valid = await campaign.call(session, "run_portfolio", {"portfolio": payload, "config": cfg(MAJORS5, VALID[0], VALID[1], interval)})
                            report["portfolios"].append({"id": rid, "kind": "portfolio", "name": name, "base_name": spec["name"], "guarded": guarded, "weights": spec["weights"], "universe": MAJORS5, "interval": interval, "portfolio": payload, "train": train, "validation": valid, "score": score(train, valid)})
                        except Exception as exc:
                            report["errors"].append({"stage": "portfolio", "id": rid, "error": str(exc)})
                        checkpoint(report)

            ranked = report["singles"] + report["portfolios"]
            ranked.sort(key=lambda row: row.get("score", -99), reverse=True)
            selected = ranked[:10]

            for row in selected:
                try:
                    if row["kind"] == "single":
                        test = await campaign.call(session, "run_backtest", {"strategy_json": row["strategy_json"], "config": cfg(row["universe"], TEST[0], TEST[1], row["interval"])})
                        neighbour_median = metric(test, "sharpe")
                        neighbours: list[dict[str, Any]] = []
                        mc = None
                        if metric(test, "sharpe", -99) >= 1.25:
                            mc = await campaign.call(session, "run_monte_carlo", {"strategy_json": row["strategy_json"], "config": cfg(row["universe"], TEST[0], TEST[1], row["interval"]), "mc_config": {"n_paths": 2000, "method": {"type": "block_bootstrap", "block_size": 24}, "rng_seed": 42}})
                    else:
                        test = await campaign.call(session, "run_portfolio", {"portfolio": row["portfolio"], "config": cfg(row["universe"], TEST[0], TEST[1], row["interval"])})
                        neighbours = []
                        neighbour_sharpes = []
                        for weights in weight_neighbours(row["weights"]):
                            payload = portfolio_payload(weights, component_docs, row["guarded"])
                            result = await campaign.call(session, "run_portfolio", {"portfolio": payload, "config": cfg(row["universe"], TEST[0], TEST[1], row["interval"])})
                            neighbours.append({"weights": weights, "result": result})
                            s = metric(result, "sharpe")
                            if math.isfinite(s):
                                neighbour_sharpes.append(s)
                        neighbour_median = median(neighbour_sharpes) if neighbour_sharpes else float("nan")
                        mc = None

                    passed = bool(
                        metric(test, "sharpe", -99) >= 1.5
                        and metric(test, "alpha", -99) > 0
                        and trades(test) >= 20
                        and metric(row["train"], "sharpe", -99) >= 0.70
                        and metric(row["validation"], "sharpe", -99) >= 0.70
                        and neighbour_median >= 1.20
                    )
                    report["finalists"].append({**row, "test": test, "neighbours": neighbours, "neighbour_median_sharpe": neighbour_median, "monte_carlo": mc, "passes": passed})
                except Exception as exc:
                    report["errors"].append({"stage": "final", "id": row["id"], "error": str(exc)})
                checkpoint(report)

    report["finalists"].sort(key=lambda row: metric(row.get("test") or {}, "sharpe", -99), reverse=True)
    report["passes"] = [row for row in report["finalists"] if row.get("passes")]
    report["best"] = report["finalists"][0] if report["finalists"] else None
    (OUT / "generation3.json").write_text(json.dumps(report, indent=2, default=str))

    lines = [
        "# Crypto alpha generation 3",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "Ensembles multi-horizon and BTC-regime filters. Selection uses 2021-2023 train + 2024 validation; 2025-2026 is frozen OOS. Fees: Binance perps; slippage: 2 bps; one-bar delay. Spot bars proxy price and funding is unavailable.",
        "",
        "| # | Kind | Candidate | TF | Train S | Valid S | OOS S | Alpha | Return | DD | Trades | Neighbour S | Pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for idx, row in enumerate(report["finalists"], 1):
        test = row["test"]
        lines.append(
            f"| {idx} | {row['kind']} | {row['name']} | {row['interval']} | "
            f"{metric(row['train'],'sharpe'):.2f} | {metric(row['validation'],'sharpe'):.2f} | "
            f"{metric(test,'sharpe'):.2f} | {metric(test,'alpha'):.3f} | "
            f"{metric(test,'total_return'):.2%} | {metric(test,'max_drawdown'):.2%} | "
            f"{trades(test)} | {row['neighbour_median_sharpe']:.2f} | {'YES' if row['passes'] else 'NO'} |"
        )
    if report["passes"]:
        best = report["passes"][0]
        lines += [
            "",
            "## Validated candidate",
            "",
            f"- Candidate: `{best['name']}`",
            f"- Kind: `{best['kind']}`",
            f"- Timeframe: `{best['interval']}`",
            f"- OOS Sharpe: `{metric(best['test'],'sharpe'):.3f}`",
            f"- OOS alpha: `{metric(best['test'],'alpha'):.4f}`",
            f"- OOS return: `{metric(best['test'],'total_return'):.2%}`",
            f"- Max drawdown: `{metric(best['test'],'max_drawdown'):.2%}`",
            f"- Trades: `{trades(best['test'])}`",
            f"- Neighbour median Sharpe: `{best['neighbour_median_sharpe']:.3f}`",
        ]
        if best["kind"] == "portfolio":
            lines.append(f"- Weights: `{json.dumps(best['weights'], sort_keys=True)}`")
    else:
        lines += ["", "## Conclusion", "", "No candidate reached and retained OOS Sharpe >= 1.5 through all gates."]
    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- `{json.dumps(err, sort_keys=True)}`" for err in report["errors"]]
    (OUT / "GENERATION3.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT / "FATAL.txt").write_text(traceback.format_exc())
        raise
