from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from sklearn.covariance import LedoitWolf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_GROUPS: dict[str, str] = {
    "SPY": "equity",
    "QQQ": "equity",
    "IWM": "equity",
    "VEA": "equity",
    "EEM": "equity",
    "VNQ": "real_estate",
    "IEF": "bond",
    "TLT": "bond",
    "GLD": "gold",
    "DBC": "commodity",
    "UUP": "usd",
    "BTC-USD": "crypto",
    "ETH-USD": "crypto",
}

DEFENSIVE_GROUPS = {"bond", "gold", "usd"}


@dataclass(frozen=True)
class ResearchConfig:
    min_train_months: int = 72
    lookback_months: int = 144
    embargo_months: int = 1
    covariance_months: int = 36
    target_vol: float = 0.10
    per_asset_cap: float = 0.25
    crypto_cap: float = 0.15
    max_assets: int = 7
    holdout_months: int = 36
    ensemble_ridge_weight: float = 0.70
    etf_cost_bps: float = 8.0
    crypto_cost_bps: float = 20.0
    monte_carlo_paths: int = 5000
    monte_carlo_block_months: int = 6
    permutation_trials: int = 1000
    trial_count: int = 24
    random_seed: int = 42
    model_mode: str = "ensemble"


@dataclass
class ResearchResult:
    weights: pd.DataFrame
    returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    audit: pd.DataFrame
    metrics: dict[str, float]
    holdout_metrics: dict[str, float]
    monte_carlo: dict[str, float]
    stress: dict[str, dict[str, float]]
    diagnostics: dict[str, Any]

    def serializable(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "holdout_metrics": self.holdout_metrics,
            "monte_carlo": self.monte_carlo,
            "stress": self.stress,
            "diagnostics": self.diagnostics,
            "audit": self.audit.reset_index().astype(str).to_dict(orient="records"),
            "weights": self.weights.reset_index().to_dict(orient="records"),
            "returns": {str(k): float(v) for k, v in self.returns.dropna().items()},
            "gross_returns": {str(k): float(v) for k, v in self.gross_returns.dropna().items()},
            "costs": {str(k): float(v) for k, v in self.costs.dropna().items()},
        }


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    std = np.nanstd(values)
    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values)
    return (values - np.nanmean(values)) / std


def _rolling_slope_tstat(series: pd.Series, window: int) -> pd.Series:
    def calc(raw: np.ndarray) -> float:
        if len(raw) != window or np.any(~np.isfinite(raw)) or np.any(raw <= 0):
            return np.nan
        y = np.log(raw)
        x = np.arange(window, dtype=float)
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = float(np.dot(x_centered, x_centered))
        if denom <= 0:
            return np.nan
        beta = float(np.dot(x_centered, y_centered) / denom)
        resid = y_centered - beta * x_centered
        dof = max(window - 2, 1)
        sigma2 = float(np.dot(resid, resid) / dof)
        se = np.sqrt(sigma2 / denom) if sigma2 > 0 else 0.0
        return beta / se if se > 1e-12 else 0.0

    return series.rolling(window, min_periods=window).apply(calc, raw=True)


def prepare_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("price frame is empty")
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.sort_index()
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.resample("ME").last().ffill(limit=2)
    valid = frame.notna().sum() >= 36
    frame = frame.loc[:, valid]
    if frame.shape[1] < 4:
        raise ValueError("at least four assets with 36 monthly observations are required")
    return frame


def build_panel(
    prices: pd.DataFrame,
    groups: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = prepare_monthly_prices(prices)
    rets = monthly.pct_change()
    groups = dict(DEFAULT_GROUPS if groups is None else groups)

    benchmark = "SPY" if "SPY" in monthly.columns else monthly.columns[0]
    regime = pd.DataFrame(index=monthly.index)
    regime["breadth_6m"] = (monthly.pct_change(6) > 0).mean(axis=1)
    regime["breadth_12m"] = (monthly.pct_change(12) > 0).mean(axis=1)
    regime["benchmark_mom_3m"] = monthly[benchmark].pct_change(3)
    regime["benchmark_mom_12m"] = monthly[benchmark].pct_change(12)
    regime["benchmark_vol_3m"] = rets[benchmark].rolling(3).std() * np.sqrt(12)
    for symbol, name in (("TLT", "bond"), ("GLD", "gold"), ("DBC", "commodity"), ("UUP", "usd")):
        regime[f"{name}_mom_6m"] = monthly[symbol].pct_change(6) if symbol in monthly else 0.0
    if "TLT" in monthly:
        regime["equity_bond_corr_12m"] = rets[benchmark].rolling(12).corr(rets["TLT"])
    else:
        regime["equity_bond_corr_12m"] = 0.0
    regime = regime.replace([np.inf, -np.inf], np.nan)

    group_names = sorted(set(groups.get(symbol, "other") for symbol in monthly.columns))
    rows: list[pd.DataFrame] = []
    for symbol in monthly.columns:
        price = monthly[symbol]
        ret = rets[symbol]
        feat = pd.DataFrame(index=monthly.index)
        feat["mom_1m"] = price.pct_change(1)
        feat["mom_3m"] = price.pct_change(3)
        feat["mom_6m"] = price.pct_change(6)
        feat["mom_12m"] = price.pct_change(12)
        feat["reversal_1m"] = -feat["mom_1m"]
        feat["vol_3m"] = ret.rolling(3).std() * np.sqrt(12)
        feat["vol_6m"] = ret.rolling(6).std() * np.sqrt(12)
        feat["vol_12m"] = ret.rolling(12).std() * np.sqrt(12)
        feat["downside_vol_6m"] = ret.where(ret < 0).rolling(6, min_periods=3).std() * np.sqrt(12)
        feat["skew_6m"] = ret.rolling(6).skew()
        feat["kurt_12m"] = ret.rolling(12).kurt()
        feat["drawdown_12m"] = price / price.rolling(12).max() - 1.0
        feat["slope_t_6m"] = _rolling_slope_tstat(price, 6)
        feat["slope_t_12m"] = _rolling_slope_tstat(price, 12)
        feat["corr_benchmark_12m"] = ret.rolling(12).corr(rets[benchmark])
        feat = feat.join(regime, how="left")
        current_group = groups.get(symbol, "other")
        for group in group_names:
            feat[f"group_{group}"] = float(group == current_group)
        feat["target"] = ret.shift(-1)
        feat["symbol"] = symbol
        feat["date"] = feat.index
        rows.append(feat.reset_index(drop=True))

    panel = pd.concat(rows, ignore_index=True).set_index(["date", "symbol"]).sort_index()
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel, monthly, rets


def _regime_scalar(
    regime_features: pd.DataFrame,
    benchmark_forward_return: pd.Series,
    train_dates: list[pd.Timestamp],
    signal_date: pd.Timestamp,
    seed: int,
) -> float:
    train = regime_features.loc[regime_features.index.intersection(train_dates)].copy()
    train = train.join(benchmark_forward_return.rename("forward"), how="inner").dropna()
    current = regime_features.loc[[signal_date]].copy()
    if len(train) < 48 or current.isna().all(axis=None):
        return 0.75
    cols = [c for c in regime_features.columns if c in train and c != "forward"]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(train[cols]))
    x_current = scaler.transform(imputer.transform(current[cols]))
    model = GaussianMixture(n_components=3, covariance_type="full", reg_covar=1e-4, random_state=seed)
    states = model.fit_predict(x_train)
    current_state = int(model.predict(x_current)[0])
    means = {state: float(train.loc[states == state, "forward"].mean()) for state in range(3)}
    ordered = sorted(means, key=means.get)
    mapping = {ordered[0]: 0.40, ordered[1]: 0.70, ordered[2]: 1.00}
    return mapping[current_state]


def _cap_and_redistribute(weights: pd.Series, cap: float) -> pd.Series:
    w = weights.clip(lower=0.0).copy()
    if w.sum() <= 0:
        return w
    w /= w.sum()
    for _ in range(12):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w.loc[over] = cap
        under = ~over
        room = (cap - w[under]).clip(lower=0.0)
        if room.sum() <= 1e-12:
            break
        w.loc[under] += excess * room / room.sum()
    return w


def allocate_weights(
    scores: pd.Series,
    trailing_returns: pd.DataFrame,
    groups: Mapping[str, str],
    regime_scalar: float,
    config: ResearchConfig,
) -> pd.Series:
    scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
    scores = scores[scores > 0].nlargest(config.max_assets)
    if scores.empty:
        return pd.Series(dtype=float)

    symbols = [s for s in scores.index if s in trailing_returns.columns]
    hist = trailing_returns[symbols].dropna(how="all")
    if len(hist) < 12:
        return pd.Series(dtype=float)
    vol = hist.std().replace(0, np.nan) * np.sqrt(12)
    symbols = [s for s in symbols if np.isfinite(vol.get(s, np.nan))]
    if not symbols:
        return pd.Series(dtype=float)
    hist = hist[symbols].fillna(0.0)
    mu = scores[symbols].clip(lower=0.0)

    inv_vol = (mu / vol[symbols]).clip(lower=0.0)
    inv_vol = inv_vol / inv_vol.sum() if inv_vol.sum() > 0 else pd.Series(1.0 / len(symbols), index=symbols)

    if len(hist) >= max(18, len(symbols) * 2):
        cov = LedoitWolf().fit(hist.to_numpy()).covariance_ * 12.0
    else:
        cov = np.diag(np.square(vol[symbols].to_numpy()))
    mv = np.linalg.pinv(cov) @ mu.to_numpy()
    mv = np.clip(mv, 0.0, None)
    mv = mv / mv.sum() if mv.sum() > 0 else inv_vol.to_numpy()

    weights = pd.Series(0.5 * inv_vol.to_numpy() + 0.5 * mv, index=symbols)
    weights = _cap_and_redistribute(weights, config.per_asset_cap)

    risky = [s for s in weights.index if groups.get(s, "other") not in DEFENSIVE_GROUPS]
    weights.loc[risky] *= float(np.clip(regime_scalar, 0.0, 1.0))

    crypto = [s for s in weights.index if groups.get(s) == "crypto"]
    crypto_total = float(weights.loc[crypto].sum()) if crypto else 0.0
    if crypto_total > config.crypto_cap and crypto_total > 0:
        weights.loc[crypto] *= config.crypto_cap / crypto_total

    active = weights.index.tolist()
    cov_active = cov[[symbols.index(s) for s in active]][:, [symbols.index(s) for s in active]]
    current_vol = float(np.sqrt(np.maximum(weights.to_numpy() @ cov_active @ weights.to_numpy(), 0.0)))
    if current_vol > 1e-12:
        weights *= min(1.0, config.target_vol / current_vol)
    if weights.sum() > 1.0:
        weights /= weights.sum()
    return weights.clip(lower=0.0)


def _fit_predict(
    train: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    config: ResearchConfig,
) -> pd.Series:
    x_train = train[feature_cols]
    y_train = train["target"].astype(float)
    x_current = current[feature_cols]

    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=8.0))
    ridge.fit(x_train, y_train)
    ridge_pred = ridge.predict(x_current)
    if config.model_mode == "ridge":
        prediction = ridge_pred
    else:
        tree = make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                learning_rate=0.04,
                max_iter=120,
                max_leaf_nodes=7,
                l2_regularization=2.0,
                min_samples_leaf=20,
                random_state=config.random_seed,
            ),
        )
        tree.fit(x_train, y_train)
        tree_pred = tree.predict(x_current)
        prediction = (
            config.ensemble_ridge_weight * _safe_zscore(ridge_pred)
            + (1.0 - config.ensemble_ridge_weight) * _safe_zscore(tree_pred)
        )
    return pd.Series(prediction, index=current.index.get_level_values("symbol"), dtype=float)


def _cost_rate(symbol: str, groups: Mapping[str, str], config: ResearchConfig) -> float:
    bps = config.crypto_cost_bps if groups.get(symbol) == "crypto" else config.etf_cost_bps
    return bps / 10_000.0


def compute_strategy_returns(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    groups: Mapping[str, str],
    config: ResearchConfig,
    cost_multiplier: float = 1.0,
    extra_delay_months: int = 0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    dates = asset_returns.index
    applied = weights.shift(extra_delay_months) if extra_delay_months else weights
    gross = pd.Series(index=dates, dtype=float)
    costs = pd.Series(index=dates, dtype=float)
    previous = pd.Series(0.0, index=asset_returns.columns)
    for signal_date, row in applied.iterrows():
        if signal_date not in dates:
            continue
        pos = dates.get_loc(signal_date)
        if not isinstance(pos, (int, np.integer)) or pos + 1 >= len(dates):
            continue
        realized_date = dates[pos + 1]
        target = row.reindex(asset_returns.columns).fillna(0.0)
        realized = asset_returns.loc[realized_date].fillna(0.0)
        gross.loc[realized_date] = float((target * realized).sum())
        delta = (target - previous).abs()
        cost = sum(float(delta[s]) * _cost_rate(s, groups, config) for s in delta.index)
        costs.loc[realized_date] = cost * cost_multiplier
        previous = target
    net = gross - costs.fillna(0.0)
    return net.dropna(), gross.dropna(), costs.reindex(gross.dropna().index).fillna(0.0)


def performance_metrics(returns: pd.Series, periods: int = 12) -> dict[str, float]:
    r = returns.dropna().astype(float)
    if len(r) < 2:
        return {"observations": float(len(r)), "sharpe": np.nan}
    wealth = (1.0 + r).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    ann_return = float(wealth.iloc[-1] ** (periods / len(r)) - 1.0) if wealth.iloc[-1] > 0 else -1.0
    ann_vol = float(r.std(ddof=1) * np.sqrt(periods))
    downside = float(r.where(r < 0).std(ddof=1) * np.sqrt(periods))
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(periods)) if r.std(ddof=1) > 0 else np.nan
    sortino = float(r.mean() * periods / downside) if downside > 0 else np.nan
    max_dd = float(drawdown.min())
    return {
        "observations": float(len(r)),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": float(ann_return / abs(max_dd)) if max_dd < 0 else np.nan,
        "win_rate": float((r > 0).mean()),
        "skew": float(skew(r, bias=False)),
        "kurtosis": float(kurtosis(r, fisher=False, bias=False)),
    }


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 3:
        return np.nan
    metrics = performance_metrics(r)
    sr = metrics["sharpe"]
    sk = metrics["skew"]
    ku = metrics["kurtosis"]
    denom = np.sqrt(max(1e-12, 1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr * sr))
    z = (sr - benchmark_sharpe) * np.sqrt(len(r) - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(returns: pd.Series, trial_count: int) -> float:
    r = returns.dropna().astype(float)
    if len(r) < 3 or trial_count < 2:
        return np.nan
    m = performance_metrics(r)
    sr = m["sharpe"]
    sk = m["skew"]
    ku = m["kurtosis"]
    sr_std = np.sqrt(max(1e-12, (1.0 - sk * sr + ((ku - 1.0) / 4.0) * sr * sr) / (len(r) - 1)))
    euler_gamma = 0.5772156649
    expected_max = sr_std * (
        (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trial_count)
        + euler_gamma * norm.ppf(1.0 - 1.0 / (trial_count * np.e))
    )
    return float(norm.cdf((sr - expected_max) / sr_std))


def block_bootstrap(
    returns: pd.Series,
    paths: int,
    block_months: int,
    seed: int,
) -> dict[str, float]:
    r = returns.dropna().to_numpy(dtype=float)
    if len(r) < 12:
        return {}
    rng = np.random.default_rng(seed)
    final = np.empty(paths)
    max_dd = np.empty(paths)
    sharpe = np.empty(paths)
    blocks_needed = int(np.ceil(len(r) / block_months))
    for i in range(paths):
        starts = rng.integers(0, len(r), size=blocks_needed)
        sample = np.concatenate([np.take(r, np.arange(s, s + block_months) % len(r)) for s in starts])[: len(r)]
        wealth = np.cumprod(1.0 + sample)
        dd = wealth / np.maximum.accumulate(wealth) - 1.0
        final[i] = wealth[-1] - 1.0
        max_dd[i] = dd.min()
        sharpe[i] = np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(12) if np.std(sample, ddof=1) > 0 else 0.0
    return {
        "paths": float(paths),
        "median_return": float(np.median(final)),
        "return_p05": float(np.quantile(final, 0.05)),
        "return_p01": float(np.quantile(final, 0.01)),
        "median_sharpe": float(np.median(sharpe)),
        "sharpe_p05": float(np.quantile(sharpe, 0.05)),
        "max_drawdown_p95_magnitude": float(abs(np.quantile(max_dd, 0.05))),
        "max_drawdown_p99_magnitude": float(abs(np.quantile(max_dd, 0.01))),
        "probability_50pct_loss": float(np.mean(final <= -0.50)),
        "probability_negative": float(np.mean(final < 0.0)),
    }


def permutation_pvalue(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    observed_mean: float,
    trials: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    for signal_date, row in weights.iterrows():
        if signal_date not in asset_returns.index:
            continue
        pos = asset_returns.index.get_loc(signal_date)
        if not isinstance(pos, (int, np.integer)) or pos + 1 >= len(asset_returns.index):
            continue
        nxt = asset_returns.iloc[pos + 1].reindex(weights.columns).fillna(0.0).to_numpy(dtype=float)
        samples.append((row.fillna(0.0).to_numpy(dtype=float), nxt))
    if not samples:
        return np.nan
    null_means = np.empty(trials)
    for i in range(trials):
        vals = [float(w @ rng.permutation(ret)) for w, ret in samples]
        null_means[i] = np.mean(vals)
    return float((1 + np.sum(null_means >= observed_mean)) / (trials + 1))


def run_walk_forward(
    prices: pd.DataFrame,
    groups: Mapping[str, str] | None = None,
    config: ResearchConfig | None = None,
) -> ResearchResult:
    config = config or ResearchConfig()
    groups = dict(DEFAULT_GROUPS if groups is None else groups)
    panel, monthly, asset_returns = build_panel(prices, groups)
    dates = list(monthly.index)
    feature_cols = [c for c in panel.columns if c != "target"]
    regime_cols = [c for c in feature_cols if c.startswith(("breadth_", "benchmark_", "bond_", "gold_", "commodity_", "usd_", "equity_bond_"))]
    regime_features = panel.reset_index().drop_duplicates("date").set_index("date")[regime_cols]
    benchmark = "SPY" if "SPY" in asset_returns.columns else asset_returns.columns[0]
    benchmark_forward = asset_returns[benchmark].shift(-1)

    weights_rows: dict[pd.Timestamp, pd.Series] = {}
    audit_rows: list[dict[str, Any]] = []
    for i in range(config.min_train_months, len(dates) - 1):
        signal_date = dates[i]
        last_train_pos = i - config.embargo_months - 1
        if last_train_pos < 0:
            continue
        first_train_pos = max(0, last_train_pos - config.lookback_months + 1)
        train_dates = dates[first_train_pos : last_train_pos + 1]
        train_mask = panel.index.get_level_values("date").isin(train_dates)
        train = panel.loc[train_mask].dropna(subset=["target"])
        current_mask = panel.index.get_level_values("date") == signal_date
        current = panel.loc[current_mask]
        current = current[current.index.get_level_values("symbol").isin(monthly.loc[signal_date].dropna().index)]
        if len(train) < 100 or current.empty:
            continue

        scores = _fit_predict(train, current, feature_cols, config)
        scalar = _regime_scalar(regime_features, benchmark_forward, train_dates, signal_date, config.random_seed)
        trailing = asset_returns.loc[:signal_date].tail(config.covariance_months)
        weights = allocate_weights(scores, trailing, groups, scalar, config)
        weights_rows[signal_date] = weights.reindex(monthly.columns).fillna(0.0)

        max_train_date = max(train_dates)
        max_label_date = dates[dates.index(max_train_date) + 1]
        audit_rows.append(
            {
                "signal_date": signal_date,
                "train_start": min(train_dates),
                "train_feature_end": max_train_date,
                "train_label_end": max_label_date,
                "embargo_months": config.embargo_months,
                "anti_lookahead_pass": bool(max_label_date < signal_date),
                "regime_scalar": scalar,
                "assets_selected": int((weights > 0).sum()),
                "gross_exposure": float(weights.sum()),
                "crypto_exposure": float(sum(weights.get(s, 0.0) for s, g in groups.items() if g == "crypto")),
            }
        )

    weights_df = pd.DataFrame.from_dict(weights_rows, orient="index").sort_index().fillna(0.0)
    weights_df.index.name = "signal_date"
    net, gross, costs = compute_strategy_returns(weights_df, asset_returns, groups, config)
    audit = pd.DataFrame(audit_rows).set_index("signal_date") if audit_rows else pd.DataFrame()
    if audit.empty or not bool(audit["anti_lookahead_pass"].all()):
        raise AssertionError("anti-lookahead audit failed")

    holdout = net.tail(config.holdout_months)
    metrics = performance_metrics(net)
    holdout_metrics = performance_metrics(holdout)
    metrics["probabilistic_sharpe_ratio"] = probabilistic_sharpe_ratio(net)
    metrics["deflated_sharpe_ratio"] = deflated_sharpe_ratio(net, config.trial_count)
    holdout_metrics["probabilistic_sharpe_ratio"] = probabilistic_sharpe_ratio(holdout)
    holdout_metrics["deflated_sharpe_ratio"] = deflated_sharpe_ratio(holdout, config.trial_count)

    double_cost, _, _ = compute_strategy_returns(weights_df, asset_returns, groups, config, cost_multiplier=2.0)
    extra_delay, _, _ = compute_strategy_returns(weights_df, asset_returns, groups, config, extra_delay_months=1)
    no_crypto_weights = weights_df.copy()
    crypto_cols = [s for s in no_crypto_weights if groups.get(s) == "crypto"]
    no_crypto_weights.loc[:, crypto_cols] = 0.0
    no_crypto, _, _ = compute_strategy_returns(no_crypto_weights, asset_returns, groups, config)
    stress = {
        "double_cost": performance_metrics(double_cost),
        "extra_month_delay": performance_metrics(extra_delay),
        "no_crypto": performance_metrics(no_crypto),
    }

    mc = block_bootstrap(
        holdout if len(holdout) >= 12 else net,
        paths=config.monte_carlo_paths,
        block_months=config.monte_carlo_block_months,
        seed=config.random_seed,
    )
    diagnostics = {
        "config": asdict(config),
        "data_start": str(monthly.index.min().date()),
        "data_end": str(monthly.index.max().date()),
        "asset_count": int(monthly.shape[1]),
        "assets": monthly.columns.tolist(),
        "anti_lookahead_pass": bool(audit["anti_lookahead_pass"].all()),
        "average_monthly_turnover": float(weights_df.diff().abs().sum(axis=1).mean()),
        "permutation_pvalue": permutation_pvalue(
            weights_df,
            asset_returns,
            observed_mean=float(gross.mean()),
            trials=config.permutation_trials,
            seed=config.random_seed,
        ),
    }
    return ResearchResult(
        weights=weights_df,
        returns=net,
        gross_returns=gross,
        costs=costs,
        audit=audit,
        metrics=metrics,
        holdout_metrics=holdout_metrics,
        monte_carlo=mc,
        stress=stress,
        diagnostics=diagnostics,
    )
