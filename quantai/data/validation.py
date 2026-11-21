"""Data validation layer. A FAILED report blocks new signals for that instrument."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from quantai.config import DataCfg
from quantai.models import (TF_SECONDS, MarketData, QualityIssue, QualityReport, QualityStatus,
                            tf_delta, utcnow)

_SEV_RANK = {QualityStatus.OK: 0, QualityStatus.DEGRADED: 1, QualityStatus.FAILED: 2}


def _expected_bar_count(index: pd.DatetimeIndex, tf: str, asset_class: str) -> int:
    if len(index) < 2:
        return len(index)
    full = pd.date_range(index[0], index[-1], freq=pd.Timedelta(seconds=TF_SECONDS[tf]))
    if asset_class == "crypto":
        return len(full)
    # Traditional markets: ignore weekends (Sat full day, Sun until evening for FX).
    wk = full[full.dayofweek < 5]
    return max(len(wk), len(index))


def validate(md: MarketData, cfg: DataCfg, asset_class: str = "crypto",
             now: datetime | None = None, live: bool = True) -> QualityReport:
    """Validate OHLCV bars. `live=False` disables staleness checks (historical research)."""
    df = md.df
    issues: list[QualityIssue] = []
    tf = md.timeframe

    if df is None or df.empty:
        rep = QualityReport(QualityStatus.FAILED, [QualityIssue("empty", QualityStatus.FAILED, "no bars returned")])
        md.quality = rep
        return rep

    cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        issues.append(QualityIssue("missing_columns", QualityStatus.FAILED, str(missing_cols)))
        rep = QualityReport(QualityStatus.FAILED, issues)
        md.quality = rep
        return rep

    # Timestamps
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        issues.append(QualityIssue("timestamp_tz", QualityStatus.FAILED, "index must be tz-aware UTC DatetimeIndex"))
    else:
        if not df.index.is_monotonic_increasing:
            issues.append(QualityIssue("timestamp_order", QualityStatus.FAILED, "timestamps not increasing"))
        dup = int(df.index.duplicated().sum())
        if dup:
            issues.append(QualityIssue("duplicates", QualityStatus.DEGRADED, "duplicate timestamps", dup))
        n = now or utcnow()
        future = int((df.index > pd.Timestamp(n)).sum())
        if future:
            issues.append(QualityIssue("future_timestamps", QualityStatus.FAILED, "bars dated in the future", future))

    # Missing values
    nan_rows = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if nan_rows:
        sev = QualityStatus.FAILED if nan_rows / len(df) > cfg.max_missing_ratio else QualityStatus.DEGRADED
        issues.append(QualityIssue("missing_values", sev, "rows with NaN prices", nan_rows))

    clean = df.dropna(subset=["open", "high", "low", "close"])
    # Non-positive prices
    nonpos = int((clean[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    if nonpos:
        issues.append(QualityIssue("non_positive_price", QualityStatus.FAILED, "price <= 0", nonpos))

    # OHLC consistency
    bad_hi = clean["high"] < clean[["open", "close"]].max(axis=1) - 1e-12
    bad_lo = clean["low"] > clean[["open", "close"]].min(axis=1) + 1e-12
    inconsistent = int((bad_hi | bad_lo | (clean["high"] < clean["low"])).sum())
    if inconsistent:
        sev = QualityStatus.FAILED if inconsistent / len(clean) > 0.01 else QualityStatus.DEGRADED
        issues.append(QualityIssue("ohlc_inconsistent", sev, "high/low do not bracket open/close", inconsistent))

    # Abnormal jumps: robust z-score of log close-to-close returns (MAD-based)
    pos = clean[clean["close"] > 0]
    if len(pos) > 30:
        # Only returns between contiguous bars: session opens / weekend gaps are not data errors.
        contiguous = pos.index.to_series().diff() == pd.Timedelta(seconds=TF_SECONDS[tf])
        r = np.log(pos["close"]).diff()[contiguous].dropna()
        med = r.rolling(100, min_periods=30).median()
        mad = (r - med).abs().rolling(100, min_periods=30).median() * 1.4826
        z = ((r - med) / mad.replace(0, np.nan)).abs()
        big = z > cfg.jump_sigma
        nxt = r.shift(-1)
        # Bad-print signature: an outsized move immediately (>=70%) reversed by the next bar.
        spikes = int((big & (np.sign(nxt) == -np.sign(r)) & (nxt.abs() >= 0.7 * r.abs())).sum())
        if spikes:
            issues.append(QualityIssue("suspected_bad_tick", QualityStatus.DEGRADED,
                                       f"{spikes} outsized moves reversed next bar (verify vs 2nd source)", spikes))
        if int(big.sum()) - spikes > 0:
            issues.append(QualityIssue("large_move", QualityStatus.OK,
                                       f"moves beyond {cfg.jump_sigma} robust sigma that persisted (informational)",
                                       int(big.sum()) - spikes))

    # Gaps. Only for 24h markets: exchange session calendars for indices/commodities/stocks are not
    # modelled yet, so their gaps cannot be distinguished from closed sessions (documented limitation).
    if isinstance(df.index, pd.DatetimeIndex) and len(df) > 2 and asset_class in ("crypto", "forex"):
        exp = _expected_bar_count(df.index, tf, asset_class)
        miss_ratio = max(0.0, 1 - len(df) / exp) if exp else 0.0
        limit = cfg.max_missing_ratio if asset_class == "crypto" else max(cfg.max_missing_ratio, 0.10)
        if miss_ratio > limit:
            issues.append(QualityIssue("gaps", QualityStatus.DEGRADED, f"missing bar ratio {miss_ratio:.1%}"))

    # Zero volume (only meaningful where provider supplies volume)
    if asset_class in ("crypto", "stock") and len(clean) and (clean["volume"] <= 0).mean() > 0.2:
        issues.append(QualityIssue("zero_volume", QualityStatus.DEGRADED, "over 20% of bars have zero volume"))

    # Staleness
    if live and isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        n = now or utcnow()
        last_close = df.index[-1] + tf_delta(tf)
        age = (pd.Timestamp(n) - last_close).total_seconds()
        limit = cfg.stale_multiplier * TF_SECONDS[tf]
        if asset_class != "crypto":
            limit += 3 * 86400  # weekends / holidays for traditional markets
        if age > limit:
            issues.append(QualityIssue("stale", QualityStatus.FAILED,
                                       f"last bar closed {age/3600:.1f}h ago (limit {limit/3600:.1f}h)"))

    status = QualityStatus.OK
    for i in issues:
        if _SEV_RANK[i.severity] > _SEV_RANK[status]:
            status = i.severity
    rep = QualityReport(status, issues)
    md.quality = rep
    return rep


def compare_providers(a: MarketData, b: MarketData, tolerance_pct: float, window: int | None = None) -> QualityReport:
    """Cross-check closes of the same instrument from two providers on overlapping bars.
    FAILED if >5% of compared bars differ by more than `tolerance_pct`, or if the LATEST common bar does
    (the bar a signal would act on)."""
    j = a.df[["close"]].join(b.df[["close"]], how="inner", lsuffix="_a", rsuffix="_b")
    if window:
        j = j.iloc[-window:]
    if j.empty:
        return QualityReport(QualityStatus.DEGRADED, [QualityIssue("no_overlap", QualityStatus.DEGRADED, "no common timestamps")])
    diff = (j["close_a"] / j["close_b"] - 1).abs() * 100
    bad = int((diff > tolerance_pct).sum())
    issues = []
    if bad:
        sev = QualityStatus.FAILED if bad / len(j) > 0.05 else QualityStatus.DEGRADED
        if diff.iloc[-1] > tolerance_pct:
            sev = QualityStatus.FAILED
        issues.append(QualityIssue("provider_discrepancy", sev,
                                   f"{a.meta.source} vs {b.meta.source}: {bad}/{len(j)} bars differ > {tolerance_pct}% "
                                   f"(max {diff.max():.2f}%, latest {diff.iloc[-1]:.2f}%)", bad))
    status = issues[0].severity if issues else QualityStatus.OK
    return QualityReport(status, issues)


def clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic cleaning: drop NaN-price rows and duplicate timestamps (keep last). Never fabricates bars."""
    out = df.dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out["volume"] = out["volume"].fillna(0.0)
    return out
