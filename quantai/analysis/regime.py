"""Market regime detection (rule-based, measurable, causal) and its historical evaluation."""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd


class Regime(str, Enum):
    STRONG_UP = "strong_uptrend"
    WEAK_UP = "weak_uptrend"
    STRONG_DOWN = "strong_downtrend"
    WEAK_DOWN = "weak_downtrend"
    RANGE = "range"
    BREAKOUT = "breakout"
    COMPRESSION = "compression"
    EXPANSION = "expansion"
    UNCLASSIFIABLE = "unclassifiable"


class VolRegime(str, Enum):
    HIGH = "high_volatility"
    NORMAL = "normal_volatility"
    LOW = "low_volatility"
    UNKNOWN = "unknown"


# Thresholds are explicit and versioned; change => new REGIME_VERSION.
REGIME_VERSION = "1.0.0"
P = dict(adx_strong=25.0, adx_weak=18.0, slope_min=0.05, compression_pct=0.15, expansion_pct=0.85,
         vol_high_pct=0.90, vol_low_pct=0.15)


def classify_row(r: pd.Series) -> tuple[Regime, VolRegime]:
    needed = ["adx", "ema50_slope", "ema200", "close", "bbw_pct", "rv_pct", "dc_high", "dc_low"]
    if any(pd.isna(r.get(k)) for k in needed):
        return Regime.UNCLASSIFIABLE, VolRegime.UNKNOWN
    vol = VolRegime.HIGH if r.rv_pct >= P["vol_high_pct"] else VolRegime.LOW if r.rv_pct <= P["vol_low_pct"] else VolRegime.NORMAL
    broke_up, broke_dn = r.close > r.dc_high, r.close < r.dc_low
    if (broke_up or broke_dn) and r.bbw_pct < 0.5:
        return Regime.BREAKOUT, vol
    if r.bbw_pct <= P["compression_pct"] and r.adx < P["adx_weak"]:
        return Regime.COMPRESSION, vol
    up = r.ema50_slope > P["slope_min"] and r.close > r.ema200
    dn = r.ema50_slope < -P["slope_min"] and r.close < r.ema200
    if r.adx >= P["adx_strong"]:
        if up:
            return Regime.STRONG_UP, vol
        if dn:
            return Regime.STRONG_DOWN, vol
    if r.adx >= P["adx_weak"]:
        if r.ema50_slope > P["slope_min"]:
            return Regime.WEAK_UP, vol
        if r.ema50_slope < -P["slope_min"]:
            return Regime.WEAK_DOWN, vol
    if r.bbw_pct >= P["expansion_pct"]:
        return Regime.EXPANSION, vol
    if r.adx < P["adx_weak"]:
        return Regime.RANGE, vol
    return Regime.UNCLASSIFIABLE, vol


def classify(feat: pd.DataFrame) -> pd.DataFrame:
    """Vectorised per-bar regime labels; identical logic to `classify_row` (tested)."""
    g = {k: feat[k].to_numpy(dtype=float) for k in ["adx", "ema50_slope", "ema200", "close", "bbw_pct", "rv_pct", "dc_high", "dc_low"]}
    na = np.zeros(len(feat), dtype=bool)
    for v in g.values():
        na |= np.isnan(v)
    with np.errstate(invalid="ignore"):
        vol = np.where(g["rv_pct"] >= P["vol_high_pct"], VolRegime.HIGH.value,
                       np.where(g["rv_pct"] <= P["vol_low_pct"], VolRegime.LOW.value, VolRegime.NORMAL.value))
        broke = ((g["close"] > g["dc_high"]) | (g["close"] < g["dc_low"])) & (g["bbw_pct"] < 0.5)
        comp = (g["bbw_pct"] <= P["compression_pct"]) & (g["adx"] < P["adx_weak"])
        up = (g["ema50_slope"] > P["slope_min"]) & (g["close"] > g["ema200"])
        dn = (g["ema50_slope"] < -P["slope_min"]) & (g["close"] < g["ema200"])
        strong = g["adx"] >= P["adx_strong"]
        weak = g["adx"] >= P["adx_weak"]
        conds = [na, broke, comp, strong & up, strong & dn,
                 weak & (g["ema50_slope"] > P["slope_min"]), weak & (g["ema50_slope"] < -P["slope_min"]),
                 g["bbw_pct"] >= P["expansion_pct"], g["adx"] < P["adx_weak"]]
        choices = [Regime.UNCLASSIFIABLE.value, Regime.BREAKOUT.value, Regime.COMPRESSION.value, Regime.STRONG_UP.value,
                   Regime.STRONG_DOWN.value, Regime.WEAK_UP.value, Regime.WEAK_DOWN.value, Regime.EXPANSION.value,
                   Regime.RANGE.value]
        reg = np.select(conds, choices, default=Regime.UNCLASSIFIABLE.value)
    vol = np.where(na, VolRegime.UNKNOWN.value, vol)
    return pd.DataFrame({"regime": reg, "vol_regime": vol}, index=feat.index)


def evaluate(feat: pd.DataFrame, regimes: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
    """Historical calibration: forward return / volatility statistics per regime label.

    A useful classifier should show (a) directional forward drift that is positive in up-trend
    labels and negative in down-trend labels, and (b) higher forward |return| in high-vol labels.
    The output is evidence to inspect, not proof of edge.
    """
    fwd = np.log(feat["close"].shift(-horizon) / feat["close"])
    fwd_abs = fwd.abs()
    atr_norm = fwd / (feat["atr"] * np.sqrt(horizon) / feat["close"])
    df = pd.DataFrame({"regime": regimes["regime"], "fwd": fwd, "fwd_abs": fwd_abs, "fwd_atr": atr_norm}).dropna()
    g = df.groupby("regime")
    out = pd.DataFrame({
        "n": g.size(),
        "share": g.size() / len(df),
        "mean_fwd_ret_pct": g["fwd"].mean() * 100,
        "hit_up": g["fwd"].apply(lambda x: (x > 0).mean()),
        "mean_abs_ret_pct": g["fwd_abs"].mean() * 100,
        "mean_fwd_atr_units": g["fwd_atr"].mean(),
        "t_stat": g["fwd"].apply(lambda x: x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan),
    })
    # Note: overlapping horizons inflate t-stats; treat as indicative only.
    return out.sort_values("n", ascending=False)
