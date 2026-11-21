"""Multi-timeframe technical analysis report."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantai.analysis.indicators import add_features
from quantai.analysis.regime import classify_row
from quantai.models import TF_ORDER


@dataclass
class TFView:
    timeframe: str
    bar_close_time: str
    close: float
    trend: str            # up / down / neutral (from structure + EMAs)
    regime: str
    vol_regime: str
    rsi: float
    adx: float
    atr: float
    atr_pct: float
    macd_hist: float
    above_ema200: bool
    rel_volume: float | None
    levels: dict[str, float] = field(default_factory=dict)


@dataclass
class TechnicalReport:
    symbol: str
    views: list[TFView]
    alignment: float      # -1 (all down) .. +1 (all up), higher TFs weighted more
    bias: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "alignment": self.alignment, "bias": self.bias, "notes": self.notes,
                "views": [asdict(v) for v in self.views]}


def _trend(r: pd.Series) -> str:
    if pd.isna(r.ema50) or pd.isna(r.ema20):
        return "neutral"
    if r.close > r.ema20 > r.ema50 and r.ema50_slope > 0:
        return "up"
    if r.close < r.ema20 < r.ema50 and r.ema50_slope < 0:
        return "down"
    return "neutral"


def key_levels(df: pd.DataFrame, tf: str) -> dict[str, float]:
    """Previous day/week high-low and recent swing levels, from completed bars only."""
    lv: dict[str, float] = {}
    if tf == "1d" and len(df) > 6:
        lv["prev_day_high"], lv["prev_day_low"] = float(df["high"].iloc[-2]), float(df["low"].iloc[-2])
        wk = df.resample("W-SUN").agg({"high": "max", "low": "min"}).dropna()
        if len(wk) >= 2:
            lv["prev_week_high"], lv["prev_week_low"] = float(wk["high"].iloc[-2]), float(wk["low"].iloc[-2])
    if len(df) >= 20:
        lv["swing_high_20"] = float(df["high"].iloc[-20:].max())
        lv["swing_low_20"] = float(df["low"].iloc[-20:].min())
    if len(df) >= 60:
        # Simple support/resistance: most-touched price bins in the last 60 bars (rough volume-profile proxy)
        win = df.iloc[-60:]
        prices = np.concatenate([win["high"].values, win["low"].values])
        hist, edges = np.histogram(prices, bins=12)
        top = edges[np.argmax(hist)]
        lv["congestion_zone"] = float(top + (edges[1] - edges[0]) / 2)
    return lv


TF_WEIGHT = {"1d": 0.35, "4h": 0.25, "1h": 0.2, "15m": 0.12, "5m": 0.08}


def analyze(symbol: str, frames: dict[str, pd.DataFrame]) -> TechnicalReport:
    views: list[TFView] = []
    notes: list[str] = []
    for tf in TF_ORDER:
        if tf not in frames or frames[tf].empty:
            continue
        f = frames[tf] if "atr" in frames[tf].columns else add_features(frames[tf])
        r = f.iloc[-1]
        reg, vol = classify_row(r)
        views.append(TFView(
            timeframe=tf, bar_close_time=str(f.index[-1]), close=float(r.close), trend=_trend(r),
            regime=reg.value, vol_regime=vol.value, rsi=float(r.rsi), adx=float(r.adx), atr=float(r.atr),
            atr_pct=float(r.atr_pct), macd_hist=float(r.macd_hist), above_ema200=bool(r.close > r.ema200) if pd.notna(r.ema200) else False,
            rel_volume=float(r.rel_volume) if pd.notna(r.rel_volume) else None, levels=key_levels(f, tf)))
    score, wsum = 0.0, 0.0
    for v in views:
        w = TF_WEIGHT.get(v.timeframe, 0.1)
        score += w * {"up": 1, "down": -1, "neutral": 0}[v.trend]
        wsum += w
    align = score / wsum if wsum else 0.0
    if views:
        htf = views[0]
        for v in views[1:]:
            if v.trend != "neutral" and htf.trend != "neutral" and v.trend != htf.trend:
                notes.append(f"{v.timeframe} trend ({v.trend}) conflicts with {htf.timeframe} ({htf.trend}); "
                             "higher timeframe takes precedence")
        if htf.rsi > 75 or htf.rsi < 25:
            notes.append(f"{htf.timeframe} RSI stretched at {htf.rsi:.0f}")
    bias = "bullish" if align > 0.4 else "bearish" if align < -0.4 else "neutral"
    return TechnicalReport(symbol, views, round(align, 3), bias, notes)
