"""Strategy 1+3 — Trend following via pullback entries within an established trend."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantai.analysis.frame import FeatureFrame

from quantai.models import Direction
from quantai.strategies.base import Strategy, registry

UP = {"strong_uptrend", "weak_uptrend"}
DOWN = {"strong_downtrend", "weak_downtrend"}


@registry.register
class TrendPullback(Strategy):
    name = "trend_pullback"
    version = "1.0.0"
    family = "trend_following"
    description = ("Long when: close>EMA200, EMA20>EMA50, a bar in the last `lookback` bars traded within "
                   "`touch_atr`xATR of EMA20 (pullback), RSI in [rsi_lo, rsi_hi] at the pullback, and the "
                   "signal bar closes above the prior bar high and above EMA20 (resumption). Stop below the "
                   "pullback low minus `stop_buffer_atr`xATR (min `min_stop_atr`xATR). Target `rr`xR. "
                   "Time stop `max_bars`. Mirror for shorts.")
    default_params = dict(lookback=4, touch_atr=0.5, rsi_lo=38.0, rsi_hi=62.0, stop_buffer_atr=0.5,
                          min_stop_atr=1.0, rr=2.0, max_bars=20, min_atr_pct=0.0, max_atr_pct=0.2,
                          min_rel_volume=0.0, blocked_hours_utc=[])
    allowed_regimes_long = UP
    allowed_regimes_short = DOWN

    def raw_signal(self, f: FeatureFrame, i: int):
        p = self.params
        lb = p["lookback"]
        if i < 200 + lb:
            return None
        r, prev = f.row(i), f.row(i - 1)
        if any(pd.isna(r[k]) for k in ("ema20", "ema50", "ema200", "atr", "rsi")):
            return None
        lows, highs = f.col("low")[i - lb:i], f.col("high")[i - lb:i]
        e20, atr_w, rsi_w = f.col("ema20")[i - lb:i], f.col("atr")[i - lb:i], f.col("rsi")[i - lb:i]
        a = r.atr
        er = r.er20 if pd.notna(r.er20) else 0.0
        q = 0.4 + 0.3 * min(1, max(0, (r.adx - 18) / 20)) + 0.3 * min(1, max(0, er))
        # long
        if r.close > r.ema200 and r.ema20 > r.ema50 and r.close > r.ema20 and r.close > prev.high:
            touched = bool(np.any(lows <= e20 + p["touch_atr"] * atr_w))
            rsi_ok = bool(np.any((rsi_w >= p["rsi_lo"]) & (rsi_w <= p["rsi_hi"])))
            if touched and rsi_ok:
                stop = min(lows.min() - p["stop_buffer_atr"] * a, r.close - p["min_stop_atr"] * a)
                return Direction.LONG, stop, p["rr"], q, {"pullback_low": float(lows.min())}
        # short
        if r.close < r.ema200 and r.ema20 < r.ema50 and r.close < r.ema20 and r.close < prev.low:
            touched = bool(np.any(highs >= e20 - p["touch_atr"] * atr_w))
            rsi_ok = bool(np.any((rsi_w >= 100 - p["rsi_hi"]) & (rsi_w <= 100 - p["rsi_lo"])))
            if touched and rsi_ok:
                stop = max(highs.max() + p["stop_buffer_atr"] * a, r.close + p["min_stop_atr"] * a)
                return Direction.SHORT, stop, p["rr"], q, {"pullback_high": float(highs.max())}
        return None
