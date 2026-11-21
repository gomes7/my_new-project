"""Strategy 5 — Time-series momentum (long-horizon trend following) with a trailing stop.

Hypothesis source (docs/VALIDATION.md): regime labels carry volatility but little short-horizon
directional information, and short-horizon trend rules showed no OOS edge. Time-series momentum
over months is a separately documented effect in futures/FX literature; it is tested here as a
NEW strategy, through the same gate, not as a tweak of trend_pullback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantai.analysis.frame import FeatureFrame
from quantai.models import Direction
from quantai.strategies.base import Strategy, registry

NOT_TRENDLESS = {"strong_uptrend", "weak_uptrend", "strong_downtrend", "weak_downtrend", "breakout", "expansion",
                 "unclassifiable"}


@registry.register
class TimeSeriesMomentum(Strategy):
    name = "tsmom"
    version = "1.0.0"
    family = "momentum"
    description = ("Long when the `long_lb`-bar AND `short_lb`-bar returns are both positive, close > EMA200, and the "
                   "close sets a new `breakout_lb`-bar closing high (entry timing). Initial stop `stop_atr`xATR; "
                   "chandelier trailing stop `trail_atr`xATR from the best price since entry; no partial exit; "
                   "time stop `max_bars`; distant target (`rr`xR) so exits are normally by the trail. Mirror for shorts. "
                   "Not taken in range/compression regimes.")
    default_params = dict(long_lb=120, short_lb=40, breakout_lb=20, stop_atr=3.0, trail_atr=3.0, rr=8.0,
                          max_bars=120, partial=False, min_atr_pct=0.0, max_atr_pct=0.2, min_rel_volume=0.0,
                          blocked_hours_utc=[])
    allowed_regimes_long = NOT_TRENDLESS
    allowed_regimes_short = NOT_TRENDLESS

    def raw_signal(self, f: FeatureFrame, i: int):
        p = self.params
        lb, sb, bb = p["long_lb"], p["short_lb"], p["breakout_lb"]
        if i < max(lb, 200) + 1:
            return None
        close = f.col("close")
        r = f.row(i)
        if any(pd.isna(r[k]) for k in ("ema200", "atr")) or r.atr <= 0:
            return None
        m_long, m_short = close[i] / close[i - lb] - 1, close[i] / close[i - sb] - 1
        prior = close[i - bb:i]
        q = float(np.clip(0.5 + 0.5 * min(abs(m_long) / (r.atr_pct * np.sqrt(lb)), 1.0), 0, 1))
        if m_long > 0 and m_short > 0 and r.close > r.ema200 and close[i] > prior.max():
            return Direction.LONG, r.close - p["stop_atr"] * r.atr, p["rr"], q, {"mom_long": float(m_long)}
        if m_long < 0 and m_short < 0 and r.close < r.ema200 and close[i] < prior.min():
            return Direction.SHORT, r.close + p["stop_atr"] * r.atr, p["rr"], q, {"mom_long": float(m_long)}
        return None
