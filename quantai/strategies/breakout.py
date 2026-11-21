"""Strategy 2+6 — Compression breakout / volatility expansion."""
from __future__ import annotations

import pandas as pd

from quantai.analysis.frame import FeatureFrame

from quantai.models import Direction
from quantai.strategies.base import Strategy, registry


@registry.register
class CompressionBreakout(Strategy):
    name = "compression_breakout"
    version = "1.0.0"
    family = "breakout"
    description = ("Requires Bollinger-width percentile <= `compression_pct` on at least one of the previous "
                   "`compression_lookback` bars (compression), then a close beyond the prior 20-bar Donchian "
                   "channel, closing in the outer `close_location` of the bar range, with relative volume >= "
                   "`min_rel_volume` where volume exists. Stop `stop_atr`xATR from entry. Target `rr`xR. "
                   "Time stop `max_bars`. Avoids breakouts against a strong opposite trend.")
    default_params = dict(compression_pct=0.2, compression_lookback=5, close_location=0.3, stop_atr=1.5,
                          rr=2.0, max_bars=15, min_atr_pct=0.0, max_atr_pct=0.2, min_rel_volume=1.1,
                          blocked_hours_utc=[])
    allowed_regimes_long = {"breakout", "compression", "expansion", "weak_uptrend", "strong_uptrend", "range"}
    allowed_regimes_short = {"breakout", "compression", "expansion", "weak_downtrend", "strong_downtrend", "range"}

    def raw_signal(self, f: FeatureFrame, i: int):
        p = self.params
        if i < 60:
            return None
        r = f.row(i)
        if any(pd.isna(r[k]) for k in ("dc_high", "dc_low", "atr", "bbw_pct")):
            return None
        prior = f.col("bbw_pct")[i - p["compression_lookback"]:i]
        if not (prior <= p["compression_pct"]).any():
            return None
        rng = r.high - r.low
        if rng <= 0:
            return None
        loc = (r.close - r.low) / rng
        q_base = 0.5 + 0.25 * (1 - float(prior.min()) / max(p["compression_pct"], 1e-9))
        vol_bonus = 0.25 * min(1.0, max(0.0, (r.rel_volume - 1) / 1.5)) if pd.notna(r.rel_volume) else 0.0
        if r.close > r.dc_high and loc >= 1 - p["close_location"]:
            return Direction.LONG, r.close - p["stop_atr"] * r.atr, p["rr"], q_base + vol_bonus, {"level": float(r.dc_high)}
        if r.close < r.dc_low and loc <= p["close_location"]:
            return Direction.SHORT, r.close + p["stop_atr"] * r.atr, p["rr"], q_base + vol_bonus, {"level": float(r.dc_low)}
        return None
