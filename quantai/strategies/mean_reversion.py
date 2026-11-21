"""Strategy 4 — Mean reversion in non-trending regimes."""
from __future__ import annotations

import pandas as pd

from quantai.analysis.frame import FeatureFrame

from quantai.models import Direction
from quantai.strategies.base import Strategy, registry


@registry.register
class RangeMeanReversion(Strategy):
    name = "range_mean_reversion"
    version = "1.0.0"
    family = "mean_reversion"
    description = ("Only in range/compression regimes with ADX < `max_adx`. Long when the PREVIOUS bar closed "
                   "with z-score <= -`z_entry` and RSI <= `rsi_os`, and the signal bar confirms by closing up "
                   "(close > open and close > previous close). Stop below the stretch low minus "
                   "`stop_buffer_atr`xATR (min `min_stop_atr`xATR). Target = 20-bar mean, trade rejected if that "
                   "gives reward:risk < `min_rr`. Time stop `max_bars`. Mirror for shorts.")
    default_params = dict(z_entry=2.0, rsi_os=30.0, max_adx=22.0, stop_buffer_atr=0.5, min_stop_atr=1.0,
                          min_rr=1.0, max_bars=10, min_atr_pct=0.0, max_atr_pct=0.2, min_rel_volume=0.0,
                          blocked_hours_utc=[])
    allowed_regimes_long = {"range", "compression"}
    allowed_regimes_short = {"range", "compression"}

    def raw_signal(self, f: FeatureFrame, i: int):
        p = self.params
        if i < 60:
            return None
        r, prev = f.row(i), f.row(i - 1)
        if any(pd.isna(x) for x in (prev.zscore, prev.rsi, r.adx, r.bb_mid, r.atr)) or r.adx >= p["max_adx"]:
            return None
        if prev.zscore <= -p["z_entry"] and prev.rsi <= p["rsi_os"] and r.close > r.open and r.close > prev.close:
            stop = min(min(prev.low, r.low) - p["stop_buffer_atr"] * r.atr, r.close - p["min_stop_atr"] * r.atr)
            rr = (r.bb_mid - r.close) / (r.close - stop)
            if rr >= p["min_rr"]:
                q = 0.5 + 0.25 * min(1, (-prev.zscore - p["z_entry"]) / 1.0) + 0.25 * min(1, (p["max_adx"] - r.adx) / 10)
                return Direction.LONG, stop, float(rr), q, {"z": float(prev.zscore)}
        if prev.zscore >= p["z_entry"] and prev.rsi >= 100 - p["rsi_os"] and r.close < r.open and r.close < prev.close:
            stop = max(max(prev.high, r.high) + p["stop_buffer_atr"] * r.atr, r.close + p["min_stop_atr"] * r.atr)
            rr = (r.close - r.bb_mid) / (stop - r.close)
            if rr >= p["min_rr"]:
                q = 0.5 + 0.25 * min(1, (prev.zscore - p["z_entry"]) / 1.0) + 0.25 * min(1, (p["max_adx"] - r.adx) / 10)
                return Direction.SHORT, stop, float(rr), q, {"z": float(prev.zscore)}
        return None
