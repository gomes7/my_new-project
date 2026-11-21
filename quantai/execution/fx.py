"""Currency conversion from observed FX closes only (no assumed rates)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from quantai.models import TF_SECONDS


class FxConverter:
    """Converts a quote currency to the account currency using the latest CLOSED FX bar at time t."""

    def __init__(self, account_ccy: str = "USD"):
        self.account_ccy = account_ccy
        self._series: dict[str, pd.Series] = {}   # pair "EURUSD" -> closes indexed by bar CLOSE time

    def add_pair(self, pair: str, df: pd.DataFrame, timeframe: str) -> None:
        s = df["close"].copy()
        s.index = s.index + pd.Timedelta(seconds=TF_SECONDS[timeframe])
        self._series[pair] = s.sort_index()

    def rate(self, ccy: str, t: datetime | pd.Timestamp) -> float | None:
        if ccy == self.account_ccy:
            return 1.0
        t = pd.Timestamp(t)
        direct, inverse = f"{ccy}{self.account_ccy}", f"{self.account_ccy}{ccy}"
        for pair, inv in ((direct, False), (inverse, True)):
            s = self._series.get(pair)
            if s is not None:
                s = s[s.index <= t]
                if not s.empty:
                    v = float(s.iloc[-1])
                    return 1 / v if inv else v
        return None
