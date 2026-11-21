"""Fast, read-only bar access for per-bar loops (backtest, paper trading, strategies)."""
from __future__ import annotations

import numpy as np
import pandas as pd


class Row(dict):
    """dict with attribute access; `.name` is the bar open timestamp (like a pandas row)."""
    __slots__ = ("name",)

    def __getattr__(self, k: str):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e


class FeatureFrame:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.index = df.index
        self._cols = {c: df[c].to_numpy() for c in df.columns}
        self._records = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.df)

    @property
    def columns(self):
        return self.df.columns

    def row(self, i: int) -> Row:
        r = Row(self._records[i])
        r.name = self.index[i]
        return r

    def col(self, c: str) -> np.ndarray:
        return self._cols[c]
