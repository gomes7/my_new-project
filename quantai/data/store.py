"""SQLite bar cache. (The journal lives in quantai/journal.py and is append-only.)"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from quantai.models import DataMeta, DelayStatus, MarketData

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
  symbol TEXT NOT NULL, timeframe TEXT NOT NULL, source TEXT NOT NULL, ts INTEGER NOT NULL,
  open REAL, high REAL, low REAL, close REAL, volume REAL, retrieved_at TEXT NOT NULL,
  PRIMARY KEY (symbol, timeframe, source, ts)
);
"""


class BarStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save(self, md: MarketData) -> int:
        if md.df.empty:
            return 0
        rows = [(md.symbol, md.timeframe, md.meta.source, int(ts.timestamp()), r.open, r.high, r.low, r.close,
                 r.volume, md.meta.retrieved_at.isoformat()) for ts, r in md.df.iterrows()]
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        return len(rows)

    def load(self, symbol: str, timeframe: str, source: str) -> MarketData:
        with self._conn() as c:
            df = pd.read_sql_query(
                "SELECT ts, open, high, low, close, volume, retrieved_at FROM bars WHERE symbol=? AND timeframe=? "
                "AND source=? ORDER BY ts", c, params=(symbol, timeframe, source))
        retrieved = pd.Timestamp(df["retrieved_at"].max()).to_pydatetime() if not df.empty else pd.Timestamp.now("UTC").to_pydatetime()
        df.index = pd.to_datetime(df.pop("ts"), unit="s", utc=True)
        df.index.name = "time"
        df = df.drop(columns=["retrieved_at"])
        meta = DataMeta(source=source, provider_symbol=symbol, timeframe=timeframe, retrieved_at=retrieved,
                        delay_status=DelayStatus.HISTORICAL, notes="loaded from local cache")
        return MarketData(symbol=symbol, df=df, meta=meta)
