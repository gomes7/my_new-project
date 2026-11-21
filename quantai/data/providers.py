"""Market data providers.

Implemented and verified against live endpoints in development:
  * KrakenProvider   — Kraken public REST OHLC (spot crypto, exchange-native, no key).
                       Limit: returns at most the 720 most recent bars per interval.
  * YahooChartProvider — Yahoo Finance chart endpoint (FX, indices, futures). UNOFFICIAL,
                       undocumented, may change or rate-limit; delay not guaranteed. Research only;
                       check Yahoo's terms before any non-personal use.
  * FredProvider     — FRED public CSV (macro series, no key).
  * CsvProvider      — local CSV files (any licensed vendor export).
Interface-only (need a subscription/key): see PaidProviderStub.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantai.config import InstrumentSpec, Secrets
from quantai.data.http import ProviderError, get
from quantai.data.validation import clean_bars
from quantai.models import TF_SECONDS, DataMeta, DelayStatus, MarketData, utcnow

COLS = ["open", "high", "low", "close", "volume"]


def drop_incomplete(df: pd.DataFrame, tf: str, now: datetime) -> pd.DataFrame:
    """Keep only bars whose close time <= now. Prevents acting on a still-forming bar."""
    if df.empty:
        return df
    close_t = df.index + pd.Timedelta(seconds=TF_SECONDS[tf])
    return df[close_t <= pd.Timestamp(now)]


def resample_ohlcv(df: pd.DataFrame, tf: str, anchor: pd.Timedelta | None = None) -> pd.DataFrame:
    """Aggregate to a higher timeframe. Buckets are anchored on the source bars' own offset
    (e.g. index hourly bars starting at :30) so that no source bar straddles a bucket close,
    which would otherwise leak post-close data into the aggregated bar."""
    if df.empty:
        return df
    rule = pd.Timedelta(seconds=TF_SECONDS[tf])
    step = df.index.to_series().diff().median()
    offset = pd.Timedelta(seconds=int(df.index[0].timestamp()) % int(step.total_seconds())) if pd.notna(step) else pd.Timedelta(0)
    if anchor is not None:
        offset = anchor
    out = df.resample(rule, origin="epoch", offset=offset, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return out.dropna(subset=["open", "close"])


class MarketDataProvider(ABC):
    name: str = "base"
    delay_status: DelayStatus = DelayStatus.UNVERIFIED

    @abstractmethod
    def fetch_ohlcv(self, provider_symbol: str, timeframe: str, start: datetime | None = None) -> pd.DataFrame:
        """Return raw OHLCV with tz-aware UTC DatetimeIndex of bar open times."""

    def get_bars(self, spec: InstrumentSpec, timeframe: str, start: datetime | None = None,
                 provider_symbol: str | None = None) -> MarketData:
        now = utcnow()
        psym = provider_symbol or spec.provider_symbol
        notes = ""
        if timeframe == "1d" and spec.daily_from_intraday:
            raw = self.fetch_ohlcv(psym, "1h", start)
            raw = resample_ohlcv(clean_bars(raw), "1d", anchor=pd.Timedelta(hours=22)) if not raw.empty else raw
            notes = "1d built from 1h bars, day boundary 22:00 UTC (vendor daily bars failed validation)"
        else:
            raw = self.fetch_ohlcv(psym, timeframe, start)
        df = drop_incomplete(clean_bars(raw), timeframe, now) if not raw.empty else raw
        meta = DataMeta(source=self.name, provider_symbol=psym, timeframe=timeframe,
                        retrieved_at=now, delay_status=self.delay_status, notes=notes)
        return MarketData(symbol=spec.symbol, df=df, meta=meta)


class KrakenProvider(MarketDataProvider):
    name = "kraken"
    delay_status = DelayStatus.REALTIME   # exchange-native for Kraken's own book
    BASE = "https://api.kraken.com/0/public"
    INTERVALS = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

    def fetch_ohlcv(self, provider_symbol: str, timeframe: str, start: datetime | None = None) -> pd.DataFrame:
        if timeframe not in self.INTERVALS:
            raise ProviderError(f"kraken: unsupported timeframe {timeframe}")
        params = {"pair": provider_symbol, "interval": self.INTERVALS[timeframe]}
        if start:
            params["since"] = int(start.timestamp())
        js = get(f"{self.BASE}/OHLC", params=params).json()
        if js.get("error"):
            raise ProviderError(f"kraken error: {js['error']}")
        result = {k: v for k, v in js["result"].items() if k != "last"}
        if not result:
            raise ProviderError("kraken: empty result")
        rows = next(iter(result.values()))
        df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
        df.index = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
        df = df[COLS].astype(float)
        df.index.name = "time"
        return df

    def ticker(self, provider_symbol: str) -> dict:
        """Best bid/ask and last trade (real-time, Kraken book only)."""
        js = get(f"{self.BASE}/Ticker", params={"pair": provider_symbol}).json()
        if js.get("error"):
            raise ProviderError(f"kraken error: {js['error']}")
        t = next(iter(js["result"].values()))
        bid, ask = float(t["b"][0]), float(t["a"][0])
        return {"bid": bid, "ask": ask, "spread": ask - bid, "last": float(t["c"][0]),
                "retrieved_at": utcnow().isoformat(), "source": self.name}


class YahooChartProvider(MarketDataProvider):
    name = "yahoo"
    delay_status = DelayStatus.UNVERIFIED
    BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
    # Yahoo-native intervals and the max range each supports
    NATIVE = {"5m": ("5m", "60d"), "15m": ("15m", "60d"), "1h": ("60m", "730d"), "1d": ("1d", "max")}

    def fetch_ohlcv(self, provider_symbol: str, timeframe: str, start: datetime | None = None) -> pd.DataFrame:
        if timeframe == "4h":
            return resample_ohlcv(self.fetch_ohlcv(provider_symbol, "1h", start), "4h")
        if timeframe not in self.NATIVE:
            raise ProviderError(f"yahoo: unsupported timeframe {timeframe}")
        interval, rng = self.NATIVE[timeframe]
        params: dict = {"interval": interval, "includePrePost": "false", "events": "div,splits"}
        if timeframe == "1d":
            # NB: range=max silently returns MONTHLY bars; always request explicit periods.
            p1 = start or (utcnow() - pd.Timedelta(days=365 * 12))
            params["period1"] = int(p1.timestamp())
            params["period2"] = int(utcnow().timestamp())
        else:
            params["range"] = rng
        js = get(f"{self.BASE}/{provider_symbol}", params=params,
                 headers={"User-Agent": "Mozilla/5.0 (research)"}).json()
        chart = js.get("chart", {})
        if chart.get("error"):
            raise ProviderError(f"yahoo error: {chart['error']}")
        res = (chart.get("result") or [None])[0]
        if not res or "timestamp" not in res:
            raise ProviderError(f"yahoo: no data for {provider_symbol}")
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({c: q.get(c) for c in COLS}, index=pd.to_datetime(res["timestamp"], unit="s", utc=True))
        df.index.name = "time"
        df = df.astype(float)
        # Yahoo appends the in-progress bar stamped with the last trade time (non-zero seconds): drop it.
        df = df[(df.index.second == 0)]
        if timeframe == "1d":
            # Normalise daily bars to 00:00 UTC of the trading date (exchange-local date).
            tz = res.get("meta", {}).get("exchangeTimezoneName", "UTC")
            local_dates = df.index.tz_convert(tz).normalize().tz_localize(None)
            df.index = pd.DatetimeIndex(local_dates).tz_localize("UTC")
            df = df[~df.index.duplicated(keep="last")]
        return df


class FredProvider:
    """FRED public CSV (no key). Returns a Series indexed by observation date (UTC)."""
    name = "fred"
    BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def series(self, series_id: str, start: str | None = None) -> pd.Series:
        params = {"id": series_id}
        if start:
            params["cosd"] = start
        text = get(self.BASE, params=params, timeout=30).text
        df = pd.read_csv(io.StringIO(text))
        if df.shape[1] < 2:
            raise ProviderError(f"fred: unexpected payload for {series_id}")
        date_col, val_col = df.columns[0], df.columns[1]
        s = pd.to_numeric(df[val_col], errors="coerce")
        s.index = pd.to_datetime(df[date_col], utc=True)
        s.name = series_id
        return s.dropna()


class CsvProvider(MarketDataProvider):
    """CSV with columns time,open,high,low,close[,volume]. `provider_symbol` = file stem."""
    name = "csv"
    delay_status = DelayStatus.HISTORICAL

    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def fetch_ohlcv(self, provider_symbol: str, timeframe: str, start: datetime | None = None) -> pd.DataFrame:
        path = self.directory / f"{provider_symbol}_{timeframe}.csv"
        if not path.exists():
            path = self.directory / f"{provider_symbol}.csv"
        if not path.exists():
            raise ProviderError(f"csv: {path} not found")
        df = pd.read_csv(path)
        tcol = next(c for c in df.columns if c.lower() in ("time", "timestamp", "date", "datetime"))
        idx = pd.to_datetime(df[tcol], utc=True)
        df.columns = [c.lower() for c in df.columns]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        out = df[COLS].astype(float)
        out.index = pd.DatetimeIndex(idx, name="time")
        if start:
            out = out[out.index >= pd.Timestamp(start)]
        return out


class PaidProviderStub(MarketDataProvider):
    """Integration point for licensed real-time feeds (e.g. Polygon, OANDA v20, Twelve Data).

    Not implemented: requires an account, an API key in QUANTAI_<NAME>_API_KEY and a
    vendor-specific adapter. Raising here is deliberate — no silent fallback to stale data.
    """

    def __init__(self, name: str):
        self.name = name

    def fetch_ohlcv(self, provider_symbol: str, timeframe: str, start: datetime | None = None) -> pd.DataFrame:
        key = Secrets.get(f"{self.name.upper()}_API_KEY")
        if not key:
            raise ProviderError(f"{self.name}: no API key configured (set QUANTAI_{self.name.upper()}_API_KEY)")
        raise ProviderError(f"{self.name}: adapter not implemented yet")


def build_provider(name: str, csv_dir: Path | None = None) -> MarketDataProvider:
    if name == "kraken":
        return KrakenProvider()
    if name == "yahoo":
        return YahooChartProvider()
    if name == "csv":
        return CsvProvider(csv_dir or Path("data"))
    return PaidProviderStub(name)


def to_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
