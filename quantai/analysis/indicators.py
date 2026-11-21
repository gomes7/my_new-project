"""Causal technical indicators (value at bar t uses only bars <= t)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = wilder(d.clip(lower=0), n)
    dn = wilder((-d).clip(lower=0), n)
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna())


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    pdi = 100 * wilder(plus_dm, n) / tr
    mdi = 100 * wilder(minus_dm, n) / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": wilder(dx, n), "plus_di": pdi, "minus_di": mdi})


def bollinger(s: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    m = sma(s, n)
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"bb_mid": m, "bb_up": m + k * sd, "bb_dn": m - k * sd,
                         "bb_width": (2 * k * sd) / m, "zscore": (s - m) / sd.replace(0, np.nan)})


def donchian(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Channel of the PRIOR n bars (excludes current bar) so a breakout compares against the past."""
    return pd.DataFrame({"dc_high": df["high"].shift(1).rolling(n, min_periods=n).max(),
                         "dc_low": df["low"].shift(1).rolling(n, min_periods=n).min()})


def rolling_vwap(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    v = df["volume"].replace(0, np.nan)
    return (tp * v).rolling(n, min_periods=n).sum() / v.rolling(n, min_periods=n).sum()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP reset each UTC day (intraday timeframes)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.floor("1D")
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return pv / vv


def realized_vol(s: pd.Series, n: int = 20) -> pd.Series:
    return np.log(s.where(s > 0)).diff().rolling(n, min_periods=n).std()


def rolling_percentile(s: pd.Series, n: int = 252) -> pd.Series:
    """Percentile rank of the current value within the trailing window (causal)."""
    return s.rolling(n, min_periods=max(30, n // 4)).rank(pct=True)


def efficiency_ratio(s: pd.Series, n: int = 20) -> pd.Series:
    change = (s - s.shift(n)).abs()
    vol = s.diff().abs().rolling(n, min_periods=n).sum()
    return change / vol.replace(0, np.nan)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Standard causal feature set used by regime detection and strategies."""
    out = df.copy()
    c = out["close"]
    out["ema20"], out["ema50"], out["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    out["atr"] = atr(out, 14)
    out["atr_pct"] = out["atr"] / c
    out["rsi"] = rsi(c, 14)
    out = out.join(macd(c)).join(adx(out)).join(bollinger(c)).join(donchian(out, 20))
    out["vwap20"] = rolling_vwap(out, 20)
    out["rv20"] = realized_vol(c, 20)
    out["rv_pct"] = rolling_percentile(out["rv20"], 252)
    out["bbw_pct"] = rolling_percentile(out["bb_width"], 252)
    out["er20"] = efficiency_ratio(c, 20)
    out["ema50_slope"] = (out["ema50"] - out["ema50"].shift(5)) / (5 * out["atr"])
    vol_ma = out["volume"].rolling(20, min_periods=20).mean()
    out["rel_volume"] = out["volume"] / vol_ma.replace(0, np.nan)
    out["mom20"] = c / c.shift(20) - 1
    return out
