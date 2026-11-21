from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantai.config import ExecutionCfg, InstrumentSpec, Settings, get_instruments, get_settings

FIX = Path(__file__).parent / "fixtures"


def load_fixture(sym: str) -> pd.DataFrame:
    df = pd.read_csv(FIX / f"{sym}_1d.csv")
    df.index = pd.to_datetime(df.pop("time"), utc=True)
    df.index.name = "time"
    return df


@pytest.fixture
def real_daily() -> dict[str, pd.DataFrame]:
    return {s: load_fixture(s) for s in ("BTCUSD", "EURUSD", "SPX", "GOLD")}


@pytest.fixture
def settings() -> Settings:
    return get_settings().model_copy(deep=True)


@pytest.fixture
def instruments() -> dict[str, InstrumentSpec]:
    return get_instruments()


@pytest.fixture
def usd_spec() -> InstrumentSpec:
    return InstrumentSpec(symbol="TEST", asset_class="crypto", provider="csv", provider_symbol="TEST",
                          quote_currency="USD", contract_size=1, qty_step=0.001, min_qty=0.001, spread=0.0,
                          commission_bps=0.0, risk_group="g1")


@pytest.fixture
def exec_cfg() -> ExecutionCfg:
    return ExecutionCfg(slippage_atr_fraction=0.0, partial_exit_fraction=0.0)


def synthetic_ohlcv(n: int = 600, drift: float = 0.0, vol: float = 0.01, seed: int = 1, start="2022-01-01",
                    freq="1D", price=100.0) -> pd.DataFrame:
    """SYNTHETIC test data (geometric random walk). For unit tests of mechanics only."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = price * np.exp(np.cumsum(r))
    open_ = np.concatenate([[price], close[:-1]])
    spread = np.abs(rng.normal(0, vol / 2, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.integers(100, 1000, n).astype(float)}, index=idx)


def utc(*a) -> datetime:
    return datetime(*a, tzinfo=timezone.utc)


TODAY = date(2026, 1, 5)
