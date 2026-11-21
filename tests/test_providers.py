import json
from datetime import timedelta

import pandas as pd
import pytest

from quantai.config import get_instruments
from quantai.data import providers as P
from quantai.data.http import ProviderError
from quantai.models import utcnow
from tests.conftest import FIX, synthetic_ohlcv


class FakeResp:
    def __init__(self, js):
        self._js = js
        self.text = json.dumps(js)

    def json(self):
        return self._js


def test_drop_incomplete_removes_forming_bar():
    now = utcnow()
    idx = pd.date_range(end=now.replace(minute=0, second=0, microsecond=0), periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}, index=idx, dtype=float)
    out = P.drop_incomplete(df, "1h", now)
    assert out.index[-1] + timedelta(hours=1) <= now and len(out) == 4


def test_resample_anchors_to_source_offset_no_leak():
    # hourly bars starting at :30 (like US cash indices)
    idx = pd.date_range("2026-01-05 13:30", periods=8, freq="1h", tz="UTC")
    df = pd.DataFrame({"open": range(8), "high": range(8), "low": range(8), "close": range(8), "volume": 1.0}, index=idx, dtype=float)
    out = P.resample_ohlcv(df, "4h")
    assert all(t.minute == 30 for t in out.index)
    # every source bar must end at or before its bucket close
    for t in out.index:
        members = df[(df.index >= t) & (df.index < t + pd.Timedelta(hours=4))]
        assert (members.index + pd.Timedelta(hours=1)).max() <= t + pd.Timedelta(hours=4)


def test_kraken_parser(monkeypatch):
    js = json.load(open(FIX / "kraken_ohlc_sample.json"))
    monkeypatch.setattr(P, "get", lambda *a, **k: FakeResp(js))
    df = P.KrakenProvider().fetch_ohlcv("XBTUSD", "1d")
    assert list(df.columns) == P.COLS and df.index.tz is not None and len(df) == 30
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()


def test_kraken_error(monkeypatch):
    monkeypatch.setattr(P, "get", lambda *a, **k: FakeResp({"error": ["EQuery:Unknown asset pair"], "result": {}}))
    with pytest.raises(ProviderError):
        P.KrakenProvider().fetch_ohlcv("BAD", "1d")


def test_yahoo_parser_drops_partial_bar(monkeypatch):
    js = json.load(open(FIX / "yahoo_chart_sample.json"))
    monkeypatch.setattr(P, "get", lambda *a, **k: FakeResp(js))
    df = P.YahooChartProvider().fetch_ohlcv("GC=F", "1h")
    assert len(df) > 10 and (df.index.second == 0).all()


def test_paid_stub_never_returns_data(monkeypatch):
    monkeypatch.delenv("QUANTAI_POLYGON_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="no API key"):
        P.PaidProviderStub("polygon").fetch_ohlcv("X", "1d")


def test_csv_provider(tmp_path):
    df = synthetic_ohlcv(50)
    df.to_csv(tmp_path / "ABC_1d.csv", index_label="time")
    out = P.CsvProvider(tmp_path).fetch_ohlcv("ABC", "1d")
    assert len(out) == 50 and out.index.tz is not None


def test_runtime_never_serves_cache_as_live(tmp_path, monkeypatch):
    from quantai.runtime import Runtime
    rt = Runtime(data_dir=tmp_path)
    spec = get_instruments()["BTCUSD"]

    def boom(*a, **k):
        raise ProviderError("down")
    monkeypatch.setattr(P, "get", boom)
    md = rt.fetch(spec, "1d")
    assert md.quality.blocks_signals and any(i.code == "api_failure" for i in md.quality.issues)
    assert rt.api_status["kraken"]["ok"] is False


@pytest.mark.network
def test_live_kraken():
    md = P.KrakenProvider().get_bars(get_instruments()["BTCUSD"], "1d")
    assert len(md.df) > 100


@pytest.mark.network
def test_live_yahoo_fx_daily_built_from_hourly():
    md = P.YahooChartProvider().get_bars(get_instruments()["EURUSD"], "1d")
    assert "built from 1h" in md.meta.notes and len(md.df) > 300
