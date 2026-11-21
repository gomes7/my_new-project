from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from quantai.intel.calendar import CalendarEvent, YamlCalendar, blackout_reason
from quantai.intel.macro import classify_macro
from quantai.intel.news import NewsItem, asset_event_risk, parse_feed, score_events
from quantai.intel.reaction import measure_reaction
from tests.conftest import FIX, synthetic_ohlcv

NOW = datetime(2026, 11, 20, 15, tzinfo=timezone.utc)


def test_parse_and_score_feed():
    items = parse_feed((FIX / "sample_feed.xml").read_text(), "testwire", 0.8)
    assert len(items) == 4
    ev = score_events(items, {"EURUSD", "SPX", "WTI", "GOLD"}, now=NOW)
    titles = {e.title: e for e in ev}
    assert "Local bakery wins award" not in titles          # irrelevant -> dropped
    fed = titles["Fed holds rates steady, signals patience on inflation"]
    assert "central_bank" in fed.categories and 0 <= fed.score <= 100
    rumor = titles["Chipmaker reportedly considering acquisition of rival"]
    assert rumor.event_type == "rumor" and not rumor.confirmed
    oil = titles["Oil surges after unexpected OPEC supply cut"]
    assert oil.volatility_impact == "high" and "WTI" in oil.assets
    assert all(e.priced_in == "unknown" for e in ev)
    assert asset_event_risk(ev, "WTI", min_score=0) > 0


def test_confirmation_requires_distinct_sources():
    a = NewsItem("ECB raises rates by 25bp in surprise move", "wireA", "", NOW, credibility=0.9)
    b = NewsItem("ECB raises rates 25bp in surprise move", "wireB", "", NOW, credibility=0.9)
    c = NewsItem("ECB raises rates by 25bp in surprise move", "wireA", "", NOW, credibility=0.9)
    ev = {e.source: e for e in score_events([a, b], {"EURUSD"}, now=NOW)}
    assert ev["wireA"].confirmed and ev["wireA"].confirmation_count == 2
    assert not score_events([a, c], {"EURUSD"}, now=NOW)[0].confirmed


def test_feed_with_entities_refused():
    with pytest.raises(ValueError):
        parse_feed('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><rss></rss>', "x")


def _monthly(vals, end="2026-10-01"):
    return pd.Series(vals, index=pd.date_range(end=end, periods=len(vals), freq="MS", tz="UTC"))


def _daily(v, n=30):
    return pd.Series([v] * n, index=pd.date_range(end="2026-11-19", periods=n, freq="1D", tz="UTC"))


def test_macro_recession_via_sahm_rule():
    un = _monthly([3.5] * 12 + [3.8, 4.1, 4.4])
    cpi = _monthly(list(np.linspace(300, 306, 15)))
    m = classify_macro({"UNRATE": un, "CPIAUCSL": cpi, "VIXCLS": _daily(30), "BAMLH0A0HYM2": _daily(6)})
    assert m.growth == "recession" and "recession" in m.labels and m.risk == "risk_off"
    assert any(e.series == "UNRATE" and "Sahm" in e.note for e in m.evidence)
    assert "WALCL" in m.missing_series


def test_macro_disinflationary_expansion_and_risk_on():
    un = _monthly([4.0] * 15)
    cpi = _monthly([300 * (1.03 ** (k / 12)) for k in range(12)] + [300 * 1.03 + 0.1 * k for k in range(1, 4)])
    m = classify_macro({"UNRATE": un, "CPIAUCSL": cpi, "VIXCLS": _daily(13.5), "BAMLH0A0HYM2": _daily(3.0)})
    assert m.growth == "expansion" and m.inflation.endswith("falling")
    assert "disinflationary_expansion" in m.labels and m.risk == "risk_on"
    assert m.bias("index") > 0 and m.bias("forex") == 0


def test_calendar_blackout(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("events:\n  - {title: CPI, time: '2026-11-20T13:30:00Z', currency: USD, impact: high}\n"
                 "  - {title: BoJ, time: '2026-11-20T03:00:00Z', currency: JPY, impact: high}\n")
    ev = YamlCalendar(p).events()
    t = datetime(2026, 11, 20, 13, 15, tzinfo=timezone.utc)
    assert blackout_reason(ev, "EURUSD", t, 30).startswith("CPI")
    assert blackout_reason(ev, "USDJPY", datetime(2026, 11, 20, 3, 10, tzinfo=timezone.utc), 30).startswith("BoJ")
    assert blackout_reason(ev, "DAX", datetime(2026, 11, 20, 3, 10, tzinfo=timezone.utc), 30) is None
    assert blackout_reason(ev, "EURUSD", t + timedelta(hours=2), 30) is None


def test_reaction_can_contradict_headline():
    df = synthetic_ohlcv(100, vol=0.002, freq="1h", start="2026-11-16")
    t = df.index[80]
    df.loc[df.index >= t, ["open", "high", "low", "close"]] *= 0.97   # sharp drop after "bullish" news
    r = measure_reaction(df, "X", t.to_pydatetime(), hypothesis_sign=+1)
    assert r.significant and r.reaction == "down" and r.agrees_with_hypothesis is False
