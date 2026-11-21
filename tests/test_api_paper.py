import pandas as pd
import pytest
from fastapi.testclient import TestClient

import quantai.strategies as S
from quantai.api.app import create_app
from quantai.config import get_instruments
from quantai.data.providers import CsvProvider
from quantai.models import DelayStatus
from quantai.paper import PaperTrader
from quantai.runtime import Runtime
from tests.conftest import load_fixture


class FrozenCsv(CsvProvider):
    """Serves a real fixture truncated to `upto` bars; marked HISTORICAL (it is not a live feed)."""
    name = "kraken"
    delay_status = DelayStatus.HISTORICAL

    def __init__(self, df):
        self.df = df
        self.upto = len(df) - 10

    def fetch_ohlcv(self, provider_symbol, timeframe, start=None):
        return self.df.iloc[:self.upto]


@pytest.fixture
def rt(tmp_path, monkeypatch):
    r = Runtime(data_dir=tmp_path)
    df = load_fixture("BTCUSD")
    prov = FrozenCsv(df)
    r._providers["kraken"] = prov
    # live staleness check would (correctly) fail on frozen data: validate in research mode
    import quantai.runtime as R
    orig = R.validate
    monkeypatch.setattr(R, "validate", lambda md, cfg, ac, live=True: orig(md, cfg, ac, live=False))
    return r


def test_paper_cycle_same_pipeline_and_journal(rt):
    strats = [S.registry.create(n) for n in S.registry.names()]
    pt = PaperTrader(rt, ["BTCUSD"], strats, "1d", get_instruments())
    rep = pt.run_once()
    assert rep["equity"] == rt.settings.account.starting_capital
    # second run on the same bar: no duplicate decision
    n_sig = len(rt.journal.entries("signal"))
    pt.run_once()
    assert len(rt.journal.entries("signal")) == n_sig
    # advance the frozen feed bar by bar; paper trader processes each closed bar exactly once
    prov = rt._providers["kraken"]
    for _ in range(9):
        prov.upto += 1
        pt.run_once()
    st = pt.state
    assert st.last_bar["BTCUSD"] == prov.df.index[prov.upto - 1]
    snap = rt.read_json("paper_snapshot.json")
    assert {"equity", "open_positions", "pending", "equity_curve"} <= set(snap)
    assert rt.journal.verify()[0]
    # persistence round-trip
    pt2 = PaperTrader(rt, ["BTCUSD"], strats, "1d", get_instruments())
    assert pt2.state.last_bar == st.last_bar and pt2.state.broker.cash == st.broker.cash


def test_kill_switch_cancels_pending_keeps_monitoring(rt):
    pt = PaperTrader(rt, ["BTCUSD"], [], "1d", get_instruments())
    from quantai.models import Direction, RiskDecision, SignalCandidate
    s = SignalCandidate("BTCUSD", "x", "1", Direction.LONG, pd.Timestamp.now("UTC").to_pydatetime(), 100, 90, 120, 5,
                        "range", .5, "1d")
    pt.state.broker.submit(s, RiskDecision(True, 0.01, 0.1, 0.0))
    rt.kill.activate("test")
    rep = pt.run_once()
    assert not pt.state.broker.pending and rep["approved"] == 0
    assert rt.journal.entries("cancel")[0]["payload"]["reason"] == "kill_switch"


def test_api_endpoints_and_auth(rt, monkeypatch):
    c = TestClient(create_app(rt))
    h = c.get("/health").json()
    assert h["live_trading"].startswith("not implemented") and h["journal_integrity"]["ok"]
    for path in ("/api/overview", "/api/scan", "/api/macro", "/api/performance", "/api/journal", "/api/journal/analysis",
                 "/api/alerts", "/"):
        assert c.get(path).status_code == 200, path
    monkeypatch.delenv("QUANTAI_ADMIN_TOKEN", raising=False)
    assert c.post("/api/killswitch", json={"active": True, "reason": "x"}).status_code == 403
    monkeypatch.setenv("QUANTAI_ADMIN_TOKEN", "a-long-admin-token")
    assert c.post("/api/killswitch", json={"active": True, "reason": "x"}, headers={"x-admin-token": "wrong"}).status_code == 403
    r = c.post("/api/killswitch", json={"active": True, "reason": "x"}, headers={"x-admin-token": "a-long-admin-token"})
    assert r.status_code == 200 and rt.kill.active
    assert c.get("/health").json()["kill_switch"]["active"]
