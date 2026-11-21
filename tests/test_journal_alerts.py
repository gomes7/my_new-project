import sqlite3

import pytest

from quantai.alerts import Alert, AlertManager
from quantai.config import AlertsCfg, RiskCfg
from quantai.journal import Journal, analyze
from quantai.killswitch import KillSwitch
from quantai.monitoring import data_alerts, performance_alerts
from quantai.models import QualityIssue, QualityReport, QualityStatus
import pandas as pd


def test_journal_append_only_and_chain(tmp_path):
    j = Journal(tmp_path / "j.db")
    for k in range(5):
        j.append("note", {"k": k})
    assert j.verify() == (True, None)
    con = sqlite3.connect(tmp_path / "j.db")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("UPDATE journal SET payload='{}' WHERE seq=2")
    with pytest.raises(sqlite3.DatabaseError):
        con.execute("DELETE FROM journal WHERE seq=2")
    # Simulate an attacker who drops the triggers and edits history: the hash chain detects it.
    con.execute("DROP TRIGGER journal_no_update")
    con.execute("UPDATE journal SET payload='{\"k\": 99}' WHERE seq=3")
    con.commit()
    assert j.verify() == (False, 3)


def test_journal_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        Journal(tmp_path / "j.db").append("whatever", {})


def test_journal_analysis(tmp_path):
    j = Journal(tmp_path / "j.db")
    j.append("signal", {"decision": "rejected", "reasons": ["filter:regime:range"]})
    j.append("signal", {"decision": "approved", "reasons": []})
    for r in [-1, -1, 1.5, -1, 2]:
        j.append("outcome", {"strategy": "s", "regime": "range", "r_multiple": r, "exit_time": "2026-01-05T00:00:00+00:00",
                             "exit_reason": "stop", "mfe_r": 1.2, "mae_r": 0.9})
    a = analyze(j)
    assert a["approval_rate"] == 0.5 and a["by_strategy"]["s"]["count"] == 5
    assert a["overtrading_days"] == 1 and a["poor_exits_mfe_ge_1R_closed_le_0"] == 3


def test_kill_switch(tmp_path):
    j = Journal(tmp_path / "j.db")
    ks = KillSwitch(tmp_path, j)
    assert not ks.active
    ks.activate("test")
    assert ks.active and ks.status()["reason"] == "test"
    ks.deactivate("done")
    assert not ks.active and len(j.entries("killswitch")) == 2


def test_alert_cooldown_and_fields(tmp_path):
    am = AlertManager(AlertsCfg(cooldown_seconds=3600), Journal(tmp_path / "j.db"), webhook_url="")
    a = Alert("data_feed", "stale", "no update", symbol="X", price=1.0, risk="n/a", action="blocked")
    assert am.emit(a) and not am.emit(a)                 # deduplicated
    assert am.emit(Alert("data_feed", "stale", "x", symbol="X", severity="critical"))  # critical bypasses cooldown
    t = a.text()
    for part in ("data_feed", "X", "stale", "no update", "price=1", "risk=n/a", "action=blocked"):
        assert part in t
    with pytest.raises(ValueError):
        am.emit(Alert("bogus", "s", "r"))


def test_monitoring_alerts():
    eq = pd.Series([100_000, 99_000, 95_500.0], index=pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC"))
    al = performance_alerts([], eq, 96_400, RiskCfg())
    sits = {a.situation for a in al}
    assert "daily_loss_approaching" in sits            # 0.93% of a 1% daily limit
    assert "drawdown_increasing" in sits               # 4.5% vs 8% limit (>= 50% threshold)
    calm = performance_alerts([], eq.iloc[:1], 100_000, RiskCfg())
    assert calm == []
    q = QualityReport(QualityStatus.FAILED, [QualityIssue("stale", QualityStatus.FAILED, "old")])
    assert data_alerts("X", q)[0].severity == "critical"


def test_research_proposals_are_never_activated():
    from quantai.research import suggest
    val = {"strategies": {
        "a": {"oos_trades": 6, "ci95": (-1, 0.2)},
        "b": {"oos_trades": 200, "ci95": (-0.3, -0.04), "folds": []},
        "c": {"oos_trades": 200, "ci95": (-0.1, 0.1), "param_stability": {"rr": {"unique": 3}},
              "folds": [{"test_expectancy_r": .2}, {"test_expectancy_r": .1}, {"test_expectancy_r": -.2}, {"test_expectancy_r": -.3}]}}}
    props = suggest(val, {"weak_regimes": ["breakout"]})
    kinds = {(p.strategy, p.kind) for p in props}
    assert ("a", "redesign") in kinds and ("b", "retire") in kinds and ("c", "filter") in kinds
    assert ("*", "filter") in kinds and all(p.stage == "proposed" for p in props)
