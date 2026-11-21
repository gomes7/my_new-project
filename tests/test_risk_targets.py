from datetime import datetime, timedelta, timezone

import pytest

from quantai.config import RiskCfg
from quantai.models import Direction, Position, QualityReport, QualityStatus, SignalCandidate
from quantai.risk.engine import AccountState, RiskContext, RiskEngine
from quantai.targets import NO_TRADE, daily_target
from tests.conftest import TODAY

T = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)
OK = QualityReport(QualityStatus.OK)


def sig(entry=100.0, stop=98.0, atr=2.0, d=Direction.LONG, sym="TEST"):
    return SignalCandidate(sym, "s", "1", d, T, entry, stop, entry + 2 * (entry - stop) * d.value, atr, "range", 0.5, "1d")


def acct(eq=100_000.0, **kw):
    a = AccountState(eq, eq, eq, eq, TODAY)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def ctx(**kw):
    return RiskContext(data_quality=OK, **kw)


def test_position_size_formula(usd_spec):
    eng = RiskEngine(RiskCfg())
    d = eng.evaluate(sig(), usd_spec, acct(), ctx())
    assert d.approved
    # 0.25% of 100k = 250; stop distance 2.0 (zero costs in this spec, zero slippage fraction aside)
    per_unit = 2.0 + eng.estimated_cost_per_unit(usd_spec, sig())
    assert d.quantity == pytest.approx(int(250 / per_unit / 0.001) * 0.001)
    assert d.risk_amount <= 250 + 1e-6 and d.risk_pct <= 0.25 + 1e-9


def test_costs_reduce_size(usd_spec):
    eng = RiskEngine(RiskCfg())
    cheap = eng.evaluate(sig(), usd_spec, acct(), ctx()).quantity
    costly = eng.evaluate(sig(), usd_spec.model_copy(update={"commission_bps": 40, "spread": 0.2}), acct(), ctx()).quantity
    assert costly < cheap


@pytest.mark.parametrize("acct_kw,reason", [
    ({"start_of_day_equity": 101_100.0}, "daily_loss_limit_reached"),
    ({"start_of_week_equity": 103_000.0}, "weekly_loss_limit_reached"),
    ({"peak_equity": 110_000.0}, "max_drawdown_reached"),
    ({"consecutive_losses": 4}, "max_consecutive_losses"),
])
def test_account_limits_block(usd_spec, acct_kw, reason):
    d = RiskEngine(RiskCfg()).evaluate(sig(), usd_spec, acct(**acct_kw), ctx())
    assert not d.approved and reason in d.reasons and d.quantity == 0


def test_cooldown_after_loss_streak(usd_spec):
    cfg = RiskCfg()
    a = acct()
    for _ in range(cfg.max_consecutive_losses):
        a.register_closed_trade(-10, T, cfg)
    assert a.consecutive_losses == 0 and a.lockout_until == T + timedelta(hours=cfg.consecutive_loss_cooldown_hours)
    d = RiskEngine(cfg).evaluate(sig(), usd_spec, a, ctx(), now=T + timedelta(hours=1))
    assert not d.approved and any(r.startswith("consecutive_loss_cooldown") for r in d.reasons)
    assert RiskEngine(cfg).evaluate(sig(), usd_spec, a, ctx(), now=T + timedelta(hours=100)).approved


@pytest.mark.parametrize("c,reason", [
    ({"kill_switch": True}, "kill_switch_active"),
    ({"event_blackout": "CPI"}, "event_blackout:CPI"),
    ({"data_quality": QualityReport(QualityStatus.FAILED)}, "data_quality_failed:"),
    ({"data_quality": None}, "data_quality_unknown"),
    ({"current_spread": 1.0}, "abnormal_spread"),
])
def test_context_blocks(usd_spec, c, reason):
    base = {"data_quality": OK}
    base.update(c)
    d = RiskEngine(RiskCfg()).evaluate(sig(), usd_spec, acct(), RiskContext(**base))
    assert not d.approved and reason in d.reasons


def test_stop_sanity(usd_spec):
    eng = RiskEngine(RiskCfg())
    assert any(r.startswith("stop_too_tight") for r in eng.evaluate(sig(stop=99.9), usd_spec, acct(), ctx()).reasons)
    assert any(r.startswith("stop_too_wide") for r in eng.evaluate(sig(stop=80), usd_spec, acct(), ctx()).reasons)


def test_missing_fx_conversion_blocks(usd_spec):
    jpy = usd_spec.model_copy(update={"quote_currency": "JPY"})
    d = RiskEngine(RiskCfg()).evaluate(sig(), jpy, acct(), ctx())
    assert not d.approved and "no_fx_conversion_JPY" in d.reasons
    assert RiskEngine(RiskCfg()).evaluate(sig(), jpy, acct(), ctx(fx_to_account=0.0066)).approved


def test_leverage_cap(usd_spec):
    # very tight but valid stop relative to ATR -> huge size -> leverage breach
    d = RiskEngine(RiskCfg(min_stop_atr=0.0)).evaluate(sig(stop=99.99, atr=0.02), usd_spec, acct(), ctx())
    assert any(r.startswith("leverage_exceeded") for r in d.reasons)


def _pos(sym, d=Direction.LONG, risk_per_unit=2.0, qty=100.0):
    return Position("p", sym, "s", d, qty, 100.0, T, 100 - risk_per_unit * d.value, 100 - risk_per_unit * d.value,
                    110, risk_per_unit * qty, 1.0, "range", "x", 10)


def test_correlated_exposure_and_open_risk(usd_spec, instruments):
    eng = RiskEngine(RiskCfg())
    other = usd_spec.model_copy(update={"symbol": "OTHER"})
    a = acct(open_positions=[_pos("OTHER", qty=300)])  # 600 open risk in same group
    d = eng.evaluate(sig(), usd_spec, a, ctx(instruments={"OTHER": other, "TEST": usd_spec}))
    assert "correlated_exposure_exceeded" in d.reasons
    # opposite direction in the same group is not "same-way" exposure
    a2 = acct(open_positions=[_pos("OTHER", d=Direction.SHORT, qty=300)])
    assert "correlated_exposure_exceeded" not in eng.evaluate(sig(), usd_spec, a2, ctx(instruments={"OTHER": other})).reasons
    # measured correlation also counts
    a3 = acct(open_positions=[_pos("ZZZ", qty=300)])
    d3 = eng.evaluate(sig(), usd_spec, a3, ctx(correlations={("TEST", "ZZZ"): 0.9}))
    assert "correlated_exposure_exceeded" in d3.reasons


def test_remaining_daily_budget(usd_spec):
    a = acct(equity=99_200.0, start_of_day_equity=100_000.0)   # -0.8% today, budget left 200 < 248 risk
    d = RiskEngine(RiskCfg()).evaluate(sig(), usd_spec, a, ctx())
    assert "exceeds_remaining_daily_risk_budget" in d.reasons


def test_duplicate_symbol_and_max_positions(usd_spec):
    eng = RiskEngine(RiskCfg(max_risk_per_group_pct=10, max_total_open_risk_pct=10))
    a = acct(open_positions=[_pos("TEST", qty=1)])
    assert "position_already_open_on_symbol" in eng.evaluate(sig(), usd_spec, a, ctx()).reasons
    a = acct(open_positions=[_pos(f"S{i}", qty=1) for i in range(4)])
    assert "max_open_positions" in eng.evaluate(sig(), usd_spec, a, ctx()).reasons


def test_target_engine_never_forces_trades():
    r = daily_target(100_000, 100_000, 0.3, 0.25, 1.0, 3, expectancy_r=0.1, expected_trades_per_day=0.5, valid_setups_now=0)
    assert r.message == NO_TRADE and r.status == "no_trade"
    r = daily_target(100_000, 100_000, 0.3, 0.25, 1.0, 3, expectancy_r=0.1, expected_trades_per_day=0.5, valid_setups_now=1)
    assert not r.compatible and "NOT increased" in r.message
    assert r.risk_per_trade_amount == 250.0          # unchanged by the target
    assert r.risk_needed_per_trade_pct > 0.25        # reported, never applied
    r = daily_target(100_000, 100_000, 0.3, 0.25, 1.0, 3, None, None, 2)
    assert r.status == "unvalidated"
