from datetime import datetime, timezone

import pandas as pd
import pytest

from quantai.analysis.frame import Row
from quantai.config import ExecutionCfg
from quantai.execution.fx import FxConverter
from quantai.execution.sim import SimBroker
from quantai.models import Direction, RiskDecision, SignalCandidate

T0 = pd.Timestamp("2026-01-05", tz="UTC")
DAY = 86400


def bar(o, h, l, c, t):
    r = Row(open=o, high=h, low=l, close=c, volume=1.0)
    r.name = t
    return r


def mk(spec, cfg=None, cash=100_000.0):
    return SimBroker({"TEST": spec}, cfg or ExecutionCfg(slippage_atr_fraction=0.0, partial_exit_fraction=0.0), cash,
                     lambda c, t: 1.0)


def submit(br, d=Direction.LONG, entry=100.0, stop=98.0, rr=2.0, qty=10.0, max_bars=20):
    s = SignalCandidate("TEST", "s", "1", d, T0.to_pydatetime(), entry, stop, entry + rr * (entry - stop) * d.value * 1,
                        2.0, "range", 0.5, "1d", max_bars=max_bars)
    if d == Direction.SHORT:
        s.target = entry - rr * (stop - entry)
    br.submit(s, RiskDecision(True, qty, 20.0, 0.02))
    return s


def test_rejects_unapproved(usd_spec):
    br = mk(usd_spec)
    s = SignalCandidate("TEST", "s", "1", Direction.LONG, T0, 100, 98, 104, 2, "r", .5, "1d")
    with pytest.raises(PermissionError):
        br.submit(s, RiskDecision(False, 0, 0, 0))


def test_entry_next_open_with_costs_and_target(usd_spec):
    spec = usd_spec.model_copy(update={"spread": 0.2, "commission_bps": 10})
    br = mk(spec, ExecutionCfg(slippage_atr_fraction=0.1, partial_exit_fraction=0.0))
    submit(br)
    ev = br.on_bar("TEST", T0, bar(100.5, 101, 100, 100.8, T0), atr=2.0, bar_seconds=DAY)
    fill = 100.5 + 0.1 + 0.2       # open + half spread + slippage (0.1 x ATR 2)
    assert ev[0].kind == "filled" and ev[0].detail["price"] == pytest.approx(fill)
    p = br.positions["TEST"]
    assert p.target == pytest.approx(fill + 2 * (fill - 98))
    t1 = T0 + pd.Timedelta(days=1)
    ev = br.on_bar("TEST", t1, bar(101, p.target + 1, 100.9, 105, t1), atr=2.0, bar_seconds=DAY)
    assert ev[-1].detail["reason"] == "target"
    tr = br.closed[0]
    exit_px = p.target - 0.1
    gross = (exit_px - fill) * 10
    fees = fill * 10 * 10 / 1e4 + exit_px * 10 * 10 / 1e4
    assert tr.pnl == pytest.approx(gross - fees)
    assert br.cash == pytest.approx(100_000 + tr.pnl)


def test_stop_assumed_first_when_both_hit(usd_spec):
    br = mk(usd_spec)
    submit(br)
    br.on_bar("TEST", T0, bar(100, 100.5, 99.5, 100, T0), 2.0, DAY)
    t1 = T0 + pd.Timedelta(days=1)
    br.on_bar("TEST", t1, bar(100, 110, 97, 100, t1), 2.0, DAY)   # both 104 target and 98 stop inside bar
    assert br.closed[0].exit_reason == "stop" and br.closed[0].r_multiple == pytest.approx(-1.0)


def test_gap_through_stop_fills_at_open(usd_spec):
    br = mk(usd_spec)
    submit(br)
    br.on_bar("TEST", T0, bar(100, 100.5, 99.5, 100, T0), 2.0, DAY)
    t1 = T0 + pd.Timedelta(days=1)
    br.on_bar("TEST", t1, bar(95, 96, 94, 95, t1), 2.0, DAY)
    tr = br.closed[0]
    assert tr.exit_reason == "stop_gap" and tr.exit_price == 95 and tr.r_multiple == pytest.approx(-2.5)


def test_entry_cancelled_if_open_beyond_stop(usd_spec):
    br = mk(usd_spec)
    submit(br)
    ev = br.on_bar("TEST", T0, bar(97, 98, 96, 97, T0), 2.0, DAY)
    assert ev[0].kind == "cancelled" and not br.positions


def test_entry_bar_target_not_credited(usd_spec):
    br = mk(usd_spec)
    submit(br)
    br.on_bar("TEST", T0, bar(100, 120, 99.5, 101, T0), 2.0, DAY)   # target inside the entry bar: ignored
    assert "TEST" in br.positions


def test_partial_exit_and_breakeven(usd_spec):
    br = mk(usd_spec, ExecutionCfg(slippage_atr_fraction=0.0, partial_exit_r=1.0, partial_exit_fraction=0.5))
    submit(br, qty=10)
    br.on_bar("TEST", T0, bar(100, 100.5, 99.5, 100, T0), 2.0, DAY)
    t1 = T0 + pd.Timedelta(days=1)
    ev = br.on_bar("TEST", t1, bar(100, 102.5, 99.9, 102, t1), 2.0, DAY)
    assert ev[0].kind == "partial" and br.positions["TEST"].quantity == pytest.approx(5)
    assert br.positions["TEST"].stop == 100
    t2 = t1 + pd.Timedelta(days=1)
    br.on_bar("TEST", t2, bar(101, 101, 99, 99.5, t2), 2.0, DAY)
    tr = br.closed[0]
    assert tr.exit_reason == "breakeven_stop" and tr.pnl == pytest.approx(5 * 2.0)   # partial +2 x 5 units, rest flat


def test_time_stop_and_short(usd_spec):
    br = mk(usd_spec)
    submit(br, d=Direction.SHORT, entry=100, stop=102, max_bars=2)
    t = T0
    br.on_bar("TEST", t, bar(100, 100.5, 99.5, 100, t), 2.0, DAY)
    for k in range(1, 3):
        t = T0 + pd.Timedelta(days=k)
        br.on_bar("TEST", t, bar(99.5, 100, 99, 99.5, t), 2.0, DAY)
    tr = br.closed[0]
    assert tr.exit_reason == "time_stop" and tr.pnl == pytest.approx(0.5 * 10)


def test_holding_cost_charged_daily(usd_spec):
    spec = usd_spec.model_copy(update={"holding_cost_bps_per_day": 10})
    br = mk(spec)
    submit(br)
    br.on_bar("TEST", T0, bar(100, 100.5, 99.5, 100, T0), 2.0, DAY)
    assert br.positions["TEST"].entry_costs == pytest.approx(100 * 10 * 10 / 1e4)


def test_fx_converter_uses_only_closed_bars():
    fx = FxConverter("USD")
    idx = pd.date_range("2026-01-01", periods=3, freq="1D", tz="UTC")
    fx.add_pair("USDJPY", pd.DataFrame({"close": [150.0, 151.0, 152.0]}, index=idx), "1d")
    assert fx.rate("JPY", idx[0]) is None                      # first bar not closed yet
    assert fx.rate("JPY", idx[1]) == pytest.approx(1 / 150)
    assert fx.rate("EUR", idx[2]) is None and fx.rate("USD", idx[0]) == 1.0


def test_trailing_stop_ratchets_and_exits(usd_spec):
    br = mk(usd_spec)
    s = submit(br, entry=100, stop=94, rr=50, qty=10)
    s.trail_atr, s.allow_partial = 2.0, False
    br.on_bar("TEST", T0, bar(100, 101, 99, 100.5, T0), atr=1.0, bar_seconds=DAY)
    p = br.positions["TEST"]
    assert p.stop == 99.0                      # max(94, 101 - 2*1) after the entry bar
    t1 = T0 + pd.Timedelta(days=1)
    br.on_bar("TEST", t1, bar(101, 106, 100.5, 105.5, t1), atr=1.0, bar_seconds=DAY)
    assert p.stop == 104.0                     # ratcheted to 106 - 2
    t2 = t1 + pd.Timedelta(days=1)
    br.on_bar("TEST", t2, bar(105, 105.2, 103, 103.5, t2), atr=1.0, bar_seconds=DAY)
    tr = br.closed[0]
    assert tr.exit_reason == "trailing_stop" and tr.exit_price == 104.0 and tr.pnl == pytest.approx(40)


def test_trailing_stop_never_loosens(usd_spec):
    br = mk(usd_spec)
    s = submit(br, d=Direction.SHORT, entry=100, stop=106, rr=50, qty=1)
    s.trail_atr = 2.0
    br.on_bar("TEST", T0, bar(100, 101, 97, 98, T0), atr=1.0, bar_seconds=DAY)
    p = br.positions["TEST"]
    assert p.stop == 99.0                      # min(106, 97 + 2)
    t1 = T0 + pd.Timedelta(days=1)
    br.on_bar("TEST", t1, bar(98, 98.5, 97.5, 98, t1), atr=5.0, bar_seconds=DAY)   # wider ATR must not loosen
    assert p.stop == 99.0
