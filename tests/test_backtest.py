import numpy as np
import pandas as pd
import pytest

import quantai.strategies as S
from quantai.backtest import metrics
from quantai.backtest.engine import run_backtest
from quantai.backtest.robustness import monte_carlo, promotion_gate, walk_forward
from quantai.config import PromotionCfg
from tests.conftest import synthetic_ohlcv


def strats():
    return [S.registry.create(n) for n in S.registry.names()]


def test_deterministic_on_real_fixture(real_daily, instruments, settings):
    a = run_backtest(real_daily, "1d", strats(), instruments, settings)
    b = run_backtest(real_daily, "1d", strats(), instruments, settings)
    assert len(a.trades) == len(b.trades) > 0
    assert a.equity.iloc[-1] == b.equity.iloc[-1]


def test_accounting_identity(real_daily, instruments, settings):
    """Final equity == start + closed P&L + open partial P&L - open entry/carry costs + unrealised."""
    r = run_backtest(real_daily, "1d", strats(), instruments, settings)
    expected = (settings.account.starting_capital + sum(t.pnl for t in r.trades)
                + sum(p.realized_pnl - p.entry_costs for p in r.open_positions) + r.final_unrealized)
    assert r.equity.iloc[-1] == pytest.approx(expected, abs=1e-6)
    assert all(t.fees >= 0 for t in r.trades)


def test_every_trade_respects_risk_cap(real_daily, instruments, settings):
    r = run_backtest(real_daily, "1d", strats(), instruments, settings)
    for t in r.trades:
        assert t.risk_amount <= settings.account.starting_capital * 1.2 * settings.risk.max_risk_per_trade_pct / 100
    approved = [s for s in r.signals if s.decision == "approved"]
    assert all(s.quantity > 0 for s in approved)
    assert all(s.reasons for s in r.signals if s.decision == "rejected")


def test_trade_window_respected(real_daily, instruments, settings):
    start = real_daily["SPX"].index[450]
    r = run_backtest(real_daily, "1d", strats(), instruments, settings, trade_start=start)
    assert all(pd.Timestamp(t.entry_time) >= start for t in r.trades)


def test_failed_data_symbol_skipped(instruments, settings):
    df = synthetic_ohlcv(400)
    df.iloc[10:30, df.columns.get_loc("high")] = 0.0
    r = run_backtest({"BTCUSD": df}, "1d", strats(), instruments, settings)
    assert "BTCUSD" in r.skipped_symbols and r.trades == []


def test_metrics_basic():
    eq = pd.Series([100, 110, 99, 120.0], index=pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC"))
    dd_pct, dd_amt = metrics.max_drawdown(eq)
    assert dd_amt == 11 and abs(dd_pct - 10) < 1e-9


def test_monte_carlo_sanity():
    mc = monte_carlo([1.0] * 50, 1.0, runs=200)
    assert mc.max_dd_pct_p95 == 0 and mc.final_return_pct_p50 > 60
    mc2 = monte_carlo([-1.0, 1.0] * 50, 1.0, runs=500)
    assert mc2.max_dd_pct_p95 > 0 and 0 <= mc2.prob_dd_exceeds["10"] <= 1


def test_walk_forward_is_out_of_sample(real_daily, instruments, settings):
    wf = walk_forward(real_daily, "1d", "trend_pullback", {"rr": [1.5, 2.0]}, instruments, settings, n_folds=2,
                      min_train_trades=1)
    assert len(wf.folds) == 2
    for f, nxt in zip(wf.folds, wf.folds[1:]):
        assert f.test[1] == nxt.test[0]
    first_test_start = pd.Timestamp(wf.folds[0].test[0])
    assert all(pd.Timestamp(t.entry_time) >= first_test_start for t in wf.oos_trades)


def test_promotion_gate_requires_significance():
    class WF:  # minimal stand-in
        oos_summary = {"trades": 200, "profit_factor": 1.3, "expectancy_r": 0.1, "expectancy_r_ci95": (-0.02, 0.22)}
    mc = monte_carlo(list(np.random.default_rng(1).normal(0.1, 1, 200)), 0.25, runs=200)
    g = promotion_gate(WF(), mc, None, PromotionCfg())
    assert not g.promote and "bootstrap_ci95_lower>0" in g.failed
    WF.oos_summary = {**WF.oos_summary, "expectancy_r_ci95": (0.01, 0.2)}
    assert promotion_gate(WF(), mc, None, PromotionCfg()).promote


def test_score_calibration_monotonic_flag():
    from quantai.ranking import calibrate, is_monotonic
    cal = calibrate([65] * 30 + [75] * 30 + [85] * 30, [-0.2] * 30 + [0.0] * 30 + [0.3] * 30)
    assert is_monotonic(cal) is True
    cal = calibrate([65] * 30 + [85] * 30, [0.3] * 30 + [-0.2] * 30)
    assert is_monotonic(cal) is False
    assert is_monotonic(calibrate([85] * 5, [1.0] * 5)) is None
