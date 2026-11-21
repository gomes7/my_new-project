import pandas as pd

import quantai.strategies as S
from quantai.analysis.frame import FeatureFrame
from quantai.backtest.engine import prepare
from quantai.models import Direction


def _signals(df, strat):
    f = FeatureFrame(prepare(df))
    out = {}
    for i in range(len(f)):
        sig, fr = strat.signal_at(f, i, "X", f.col("regime")[i], "1d", 86400)
        if sig is not None:
            out[f.index[i]] = sig
    return out


def test_registry_and_versioning():
    assert set(S.registry.names()) >= {"trend_pullback", "compression_breakout", "range_mean_reversion"}
    a, b = S.registry.create("trend_pullback"), S.registry.create("trend_pullback", rr=3.0)
    assert a.fingerprint != b.fingerprint and a.spec()["description"]


def test_unknown_param_rejected():
    import pytest
    with pytest.raises(ValueError):
        S.registry.create("trend_pullback", nonsense=1)


def test_no_lookahead_signals_identical_when_future_removed(real_daily):
    """A signal at bar i must be identical whether or not bars > i exist."""
    df = real_daily["SPX"]
    for name in S.registry.names():
        strat = S.registry.create(name)
        full = _signals(df, strat)
        cut_at = len(df) - 120
        part = _signals(df.iloc[:cut_at], strat)
        common = [t for t in part if t in full]
        assert len(common) == len(part)
        for t in common:
            assert (part[t].direction, round(part[t].stop, 8), round(part[t].target, 8)) == \
                   (full[t].direction, round(full[t].stop, 8), round(full[t].target, 8))


def test_stops_on_correct_side(real_daily):
    for df in real_daily.values():
        for name in S.registry.names():
            for sig in _signals(df, S.registry.create(name)).values():
                if sig.direction == Direction.LONG:
                    assert sig.stop < sig.entry_ref < sig.target
                else:
                    assert sig.target < sig.entry_ref < sig.stop
                assert 0 <= sig.setup_quality <= 1


def test_regime_filter_enforced(real_daily):
    strat = S.registry.create("range_mean_reversion")
    f = FeatureFrame(prepare(real_daily["GOLD"]))
    for i in range(len(f)):
        reg = f.col("regime")[i]
        sig, fr = strat.signal_at(f, i, "X", reg, "1d", 86400)
        if sig is not None and fr.passed:
            assert reg in strat.allowed_regimes_long | strat.allowed_regimes_short


def test_signal_time_is_bar_close(real_daily):
    df = real_daily["BTCUSD"]
    sigs = _signals(df, S.registry.create("compression_breakout"))
    for t, s in sigs.items():
        assert pd.Timestamp(s.signal_time) == t + pd.Timedelta(days=1)
