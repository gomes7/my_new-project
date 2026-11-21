import numpy as np
import pandas as pd

from quantai.analysis import indicators as I
from quantai.analysis.regime import Regime, classify, classify_row, evaluate
from quantai.analysis.technical import analyze
from quantai.backtest.engine import prepare
from tests.conftest import synthetic_ohlcv


def test_features_are_causal():
    """Appending future bars must not change any feature value at earlier bars."""
    df = synthetic_ohlcv(500, seed=3)
    full = I.add_features(df)
    cut = I.add_features(df.iloc[:400])
    cols = [c for c in cut.columns if c not in ("open", "high", "low", "close", "volume")]
    pd.testing.assert_frame_equal(full.iloc[:400][cols], cut[cols], check_exact=False, rtol=1e-9)


def test_rsi_bounds_and_atr_positive():
    f = I.add_features(synthetic_ohlcv(300))
    r = f["rsi"].dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert (f["atr"].dropna() > 0).all()


def test_true_range_known_values():
    df = pd.DataFrame({"open": [10, 11], "high": [12, 13], "low": [9, 10.5], "close": [11, 12]}, dtype=float)
    tr = I.true_range(df)
    assert tr.iloc[0] == 3 and tr.iloc[1] == 2.5


def test_donchian_excludes_current_bar():
    df = synthetic_ohlcv(50)
    dc = I.donchian(df, 20)
    assert dc["dc_high"].iloc[30] == df["high"].iloc[10:30].max()


def test_vectorised_regime_equals_rowwise(real_daily):
    for df in real_daily.values():
        f = I.add_features(df)
        vec = classify(f)
        row = [classify_row(r)[0].value for _, r in f.iterrows()]
        assert list(vec["regime"]) == row


def test_trending_series_classified_as_uptrend():
    df = synthetic_ohlcv(600, drift=0.004, vol=0.006, seed=5)
    reg = classify(I.add_features(df))["regime"].iloc[-150:]
    assert reg.isin([Regime.STRONG_UP.value, Regime.WEAK_UP.value, Regime.BREAKOUT.value]).mean() > 0.6


def test_regime_evaluation_table(real_daily):
    f = I.add_features(real_daily["SPX"])
    ev = evaluate(f, classify(f), 10)
    assert {"n", "mean_fwd_ret_pct", "hit_up", "t_stat"} <= set(ev.columns)
    assert ev["n"].sum() > 300


def test_technical_report_mtf(real_daily):
    df = real_daily["BTCUSD"]
    rep = analyze("BTCUSD", {"1d": prepare(df), "4h": prepare(df.iloc[-300:])})
    assert -1 <= rep.alignment <= 1 and rep.views[0].timeframe == "1d"
    assert "prev_day_high" in rep.views[0].levels
