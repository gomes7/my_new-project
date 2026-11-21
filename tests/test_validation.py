import pandas as pd

from quantai.config import DataCfg
from quantai.data.validation import clean_bars, compare_providers, validate
from quantai.models import DataMeta, DelayStatus, MarketData, QualityStatus, utcnow
from tests.conftest import synthetic_ohlcv, utc


def md(df, tf="1d", src="t"):
    return MarketData("X", df, DataMeta(src, "X", tf, utcnow(), DelayStatus.HISTORICAL))


def codes(rep):
    return {i.code for i in rep.issues}


def test_clean_synthetic_passes():
    rep = validate(md(synthetic_ohlcv(300)), DataCfg(), "crypto", live=False)
    assert rep.status in (QualityStatus.OK, QualityStatus.DEGRADED) and not rep.blocks_signals


def test_empty_fails():
    rep = validate(md(pd.DataFrame()), DataCfg())
    assert rep.blocks_signals and "empty" in codes(rep)


def test_ohlc_inconsistency_detected():
    df = synthetic_ohlcv(200)
    df.iloc[10:20, df.columns.get_loc("high")] = df["low"].iloc[10:20] - 1
    rep = validate(md(df), DataCfg(), live=False)
    assert "ohlc_inconsistent" in codes(rep) and rep.blocks_signals


def test_duplicates_and_nans():
    df = synthetic_ohlcv(200)
    df = pd.concat([df, df.iloc[[5]]]).sort_index()
    df.iloc[50, 0] = float("nan")
    rep = validate(md(df), DataCfg(), live=False)
    assert {"duplicates", "missing_values"} <= codes(rep)


def test_non_positive_price_blocks():
    df = synthetic_ohlcv(200)
    df.iloc[100, df.columns.get_loc("close")] = -1.0
    df.iloc[100, df.columns.get_loc("low")] = -2.0
    assert validate(md(df), DataCfg(), live=False).blocks_signals


def test_stale_data_blocks_live_but_not_research():
    df = synthetic_ohlcv(300, start="2020-01-01")
    assert "stale" in codes(validate(md(df), DataCfg(), "crypto", live=True))
    assert "stale" not in codes(validate(md(df), DataCfg(), "crypto", live=False))


def test_future_timestamps_block():
    df = synthetic_ohlcv(50, start=(pd.Timestamp.now("UTC") + pd.Timedelta(days=5)).strftime("%Y-%m-%d"))
    assert "future_timestamps" in codes(validate(md(df), DataCfg(), live=False))


def test_reverting_spike_flagged_as_bad_tick_persistent_move_is_informational():
    df = synthetic_ohlcv(300, vol=0.005)
    i = 200
    df.iloc[i, df.columns.get_loc("close")] *= 1.3
    df.iloc[i, df.columns.get_loc("high")] = df["close"].iloc[i]
    rep = validate(md(df), DataCfg(), "crypto", live=False)
    assert "suspected_bad_tick" in codes(rep)
    df2 = synthetic_ohlcv(300, vol=0.005)
    df2.iloc[200:, :4] *= 1.3  # level shift that persists (a real move)
    rep2 = validate(md(df2), DataCfg(), "crypto", live=False)
    assert "suspected_bad_tick" not in codes(rep2)
    assert any(i.code == "large_move" and i.severity == QualityStatus.OK for i in rep2.issues)


def test_provider_discrepancy():
    a = synthetic_ohlcv(100)
    b = a.copy()
    b["close"] *= 1.02
    rep = compare_providers(md(a, src="a"), md(b, src="b"), 0.5)
    assert rep.blocks_signals and "provider_discrepancy" in codes(rep)
    assert compare_providers(md(a), md(a.copy()), 0.5).status == QualityStatus.OK


def test_clean_bars_never_fabricates():
    df = synthetic_ohlcv(100)
    df.iloc[10, 0] = float("nan")
    out = clean_bars(df)
    assert len(out) == 99 and out.index.isin(df.index).all()


def test_tz_naive_index_fails():
    df = synthetic_ohlcv(50)
    df.index = df.index.tz_localize(None)
    assert validate(md(df), DataCfg(), live=False).blocks_signals


def test_real_fixture_quality(real_daily):
    for sym, df in real_daily.items():
        ac = {"BTCUSD": "crypto", "EURUSD": "forex"}.get(sym, "index")
        assert not validate(md(df), DataCfg(), ac, live=False).blocks_signals, sym


def test_utc_helper():
    assert utc(2026, 1, 1).tzinfo is not None
