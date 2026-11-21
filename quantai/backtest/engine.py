"""Event-driven multi-instrument backtester.

Per bar t (all instruments, chronological):
  1. SimBroker.on_bar: fill pending entries at t's open, manage stops/targets inside bar t.
  2. Account bookkeeping (day / week roll, equity, drawdown, consecutive losses).
  3. At bar t's CLOSE: strategies -> selection -> risk engine -> pending order for bar t+1.
Features are causal, so computing them over the full series does not leak future data;
`trade_start`/`trade_end` restrict when new positions may be opened (train/test windows).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from quantai.analysis.frame import FeatureFrame
from quantai.analysis.indicators import add_features
from quantai.analysis.regime import classify
from quantai.config import InstrumentSpec, Settings
from quantai.data.validation import validate
from quantai.execution.fx import FxConverter
from quantai.execution.sim import SimBroker
from quantai.models import TF_SECONDS, ClosedTrade, DataMeta, DelayStatus, MarketData, QualityReport, utcnow
from quantai.decision import SignalLog, decide
from quantai.risk.engine import AccountState, RiskEngine
from quantai.selection import StrategySelector
from quantai.strategies.base import Strategy


@dataclass
class BacktestResult:
    trades: list[ClosedTrade]
    equity: pd.Series
    signals: list[SignalLog]
    quality: dict[str, dict]
    skipped_symbols: dict[str, str]
    config: dict = field(default_factory=dict)
    open_positions: list = field(default_factory=list)
    final_unrealized: float = 0.0


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    f = add_features(df)
    return f.join(classify(f))


def run_backtest(data: dict[str, pd.DataFrame], timeframe: str, strategies: list[Strategy],
                 instruments: dict[str, InstrumentSpec], settings: Settings,
                 trade_start: pd.Timestamp | None = None, trade_end: pd.Timestamp | None = None,
                 selector: StrategySelector | None = None, min_score: float | None = None,
                 prepared: dict[str, pd.DataFrame] | None = None, fx_data: dict[str, pd.DataFrame] | None = None) -> BacktestResult:
    bar_s = TF_SECONDS[timeframe]
    risk = RiskEngine(settings.risk, settings.execution)
    selector = selector or StrategySelector()
    acct_ccy = settings.account.currency

    fx = FxConverter(acct_ccy)
    for pair, df in {**data, **(fx_data or {})}.items():
        if len(pair) == 6 and pair.isalpha():
            fx.add_pair(pair, df, timeframe)

    feats: dict[str, FeatureFrame] = {}
    quality: dict[str, QualityReport] = {}
    skipped: dict[str, str] = {}
    for sym, df in data.items():
        spec = instruments[sym]
        md = MarketData(sym, df, DataMeta("backtest", sym, timeframe, utcnow(), DelayStatus.HISTORICAL))
        q = validate(md, settings.data, spec.asset_class, live=False)
        quality[sym] = q
        if q.blocks_signals:
            skipped[sym] = "data_quality_failed: " + "; ".join(f"{i.code}({i.count})" for i in q.issues)
            continue
        feats[sym] = FeatureFrame(prepared[sym] if prepared and sym in prepared else prepare(df))

    broker = SimBroker(instruments, settings.execution, settings.account.starting_capital, fx.rate)
    start_cap = settings.account.starting_capital
    timeline = sorted(set().union(*[f.index for f in feats.values()])) if feats else []
    pos_index = {s: {t: i for i, t in enumerate(f.index)} for s, f in feats.items()}

    acct = AccountState(start_cap, start_cap, start_cap, start_cap, day=None)  # type: ignore[arg-type]
    eq_t, eq_v = [], []
    signals: list[SignalLog] = []
    n_closed = 0

    for t in timeline:
        t_close = t + pd.Timedelta(seconds=bar_s)
        acct.roll(t_close.to_pydatetime(), broker.equity())

        for sym, f in feats.items():
            i = pos_index[sym].get(t)
            if i is None:
                continue
            broker.on_bar(sym, t, f.row(i), float(f.col("atr")[i - 1]) if i > 0 else float("nan"), bar_s)

        # closed-trade bookkeeping
        for tr in broker.closed[n_closed:]:
            acct.register_closed_trade(tr.pnl, pd.Timestamp(tr.exit_time).to_pydatetime(), settings.risk)
        n_closed = len(broker.closed)
        eqv = broker.equity()
        acct.equity = eqv
        acct.peak_equity = max(acct.peak_equity, eqv)
        acct.open_positions = broker.exposure_positions(t_close)
        eq_t.append(t_close); eq_v.append(eqv)

        if (trade_start is not None and t_close < trade_start) or (trade_end is not None and t_close > trade_end):
            continue
        for sym, f in feats.items():
            i = pos_index[sym].get(t)
            if i is None or sym in broker.positions or sym in broker.pending:
                continue
            spec = instruments[sym]
            bd = decide(sym, f, i, t_close.to_pydatetime(), timeframe, bar_s, strategies, selector, risk, acct, spec,
                        quality[sym], fx.rate(spec.quote_currency, t_close), settings, instruments, min_score)
            signals += bd.logs
            if bd.approved:
                sig, dec, sb = bd.approved
                broker.submit(sig, dec, sb.score)
                acct.trades_today += 1
                # Reserve the risk immediately so later same-bar candidates see it.
                acct.open_positions = broker.exposure_positions(t_close)

    equity = pd.Series(eq_v, index=pd.DatetimeIndex(eq_t), name="equity") if eq_t else pd.Series(dtype=float)
    return BacktestResult(broker.closed, equity, signals, {k: v.to_dict() for k, v in quality.items()}, skipped,
                          {"timeframe": timeframe, "strategies": [s.spec() for s in strategies],
                           "trade_start": str(trade_start), "trade_end": str(trade_end)},
                          list(broker.positions.values()), broker.unrealized())
