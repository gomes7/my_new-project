"""Multi-asset scanner: data -> validation -> MTF technicals -> regime -> strategies -> score -> risk."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from quantai.analysis.frame import FeatureFrame
from quantai.analysis.technical import analyze
from quantai.backtest.engine import prepare
from quantai.config import InstrumentSpec, get_instruments
from quantai.data.http import ProviderError
from quantai.decision import MarketContext, decide
from quantai.intel.calendar import CalendarEvent, blackout_reason
from quantai.intel.macro import MacroRegime
from quantai.intel.news import ScoredEvent, asset_event_risk
from quantai.models import TF_SECONDS, utcnow
from quantai.risk.engine import AccountState, RiskEngine
from quantai.runtime import Runtime
from quantai.selection import StrategySelector
from quantai.strategies.base import Strategy
from quantai.targets import NO_TRADE


@dataclass
class ScanResult:
    generated_at: str
    primary_timeframe: str
    markets: list[dict] = field(default_factory=list)
    opportunities: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    message: str = ""


def scan(rt: Runtime, symbols: list[str], strategies: list[Strategy], acct: AccountState,
         primary_tf: str = "1d", context_tfs: tuple[str, ...] = ("1d", "4h", "1h"),
         macro: MacroRegime | None = None, events: list[ScoredEvent] | None = None,
         calendar: list[CalendarEvent] | None = None, selector: StrategySelector | None = None,
         instruments: dict[str, InstrumentSpec] | None = None, journal: bool = True) -> ScanResult:
    st = rt.settings
    ins = instruments or get_instruments()
    risk = RiskEngine(st.risk, st.execution)
    selector = selector or StrategySelector()
    now = utcnow()
    res = ScanResult(now.isoformat(), primary_tf)
    fx_cache: dict[str, float | None] = {}

    def fx_rate(ccy: str) -> float | None:
        if ccy == st.account.currency:
            return 1.0
        if ccy not in fx_cache:
            fx_cache[ccy] = None
            for pair, inv in ((f"{ccy}{st.account.currency}", False), (f"{st.account.currency}{ccy}", True)):
                if pair in ins:
                    md = rt.fetch(ins[pair], "1h")
                    if not md.df.empty and not md.quality.blocks_signals:
                        v = float(md.df["close"].iloc[-1])
                        fx_cache[ccy] = 1 / v if inv else v
                        break
        return fx_cache[ccy]

    for sym in symbols:
        spec = ins[sym]
        frames, qual = {}, {}
        for tf in dict.fromkeys((primary_tf, *context_tfs)):
            md = rt.fetch(spec, tf, crosscheck=(tf == primary_tf))
            qual[tf] = md.quality
            if not md.df.empty:
                frames[tf] = md
        pmd = frames.get(primary_tf)
        mkt = {"symbol": sym, "asset_class": spec.asset_class, "provider": spec.provider,
               "quality": {tf: (q.status.value if q else "unknown") for tf, q in qual.items()},
               "quality_issues": {tf: [i.code for i in q.issues] for tf, q in qual.items() if q and q.issues}}
        if pmd is None:
            mkt["status"] = "no_data"
            res.markets.append(mkt)
            continue
        feats = {tf: prepare(m.df) for tf, m in frames.items() if len(m.df) >= 30}
        tech = analyze(sym, feats)
        pf = feats.get(primary_tf)
        if pf is None or len(pf) < 210:
            mkt["status"] = "insufficient_history"
            res.markets.append(mkt)
            continue
        f = FeatureFrame(pf)
        i = len(f) - 1
        last = f.row(i)
        spread = None
        if spec.provider == "kraken":
            try:
                spread = rt.provider("kraken").ticker(spec.provider_symbol)["spread"]  # type: ignore[attr-defined]
            except ProviderError:
                spread = None
        mctx = MarketContext(
            mtf_alignment=tech.alignment, macro_bias=macro.bias(spec.asset_class, sym) if macro else None,
            event_risk=asset_event_risk(events, sym) if events is not None else None,
            event_blackout=blackout_reason(calendar or [], sym, now, st.risk.event_blackout_minutes),
            current_spread=spread, kill_switch=rt.kill.active)
        t_close = f.index[i].to_pydatetime() + timedelta(seconds=TF_SECONDS[primary_tf])
        bd = decide(sym, f, i, t_close, primary_tf, TF_SECONDS[primary_tf], strategies, selector, risk, acct, spec,
                    pmd.quality, fx_rate(spec.quote_currency), st, ins, st.ranking.min_score_to_trade, mctx)
        mkt.update(status="ok", last_close=float(last.close), bar_close_time=t_close.isoformat(),
                   delay_status=pmd.meta.delay_status.value, regime=last.regime, vol_regime=last.vol_regime,
                   atr=float(last.atr), bias=tech.bias, mtf_alignment=tech.alignment, notes=tech.notes,
                   live_spread=spread, data_notes=pmd.meta.notes, technical=tech.to_dict())
        res.markets.append(mkt)
        for lg in bd.logs:
            row = {"symbol": sym, "strategy": lg.strategy, "direction": "long" if lg.direction == 1 else "short",
                   "regime": lg.regime, "entry_ref": lg.entry_ref, "entry_zone": [lg.entry_ref - 0.1 * float(last.atr),
                                                                                 lg.entry_ref + 0.1 * float(last.atr)],
                   "stop": lg.stop, "target": lg.target,
                   "reward_risk": abs(lg.target - lg.entry_ref) / abs(lg.entry_ref - lg.stop) if lg.entry_ref != lg.stop else 0,
                   "score": lg.score, "score_detail": lg.score_detail, "decision": lg.decision, "reasons": lg.reasons,
                   "quantity": lg.quantity, "risk_amount": lg.risk_amount, "signal_id": lg.signal_id,
                   "time": lg.time.isoformat()}
            (res.opportunities if lg.decision == "approved" else res.rejected).append(row)
            if journal:
                rt.journal.append("signal", {**row, "source": "scanner", "risk_checks": lg.risk_checks},
                                  ref_id=lg.signal_id, symbol=sym, strategy=lg.strategy)
    res.opportunities.sort(key=lambda r: r["score"] or 0, reverse=True)
    res.rejected.sort(key=lambda r: r["score"] or 0, reverse=True)
    res.message = (f"{len(res.opportunities)} approved opportunit{'y' if len(res.opportunities) == 1 else 'ies'}"
                   if res.opportunities else NO_TRADE)
    if journal:
        rt.journal.append("scan", {"generated_at": res.generated_at, "symbols": symbols, "message": res.message,
                                   "approved": len(res.opportunities), "rejected": len(res.rejected)})
    return res
