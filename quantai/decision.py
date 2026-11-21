"""Single per-bar decision pipeline shared by backtest, paper trading and the scanner.

strategies -> filters -> selection -> score -> risk engine. Every candidate, including rejected
ones, is returned as a SignalLog so it can be journaled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from quantai.analysis.frame import FeatureFrame
from quantai.config import InstrumentSpec, Settings
from quantai.models import QualityReport, RiskDecision, SignalCandidate
from quantai.ranking import ScoreBreakdown, score_opportunity
from quantai.risk.engine import AccountState, RiskContext, RiskEngine
from quantai.selection import StrategySelector
from quantai.strategies.base import Strategy


@dataclass
class SignalLog:
    time: datetime
    symbol: str
    strategy: str
    direction: int
    regime: str
    entry_ref: float
    stop: float
    target: float
    score: float | None
    decision: str          # approved | rejected
    reasons: list[str]
    quantity: float = 0.0
    signal_id: str = ""
    risk_amount: float = 0.0
    score_detail: dict = field(default_factory=dict)
    risk_checks: dict = field(default_factory=dict)


@dataclass
class BarDecision:
    logs: list[SignalLog]
    approved: tuple[SignalCandidate, RiskDecision, ScoreBreakdown] | None = None
    best_candidate: tuple[SignalCandidate, ScoreBreakdown, RiskDecision] | None = None


@dataclass
class MarketContext:
    """Optional context; None values are neutral in scoring and reported as unavailable."""
    mtf_alignment: float | None = None
    macro_bias: float | None = None
    event_risk: float | None = None
    event_blackout: str | None = None
    current_spread: float | None = None
    kill_switch: bool = False
    correlations: dict = field(default_factory=dict)


def _log(sig: SignalCandidate, t: datetime, decision: str, reasons: list[str], **kw) -> SignalLog:
    return SignalLog(t, sig.symbol, sig.strategy, sig.direction.value, sig.regime, sig.entry_ref, sig.stop, sig.target,
                     kw.pop("score", None), decision, reasons, signal_id=sig.id, **kw)


def decide(sym: str, f: FeatureFrame, i: int, t_close: datetime, timeframe: str, bar_seconds: int,
           strategies: list[Strategy], selector: StrategySelector, risk: RiskEngine, acct: AccountState,
           spec: InstrumentSpec, quality: QualityReport | None, fx_to_account: float | None, settings: Settings,
           instruments: dict[str, InstrumentSpec], min_score: float | None = None,
           mctx: MarketContext | None = None) -> BarDecision:
    mctx = mctx or MarketContext()
    regime = f.col("regime")[i]
    logs: list[SignalLog] = []
    cands: list[SignalCandidate] = []
    for strat in strategies:
        sig, fr = strat.signal_at(f, i, sym, regime, timeframe, bar_seconds)
        if sig is None:
            continue
        if fr is not None and not fr.passed:
            logs.append(_log(sig, t_close, "rejected", ["filter:" + x for x in fr.failed]))
        else:
            cands.append(sig)
    if not cands:
        return BarDecision(logs)
    sel = selector.select(cands)
    logs += [_log(s, t_close, "rejected", [why]) for s, why in sel.rejected]
    sig = sel.chosen
    if sig is None:
        return BarDecision(logs)
    sb = score_opportunity(sig, f.row(i), settings.ranking.thresholds, mtf_alignment=mctx.mtf_alignment,
                           validated_expectancy_r=selector.validated_expectancy(sig.strategy, regime),
                           spread=mctx.current_spread if mctx.current_spread is not None else spec.spread_at(sig.entry_ref),
                           macro_bias=mctx.macro_bias, event_risk=mctx.event_risk, quality=quality)
    reasons: list[str] = []
    if min_score is not None and sb.score < min_score:
        reasons.append(f"score_below_threshold({sb.score}<{min_score})")
    ctx = RiskContext(data_quality=quality, fx_to_account=fx_to_account, current_spread=mctx.current_spread,
                      kill_switch=mctx.kill_switch, event_blackout=mctx.event_blackout, correlations=mctx.correlations,
                      instruments=instruments, account_currency=settings.account.currency,
                      max_trades_per_day=settings.target.max_trades_per_day)
    dec = risk.evaluate(sig, spec, acct, ctx, now=t_close)
    reasons += dec.reasons
    ok = dec.approved and not reasons
    logs.append(_log(sig, t_close, "approved" if ok else "rejected", reasons, score=sb.score,
                     quantity=dec.quantity if ok else 0.0, risk_amount=dec.risk_amount if ok else 0.0,
                     score_detail=sb.to_dict(), risk_checks=dec.checks))
    return BarDecision(logs, (sig, dec, sb) if ok else None, (sig, sb, dec))
