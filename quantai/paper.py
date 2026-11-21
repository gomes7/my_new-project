"""Paper trading: live data, simulated orders, same decision pipeline and simulator as backtests.

State (SimBroker + AccountState + last processed bar per symbol) is persisted with pickle in the
local data directory (trusted, operator-owned file). Every event is also journaled.
"""
from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from quantai.alerts import Alert, AlertManager
from quantai.analysis.frame import FeatureFrame
from quantai.backtest.engine import prepare
from quantai.config import InstrumentSpec, get_instruments
from quantai.decision import MarketContext, decide
from quantai.execution.sim import SimBroker
from quantai.models import TF_SECONDS, utcnow
from quantai.monitoring import data_alerts, performance_alerts
from quantai.risk.engine import AccountState, RiskEngine
from quantai.runtime import Runtime
from quantai.selection import StrategySelector
from quantai.strategies.base import Strategy


@dataclass
class PaperState:
    broker: SimBroker
    acct: AccountState
    last_bar: dict[str, pd.Timestamp] = field(default_factory=dict)
    equity_curve: list[tuple[str, float]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: utcnow().isoformat())
    fx_last: dict[str, float] = field(default_factory=dict)
    decided_bar: dict[str, pd.Timestamp] = field(default_factory=dict)  # one decision per closed bar (= backtest)
    n_closed_journaled: int = 0


class PaperTrader:
    STATE = "paper_state.pkl"

    def __init__(self, rt: Runtime, symbols: list[str], strategies: list[Strategy], timeframe: str = "1d",
                 instruments: dict[str, InstrumentSpec] | None = None, selector: StrategySelector | None = None,
                 alerts: AlertManager | None = None):
        if rt.settings.mode not in ("research", "paper"):
            raise PermissionError("unsupported mode")
        self.rt = rt
        self.symbols = symbols
        self.strategies = strategies
        self.tf = timeframe
        self.ins = instruments or get_instruments()
        self.selector = selector or StrategySelector()
        self.risk = RiskEngine(rt.settings.risk, rt.settings.execution)
        self.alerts = alerts or AlertManager(rt.settings.alerts, rt.journal)
        self.state = self._load() or self._new()

    # ---------------------------------------------------------------- persistence
    def _fx(self, ccy: str, _t: datetime) -> float | None:
        return 1.0 if ccy == self.rt.settings.account.currency else self.state.fx_last.get(ccy)

    def _new(self) -> PaperState:
        cap = self.rt.settings.account.starting_capital
        now = utcnow()
        st = PaperState(None, AccountState(cap, cap, cap, cap, day=now.date()))  # type: ignore[arg-type]
        st.broker = SimBroker(self.ins, self.rt.settings.execution, cap, lambda c, t: self._fx(c, t))
        return st

    def _load(self) -> PaperState | None:
        p = self.rt.data_dir / self.STATE
        if not p.exists():
            return None
        with open(p, "rb") as f:
            st: PaperState = pickle.load(f)
        st.broker.fx_rate = lambda c, t: self._fx(c, t)
        return st

    def save(self) -> None:
        p = self.rt.data_dir / self.STATE
        fx = self.state.broker.fx_rate
        self.state.broker.fx_rate = None  # lambdas are not picklable
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.state, f)
        tmp.replace(p)
        self.state.broker.fx_rate = fx

    # ---------------------------------------------------------------- cycle
    def _refresh_fx(self) -> None:
        acc = self.rt.settings.account.currency
        for ccy in {self.ins[s].quote_currency for s in self.symbols} - {acc}:
            for pair, inv in ((f"{ccy}{acc}", False), (f"{acc}{ccy}", True)):
                if pair in self.ins:
                    md = self.rt.fetch(self.ins[pair], "1h")
                    if not md.df.empty and not md.quality.blocks_signals:
                        v = float(md.df["close"].iloc[-1])
                        self.state.fx_last[ccy] = 1 / v if inv else v
                        break

    def run_once(self) -> dict:
        rt, st, br, acct = self.rt, self.state, self.state.broker, self.state.acct
        now = utcnow()
        bar_s = TF_SECONDS[self.tf]
        report: dict = {"at": now.isoformat(), "events": [], "signals": 0, "approved": 0, "blocked_symbols": {}}
        if rt.kill.active and br.pending:
            n = br.cancel_all_pending("kill_switch", now)
            rt.journal.append("cancel", {"reason": "kill_switch", "count": n})
        self._refresh_fx()
        for sym in self.symbols:
            spec = self.ins[sym]
            md = rt.fetch(spec, self.tf)
            for a in data_alerts(sym, md.quality):
                self.alerts.emit(a)
            if md.df.empty:
                report["blocked_symbols"][sym] = "no_data"
                continue
            f = FeatureFrame(prepare(md.df))
            last_seen = st.last_bar.get(sym)
            new_idx = [i for i, t in enumerate(f.index) if last_seen is None or t > last_seen]
            if last_seen is None:
                new_idx = new_idx[-1:]   # first run: no historical replay, start from the latest closed bar
            for i in new_idx:
                t = f.index[i]
                if last_seen is not None:   # bars after the first one are traded through the simulator
                    for ev in br.on_bar(sym, t, f.row(i), float(f.col("atr")[i - 1]) if i > 0 else float("nan"), bar_s):
                        rt.journal.append({"filled": "fill", "cancelled": "cancel", "partial": "partial",
                                           "closed": "exit"}[ev.kind], {**ev.detail, "time": ev.time}, symbol=sym)
                        report["events"].append({"kind": ev.kind, "symbol": sym, **ev.detail})
                        if ev.kind == "closed":
                            cat = "stop_loss" if "stop" in ev.detail["reason"] else "take_profit" if ev.detail["reason"] == "target" else "position_risk"
                            self.alerts.emit(Alert(cat, ev.detail["reason"], f"closed at {ev.detail['price']:g}, "
                                                   f"P&L {ev.detail['pnl']:.2f} ({ev.detail['r']:.2f}R)", symbol=sym,
                                                   price=ev.detail["price"], severity="info"))
                st.last_bar[sym] = t
            for tr in br.closed[st.n_closed_journaled:]:
                rt.journal.append("outcome", asdict(tr), ref_id=tr.signal_id, symbol=tr.symbol, strategy=tr.strategy)
                acct.register_closed_trade(tr.pnl, pd.Timestamp(tr.exit_time).to_pydatetime(), rt.settings.risk)
            st.n_closed_journaled = len(br.closed)
            acct.equity = br.equity()
            acct.peak_equity = max(acct.peak_equity, acct.equity)
            acct.roll(now, acct.equity)
            acct.open_positions = br.exposure_positions(now)
            if md.quality.blocks_signals:
                report["blocked_symbols"][sym] = "data_quality_failed"
                continue
            i = len(f) - 1
            if st.decided_bar.get(sym) == f.index[i]:
                continue   # this closed bar was already evaluated
            st.decided_bar[sym] = f.index[i]
            if sym in br.positions or sym in br.pending or len(f) < 210:
                continue
            t_close = f.index[i].to_pydatetime() + timedelta(seconds=bar_s)
            bd = decide(sym, f, i, t_close, self.tf, bar_s, self.strategies, self.selector, self.risk, acct, spec,
                        md.quality, self._fx(spec.quote_currency, now), rt.settings, self.ins,
                        rt.settings.ranking.min_score_to_trade, MarketContext(kill_switch=rt.kill.active))
            for lg in bd.logs:
                rt.journal.append("signal", {**asdict(lg), "source": "paper"}, ref_id=lg.signal_id, symbol=sym,
                                  strategy=lg.strategy)
                report["signals"] += 1
            if bd.approved:
                sig, dec, sb = bd.approved
                br.submit(sig, dec, sb.score)
                acct.trades_today += 1
                acct.open_positions = br.exposure_positions(now)
                report["approved"] += 1
                rt.journal.append("order", {"signal_id": sig.id, "qty": dec.quantity, "stop": sig.stop,
                                            "target": sig.target, "risk": dec.risk_amount, "fill": "next bar open"},
                                  ref_id=sig.id, symbol=sym, strategy=sig.strategy)
                self.alerts.emit(Alert("setup_confirmed", f"{sig.strategy} {'long' if sig.direction.value == 1 else 'short'}",
                                       f"score {sb.score}, R:R {sig.reward_risk:.2f}", symbol=sym, price=sig.entry_ref,
                                       risk=f"{dec.risk_amount:.2f} {rt.settings.account.currency}",
                                       action="paper order placed for next bar open"))
        st.equity_curve.append((now.isoformat(), br.equity()))
        eq = pd.Series([v for _, v in st.equity_curve], index=pd.to_datetime([t for t, _ in st.equity_curve], utc=True))
        for a in performance_alerts(br.closed, eq, acct.start_of_day_equity, rt.settings.risk):
            self.alerts.emit(a)
        self.save()
        snap = self.snapshot()
        rt.write_json("paper_snapshot.json", snap)
        rt.journal.append("paper_state", {k: snap[k] for k in ("equity", "cash", "open_positions", "pending", "closed_trades")})
        report["equity"] = snap["equity"]
        return report

    def snapshot(self) -> dict:
        br, acct = self.state.broker, self.state.acct
        positions = []
        for p in br.positions.values():
            px = br.last_price.get(p.symbol, p.entry_price)
            spec = self.ins[p.symbol]
            positions.append({"symbol": p.symbol, "strategy": p.strategy, "direction": "long" if p.direction.value == 1 else "short",
                              "qty": p.quantity, "entry": p.entry_price, "current": px, "stop": p.stop, "target": p.target,
                              "unrealized": (px - p.entry_price) * p.direction.value * p.quantity * spec.contract_size * p.fx_to_account,
                              "risk": p.risk_amount, "bars_held": p.bars_held, "status": "partial_taken" if p.partial_taken else "open"})
        return {"at": utcnow().isoformat(), "equity": br.equity(), "cash": br.cash,
                "starting_capital": self.rt.settings.account.starting_capital,
                "start_of_day_equity": acct.start_of_day_equity, "peak_equity": acct.peak_equity,
                "drawdown_pct": acct.drawdown_pct, "day_pnl": acct.day_pnl, "open_risk": acct.open_risk(),
                "open_positions": positions, "pending": [{"symbol": s, "strategy": pe.signal.strategy, "qty": pe.decision.quantity}
                                                         for s, pe in br.pending.items()],
                "closed_trades": len(br.closed), "equity_curve": self.state.equity_curve[-500:],
                "trades": [asdict(t) for t in br.closed[-200:]], "kill_switch": self.rt.kill.status(),
                "started_at": self.state.started_at}
