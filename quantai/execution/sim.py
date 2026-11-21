"""Conservative bar-based execution simulator shared by backtesting and paper trading.

Assumptions (all explicit, all adverse where ambiguous):
  * Prices are mid/last; we pay half the configured spread on every fill.
  * Market and stop fills suffer adverse slippage = slippage_atr_fraction x ATR.
  * Entries fill at the NEXT bar open. If that open is already through the stop, the entry is cancelled.
  * Gaps through a stop fill at the (worse) open.
  * If stop and target/partial levels are both inside one bar, the STOP is assumed to fill first.
  * Take-profit is a limit: no positive slippage, half spread paid.
  * Commission = commission_bps of notional per side. Holding cost charged per UTC day boundary crossed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import pandas as pd

from quantai.config import ExecutionCfg, InstrumentSpec
from quantai.models import ClosedTrade, Position, RiskDecision, SignalCandidate, new_id


@dataclass
class PendingEntry:
    signal: SignalCandidate
    decision: RiskDecision
    score: float | None = None


@dataclass
class SimEvent:
    kind: str           # filled | cancelled | partial | closed
    time: datetime
    symbol: str
    detail: dict = field(default_factory=dict)


class SimBroker:
    def __init__(self, instruments: dict[str, InstrumentSpec], exec_cfg: ExecutionCfg, starting_cash: float,
                 fx_rate: Callable[[str, datetime], float | None]):
        self.instruments = instruments
        self.cfg = exec_cfg
        self.cash = starting_cash
        self.fx_rate = fx_rate
        self.positions: dict[str, Position] = {}
        self.pending: dict[str, PendingEntry] = {}
        self.closed: list[ClosedTrade] = []
        self.last_price: dict[str, float] = {}
        self.events: list[SimEvent] = []

    # ------------------------------------------------------------------ orders
    def submit(self, sig: SignalCandidate, decision: RiskDecision, score: float | None = None) -> None:
        if not decision.approved:
            raise PermissionError("SimBroker only accepts risk-approved orders")
        self.pending[sig.symbol] = PendingEntry(sig, decision, score)

    def cancel_all_pending(self, reason: str, t: datetime) -> int:
        n = len(self.pending)
        for sym in list(self.pending):
            self.events.append(SimEvent("cancelled", t, sym, {"reason": reason}))
        self.pending.clear()
        return n

    def exposure_positions(self, fx_at: datetime) -> list[Position]:
        """Open positions plus pending entries (as provisional positions) for risk accounting."""
        out = list(self.positions.values())
        for sym, pe in self.pending.items():
            s = pe.signal
            fx = self.fx_rate(self.instruments[sym].quote_currency, fx_at) or 0.0
            out.append(Position(id="pending", symbol=sym, strategy=s.strategy, direction=s.direction,
                                quantity=pe.decision.quantity, entry_price=s.entry_ref, entry_time=s.signal_time,
                                stop=s.stop, initial_stop=s.stop, target=s.target, risk_amount=pe.decision.risk_amount,
                                fx_to_account=fx, regime=s.regime, signal_id=s.id, max_bars=s.max_bars))
        return out

    # ------------------------------------------------------------------ valuation
    def unrealized(self) -> float:
        tot = 0.0
        for p in self.positions.values():
            px = self.last_price.get(p.symbol, p.entry_price)
            spec = self.instruments[p.symbol]
            tot += (px - p.entry_price) * p.direction.value * p.quantity * spec.contract_size * p.fx_to_account
        return tot

    def equity(self) -> float:
        return self.cash + self.unrealized()

    # ------------------------------------------------------------------ bar processing
    def on_bar(self, symbol: str, t_open: pd.Timestamp, bar: pd.Series, atr: float, bar_seconds: int) -> list[SimEvent]:
        spec = self.instruments[symbol]
        ev: list[SimEvent] = []
        slip = self.cfg.slippage_atr_fraction * (atr if atr == atr else 0.0)
        half = spec.spread_at(float(bar.open)) / 2
        fx = self.fx_rate(spec.quote_currency, t_open)
        t_close = (t_open + pd.Timedelta(seconds=bar_seconds)).to_pydatetime()

        # 1) pending entry at this bar's open
        pe = self.pending.pop(symbol, None)
        if pe is not None:
            s = pe.signal
            d = s.direction.value
            fill = float(bar.open) + d * (half + slip)
            if fx is None:
                ev.append(SimEvent("cancelled", t_open.to_pydatetime(), symbol, {"reason": "no_fx_rate", "signal": s.id}))
            elif (d == 1 and fill <= s.stop) or (d == -1 and fill >= s.stop):
                ev.append(SimEvent("cancelled", t_open.to_pydatetime(), symbol, {"reason": "gapped_through_stop", "signal": s.id}))
            else:
                rr = s.reward_risk
                target = fill + d * rr * abs(fill - s.stop)
                fee = fill * pe.decision.quantity * spec.contract_size * spec.commission_bps / 10_000 * fx
                self.cash -= fee
                self.positions[symbol] = Position(
                    id=new_id("pos"), symbol=symbol, strategy=s.strategy, direction=s.direction,
                    quantity=pe.decision.quantity, entry_price=fill, entry_time=t_open.to_pydatetime(), stop=s.stop,
                    initial_stop=s.stop, target=target,
                    risk_amount=abs(fill - s.stop) * pe.decision.quantity * spec.contract_size * fx,
                    fx_to_account=fx, regime=s.regime, signal_id=s.id, max_bars=s.max_bars, fees_paid=fee,
                    entry_costs=fee, score=pe.score)
                ev.append(SimEvent("filled", t_open.to_pydatetime(), symbol,
                                   {"price": fill, "qty": pe.decision.quantity, "stop": s.stop, "target": target, "signal": s.id}))

        # 2) manage open position through this bar
        p = self.positions.get(symbol)
        if p is not None:
            if fx is not None:
                p.fx_to_account = fx
            d = p.direction.value
            o, h, lo, c = float(bar.open), float(bar.high), float(bar.low), float(bar.close)
            fav, adv = (h - p.entry_price) * d if d == 1 else (p.entry_price - lo), (p.entry_price - lo) if d == 1 else (h - p.entry_price)
            p.mfe, p.mae = max(p.mfe, fav), max(p.mae, adv)
            r_dist = abs(p.entry_price - p.initial_stop)
            stop_hit = (lo <= p.stop) if d == 1 else (h >= p.stop)
            gap_stop = (o <= p.stop) if d == 1 else (o >= p.stop)
            tgt_hit = (h >= p.target + half) if d == 1 else (lo <= p.target - half)
            partial_lvl = p.entry_price + d * self.cfg.partial_exit_r * r_dist
            partial_hit = (not p.partial_taken and self.cfg.partial_exit_fraction > 0 and
                           ((h >= partial_lvl + half) if d == 1 else (lo <= partial_lvl - half)))
            if p.entry_time == t_open.to_pydatetime():
                # Entry bar: intrabar ordering unknown -> only allow the stop (pessimistic).
                tgt_hit = partial_hit = False
                gap_stop = False
            if gap_stop:
                ev.append(self._close(p, o - d * (half + slip), t_open.to_pydatetime(), "stop_gap"))
            elif stop_hit:
                ev.append(self._close(p, p.stop - d * (half + slip), t_close, "stop" if p.stop == p.initial_stop else "breakeven_stop"))
            elif tgt_hit:
                ev.append(self._close(p, p.target - d * half, t_close, "target"))
            else:
                if partial_hit:
                    ev.append(self._partial(p, partial_lvl - d * half, t_close))
                p.bars_held += 1
                if p.bars_held >= p.max_bars and symbol in self.positions:
                    ev.append(self._close(p, c - d * (half + slip), t_close, "time_stop"))
            # holding cost when crossing a UTC day boundary
            if symbol in self.positions and spec.holding_cost_bps_per_day > 0 and t_close.date() != t_open.date():
                cost = c * p.quantity * spec.contract_size * spec.holding_cost_bps_per_day / 10_000 * p.fx_to_account
                self.cash -= cost
                p.fees_paid += cost
                p.entry_costs += cost
        self.last_price[symbol] = float(bar.close)
        self.events.extend(ev)
        return ev

    def _pnl(self, p: Position, qty: float, price: float) -> tuple[float, float]:
        spec = self.instruments[p.symbol]
        gross = (price - p.entry_price) * p.direction.value * qty * spec.contract_size * p.fx_to_account
        fee = price * qty * spec.contract_size * spec.commission_bps / 10_000 * p.fx_to_account
        return gross, fee

    def _partial(self, p: Position, price: float, t: datetime) -> SimEvent:
        spec = self.instruments[p.symbol]
        q = round(int(p.quantity * self.cfg.partial_exit_fraction / spec.qty_step + 1e-9) * spec.qty_step, 10)
        if q < spec.min_qty or p.quantity - q < spec.min_qty - 1e-12:
            return SimEvent("partial", t, p.symbol, {"skipped": "size_too_small"})
        gross, fee = self._pnl(p, q, price)
        self.cash += gross - fee
        p.realized_pnl += gross - fee
        p.fees_paid += fee
        p.quantity -= q
        p.partial_taken = True
        p.stop = p.entry_price  # move to breakeven
        return SimEvent("partial", t, p.symbol, {"price": price, "qty": q, "new_stop": p.stop})

    def _close(self, p: Position, price: float, t: datetime, reason: str) -> SimEvent:
        gross, fee = self._pnl(p, p.quantity, price)
        self.cash += gross - fee
        total = p.realized_pnl + gross - fee - p.entry_costs
        r_dist = abs(p.entry_price - p.initial_stop)
        init_risk = p.risk_amount if p.risk_amount > 0 else 1e-9
        tr = ClosedTrade(
            id=p.id, symbol=p.symbol, strategy=p.strategy, direction=p.direction.value,
            quantity=p.quantity, entry_time=p.entry_time, exit_time=t, entry_price=p.entry_price, exit_price=price,
            pnl=total, r_multiple=total / init_risk, risk_amount=p.risk_amount, fees=p.fees_paid + fee,
            exit_reason=reason, regime=p.regime, bars_held=p.bars_held,
            mfe_r=p.mfe / r_dist if r_dist else 0.0, mae_r=p.mae / r_dist if r_dist else 0.0,
            signal_id=p.signal_id, score=p.score)
        self.closed.append(tr)
        del self.positions[p.symbol]
        return SimEvent("closed", t, p.symbol, {"price": price, "reason": reason, "pnl": total, "r": tr.r_multiple})
