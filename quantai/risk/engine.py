"""Risk engine — independent of strategies; every trade must pass `evaluate`.

Nothing in the strategy layer can bypass this: the simulator/paper broker only opens positions
from an approved RiskDecision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from quantai.config import ExecutionCfg, InstrumentSpec, RiskCfg
from quantai.models import Position, QualityReport, RiskDecision, SignalCandidate


@dataclass
class AccountState:
    equity: float
    start_of_day_equity: float
    start_of_week_equity: float
    peak_equity: float
    day: date
    consecutive_losses: int = 0
    trades_today: int = 0
    open_positions: list[Position] = field(default_factory=list)
    lockout_until: datetime | None = None
    week: tuple[int, int] | None = None

    def roll(self, t: datetime, equity: float) -> None:
        """Start-of-day / start-of-week bookkeeping (UTC)."""
        if self.day != t.date():
            self.day, self.start_of_day_equity, self.trades_today = t.date(), equity, 0
        wk = tuple(t.isocalendar()[:2])
        if self.week != wk:
            self.week, self.start_of_week_equity = wk, equity

    def register_closed_trade(self, pnl: float, t: datetime, cfg: RiskCfg) -> None:
        """Update the loss streak; on hitting the limit start a cooldown and reset the counter."""
        self.consecutive_losses = self.consecutive_losses + 1 if pnl <= 0 else 0
        if self.consecutive_losses >= cfg.max_consecutive_losses:
            self.lockout_until = t + timedelta(hours=cfg.consecutive_loss_cooldown_hours)
            self.consecutive_losses = 0

    @property
    def day_pnl(self) -> float:
        return self.equity - self.start_of_day_equity

    @property
    def drawdown_pct(self) -> float:
        return 0.0 if self.peak_equity <= 0 else (self.peak_equity - self.equity) / self.peak_equity * 100

    def open_risk(self) -> float:
        return sum(open_position_risk(p) for p in self.open_positions)


def open_position_risk(p: Position) -> float:
    """Remaining risk to stop in account currency (0 once stop is at/through breakeven)."""
    per_unit = (p.entry_price - p.stop) * p.direction.value
    return max(0.0, per_unit * p.quantity * p.fx_to_account)


@dataclass
class RiskContext:
    data_quality: QualityReport | None = None
    fx_to_account: float | None = None          # value of 1 unit of quote ccy in account ccy
    current_spread: float | None = None
    kill_switch: bool = False
    event_blackout: str | None = None           # reason string if inside a blackout window
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    max_trades_per_day: int | None = None
    instruments: dict[str, InstrumentSpec] = field(default_factory=dict)
    account_currency: str = "USD"


def round_down(x: float, step: float) -> float:
    return math.floor(x / step + 1e-9) * step


class RiskEngine:
    def __init__(self, cfg: RiskCfg, exec_cfg: ExecutionCfg | None = None):
        self.cfg = cfg
        self.exec_cfg = exec_cfg or ExecutionCfg()

    @property
    def risk_pct(self) -> float:
        return self.cfg.risk_per_trade_pct

    def estimated_cost_per_unit(self, spec: InstrumentSpec, sig: SignalCandidate) -> float:
        """Conservative round-trip cost per unit in QUOTE currency: spread + 2x slippage + 2x commission."""
        spread = spec.spread_at(sig.entry_ref)
        slip = self.exec_cfg.slippage_atr_fraction * sig.atr
        comm = sig.entry_ref * spec.commission_bps / 10_000
        return (spread + 2 * slip + 2 * comm) * spec.contract_size

    def evaluate(self, sig: SignalCandidate, spec: InstrumentSpec, acct: AccountState, ctx: RiskContext,
                 now: datetime | None = None) -> RiskDecision:
        c = self.cfg
        reasons: list[str] = []
        checks: dict = {}

        if ctx.kill_switch:
            reasons.append("kill_switch_active")
        if ctx.data_quality is None:
            reasons.append("data_quality_unknown")
        elif ctx.data_quality.blocks_signals:
            reasons.append("data_quality_failed:" + ",".join(i.code for i in ctx.data_quality.issues))
        if ctx.event_blackout:
            reasons.append(f"event_blackout:{ctx.event_blackout}")

        # Account-level limits
        sod = acct.start_of_day_equity
        day_loss_pct = -acct.day_pnl / sod * 100 if sod > 0 else 0.0
        week_loss_pct = (acct.start_of_week_equity - acct.equity) / acct.start_of_week_equity * 100 if acct.start_of_week_equity > 0 else 0
        checks.update(day_loss_pct=round(day_loss_pct, 4), week_loss_pct=round(week_loss_pct, 4),
                      drawdown_pct=round(acct.drawdown_pct, 4))
        if day_loss_pct >= c.max_daily_loss_pct:
            reasons.append("daily_loss_limit_reached")
        if week_loss_pct >= c.max_weekly_loss_pct:
            reasons.append("weekly_loss_limit_reached")
        if acct.drawdown_pct >= c.max_drawdown_pct:
            reasons.append("max_drawdown_reached")
        t_now = now or sig.signal_time
        if acct.lockout_until is not None and t_now < acct.lockout_until:
            reasons.append(f"consecutive_loss_cooldown_until_{acct.lockout_until.isoformat()}")
        if acct.consecutive_losses >= c.max_consecutive_losses:
            reasons.append("max_consecutive_losses")
        if len(acct.open_positions) >= c.max_open_positions:
            reasons.append("max_open_positions")
        if ctx.max_trades_per_day is not None and acct.trades_today >= ctx.max_trades_per_day:
            reasons.append("max_trades_per_day")
        if any(p.symbol == sig.symbol for p in acct.open_positions):
            reasons.append("position_already_open_on_symbol")

        # Stop validity
        dist = sig.stop_distance
        if sig.atr <= 0 or dist <= 0:
            reasons.append("stop_undefined")
        else:
            stop_atr = dist / sig.atr
            checks["stop_atr"] = round(stop_atr, 3)
            if stop_atr < c.min_stop_atr:
                reasons.append(f"stop_too_tight({stop_atr:.2f}ATR)")
            if stop_atr > c.max_stop_atr:
                reasons.append(f"stop_too_wide({stop_atr:.2f}ATR)")

        # Execution conditions
        spread = ctx.current_spread if ctx.current_spread is not None else spec.spread_at(sig.entry_ref)
        if sig.atr > 0:
            checks["spread_to_atr"] = round(spread / sig.atr, 4)
            if spread / sig.atr > c.max_spread_to_atr:
                reasons.append("abnormal_spread")

        # Currency conversion: never assume
        fx = 1.0 if spec.quote_currency == ctx.account_currency else ctx.fx_to_account
        if fx is None or fx <= 0:
            reasons.append(f"no_fx_conversion_{spec.quote_currency}")
            fx = 0.0

        # Sizing
        mult = c.vol_risk_multipliers.get(sig.vol_regime, min(c.vol_risk_multipliers.values())) if c.vol_scaling else 1.0
        checks["vol_risk_multiplier"] = mult
        risk_amount = acct.equity * self.risk_pct / 100 * mult
        qty = 0.0
        if fx > 0 and dist > 0:
            per_unit = (dist * spec.contract_size + self.estimated_cost_per_unit(spec, sig)) * fx
            qty = round_down(risk_amount / per_unit, spec.qty_step)
            checks["per_unit_risk"] = per_unit
            if qty < spec.min_qty:
                reasons.append("size_below_min_qty")
                qty = 0.0
            actual_risk = qty * per_unit
            notional = qty * sig.entry_ref * spec.contract_size * fx
            lev = notional / acct.equity if acct.equity > 0 else math.inf
            checks.update(notional=round(notional, 2), leverage=round(lev, 3))
            if lev > c.max_leverage:
                reasons.append(f"leverage_exceeded({lev:.2f}x)")
            if actual_risk > acct.equity * c.max_risk_per_trade_pct / 100 * 1.0001 and not c.exceptional_justification:
                reasons.append("oversized")
            risk_amount = actual_risk

        # Remaining daily risk budget (realised day loss + open risk + this trade)
        budget = acct.start_of_day_equity * c.max_daily_loss_pct / 100 + min(0.0, acct.day_pnl) - acct.open_risk()
        checks["daily_budget_remaining"] = round(budget, 2)
        if risk_amount > budget + 1e-9:
            reasons.append("exceeds_remaining_daily_risk_budget")
        total_open = acct.open_risk() + risk_amount
        if total_open > acct.equity * c.max_total_open_risk_pct / 100 + 1e-9:
            reasons.append("total_open_risk_exceeded")

        # Correlated exposure
        group_risk = risk_amount
        for p in acct.open_positions:
            pspec = ctx.instruments.get(p.symbol)
            same_group = pspec is not None and pspec.risk_group == spec.risk_group
            rho = ctx.correlations.get((sig.symbol, p.symbol), ctx.correlations.get((p.symbol, sig.symbol)))
            corr_same_way = rho is not None and abs(rho) >= 0.7 and rho * p.direction.value * sig.direction.value > 0
            if (same_group and p.direction == sig.direction) or corr_same_way:
                group_risk += open_position_risk(p)
        checks["correlated_risk"] = round(group_risk, 2)
        if group_risk > acct.equity * c.max_risk_per_group_pct / 100 + 1e-9:
            reasons.append("correlated_exposure_exceeded")

        approved = not reasons and qty > 0
        return RiskDecision(approved=approved, quantity=qty if approved else 0.0, risk_amount=risk_amount if approved else 0.0,
                            risk_pct=(risk_amount / acct.equity * 100) if approved and acct.equity else 0.0,
                            reasons=reasons, checks=checks)
