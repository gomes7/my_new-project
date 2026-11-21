"""Daily profit target engine. Reports feasibility; NEVER changes risk or forces trades."""
from __future__ import annotations

from dataclasses import asdict, dataclass

NO_TRADE = "NO TRADE — NO VALID SETUP"


@dataclass
class TargetReport:
    starting_capital: float
    current_capital: float
    daily_target_amount: float
    daily_target_pct: float
    required_return_pct: float
    risk_per_trade_amount: float
    daily_risk_budget: float
    max_trades: int
    expectancy_r: float | None
    expected_trades_per_day: float | None
    expected_daily_pnl: float | None
    feasibility_ratio: float | None
    risk_needed_per_trade_pct: float | None
    compatible: bool
    status: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def daily_target(starting_capital: float, current_capital: float, target_pct: float, risk_per_trade_pct: float,
                 max_daily_loss_pct: float, max_trades: int, expectancy_r: float | None,
                 expected_trades_per_day: float | None, valid_setups_now: int, target_amount: float | None = None) -> TargetReport:
    tgt = target_amount if target_amount is not None else current_capital * target_pct / 100
    rpt = current_capital * risk_per_trade_pct / 100
    budget = current_capital * max_daily_loss_pct / 100
    exp_pnl = feas = need = None
    if expectancy_r is not None and expected_trades_per_day is not None:
        ntr = min(expected_trades_per_day, max_trades)
        exp_pnl = expectancy_r * rpt * ntr
        feas = exp_pnl / tgt if tgt > 0 else None
        need = (tgt / (expectancy_r * ntr) / current_capital * 100) if expectancy_r > 0 and ntr > 0 else None
    compatible = feas is not None and feas >= 1.0
    if valid_setups_now == 0:
        status, msg = "no_trade", NO_TRADE
    elif expectancy_r is None:
        status, msg = "unvalidated", "Strategy expectancy not validated; target feasibility unknown. Risk unchanged."
    elif not compatible:
        status = "target_not_compatible"
        msg = (f"Expected daily P&L {exp_pnl:,.2f} < target {tgt:,.2f}. Target would require "
               f"{need:.2f}% risk/trade" if need else "Expectancy <= 0: target unreachable without edge")
        msg += " — risk is NOT increased; trade only valid setups."
    else:
        status, msg = "feasible", "Target consistent with validated expectancy; still only valid setups are traded."
    return TargetReport(starting_capital, current_capital, round(tgt, 2), target_pct,
                        round(tgt / current_capital * 100, 4) if current_capital else 0.0, round(rpt, 2), round(budget, 2),
                        max_trades, expectancy_r, expected_trades_per_day,
                        None if exp_pnl is None else round(exp_pnl, 2), None if feas is None else round(feas, 3),
                        None if need is None else round(need, 3), compatible, status, msg)
