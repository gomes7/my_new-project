"""Performance & health monitoring that produces alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantai.alerts import Alert
from quantai.backtest.metrics import max_drawdown
from quantai.config import RiskCfg
from quantai.models import ClosedTrade, QualityReport, QualityStatus


def performance_alerts(trades: list[ClosedTrade], equity: pd.Series, start_of_day_equity: float, risk: RiskCfg,
                       expected_expectancy_r: float | None = None, expected_ci_low: float | None = None,
                       window: int = 20) -> list[Alert]:
    alerts: list[Alert] = []
    if not equity.empty:
        dd_pct, _ = max_drawdown(equity)
        cur_dd = float((equity.cummax().iloc[-1] - equity.iloc[-1]) / equity.cummax().iloc[-1] * 100)
        if cur_dd >= 0.5 * risk.max_drawdown_pct:
            alerts.append(Alert("performance", "drawdown_increasing", f"current drawdown {cur_dd:.2f}% "
                                f"(limit {risk.max_drawdown_pct}%)", risk=f"{cur_dd:.2f}%",
                                severity="critical" if cur_dd >= 0.8 * risk.max_drawdown_pct else "warning",
                                action="review open risk; engine halts at limit"))
        day_loss = (start_of_day_equity - float(equity.iloc[-1])) / start_of_day_equity * 100 if start_of_day_equity else 0
        if day_loss >= risk.max_daily_loss_pct:
            alerts.append(Alert("daily_loss_limit", "daily_loss_limit_reached", f"day loss {day_loss:.2f}%",
                                severity="critical", action="no new trades until next session"))
        elif day_loss >= 0.75 * risk.max_daily_loss_pct:
            alerts.append(Alert("risk_limit_approaching", "daily_loss_approaching", f"day loss {day_loss:.2f}% of "
                                f"{risk.max_daily_loss_pct}% limit", severity="warning", action="reduce activity"))
    if len(trades) >= window:
        r = np.array([t.r_multiple for t in trades[-window:]])
        if r.mean() < 0:
            alerts.append(Alert("performance", "rolling_expectancy_negative",
                                f"last {window} trades expectancy {r.mean():.2f}R", severity="warning",
                                action="review strategy; consider pausing"))
        if expected_ci_low is not None and r.mean() < expected_ci_low:
            alerts.append(Alert("performance", "outside_expected_parameters",
                                f"rolling {r.mean():.2f}R below validated CI lower bound {expected_ci_low:.2f}R",
                                severity="warning", action="strategy may have degraded; re-validate"))
    return alerts


def data_alerts(symbol: str, q: QualityReport) -> list[Alert]:
    if q.status == QualityStatus.OK:
        return []
    sev = "critical" if q.status == QualityStatus.FAILED else "info"
    return [Alert("data_feed", f"data_quality_{q.status.value}", "; ".join(f"{i.code}: {i.detail}" for i in q.issues),
                  symbol=symbol, severity=sev,
                  action="signals blocked for this instrument" if q.status == QualityStatus.FAILED else "none")]
