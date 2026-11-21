"""Performance metrics computed only from simulated trades / equity — never estimated."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from quantai.models import ClosedTrade


def trades_frame(trades: list[ClosedTrade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[f for f in ClosedTrade.__dataclass_fields__])
    df = pd.DataFrame([asdict(t) for t in trades])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    hour = df["entry_time"].dt.hour
    df["session"] = np.select([hour < 7, hour < 13, hour < 21], ["asia", "europe", "us"], "late_us")
    return df


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """(max drawdown %, max drawdown in currency)."""
    if equity.empty:
        return 0.0, 0.0
    peak = equity.cummax()
    dd = equity - peak
    pct = (dd / peak).min() * 100
    return float(-pct), float(-dd.min())


def summary(trades: list[ClosedTrade], equity: pd.Series, starting_capital: float, annualization_days: int = 252) -> dict:
    df = trades_frame(trades)
    n = len(df)
    out: dict = {"trades": n, "starting_capital": starting_capital}
    final = float(equity.iloc[-1]) if not equity.empty else starting_capital
    out["final_equity"] = final
    out["net_profit"] = final - starting_capital
    out["net_return_pct"] = (final / starting_capital - 1) * 100
    mdd_pct, mdd_amt = max_drawdown(equity)
    out["max_drawdown_pct"], out["max_drawdown_amount"] = mdd_pct, mdd_amt
    if n:
        wins, losses = df[df.pnl > 0], df[df.pnl <= 0]
        gp, gl = wins.pnl.sum(), -losses.pnl.sum()
        out.update(
            win_rate=len(wins) / n, loss_rate=len(losses) / n,
            profit_factor=float(gp / gl) if gl > 0 else float("inf"),
            expectancy=float(df.pnl.mean()), expectancy_r=float(df.r_multiple.mean()),
            r_std=float(df.r_multiple.std(ddof=1)) if n > 1 else 0.0,
            avg_win=float(wins.pnl.mean()) if len(wins) else 0.0, avg_loss=float(losses.pnl.mean()) if len(losses) else 0.0,
            avg_win_r=float(wins.r_multiple.mean()) if len(wins) else 0.0,
            avg_loss_r=float(losses.r_multiple.mean()) if len(losses) else 0.0,
            total_fees=float(df.fees.sum()), avg_bars_held=float(df.bars_held.mean()),
            return_on_risk=float(df.pnl.sum() / df.risk_amount.sum()) if df.risk_amount.sum() > 0 else 0.0,
            exit_reasons=df.exit_reason.value_counts().to_dict(),
        )
    else:
        out.update(win_rate=0.0, loss_rate=0.0, profit_factor=0.0, expectancy=0.0, expectancy_r=0.0, r_std=0.0,
                   avg_win=0.0, avg_loss=0.0, avg_win_r=0.0, avg_loss_r=0.0, total_fees=0.0, avg_bars_held=0.0,
                   return_on_risk=0.0, exit_reasons={})
    daily = equity.resample("1D").last().dropna() if not equity.empty else equity
    rets = daily.pct_change().dropna()
    if len(rets) > 2 and rets.std() > 0:
        out["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(annualization_days))
        downside = rets[rets < 0]
        dsd = np.sqrt((downside ** 2).mean()) if len(downside) else 0.0
        out["sortino"] = float(rets.mean() / dsd * np.sqrt(annualization_days)) if dsd > 0 else float("inf")
    else:
        out["sharpe"] = out["sortino"] = 0.0
    return out


def breakdown(trades: list[ClosedTrade], by: str) -> pd.DataFrame:
    df = trades_frame(trades)
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(by)
    res = pd.DataFrame({
        "trades": g.size(), "win_rate": g["pnl"].apply(lambda x: (x > 0).mean()),
        "expectancy_r": g["r_multiple"].mean(), "net_pnl": g["pnl"].sum(),
        "profit_factor": g["pnl"].apply(lambda x: x[x > 0].sum() / -x[x <= 0].sum() if (x <= 0).any() and x[x <= 0].sum() < 0 else np.inf),
    })
    return res.sort_values("trades", ascending=False)


def bootstrap_expectancy_ci(r: np.ndarray, runs: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    if len(r) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(r, size=(runs, len(r)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
