"""Walk-forward optimisation, Monte Carlo, parameter sensitivity, and promotion gates."""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantai.backtest import metrics
from quantai.backtest.engine import prepare, run_backtest
from quantai.config import InstrumentSpec, PromotionCfg, Settings
from quantai.models import ClosedTrade
from quantai.strategies.base import registry


def _objective(trades: list[ClosedTrade], min_trades: int) -> float:
    if len(trades) < min_trades:
        return -np.inf
    r = np.array([t.r_multiple for t in trades])
    # Penalise variance: mean R minus half a standard error (prefers robust, not lucky, params)
    return float(r.mean() - 0.5 * r.std(ddof=1) / np.sqrt(len(r)))


@dataclass
class WalkForwardFold:
    train: tuple[str, str]
    test: tuple[str, str]
    best_params: dict[str, Any]
    train_objective: float
    test_trades: int
    test_expectancy_r: float


@dataclass
class WalkForwardResult:
    strategy: str
    folds: list[WalkForwardFold]
    oos_trades: list[ClosedTrade]
    oos_summary: dict
    param_stability: dict[str, Any] = field(default_factory=dict)


def walk_forward(data: dict[str, pd.DataFrame], timeframe: str, strategy: str, grid: dict[str, list],
                 instruments: dict[str, InstrumentSpec], settings: Settings, n_folds: int = 4,
                 train_frac: float = 0.6, min_train_trades: int = 20, anchored: bool = False) -> WalkForwardResult:
    """Rolling (or anchored) walk-forward. Parameters are chosen on each train window only,
    then frozen and evaluated on the following, unseen test window. OOS trades are concatenated."""
    prepared = {s: prepare(df) for s, df in data.items()}
    idx = sorted(set().union(*[d.index for d in data.values()]))
    t0, t1 = idx[0], idx[-1]
    span = (t1 - t0)
    # Divide timeline so that every fold has a train block followed by a test block.
    test_len = span * (1 - train_frac) / n_folds
    train_len = span * train_frac
    combos = [dict(zip(grid, v)) for v in itertools.product(*grid.values())] or [{}]
    folds, oos = [], []
    for k in range(n_folds):
        test_start = t0 + train_len + k * test_len
        test_end = test_start + test_len
        train_start = t0 if anchored else test_start - train_len
        best, best_obj = combos[0], -np.inf
        for params in combos:
            strat = registry.create(strategy, **params)
            r = run_backtest(data, timeframe, [strat], instruments, settings, trade_start=train_start,
                             trade_end=test_start, prepared=prepared)
            obj = _objective(r.trades, min_train_trades)
            if obj > best_obj:
                best, best_obj = params, obj
        strat = registry.create(strategy, **best)
        r = run_backtest(data, timeframe, [strat], instruments, settings, trade_start=test_start, trade_end=test_end,
                         prepared=prepared)
        # keep only trades opened inside the test window
        tt = [t for t in r.trades if pd.Timestamp(t.entry_time) >= test_start]
        oos += tt
        folds.append(WalkForwardFold((str(train_start), str(test_start)), (str(test_start), str(test_end)), best,
                                     float(best_obj), len(tt), float(np.mean([t.r_multiple for t in tt])) if tt else 0.0))
    rs = np.array([t.r_multiple for t in oos])
    lo, hi = metrics.bootstrap_expectancy_ci(rs, settings.backtest.monte_carlo_runs, seed=settings.backtest.seed)
    pnl = pd.Series([t.pnl for t in oos], index=pd.DatetimeIndex([t.exit_time for t in oos])).sort_index() if oos else pd.Series(dtype=float)
    eq = settings.account.starting_capital + pnl.cumsum()
    summ = metrics.summary(oos, eq, settings.account.starting_capital, settings.backtest.annualization_days)
    summ.update(expectancy_r_ci95=(lo, hi))
    stab = {}
    for p in grid:
        vals = [f.best_params.get(p) for f in folds]
        stab[p] = {"chosen": vals, "unique": len(set(map(str, vals)))}
    return WalkForwardResult(strategy, folds, oos, summ, stab)


@dataclass
class MonteCarloResult:
    runs: int
    final_return_pct_p5: float
    final_return_pct_p50: float
    final_return_pct_p95: float
    max_dd_pct_p50: float
    max_dd_pct_p95: float
    prob_dd_exceeds: dict[str, float]


def monte_carlo(r_multiples: list[float], risk_pct: float, runs: int = 2000, seed: int = 7,
                n_trades: int | None = None, dd_levels: tuple[float, ...] = (10, 20, 30)) -> MonteCarloResult:
    """Bootstrap trade sequences (with replacement) at fixed fractional risk. Risk of ruin is
    reported as P(max drawdown >= level)."""
    r = np.asarray(r_multiples, dtype=float)
    if len(r) == 0:
        return MonteCarloResult(runs, 0, 0, 0, 0, 0, {str(x): float("nan") for x in dd_levels})
    n = n_trades or len(r)
    rng = np.random.default_rng(seed)
    samp = rng.choice(r, size=(runs, n), replace=True)
    growth = 1 + samp * risk_pct / 100
    eq = np.cumprod(growth, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((runs, 1)), eq], axis=1), axis=1)[:, 1:]
    dd = (1 - eq / peak).max(axis=1) * 100
    fin = (eq[:, -1] - 1) * 100
    return MonteCarloResult(runs, *[float(np.quantile(fin, q)) for q in (0.05, 0.5, 0.95)],
                            float(np.quantile(dd, 0.5)), float(np.quantile(dd, 0.95)),
                            {str(x): float((dd >= x).mean()) for x in dd_levels})


def sensitivity(data: dict[str, pd.DataFrame], timeframe: str, strategy: str, base_params: dict[str, Any],
                instruments: dict[str, InstrumentSpec], settings: Settings, rel_steps: tuple[float, ...] = (0.8, 1.2),
                params: list[str] | None = None) -> pd.DataFrame:
    """Re-run with each numeric parameter perturbed. Large swings = fragile (curve-fit) strategy."""
    prepared = {s: prepare(df) for s, df in data.items()}
    base_strat = registry.create(strategy, **base_params)
    base = run_backtest(data, timeframe, [base_strat], instruments, settings, prepared=prepared)
    base_e = float(np.mean([t.r_multiple for t in base.trades])) if base.trades else 0.0
    rows = [{"param": "(base)", "value": None, "trades": len(base.trades), "expectancy_r": base_e, "degradation": 0.0}]
    for p, v in base_strat.params.items():
        if params and p not in params:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v == 0:
            continue
        for k in rel_steps:
            nv = type(v)(round(v * k)) if isinstance(v, int) else v * k
            if nv == v:
                continue
            r = run_backtest(data, timeframe, [registry.create(strategy, **{**base_strat.params, p: nv})], instruments,
                             settings, prepared=prepared)
            e = float(np.mean([t.r_multiple for t in r.trades])) if r.trades else 0.0
            deg = (base_e - e) / abs(base_e) if base_e != 0 else float("nan")
            rows.append({"param": p, "value": nv, "trades": len(r.trades), "expectancy_r": e, "degradation": deg})
    return pd.DataFrame(rows)


@dataclass
class PromotionDecision:
    promote: bool
    stage: str
    passed: list[str]
    failed: list[str]


def promotion_gate(wf: WalkForwardResult, mc: MonteCarloResult, sens: pd.DataFrame | None, cfg: PromotionCfg) -> PromotionDecision:
    """Research -> paper gate. All criteria must pass; profitable backtests alone are insufficient."""
    s = wf.oos_summary
    checks = {
        f"oos_trades>={cfg.min_oos_trades}": s["trades"] >= cfg.min_oos_trades,
        f"oos_profit_factor>={cfg.min_oos_profit_factor}": s["profit_factor"] >= cfg.min_oos_profit_factor,
        f"oos_expectancy_r>={cfg.min_oos_expectancy_r}": s["expectancy_r"] >= cfg.min_oos_expectancy_r,
        f"mc_p95_drawdown<={cfg.max_mc_p95_drawdown_pct}%": mc.max_dd_pct_p95 <= cfg.max_mc_p95_drawdown_pct,
    }
    if cfg.require_bootstrap_lower_bound_positive:
        lo = s.get("expectancy_r_ci95", (float("nan"),))[0]
        checks["bootstrap_ci95_lower>0"] = bool(lo == lo and lo > 0)
    if sens is not None and len(sens) > 1:
        worst = sens["degradation"].iloc[1:].max()
        checks[f"sensitivity_degradation<={cfg.max_sensitivity_degradation}"] = bool(worst == worst and worst <= cfg.max_sensitivity_degradation)
    passed = [k for k, v in checks.items() if v]
    failed = [k for k, v in checks.items() if not v]
    return PromotionDecision(not failed, "paper_candidate" if not failed else "research", passed, failed)
