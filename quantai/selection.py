"""Strategy selection: regime -> applicable strategies -> validated performance -> conflict check."""
from __future__ import annotations

from dataclasses import dataclass, field

from quantai.models import SignalCandidate


@dataclass
class StrategyStats:
    """Validated (out-of-sample) statistics for a strategy in a regime."""
    strategy: str
    regime: str
    n_trades: int
    expectancy_r: float
    profit_factor: float


@dataclass
class SelectionResult:
    chosen: SignalCandidate | None
    rejected: list[tuple[SignalCandidate, str]] = field(default_factory=list)
    reason: str = ""


class StrategySelector:
    def __init__(self, stats: list[StrategyStats] | None = None, min_trades: int = 30, require_validated: bool = False):
        self.stats = {(s.strategy, s.regime): s for s in (stats or [])}
        self.min_trades = min_trades
        self.require_validated = require_validated

    def validated_expectancy(self, strategy: str, regime: str) -> float | None:
        s = self.stats.get((strategy, regime))
        if s is None or s.n_trades < self.min_trades:
            return None
        # Shrink towards 0 by sample size (simple James-Stein-like factor).
        k = s.n_trades / (s.n_trades + 50)
        return s.expectancy_r * k

    def select(self, candidates: list[SignalCandidate]) -> SelectionResult:
        if not candidates:
            return SelectionResult(None, reason="no_candidates")
        dirs = {c.direction for c in candidates}
        if len(dirs) > 1:
            return SelectionResult(None, [(c, "strategy_conflict") for c in candidates], "strategies disagree on direction")
        scored = []
        rejected = []
        for c in candidates:
            ev = self.validated_expectancy(c.strategy, c.regime)
            if self.require_validated and (ev is None or ev <= 0):
                rejected.append((c, "no_validated_positive_expectancy_in_regime"))
                continue
            scored.append(((ev if ev is not None else 0.0), c.setup_quality, c))
        if not scored:
            return SelectionResult(None, rejected, "no strategy validated for current regime")
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        best = scored[0][2]
        rejected += [(c, "lower_ranked_same_direction") for _, _, c in scored[1:]]
        return SelectionResult(best, rejected, "selected")
