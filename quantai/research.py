"""Research assistant: turns validation and journal evidence into PROPOSALS.

Proposals are journaled with stage 'proposed'. Nothing here changes configuration or activates a
strategy. A proposal advances only through: review -> implementation -> unit tests -> backtest ->
out-of-sample -> robustness -> paper trading -> explicit promotion (see backtest/robustness.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

PIPELINE = ["proposed", "reviewed", "implemented", "unit_tested", "backtested", "out_of_sample",
            "robustness", "paper", "promoted"]


@dataclass
class Proposal:
    strategy: str
    kind: str          # retire | redesign | filter | data | risk | hypothesis
    evidence: str
    suggestion: str
    stage: str = "proposed"

    def to_dict(self) -> dict:
        return asdict(self)


def suggest(validation: dict, journal_analysis: dict | None = None) -> list[Proposal]:
    out: list[Proposal] = []
    for name, v in (validation.get("strategies") or {}).items():
        lo, hi = (v.get("ci95") or (None, None))
        n = v.get("oos_trades", 0)
        if n < 30:
            out.append(Proposal(name, "redesign", f"only {n} OOS trades",
                                "Entry conditions too restrictive for this timeframe; test on a lower timeframe or "
                                "relax one condition at a time as a NEW version (do not tune on the same OOS data)."))
        elif hi is not None and hi < 0:
            out.append(Proposal(name, "retire", f"OOS expectancy CI entirely negative ({lo:.3f}..{hi:.3f}R)",
                                "Retire this version from the research universe; keep it journaled for reference."))
        elif lo is not None and lo <= 0:
            out.append(Proposal(name, "hypothesis", f"OOS expectancy CI includes zero ({lo:.3f}..{hi:.3f}R)",
                                "No demonstrated edge. Gather more OOS data or test a structurally different variant."))
        stab = v.get("param_stability") or {}
        unstable = [p for p, s in stab.items() if s.get("unique", 1) >= 3]
        if unstable:
            out.append(Proposal(name, "hypothesis", f"walk-forward chose different values for {unstable}",
                                "Parameter instability: prefer a fixed, theory-motivated value or a volatility-normalised rule."))
        fl = v.get("folds", [])
        folds = [f.get("test_expectancy_r", 0) for f in fl]
        enough = all(f.get("test_trades", 10) >= 10 for f in fl)   # ignore folds too small to mean anything
        if enough and len(folds) >= 3 and folds[-1] < 0 and folds[-2] < 0 and max(folds[:-2], default=0) > 0:
            out.append(Proposal(name, "filter", "expectancy positive early, negative in the two latest folds",
                                "Possible regime decay: test a regime/volatility filter as a new version."))
    if journal_analysis:
        for reg in journal_analysis.get("weak_regimes", []):
            out.append(Proposal("*", "filter", f"negative realised expectancy in regime '{reg}' (n>=10)",
                                f"Evaluate excluding '{reg}' for the affected strategies as a new version."))
        if journal_analysis.get("poor_exits_mfe_ge_1R_closed_le_0", 0) >= 5:
            out.append(Proposal("*", "hypothesis", "several trades reached +1R then closed at or below 0",
                                "Test exit variants (trailing stop, earlier partial) out-of-sample."))
    return out
