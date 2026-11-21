"""Opportunity ranking: explainable 0-100 score. A score is NOT a win probability.

Each component is in [0, 1] and documented. Components without data are marked `unavailable`
and given a neutral 0.5 so missing information neither helps nor hides a setup; the
explanation lists them. Weights are configuration, and calibration (score bucket vs realised R)
is computed from history by `calibrate`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantai.models import QualityReport, QualityStatus, SignalCandidate

WEIGHTS = {
    "regime_fit": 0.12, "mtf_alignment": 0.12, "technical_quality": 0.14, "reward_risk": 0.10,
    "historical_expectancy": 0.14, "execution": 0.08, "volatility_fit": 0.06, "macro_alignment": 0.07,
    "event_risk": 0.07, "data_quality": 0.05, "momentum": 0.05,
}


@dataclass
class ScoreBreakdown:
    score: float
    label: str
    components: dict[str, float]
    unavailable: list[str] = field(default_factory=list)
    explanation: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": self.score, "label": self.label, "components": self.components,
                "unavailable": self.unavailable, "explanation": self.explanation}


def label_for(score: float, th: dict[str, float]) -> str:
    if score >= th["exceptional"]:
        return "exceptional (strict validation required)"
    if score >= th["high"]:
        return "high_quality"
    if score >= th["watchlist"]:
        return "watchlist"
    if score >= th["insufficient"]:
        return "insufficient_conviction"
    return "reject"


def score_opportunity(sig: SignalCandidate, feat_row: pd.Series, thresholds: dict[str, float], *,
                      mtf_alignment: float | None = None, validated_expectancy_r: float | None = None,
                      spread: float | None = None, macro_bias: float | None = None,
                      event_risk: float | None = None, quality: QualityReport | None = None,
                      weights: dict[str, float] | None = None) -> ScoreBreakdown:
    w = weights or WEIGHTS
    d = sig.direction.value
    comp: dict[str, float] = {}
    unavailable: list[str] = []
    expl: list[str] = []

    trend_regimes = {"strong_uptrend": 1, "weak_uptrend": 0.5, "strong_downtrend": -1, "weak_downtrend": -0.5}
    rf = trend_regimes.get(sig.regime)
    comp["regime_fit"] = 0.5 + 0.5 * rf * d if rf is not None else 0.6  # strategy already regime-filtered
    if mtf_alignment is None:
        unavailable.append("mtf_alignment"); comp["mtf_alignment"] = 0.5
    else:
        comp["mtf_alignment"] = float(np.clip(0.5 + 0.5 * mtf_alignment * d, 0, 1))
        if mtf_alignment * d < -0.3:
            expl.append("higher timeframes oppose the trade direction")
    comp["technical_quality"] = sig.setup_quality
    comp["reward_risk"] = float(np.clip((sig.reward_risk - 1.0) / 2.0, 0, 1))
    if validated_expectancy_r is None:
        unavailable.append("historical_expectancy"); comp["historical_expectancy"] = 0.5
    else:
        comp["historical_expectancy"] = float(np.clip(0.5 + validated_expectancy_r / 0.6, 0, 1))
    if spread is not None and sig.atr > 0:
        comp["execution"] = float(np.clip(1 - (spread / sig.atr) / 0.25, 0, 1))
    else:
        unavailable.append("execution"); comp["execution"] = 0.5
    rv = feat_row.get("rv_pct")
    comp["volatility_fit"] = 0.5 if rv is None or pd.isna(rv) else float(1 - abs(rv - 0.5) * 1.4)
    if macro_bias is None:
        unavailable.append("macro_alignment"); comp["macro_alignment"] = 0.5
    else:
        comp["macro_alignment"] = float(np.clip(0.5 + 0.5 * macro_bias * d, 0, 1))
    if event_risk is None:
        unavailable.append("event_risk"); comp["event_risk"] = 0.5
    else:
        comp["event_risk"] = float(np.clip(1 - event_risk, 0, 1))
    if quality is None:
        unavailable.append("data_quality"); comp["data_quality"] = 0.5
    else:
        comp["data_quality"] = {QualityStatus.OK: 1.0, QualityStatus.DEGRADED: 0.6, QualityStatus.FAILED: 0.0}[quality.status]
    mom = feat_row.get("mom20")
    comp["momentum"] = 0.5 if mom is None or pd.isna(mom) else float(np.clip(0.5 + d * mom / (4 * max(feat_row.get("atr_pct", 0.01), 1e-4) * np.sqrt(20) / 2), 0, 1))

    total_w = sum(w.values())
    score = round(100 * sum(w[k] * comp[k] for k in w) / total_w, 1)
    top = sorted(comp.items(), key=lambda kv: w[kv[0]] * (kv[1] - 0.5), reverse=True)
    expl.append("strongest: " + ", ".join(f"{k}={v:.2f}" for k, v in top[:3]))
    expl.append("weakest: " + ", ".join(f"{k}={v:.2f}" for k, v in top[-3:]))
    if unavailable:
        expl.append("neutral (no data): " + ", ".join(unavailable))
    return ScoreBreakdown(score, label_for(score, thresholds), {k: round(v, 3) for k, v in comp.items()}, unavailable, expl)


def calibrate(scores: list[float], r_multiples: list[float], bins: tuple[float, ...] = (0, 60, 70, 80, 90, 101)) -> pd.DataFrame:
    """Realised outcome by score bucket. Monotonic increase in expectancy = score carries information."""
    if not scores:
        return pd.DataFrame(columns=["bucket", "n", "win_rate", "expectancy_r"])
    df = pd.DataFrame({"score": scores, "r": r_multiples})
    df["bucket"] = pd.cut(df["score"], bins=list(bins), right=False)
    g = df.groupby("bucket", observed=False)
    return pd.DataFrame({"n": g.size(), "win_rate": g["r"].apply(lambda x: (x > 0).mean() if len(x) else np.nan),
                         "expectancy_r": g["r"].mean()}).reset_index()


def is_monotonic(cal: pd.DataFrame, min_n: int = 20) -> bool | None:
    """True if expectancy rises with score across buckets holding >= min_n trades; None if < 2 such buckets."""
    c = cal[cal["n"] >= min_n]
    if len(c) < 2:
        return None
    return bool(c["expectancy_r"].is_monotonic_increasing)
