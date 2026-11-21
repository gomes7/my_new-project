"""Strategy framework.

Contract (enforced by the backtester and paper engine, identically):
  * `signal_at(feat, i)` may read rows <= i only (feat is a FeatureFrame). The signal bar is CLOSED.
  * Entry is filled at the NEXT bar's open (plus spread/slippage). Stop is a fixed price; the
    target is re-anchored to the real fill so the planned reward:risk is preserved.
  * Strategies never size positions — the risk engine does.
"""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantai.analysis.frame import FeatureFrame, Row

from quantai.models import Direction, SignalCandidate


@dataclass
class FilterResult:
    passed: bool
    failed: list[str] = field(default_factory=list)


class Strategy(ABC):
    name: str = "base"
    version: str = "0.0.0"
    family: str = ""
    description: str = ""
    default_params: dict[str, Any] = {}
    allowed_regimes_long: set[str] = set()
    allowed_regimes_short: set[str] = set()

    def __init__(self, **params: Any):
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {unknown}")
        self.params = {**self.default_params, **params}

    @property
    def fingerprint(self) -> str:
        blob = json.dumps({"name": self.name, "version": self.version, "params": self.params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "family": self.family, "params": self.params,
                "fingerprint": self.fingerprint, "description": self.description,
                "regimes_long": sorted(self.allowed_regimes_long), "regimes_short": sorted(self.allowed_regimes_short)}

    # ---- common filters -------------------------------------------------------------------
    def common_filters(self, r: Row, regime: str, direction: Direction, timeframe: str) -> FilterResult:
        failed = []
        allowed = self.allowed_regimes_long if direction == Direction.LONG else self.allowed_regimes_short
        if regime not in allowed:
            failed.append(f"regime:{regime}")
        lo, hi = self.params.get("min_atr_pct", 0.0), self.params.get("max_atr_pct", 1.0)
        if pd.isna(r.atr_pct) or not (lo <= r.atr_pct <= hi):
            failed.append("volatility")
        if r.volume > 0 and pd.notna(r.rel_volume) and r.rel_volume < self.params.get("min_rel_volume", 0.0):
            failed.append("liquidity")
        hours = self.params.get("blocked_hours_utc", [])
        if timeframe not in ("1d", "4h") and r.name.hour in hours:
            failed.append("time")
        return FilterResult(not failed, failed)

    @abstractmethod
    def raw_signal(self, feat: FeatureFrame, i: int) -> tuple[Direction, float, float, float, dict] | None:
        """Return (direction, stop, reward_risk, quality 0..1, notes) or None."""

    def signal_at(self, feat: FeatureFrame, i: int, symbol: str, regime: str, timeframe: str,
                  bar_seconds: int) -> tuple[SignalCandidate | None, FilterResult | None]:
        raw = self.raw_signal(feat, i)
        if raw is None:
            return None, None
        direction, stop, rr, quality, notes = raw
        r = feat.row(i)
        entry = float(r.close)
        dist = abs(entry - stop)
        if dist <= 0 or (direction == Direction.LONG and stop >= entry) or (direction == Direction.SHORT and stop <= entry):
            return None, FilterResult(False, ["invalid_stop"])
        target = entry + direction.value * rr * dist
        sig = SignalCandidate(
            symbol=symbol, strategy=self.name, strategy_version=f"{self.version}+{self.fingerprint}",
            direction=direction, signal_time=(feat.index[i] + pd.Timedelta(seconds=bar_seconds)).to_pydatetime(),
            entry_ref=entry, stop=float(stop), target=float(target), atr=float(r.atr), regime=regime,
            setup_quality=float(max(0.0, min(1.0, quality))), timeframe=timeframe,
            max_bars=int(self.params.get("max_bars", 20)), notes={**notes, "rr": rr})
        return sig, self.common_filters(r, regime, direction, timeframe)


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, type[Strategy]] = {}

    def register(self, cls: type[Strategy]) -> type[Strategy]:
        self._items[cls.name] = cls
        return cls

    def create(self, name: str, **params: Any) -> Strategy:
        return self._items[name](**params)

    def names(self) -> list[str]:
        return sorted(self._items)


registry = Registry()
