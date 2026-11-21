"""Core data models shared across modules."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

import pandas as pd

TF_SECONDS: dict[str, int] = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
TF_ORDER = ["1d", "4h", "1h", "15m", "5m"]  # higher -> lower


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def tf_delta(tf: str) -> timedelta:
    return timedelta(seconds=TF_SECONDS[tf])


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class QualityStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"     # usable for research, flagged
    FAILED = "failed"         # blocks new signals


class DelayStatus(str, Enum):
    REALTIME = "realtime"
    DELAYED = "delayed"
    UNVERIFIED = "unverified_delay"
    HISTORICAL = "historical"   # file / backfill, not a live feed


class Direction(int, Enum):
    LONG = 1
    SHORT = -1


@dataclass
class QualityIssue:
    code: str
    severity: QualityStatus
    detail: str
    count: int = 1


@dataclass
class QualityReport:
    status: QualityStatus
    issues: list[QualityIssue] = field(default_factory=list)
    checked_at: datetime = field(default_factory=utcnow)

    @property
    def blocks_signals(self) -> bool:
        return self.status == QualityStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status.value, "checked_at": self.checked_at.isoformat(),
                "issues": [{**asdict(i), "severity": i.severity.value} for i in self.issues]}


@dataclass
class DataMeta:
    source: str
    provider_symbol: str
    timeframe: str
    retrieved_at: datetime
    delay_status: DelayStatus
    timezone: str = "UTC"
    notes: str = ""


@dataclass
class MarketData:
    """OHLCV bars indexed by bar OPEN time (UTC). Only fully closed bars are kept."""
    symbol: str
    df: pd.DataFrame
    meta: DataMeta
    quality: QualityReport | None = None

    @property
    def timeframe(self) -> str:
        return self.meta.timeframe

    def last_close_time(self) -> datetime | None:
        if self.df.empty:
            return None
        return (self.df.index[-1] + tf_delta(self.timeframe)).to_pydatetime()


@dataclass
class SignalCandidate:
    """A strategy's proposal. Not a trade until the risk engine approves it."""
    symbol: str
    strategy: str
    strategy_version: str
    direction: Direction
    signal_time: datetime          # close time of the bar that produced the signal
    entry_ref: float               # reference price (signal bar close); fill happens next bar
    stop: float
    target: float
    atr: float
    regime: str
    setup_quality: float           # 0..1, strategy's own rule-based quality
    timeframe: str
    max_bars: int = 20
    notes: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("sig"))

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_ref - self.stop)

    @property
    def reward_risk(self) -> float:
        d = self.stop_distance
        return abs(self.target - self.entry_ref) / d if d > 0 else 0.0


@dataclass
class RiskDecision:
    approved: bool
    quantity: float
    risk_amount: float
    risk_pct: float
    reasons: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    id: str
    symbol: str
    strategy: str
    direction: Direction
    quantity: float
    entry_price: float
    entry_time: datetime
    stop: float
    initial_stop: float
    target: float
    risk_amount: float            # account currency at entry
    fx_to_account: float
    regime: str
    signal_id: str
    max_bars: int
    bars_held: int = 0
    partial_taken: bool = False
    realized_pnl: float = 0.0     # from partial exits, account ccy, net of costs
    fees_paid: float = 0.0        # all costs, for reporting
    entry_costs: float = 0.0      # entry commission + holding costs (deducted from cash, not in realized_pnl)
    mfe: float = 0.0              # max favourable excursion (price units)
    mae: float = 0.0
    score: float | None = None


@dataclass
class ClosedTrade:
    id: str
    symbol: str
    strategy: str
    direction: int
    quantity: float
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    pnl: float                     # account currency, net of fees/spread/slippage
    r_multiple: float
    risk_amount: float
    fees: float
    exit_reason: str
    regime: str
    bars_held: int
    mfe_r: float
    mae_r: float
    signal_id: str
    score: float | None = None
