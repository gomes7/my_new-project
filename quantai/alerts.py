"""Configurable alerting with de-duplication / cooldown."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime

import httpx

from quantai.config import AlertsCfg, Secrets
from quantai.journal import Journal
from quantai.logging_setup import get_logger, redact
from quantai.models import utcnow

log = get_logger("quantai.alerts")

CATEGORIES = {"opportunity", "setup_approaching", "setup_confirmed", "setup_invalidated", "stop_loss", "take_profit",
              "position_risk", "risk_limit_approaching", "daily_loss_limit", "major_event", "breaking_news",
              "abnormal_volatility", "data_feed", "api_error", "system", "performance"}


@dataclass
class Alert:
    category: str
    situation: str
    reason: str
    symbol: str | None = None
    price: float | None = None
    risk: str | None = None
    action: str = "none"
    severity: str = "info"         # info | warning | critical
    timestamp: datetime = field(default_factory=utcnow)

    def text(self) -> str:
        parts = [f"[{self.severity.upper()}] {self.timestamp:%Y-%m-%d %H:%M:%SZ} {self.category}"]
        if self.symbol:
            parts.append(self.symbol)
        parts.append(f"{self.situation} — {self.reason}")
        if self.price is not None:
            parts.append(f"price={self.price:g}")
        if self.risk:
            parts.append(f"risk={self.risk}")
        parts.append(f"action={self.action}")
        return " | ".join(parts)


class AlertManager:
    def __init__(self, cfg: AlertsCfg, journal: Journal | None = None, webhook_url: str | None = None):
        self.cfg = cfg
        self.journal = journal
        self.webhook_url = webhook_url if webhook_url is not None else Secrets.get("ALERT_WEBHOOK_URL")
        self._last: dict[tuple, float] = {}
        self.sent: list[Alert] = []

    def emit(self, a: Alert) -> bool:
        if a.category not in CATEGORIES:
            raise ValueError(f"unknown alert category {a.category}")
        if self.cfg.enabled_categories and a.category not in self.cfg.enabled_categories and a.severity != "critical":
            return False
        key = (a.category, a.symbol, a.situation)
        now = time.monotonic()
        if a.severity != "critical" and now - self._last.get(key, -1e18) < self.cfg.cooldown_seconds:
            return False
        self._last[key] = now
        self.sent.append(a)
        log.warning(a.text()) if a.severity != "info" else log.info(a.text())
        if self.journal:
            self.journal.append("alert", asdict(a), symbol=a.symbol)
        if self.webhook_url:
            try:
                httpx.post(self.webhook_url, json={"text": a.text(), **{k: str(v) for k, v in asdict(a).items()}}, timeout=5)
            except httpx.HTTPError as e:
                log.error("alert webhook failed", extra={"ctx": {"err": redact(str(e))}})
        return True
