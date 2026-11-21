"""Economic calendar: interface + YAML-backed provider; produces risk blackout windows."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from quantai.models import utcnow

CCY_ASSETS = {
    "USD": None,  # affects everything priced in / against USD
    "EUR": ["EURUSD", "EURJPY", "DAX", "CAC40"], "GBP": ["GBPUSD", "GBPJPY", "FTSE"],
    "JPY": ["USDJPY", "EURJPY", "GBPJPY"], "CHF": ["USDCHF"], "AUD": ["AUDUSD"], "CAD": ["USDCAD", "WTI"],
}


@dataclass
class CalendarEvent:
    title: str
    time: datetime
    currency: str
    impact: str
    source: str = ""

    def affects(self, symbol: str) -> bool:
        assets = CCY_ASSETS.get(self.currency)
        if assets is None:
            return True if self.currency == "USD" else False
        return symbol in assets


class YamlCalendar:
    def __init__(self, path: Path):
        self.path = Path(path)

    def events(self) -> list[CalendarEvent]:
        if not self.path.exists():
            return []
        raw = yaml.safe_load(self.path.read_text()) or {}
        out = []
        for e in raw.get("events") or []:
            t = e["time"] if isinstance(e["time"], datetime) else datetime.fromisoformat(str(e["time"]).replace("Z", "+00:00"))
            out.append(CalendarEvent(e["title"], t, e.get("currency", "USD"), e.get("impact", "high"), e.get("source", "")))
        return out

    def upcoming(self, hours: int = 48, now: datetime | None = None) -> list[CalendarEvent]:
        n = now or utcnow()
        return sorted([e for e in self.events() if n <= e.time <= n + timedelta(hours=hours)], key=lambda e: e.time)


def blackout_reason(events: list[CalendarEvent], symbol: str, now: datetime, minutes: int) -> str | None:
    for e in events:
        if e.impact == "high" and e.affects(symbol) and abs((e.time - now).total_seconds()) <= minutes * 60:
            return f"{e.title}@{e.time.isoformat()}"
    return None
