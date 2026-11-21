"""Shared runtime services: paths, journal, kill switch, cached data fetching."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from quantai.config import InstrumentSpec, Settings, get_settings
from quantai.data.http import ProviderError
from quantai.data.providers import MarketDataProvider, build_provider
from quantai.data.store import BarStore
from quantai.data.validation import compare_providers, validate
from quantai.journal import Journal
from quantai.killswitch import KillSwitch
from quantai.logging_setup import get_logger
from quantai.models import MarketData, QualityIssue, QualityReport, QualityStatus, utcnow

log = get_logger("quantai.runtime")


class Runtime:
    def __init__(self, settings: Settings | None = None, data_dir: Path | None = None):
        self.settings = settings or get_settings()
        self.data_dir = Path(data_dir) if data_dir else self.settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.data_dir / "journal.db")
        self.bars = BarStore(self.data_dir / "bars.db")
        self.kill = KillSwitch(self.data_dir, self.journal)
        self._providers: dict[str, MarketDataProvider] = {}
        self.api_status: dict[str, dict] = {}

    def provider(self, name: str) -> MarketDataProvider:
        if name not in self._providers:
            self._providers[name] = build_provider(name, self.data_dir / "csv")
        return self._providers[name]

    def fetch(self, spec: InstrumentSpec, tf: str, live: bool = True, crosscheck: bool = False) -> MarketData:
        """Fetch + validate. On provider failure returns an EMPTY, FAILED MarketData — never cached
        bars masquerading as live data."""
        prov = self.provider(spec.provider)
        try:
            md = prov.get_bars(spec, tf)
            self.api_status[spec.provider] = {"ok": True, "at": utcnow().isoformat()}
            self.bars.save(md)
        except ProviderError as e:
            self.api_status[spec.provider] = {"ok": False, "at": utcnow().isoformat(), "error": str(e)}
            log.error("provider failure", extra={"ctx": {"symbol": spec.symbol, "tf": tf, "err": str(e)}})
            md = self.bars.load(spec.symbol, tf, spec.provider)
            md.meta.notes = f"provider failed ({e}); cached bars shown for reference only"
            q = validate(md, self.settings.data, spec.asset_class, live=live)
            q.issues.append(QualityIssue("api_failure", QualityStatus.FAILED, str(e)))
            q.status = QualityStatus.FAILED
            md.quality = q
            return md
        validate(md, self.settings.data, spec.asset_class, live=live)
        if crosscheck and spec.crosscheck and not md.quality.blocks_signals:
            self._crosscheck(spec, tf, md)
        return md

    def _crosscheck(self, spec: InstrumentSpec, tf: str, md: MarketData) -> None:
        """Compare against an independent provider; a FAILED discrepancy blocks signals, an unavailable
        second source only degrades (the primary data itself passed validation)."""
        cc = spec.crosscheck or {}
        try:
            other = self.provider(cc["provider"]).get_bars(spec, tf, provider_symbol=cc["provider_symbol"])
        except ProviderError as e:
            md.quality.issues.append(QualityIssue("crosscheck_unavailable", QualityStatus.DEGRADED, str(e)))
            if md.quality.status == QualityStatus.OK:
                md.quality.status = QualityStatus.DEGRADED
            return
        rep = compare_providers(md, other, self.settings.data.provider_discrepancy_pct, window=60)
        md.quality.issues += rep.issues
        rank = {QualityStatus.OK: 0, QualityStatus.DEGRADED: 1, QualityStatus.FAILED: 2}
        if rank[rep.status] > rank[md.quality.status]:
            md.quality.status = rep.status

    def write_json(self, name: str, obj: dict) -> Path:
        p = self.data_dir / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj, default=str, indent=1))
        tmp.replace(p)
        return p

    def read_json(self, name: str) -> dict | None:
        p = self.data_dir / name
        return json.loads(p.read_text()) if p.exists() else None


def quality_summary(q: QualityReport | None) -> dict:
    return {"status": "unknown"} if q is None else q.to_dict()


def iso(t: datetime | None) -> str | None:
    return t.isoformat() if t else None
