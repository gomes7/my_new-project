"""Macro regime engine — rule-based on measured FRED series (no subjective LLM judgement).

Every label carries the evidence (value, observation date) that produced it. Thresholds are
explicit and versioned. The asset-class bias mapping at the bottom is a PRIOR (economic
reasoning), not a validated relationship; it receives a small weight in ranking.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from quantai.data.providers import FredProvider
from quantai.models import utcnow

MACRO_VERSION = "1.0.0"
SERIES = {
    "CPIAUCSL": "CPI all items (index, monthly)",
    "CPILFESL": "Core CPI (index, monthly)",
    "PCEPI": "PCE price index (monthly)",
    "UNRATE": "Unemployment rate (monthly)",
    "PAYEMS": "Nonfarm payrolls (thousands, monthly)",
    "INDPRO": "Industrial production (index, monthly)",
    "T10Y2Y": "10y-2y Treasury spread (daily)",
    "DGS10": "10y Treasury yield (daily)",
    "DFF": "Effective fed funds rate (daily)",
    "BAMLH0A0HYM2": "ICE BofA US high-yield OAS (daily)",
    "VIXCLS": "VIX close (daily)",
    "WALCL": "Fed total assets (weekly)",
    "NFCI": "Chicago Fed National Financial Conditions Index (weekly)",
    "DTWEXBGS": "Broad trade-weighted USD index (daily)",
}
TH = dict(sahm_recession=0.5, sahm_warning=0.3, cpi_high=3.0, cpi_stag=3.5, vix_crisis=35, vix_off=25, vix_on=16,
          vix_compress=13, hy_off=5.0, hy_on=4.0, walcl_13w=1.0, nfci_loose=-0.3, nfci_tight=0.0)


@dataclass
class Evidence:
    series: str
    value: float
    date: str
    note: str = ""


@dataclass
class MacroRegime:
    growth: str
    inflation: str
    liquidity: str
    risk: str
    volatility: str
    labels: list[str]
    evidence: list[Evidence] = field(default_factory=list)
    stale_series: list[str] = field(default_factory=list)
    missing_series: list[str] = field(default_factory=list)
    asof: str = ""
    version: str = MACRO_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def bias(self, asset_class: str, symbol: str = "") -> float:
        """Prior directional tilt in [-1, 1]. UNVALIDATED; documented in docs/ARCHITECTURE.md."""
        b = 0.0
        if self.risk == "risk_on":
            b += 0.5 if asset_class in ("index", "crypto", "stock") else 0.0
        elif self.risk == "risk_off":
            b += -0.5 if asset_class in ("index", "crypto", "stock") else 0.0
            if symbol == "GOLD":
                b += 0.3
        if self.liquidity == "liquidity_expansion" and asset_class in ("crypto", "index"):
            b += 0.25
        elif self.liquidity == "liquidity_contraction" and asset_class in ("crypto", "index"):
            b -= 0.25
        return max(-1.0, min(1.0, b))


def _last(s: pd.Series) -> tuple[float, str]:
    return float(s.iloc[-1]), s.index[-1].date().isoformat()


def classify_macro(data: dict[str, pd.Series], max_age_days: dict[str, int] | None = None) -> MacroRegime:
    max_age = max_age_days or {}
    ev: list[Evidence] = []
    stale, missing = [], []
    now = pd.Timestamp(utcnow())

    def get(sid: str) -> pd.Series | None:
        s = data.get(sid)
        if s is None or s.empty:
            missing.append(sid)
            return None
        lim = max_age.get(sid, 75 if sid in ("CPIAUCSL", "CPILFESL", "PCEPI", "UNRATE", "PAYEMS", "INDPRO") else 14)
        if (now - s.index[-1]).days > lim:
            stale.append(sid)
        return s

    growth = inflation = liquidity = risk = vol = "unknown"
    un = get("UNRATE")
    ip = get("INDPRO")
    sahm = None
    if un is not None and len(un) >= 15:
        m3 = un.rolling(3).mean()
        sahm = float(m3.iloc[-1] - m3.iloc[-13:-1].min())
        ev.append(Evidence("UNRATE", *_last(un), note=f"Sahm-rule gap {sahm:.2f}pp"))
    ip_yoy = None
    if ip is not None and len(ip) >= 13:
        ip_yoy = float((ip.iloc[-1] / ip.iloc[-13] - 1) * 100)
        ev.append(Evidence("INDPRO", *_last(ip), note=f"YoY {ip_yoy:.2f}%"))
    if sahm is not None:
        if sahm >= TH["sahm_recession"]:
            growth = "recession"
        elif sahm >= TH["sahm_warning"] or (ip_yoy is not None and ip_yoy < 0):
            growth = "slowing"
        else:
            growth = "expansion"

    cpi = get("CPIAUCSL")
    if cpi is not None and len(cpi) >= 13:
        yoy = float((cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100)
        ann3 = float(((cpi.iloc[-1] / cpi.iloc[-4]) ** 4 - 1) * 100)
        ev.append(Evidence("CPIAUCSL", *_last(cpi), note=f"YoY {yoy:.2f}%, 3m annualised {ann3:.2f}%"))
        trend = "rising" if ann3 > yoy + 0.25 else "falling" if ann3 < yoy - 0.25 else "stable"
        inflation = ("high_" if yoy >= TH["cpi_high"] else "moderate_") + trend

    wal, nf = get("WALCL"), get("NFCI")
    if wal is not None and len(wal) >= 14:
        ch = float((wal.iloc[-1] / wal.iloc[-14] - 1) * 100)
        ev.append(Evidence("WALCL", *_last(wal), note=f"13w change {ch:+.2f}%"))
        liquidity = "liquidity_expansion" if ch >= TH["walcl_13w"] else "liquidity_contraction" if ch <= -TH["walcl_13w"] else "liquidity_neutral"
    if nf is not None:
        v, d = _last(nf)
        ev.append(Evidence("NFCI", v, d, note="<0 = looser than average"))
        if liquidity == "liquidity_neutral" or liquidity == "unknown":
            liquidity = "liquidity_expansion" if v <= TH["nfci_loose"] else "liquidity_contraction" if v > TH["nfci_tight"] else "liquidity_neutral"

    vix, hy = get("VIXCLS"), get("BAMLH0A0HYM2")
    if vix is not None:
        v, d = _last(vix)
        ev.append(Evidence("VIXCLS", v, d))
        vol = "high_volatility_crisis" if v >= TH["vix_crisis"] else "low_volatility_compression" if v <= TH["vix_compress"] else "normal"
    if vix is not None and hy is not None:
        v, h = float(vix.iloc[-1]), float(hy.iloc[-1])
        ev.append(Evidence("BAMLH0A0HYM2", *_last(hy)))
        risk = "risk_off" if (v >= TH["vix_off"] or h >= TH["hy_off"]) else "risk_on" if (v <= TH["vix_on"] and h <= TH["hy_on"]) else "neutral"
    for sid in ("T10Y2Y", "DGS10", "DFF", "DTWEXBGS"):
        s = get(sid)
        if s is not None:
            ev.append(Evidence(sid, *_last(s)))

    labels = []
    if growth == "recession":
        labels.append("recession")
    elif growth in ("slowing",) and inflation.startswith("high"):
        cpi_yoy = next((float(e.note.split()[1].rstrip("%")) for e in ev if e.series == "CPIAUCSL"), 0)
        labels.append("stagflation" if cpi_yoy >= TH["cpi_stag"] else "slowing_growth")
    elif growth == "expansion":
        labels.append("inflationary_expansion" if inflation.endswith("rising") or inflation.startswith("high") else "disinflationary_expansion")
    for x in (risk, liquidity, vol):
        if x not in ("unknown", "neutral", "normal", "liquidity_neutral"):
            labels.append(x)
    asof = max((e.date for e in ev), default="")
    return MacroRegime(growth, inflation, liquidity, risk, vol, labels, ev, stale, missing, asof)


def fetch_macro(provider: FredProvider | None = None, start: str = "2015-01-01") -> dict[str, pd.Series]:
    p = provider or FredProvider()
    out = {}
    for sid in SERIES:
        try:
            out[sid] = p.series(sid, start=start)
        except Exception:  # noqa: BLE001 — missing series are reported, never fabricated
            out[sid] = pd.Series(dtype=float)
    return out
