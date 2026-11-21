"""News ingestion, classification and event scoring (0-100).

Headlines are treated as UNCONFIRMED information. Direction is a labelled *hypothesis*; the
market's actual reaction is measured separately (intel/reaction.py) and can contradict it.
"""
from __future__ import annotations

import hashlib
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from quantai.data.http import get
from quantai.models import utcnow

MAX_FEED_BYTES = 2_000_000

TAXONOMY: dict[str, dict] = {
    "central_bank": {"kw": ["fed ", "federal reserve", "fomc", "ecb", "boj", "bank of england", "boe", "rate decision",
                            "rate hike", "rate cut", "powell", "lagarde", "monetary policy", "quantitative"],
                     "importance": 0.95, "assets": ["EURUSD", "USDJPY", "GBPUSD", "SPX", "NDX", "GOLD", "BTCUSD"], "duration_h": 72},
    "inflation": {"kw": ["cpi", "inflation", "pce", "price index", "ppi"], "importance": 0.85,
                  "assets": ["EURUSD", "USDJPY", "SPX", "NDX", "GOLD"], "duration_h": 24},
    "employment": {"kw": ["payroll", "nonfarm", "unemployment", "jobless", "jobs report", "labor market"],
                   "importance": 0.8, "assets": ["EURUSD", "USDJPY", "SPX", "GOLD"], "duration_h": 24},
    "growth": {"kw": ["gdp", "pmi", "retail sales", "industrial production", "recession", "consumer confidence"],
               "importance": 0.7, "assets": ["SPX", "NDX", "DAX", "EURUSD"], "duration_h": 24},
    "geopolitics": {"kw": ["war", "military", "missile", "invasion", "sanction", "ceasefire", "troops", "attack",
                           "election", "coup", "tariff", "trade war", "embargo"], "importance": 0.8,
                    "assets": ["GOLD", "WTI", "BRENT", "SPX", "USDJPY", "USDCHF"], "duration_h": 96},
    "energy": {"kw": ["opec", "oil", "crude", "pipeline", "refinery", "natural gas"], "importance": 0.7,
               "assets": ["WTI", "BRENT", "USDCAD"], "duration_h": 48},
    "technology": {"kw": ["semiconductor", "chip", "nvidia", "ai ", "artificial intelligence", "data center", "quantum",
                          "cloud", "cybersecurity", "export control"], "importance": 0.6, "assets": ["NDX", "SPX"],
                   "duration_h": 48},
    "corporate": {"kw": ["earnings", "guidance", "merger", "acquisition", "ceo", "lawsuit", "antitrust", "buyback",
                         "downgrade", "upgrade"], "importance": 0.5, "assets": ["NDX", "SPX"], "duration_h": 24},
    "crypto": {"kw": ["bitcoin", "ethereum", "crypto", "stablecoin", "solana", "sec ", "etf"], "importance": 0.6,
               "assets": ["BTCUSD", "ETHUSD", "SOLUSD"], "duration_h": 48},
    "credit_liquidity": {"kw": ["default", "bank failure", "credit spread", "liquidity", "bailout", "downgrade of",
                                "treasury auction", "repo"], "importance": 0.85, "assets": ["SPX", "GOLD", "USDJPY"],
                         "duration_h": 72},
}
RUMOR_KW = ["reportedly", "sources say", "according to people", "rumor", "rumour", "considering", "could", "may ",
            "unconfirmed", "speculation"]
SCHEDULED_KW = ["scheduled", "calendar", "preview", "expected to", "forecast", "consensus"]
OUTCOME_KW = ["rose", "fell", "came in", "announced", "decided", "raised", "cut", "held", "beat", "missed", "printed"]
MAGNITUDE_KW = ["surprise*", "unexpected*", "emergency", "shock*", "record", "plung*", "surg*", "crash*", "historic",
                "default*", "collaps*", "escalat*"]
POS_KW = ["beat*", "surg*", "rall*", "cut rates", "rate cut*", "stimulus", "ceasefire", "deal", "agreement", "upgrade*", "eases"]
NEG_KW = ["miss*", "plung*", "hike*", "sanction*", "war", "wars", "attack*", "default*", "downgrade*", "tariff*", "recession",
          "collaps*"]


@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published: datetime
    summary: str = ""
    credibility: float = 0.6
    retrieved_at: datetime = field(default_factory=utcnow)

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.source}|{self.title}".encode()).hexdigest()[:16]


@dataclass
class ScoredEvent:
    id: str
    title: str
    source: str
    published: str
    event_time: str | None
    categories: list[str]
    assets: list[str]
    event_type: str                 # breaking | scheduled | rumor | confirmed_outcome | historical
    direction_hypothesis: str       # risk_positive | risk_negative | unclear  (HYPOTHESIS ONLY)
    hypothesis_confidence: float
    volatility_impact: str          # low | medium | high
    confirmation_count: int
    confirmed: bool
    priced_in: str                  # unknown until market reaction is measured
    horizon_hours: int
    score: float
    components: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


_RX_CACHE: dict[str, re.Pattern] = {}


def has_kw(text: str, kw: str) -> bool:
    """Whole-word / whole-phrase match; a trailing '*' marks an intended stem (e.g. 'escalat*')."""
    rx = _RX_CACHE.get(kw)
    if rx is None:
        k = kw.strip()
        body = re.escape(k[:-1]) + r"\w*" if k.endswith("*") else re.escape(k) + r"\b"
        rx = _RX_CACHE[kw] = re.compile(r"\b" + body, re.IGNORECASE)
    return rx.search(text) is not None


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def parse_feed(xml_text: str, source: str, credibility: float = 0.6) -> list[NewsItem]:
    """Parse RSS 2.0 or Atom. External entities are not resolved by ElementTree; size is capped."""
    if len(xml_text) > MAX_FEED_BYTES:
        raise ValueError("feed too large")
    if "<!ENTITY" in xml_text:
        raise ValueError("feed contains entity declarations; refused")
    root = ET.fromstring(xml_text)
    items: list[NewsItem] = []
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        pub = it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date")
        items.append(NewsItem(title, source, (it.findtext("link") or "").strip(), _parse_date(pub),
                              re.sub("<[^>]+>", "", it.findtext("description") or "")[:500], credibility))
    for it in root.findall("a:entry", ns):
        title = (it.findtext("a:title", namespaces=ns) or "").strip()
        link = it.find("a:link", ns)
        pub = it.findtext("a:published", namespaces=ns) or it.findtext("a:updated", namespaces=ns)
        items.append(NewsItem(title, source, link.get("href", "") if link is not None else "", _parse_date(pub),
                              re.sub("<[^>]+>", "", it.findtext("a:summary", namespaces=ns) or "")[:500], credibility))
    return [i for i in items if i.title]


def _parse_date(s: str | None) -> datetime:
    if not s:
        return utcnow()
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return utcnow()
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def fetch_feed(url: str, source: str, credibility: float) -> list[NewsItem]:
    return parse_feed(get(url, timeout=15).text, source, credibility)


def classify(item: NewsItem) -> tuple[list[str], list[str]]:
    text = f" {item.title} {item.summary} ".lower()
    cats, assets = [], []
    for c, d in TAXONOMY.items():
        if any(has_kw(text, k) for k in d["kw"]):
            cats.append(c)
            assets += [a for a in d["assets"] if a not in assets]
    return cats, assets


def event_type(item: NewsItem, now: datetime) -> str:
    t = f" {item.title} ".lower()
    if (now - item.published).total_seconds() > 7 * 86400:
        return "historical"
    if any(has_kw(t, k) for k in RUMOR_KW):
        return "rumor"
    if any(has_kw(t, k) for k in SCHEDULED_KW):
        return "scheduled"
    if any(has_kw(t, k) for k in OUTCOME_KW):
        return "confirmed_outcome"
    return "breaking"


def cluster_confirmations(items: list[NewsItem], threshold: float = 0.45) -> dict[str, int]:
    """Number of DISTINCT sources reporting a similar headline (token Jaccard)."""
    toks = {i.id: _tokens(i.title) for i in items}
    out = {}
    for a in items:
        srcs = {a.source}
        for b in items:
            if b.source in srcs:
                continue
            ta, tb = toks[a.id], toks[b.id]
            if ta and tb and len(ta & tb) / len(ta | tb) >= threshold:
                srcs.add(b.source)
        out[a.id] = len(srcs)
    return out


W = {"credibility": 0.18, "recency": 0.14, "importance": 0.18, "relevance": 0.12, "magnitude": 0.12,
     "confirmation": 0.14, "sensitivity": 0.07, "duration": 0.05}


def score_events(items: list[NewsItem], watchlist: set[str], now: datetime | None = None,
                 half_life_h: float = 6.0) -> list[ScoredEvent]:
    now = now or utcnow()
    conf = cluster_confirmations(items)
    out = []
    for it in items:
        cats, assets = classify(it)
        if not cats:
            continue
        et = event_type(it, now)
        age_h = max(0.0, (now - it.published).total_seconds() / 3600)
        title = f" {it.title} ".lower()
        mag = min(1.0, 0.3 + 0.35 * sum(has_kw(title, k) for k in MAGNITUDE_KW))
        imp = max(TAXONOMY[c]["importance"] for c in cats)
        dur = max(TAXONOMY[c]["duration_h"] for c in cats)
        n_conf = conf.get(it.id, 1)
        comp = {
            "credibility": it.credibility * (0.5 if et == "rumor" else 1.0),
            "recency": math.exp(-math.log(2) * age_h / half_life_h),
            "importance": imp,
            "relevance": min(1.0, len(set(assets) & watchlist) / 3),
            "magnitude": mag,
            "confirmation": min(1.0, (n_conf - 1) / 2) if n_conf > 1 else 0.0,
            "sensitivity": imp,   # prior; replace with measured reaction stats per category (reaction.py)
            "duration": min(1.0, dur / 96),
        }
        score = round(100 * sum(W[k] * comp[k] for k in W) / sum(W.values()), 1)
        pos, neg = sum(has_kw(title, k) for k in POS_KW), sum(has_kw(title, k) for k in NEG_KW)
        hyp = "risk_positive" if pos > neg else "risk_negative" if neg > pos else "unclear"
        out.append(ScoredEvent(
            id=it.id, title=it.title, source=it.source, published=it.published.isoformat(), event_time=None,
            categories=cats, assets=assets, event_type=et, direction_hypothesis=hyp,
            hypothesis_confidence=round(min(0.6, 0.2 + 0.1 * abs(pos - neg)), 2),
            volatility_impact="high" if mag >= 0.65 or imp >= 0.85 else "medium" if imp >= 0.6 else "low",
            confirmation_count=n_conf, confirmed=n_conf >= 2 and et != "rumor", priced_in="unknown",
            horizon_hours=dur, score=score, components={k: round(v, 3) for k, v in comp.items()}))
    return sorted(out, key=lambda e: e.score, reverse=True)


def asset_event_risk(events: list[ScoredEvent], symbol: str, min_score: float = 60) -> float:
    """0..1 event risk for an instrument from recent, relevant, high-scoring events."""
    rel = [e.score for e in events if symbol in e.assets and e.score >= min_score and e.volatility_impact != "low"]
    return min(1.0, max(rel) / 100) if rel else 0.0
