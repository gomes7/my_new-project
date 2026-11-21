"""Persistent, append-only trading journal with a tamper-evident hash chain.

UPDATE and DELETE are blocked by SQLite triggers; corrections are new entries that reference
the original (kind='correction'). `verify()` recomputes the chain to detect tampering.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quantai.models import utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS journal (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  ref_id TEXT,
  symbol TEXT,
  strategy TEXT,
  payload TEXT NOT NULL,
  prev_hash TEXT NOT NULL,
  hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_journal_kind ON journal(kind);
CREATE INDEX IF NOT EXISTS ix_journal_ref ON journal(ref_id);
CREATE TRIGGER IF NOT EXISTS journal_no_update BEFORE UPDATE ON journal
BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;
CREATE TRIGGER IF NOT EXISTS journal_no_delete BEFORE DELETE ON journal
BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;
"""

KINDS = {"signal", "order", "fill", "partial", "exit", "cancel", "outcome", "note", "lesson", "correction",
         "strategy_version", "promotion", "alert", "killswitch", "system", "scan", "paper_state", "daily_report", "proposal"}


def _default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (datetime, pd.Timestamp)):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _hash(prev: str, ts: str, kind: str, ref: str | None, payload: str) -> str:
    return hashlib.sha256(f"{prev}|{ts}|{kind}|{ref or ''}|{payload}".encode()).hexdigest()


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def append(self, kind: str, payload: Any, ref_id: str | None = None, symbol: str | None = None,
               strategy: str | None = None, ts: datetime | None = None) -> int:
        if kind not in KINDS:
            raise ValueError(f"unknown journal kind {kind}")
        body = json.dumps(payload, default=_default, sort_keys=True)
        t = (ts or utcnow()).isoformat()
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT hash FROM journal ORDER BY seq DESC LIMIT 1").fetchone()
            prev = row[0] if row else "GENESIS"
            h = _hash(prev, t, kind, ref_id, body)
            cur = c.execute("INSERT INTO journal (ts, kind, ref_id, symbol, strategy, payload, prev_hash, hash) "
                            "VALUES (?,?,?,?,?,?,?,?)", (t, kind, ref_id, symbol, strategy, body, prev, h))
            return int(cur.lastrowid)

    def append_many(self, items: Iterable[tuple[str, Any, dict]]) -> int:
        n = 0
        for kind, payload, kw in items:
            self.append(kind, payload, **kw)
            n += 1
        return n

    def entries(self, kind: str | None = None, limit: int | None = None, symbol: str | None = None) -> list[dict]:
        q = "SELECT seq, ts, kind, ref_id, symbol, strategy, payload FROM journal"
        cond, args = [], []
        if kind:
            cond.append("kind=?"); args.append(kind)
        if symbol:
            cond.append("symbol=?"); args.append(symbol)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY seq DESC"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [{"seq": r[0], "ts": r[1], "kind": r[2], "ref_id": r[3], "symbol": r[4], "strategy": r[5],
                 "payload": json.loads(r[6])} for r in rows]

    def latest(self, kind: str) -> dict | None:
        e = self.entries(kind, limit=1)
        return e[0] if e else None

    def verify(self) -> tuple[bool, int | None]:
        """Return (ok, first_bad_seq)."""
        prev = "GENESIS"
        with self._conn() as c:
            for seq, ts, kind, ref, payload, p, h in c.execute(
                    "SELECT seq, ts, kind, ref_id, payload, prev_hash, hash FROM journal ORDER BY seq"):
                if p != prev or _hash(prev, ts, kind, ref, payload) != h:
                    return False, seq
                prev = h
        return True, None


def analyze(journal: Journal) -> dict:
    """Pattern analysis over journaled signals and outcomes."""
    sig = [e["payload"] for e in journal.entries("signal")]
    out = [e["payload"] for e in journal.entries("outcome")]
    res: dict = {"signals": len(sig), "outcomes": len(out)}
    if sig:
        s = pd.DataFrame(sig)
        res["approval_rate"] = float((s["decision"] == "approved").mean())
        # Normalise parametrised reasons, e.g. "score_below_threshold(66.7<80.0)" -> "score_below_threshold"
        reasons = s.explode("reasons")["reasons"].dropna().str.replace(r"\(.*\)$|_until_.*$", "", regex=True)
        res["top_rejection_reasons"] = reasons.value_counts().head(10).to_dict()
    if out:
        o = pd.DataFrame(out)
        o["exit_time"] = pd.to_datetime(o["exit_time"], utc=True)
        res["by_strategy"] = o.groupby("strategy")["r_multiple"].agg(["count", "mean"]).round(3).to_dict("index")
        res["by_regime"] = o.groupby("regime")["r_multiple"].agg(["count", "mean"]).round(3).to_dict("index")
        res["weak_regimes"] = [k for k, v in res["by_regime"].items() if v["count"] >= 10 and v["mean"] < 0]
        per_day = o.groupby(o["exit_time"].dt.date).size()
        res["overtrading_days"] = int((per_day > 3).sum())
        if "mfe_r" in o:
            winners_given_back = o[(o["mfe_r"] >= 1.0) & (o["r_multiple"] <= 0)]
            res["poor_exits_mfe_ge_1R_closed_le_0"] = int(len(winners_given_back))
            res["poor_entries_mae_gt_0.8R_then_win"] = int(((o["mae_r"] > 0.8) & (o["r_multiple"] > 0)).sum())
        res["time_stops"] = int((o["exit_reason"] == "time_stop").sum())
    return res
