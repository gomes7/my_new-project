"""Read-mostly HTTP API + dashboard. Bind to localhost by default; mutations need QUANTAI_ADMIN_TOKEN."""
from __future__ import annotations

import hmac
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from quantai import __version__
from quantai.backtest.metrics import summary
from quantai.config import Secrets
from quantai.journal import analyze
from quantai.models import ClosedTrade, utcnow
from quantai.runtime import Runtime
from quantai.targets import daily_target

STATIC = Path(__file__).parent / "static"


def _clean(o):
    """Make NaN/inf JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
        return None
    return o


class KillReq(BaseModel):
    active: bool
    reason: str


def create_app(rt: Runtime | None = None) -> FastAPI:
    rt = rt or Runtime()
    app = FastAPI(title="QUANT AI", version=__version__, docs_url="/docs")

    @app.get("/")
    def dashboard():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        ok, bad = rt.journal.verify()
        scan = rt.read_json("latest_scan.json") or {}
        snap = rt.read_json("paper_snapshot.json") or {}
        macro = rt.read_json("latest_macro.json") or {}
        stale_scan = None
        if scan.get("generated_at"):
            stale_scan = (utcnow() - pd.Timestamp(scan["generated_at"]).to_pydatetime()).total_seconds() / 3600
        data_status = {m["symbol"]: m.get("quality", {}) for m in scan.get("markets", [])}
        degraded = {s: q for s, q in data_status.items() if any(v != "ok" for v in q.values())}
        return _clean({
            "status": "ok" if ok else "journal_integrity_failure", "version": __version__, "mode": rt.settings.mode,
            "live_trading": "not implemented (disabled)", "kill_switch": rt.kill.status(),
            "journal_integrity": {"ok": ok, "first_bad_seq": bad},
            "last_scan": scan.get("generated_at"), "scan_age_hours": stale_scan,
            "delayed_data_warning": [m["symbol"] for m in scan.get("markets", []) if m.get("delay_status") not in (None, "realtime")],
            "data_status": data_status, "degraded_data": degraded, "api_status": scan.get("api_status", {}),
            "macro_asof": macro.get("asof"), "paper_last_cycle": snap.get("at"),
            "risk_engine": "active", "errors": [e["payload"] for e in rt.journal.entries("alert", limit=50)
                                                if e["payload"].get("severity") == "critical"][:20],
        })

    @app.get("/api/overview")
    def overview():
        snap = rt.read_json("paper_snapshot.json") or {}
        st = rt.settings
        cap = snap.get("equity", st.account.starting_capital)
        bt = rt.read_json("latest_backtest.json") or {}
        exp_r = (bt.get("walk_forward") or {}).get("expectancy_r")
        scan = rt.read_json("latest_scan.json") or {}
        tgt = daily_target(st.account.starting_capital, cap, st.target.daily_target_pct, st.risk.risk_per_trade_pct,
                           st.risk.max_daily_loss_pct, st.target.max_trades_per_day, exp_r,
                           bt.get("trades_per_day"), len(scan.get("opportunities", [])))
        sod = snap.get("start_of_day_equity", cap)
        day_pnl = cap - sod
        budget = sod * st.risk.max_daily_loss_pct / 100
        eq = snap.get("equity_curve", [])
        month_pnl = None
        if eq:
            s = pd.Series([v for _, v in eq], index=pd.to_datetime([t for t, _ in eq], utc=True))
            m = s[s.index >= s.index[-1].normalize().replace(day=1)]
            month_pnl = float(s.iloc[-1] - m.iloc[0]) if len(m) else None
        return _clean({"capital": st.account.starting_capital, "equity": cap, "day_pnl": day_pnl, "month_pnl": month_pnl,
                       "daily_target": tgt.to_dict(), "daily_risk_used": max(0.0, -day_pnl) + snap.get("open_risk", 0.0),
                       "daily_risk_budget": budget, "drawdown_pct": snap.get("drawdown_pct", 0.0),
                       "open_positions": snap.get("open_positions", []), "pending": snap.get("pending", []),
                       "kill_switch": rt.kill.status(), "paper_started": snap.get("started_at")})

    @app.get("/api/scan")
    def get_scan():
        return _clean(rt.read_json("latest_scan.json") or {"message": "no scan yet — run `quantai scan`"})

    @app.get("/api/macro")
    def get_macro():
        return _clean({"macro": rt.read_json("latest_macro.json"), "events": rt.read_json("latest_events.json"),
                       "calendar": rt.read_json("latest_calendar.json")})

    @app.get("/api/performance")
    def performance():
        snap = rt.read_json("paper_snapshot.json") or {}
        trades = [ClosedTrade(**{**t}) for t in snap.get("trades", [])]
        eq = snap.get("equity_curve", [])
        s = pd.Series([v for _, v in eq], index=pd.to_datetime([t for t, _ in eq], utc=True)) if eq else pd.Series(dtype=float)
        paper = summary(trades, s, rt.settings.account.starting_capital) if eq else {}
        return _clean({"paper": paper, "equity_curve": eq, "trades": snap.get("trades", [])[-100:],
                       "backtest": rt.read_json("latest_backtest.json")})

    @app.get("/api/journal")
    def journal(kind: str | None = None, symbol: str | None = None, limit: int = 200):
        return _clean(rt.journal.entries(kind, min(limit, 2000), symbol))

    @app.get("/api/journal/analysis")
    def journal_analysis():
        return _clean(analyze(rt.journal))

    @app.get("/api/alerts")
    def alerts(limit: int = 100):
        return _clean([e["payload"] for e in rt.journal.entries("alert", min(limit, 1000))])

    @app.post("/api/killswitch")
    def killswitch(req: KillReq, x_admin_token: str | None = Header(default=None)):
        token = Secrets.get("ADMIN_TOKEN")
        if not token or not x_admin_token or not hmac.compare_digest(token, x_admin_token):
            raise HTTPException(status_code=403, detail="admin token required (QUANTAI_ADMIN_TOKEN)")
        return rt.kill.activate(req.reason, "api") if req.active else rt.kill.deactivate(req.reason, "api")

    return app
