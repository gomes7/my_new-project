import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from quantai.config import RiskCfg, Secrets, Settings
from quantai.logging_setup import redact

ROOT = Path(__file__).resolve().parents[1]


def test_default_risk_parameters_match_spec():
    r = RiskCfg()
    assert r.risk_per_trade_pct == 0.25 and r.max_risk_per_trade_pct == 0.5 and r.absolute_ceiling_pct == 1.0
    assert r.max_daily_loss_pct == 1.0 and r.max_daily_loss_ceiling_pct == 2.0


def test_risk_above_normal_requires_justification():
    with pytest.raises(ValidationError):
        RiskCfg(risk_per_trade_pct=0.8)
    assert RiskCfg(risk_per_trade_pct=0.8, exceptional_justification="documented reason").risk_per_trade_pct == 0.8


def test_risk_ceilings_cannot_be_exceeded():
    with pytest.raises(ValidationError):
        RiskCfg(absolute_ceiling_pct=2.0)
    with pytest.raises(ValidationError):
        RiskCfg(risk_per_trade_pct=1.5, exceptional_justification="x")
    with pytest.raises(ValidationError):
        RiskCfg(max_daily_loss_pct=3.0, max_daily_loss_ceiling_pct=3.0)


def test_live_mode_is_not_a_valid_setting():
    with pytest.raises(ValidationError):
        Settings(mode="live")


def test_live_trading_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("QUANTAI_LIVE_TRADING_AUTHORIZED", raising=False)
    assert Secrets.live_trading_authorized() is False


def test_redaction(monkeypatch):
    monkeypatch.setenv("QUANTAI_POLYGON_API_KEY", "supersecretvalue123")
    out = redact("GET /x?apikey=abc&token=t0k Authorization: Bearer xyz.abc and supersecretvalue123")
    assert "abc&" not in out and "t0k" not in out and "xyz.abc" not in out and "supersecretvalue123" not in out


def test_no_hardcoded_secrets_in_source():
    pat = re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*=\s*['\"][A-Za-z0-9_\-]{12,}['\"]")
    hits = [p for p in (ROOT / "quantai").rglob("*.py") if pat.search(p.read_text())]
    assert hits == []


def test_env_file_is_gitignored():
    assert ".env" in (ROOT / ".gitignore").read_text().split()
