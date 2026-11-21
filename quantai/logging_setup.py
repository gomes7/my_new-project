"""Structured JSON logging with secret redaction."""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

_KV = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)(\"?\s*[:=]\s*\"?)([^\s\"&,]+)")
_BEARER = re.compile(r"(?i)(bearer\s+)([A-Za-z0-9\-._~+/]+=*)")


def redact(text: str) -> str:
    out = _BEARER.sub(lambda m: m.group(1) + "***REDACTED***", text)
    out = _KV.sub(lambda m: m.group(0) if m.group(3).lower() == "bearer" else m.group(1) + m.group(2) + "***REDACTED***", out)
    # Also redact literal values of any QUANTAI_* secret env var.
    for k, v in os.environ.items():
        if k.startswith("QUANTAI_") and any(s in k for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "WEBHOOK")) and v and len(v) >= 6:
            out = out.replace(v, "***REDACTED***")
    return out


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload["ctx"] = extra
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str))


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(JsonFormatter())
    root.handlers = [h]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
