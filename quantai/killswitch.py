"""Kill switch: stops new trades and cancels pending simulated orders. Monitoring continues."""
from __future__ import annotations

import json
from pathlib import Path

from quantai.journal import Journal
from quantai.models import utcnow


class KillSwitch:
    def __init__(self, data_dir: Path, journal: Journal | None = None):
        self.flag = Path(data_dir) / "KILL_SWITCH"
        self.journal = journal

    @property
    def active(self) -> bool:
        return self.flag.exists()

    def activate(self, reason: str, actor: str = "operator") -> dict:
        info = {"active": True, "reason": reason, "actor": actor, "at": utcnow().isoformat()}
        self.flag.write_text(json.dumps(info))
        if self.journal:
            self.journal.append("killswitch", info)
        return info

    def deactivate(self, reason: str, actor: str = "operator") -> dict:
        info = {"active": False, "reason": reason, "actor": actor, "at": utcnow().isoformat()}
        if self.flag.exists():
            self.flag.unlink()
        if self.journal:
            self.journal.append("killswitch", info)
        return info

    def status(self) -> dict:
        if not self.active:
            return {"active": False}
        try:
            return json.loads(self.flag.read_text())
        except (OSError, ValueError):
            return {"active": True, "reason": "unreadable flag file"}
