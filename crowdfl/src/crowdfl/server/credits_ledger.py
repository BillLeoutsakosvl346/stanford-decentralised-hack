"""Simple JSON-backed credits ledger."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List


@dataclass
class CreditsLedger:
    path: Path
    lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"credits": {}, "history": [], "updated_at": None})

    def _read(self) -> Dict:
        if not self.path.exists():
            return {"credits": {}, "history": [], "updated_at": None}
        with self.path.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _write(self, payload: Dict) -> None:
        with self.path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

    def credit(self, client_id: str, amount: float, round_idx: int | None = None) -> None:
        if amount <= 0:
            return
        with self.lock:
            data = self._read()
            credits = data.setdefault("credits", {})
            credits[client_id] = float(credits.get(client_id, 0.0) + amount)
            history: List[Dict] = data.setdefault("history", [])
            history.append(
                {
                    "client_id": client_id,
                    "amount": float(amount),
                    "round": round_idx,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write(data)

    def zero_out(self) -> None:
        with self.lock:
            data = self._read()
            data["credits"] = {}
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write(data)

    def snapshot(self) -> Dict:
        with self.lock:
            data = self._read()
        credits = data.get("credits", {})
        leaderboard = sorted(credits.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "leaderboard": [
                {"client": client_id, "credit": float(amount)} for client_id, amount in leaderboard
            ],
            "updated_at": data.get("updated_at"),
            "total_credited": float(sum(credits.values())),
        }


__all__ = ["CreditsLedger"]
