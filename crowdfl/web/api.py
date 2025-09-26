"""FastAPI endpoints exposing metrics and leaderboard."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from crowdfl.server.credits_ledger import CreditsLedger

STATE_DIR = Path(os.getenv("CROWDFL_STATE_DIR", "state"))
METRICS_PATH = STATE_DIR / "metrics.json"
LEDGER_PATH = STATE_DIR / "credits.json"

app = FastAPI(title="CrowdFL Dashboard API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"]
)

ledger = CreditsLedger(LEDGER_PATH)


def _read_json(path: Path) -> Dict:
    if not path.exists():
        return {
            "round": 0,
            "val_accuracy": 0.0,
            "delta_accuracy": 0.0,
            "processed_packets": 0,
            "target_packets": 0,
        }
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid metrics file: {exc}") from exc


@app.get("/metrics")
def metrics() -> Dict:
    return _read_json(METRICS_PATH)


@app.get("/leaderboard")
def leaderboard() -> Dict:
    return ledger.snapshot()


@app.post("/payouts/mock")
def mock_payouts() -> Dict:
    snapshot = ledger.snapshot()
    ledger.zero_out()
    return {
        "payouts": snapshot.get("leaderboard", []),
        "message": "Mock payouts executed; credits reset.",
    }


__all__ = ["app"]
