"""The Odds API client for live WC 2026 odds (h2h, totals, BTTS where offered).

Requires THE_ODDS_API_KEY in the environment (free tier: 500 credits/month).
Snapshots are stored append-only with fetch timestamps (point-in-time
discipline) under data/raw/odds/.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.the-odds-api.com/v4"
SPORT = "soccer_fifa_world_cup"  # confirm exact key via list_sports() when live
RAW_DIR = Path("data/raw/odds")


def _key() -> str:
    key = os.environ.get("THE_ODDS_API_KEY")
    if not key:
        raise RuntimeError("set THE_ODDS_API_KEY (free key: the-odds-api.com)")
    return key


def list_sports() -> list[dict]:
    resp = requests.get(f"{BASE}/sports", params={"apiKey": _key()}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def snapshot_odds(sport: str = SPORT, markets: str = "h2h,totals", raw_dir: Path = RAW_DIR) -> Path:
    """Fetch current odds for all upcoming matches; append a timestamped snapshot."""
    resp = requests.get(
        f"{BASE}/sports/{sport}/odds",
        params={"apiKey": _key(), "regions": "eu", "markets": markets, "oddsFormat": "decimal"},
        timeout=60,
    )
    resp.raise_for_status()
    raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = raw_dir / f"odds_{sport}_{stamp}.json"
    dest.write_text(json.dumps(resp.json()))
    return dest
