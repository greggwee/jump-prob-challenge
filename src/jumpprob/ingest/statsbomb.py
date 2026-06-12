"""Ingest StatsBomb open data for the six backtest tournaments.

Verified against competitions.json (2026-06-11):
  WC 2018 (43, 3), WC 2022 (43, 106), Euro 2020 (55, 43), Euro 2024 (55, 282),
  Copa America 2024 (223, 282), AFCON 2023 (1267, 107).

Phase 1 pulls match metadata only (scores, dates, referees, stage). Event files
(~5 MB/match) are fetched lazily in Phase 4 and cached.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

TOURNAMENTS = {
    "wc2018": (43, 3),
    "wc2022": (43, 106),
    "euro2020": (55, 43),
    "euro2024": (55, 282),
    "copa2024": (223, 282),
    "afcon2023": (1267, 107),
}

RAW_DIR = Path("data/raw/statsbomb")


def fetch_matches(raw_dir: Path = RAW_DIR, force: bool = False) -> dict[str, Path]:
    """Download matches/{comp}/{season}.json for each tournament."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (comp, season) in TOURNAMENTS.items():
        dest = raw_dir / f"matches_{key}.json"
        if force or not dest.exists():
            url = f"{BASE_URL}/matches/{comp}/{season}.json"
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            meta = {"fetched_at": datetime.now(timezone.utc).isoformat(), "url": url}
            (raw_dir / f"matches_{key}.meta.json").write_text(json.dumps(meta))
        paths[key] = dest
    return paths


def load_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Flatten all tournaments' match metadata into one frame."""
    rows = []
    for key in TOURNAMENTS:
        data = json.loads((raw_dir / f"matches_{key}.json").read_text(encoding="utf-8"))
        for m in data:
            rows.append(
                {
                    "tournament": key,
                    "match_id": m["match_id"],
                    "date": m["match_date"],
                    "kick_off": m.get("kick_off"),
                    "home_team": m["home_team"]["home_team_name"],
                    "away_team": m["away_team"]["away_team_name"],
                    "home_score": m["home_score"],
                    "away_score": m["away_score"],
                    "stage": (m.get("competition_stage") or {}).get("name"),
                    "referee": (m.get("referee") or {}).get("name"),
                    "stadium": (m.get("stadium") or {}).get("name"),
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    expected = {"wc2018": 64, "wc2022": 64, "euro2020": 51, "euro2024": 51, "copa2024": 32, "afcon2023": 52}
    counts = df["tournament"].value_counts().to_dict()
    for t, n in expected.items():
        if counts.get(t) != n:
            raise ValueError(f"{t}: expected {n} matches, got {counts.get(t)}")
    return df.sort_values("date").reset_index(drop=True)


def fetch_events(match_id: int, raw_dir: Path = RAW_DIR) -> list[dict]:
    """Fetch (and cache) the event file for one match. Used from Phase 4 on."""
    cache = raw_dir / "events"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{match_id}.json"
    if not dest.exists():
        resp = requests.get(f"{BASE_URL}/events/{match_id}.json", timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    return json.loads(dest.read_text(encoding="utf-8"))
