"""Understat club-season per-player rates — the player-data foundation (Phase 1).

Public AJAX endpoint `POST /main/getPlayersStats/ {league, season}` returns every
player's season aggregates (games, minutes, shots, xG, npxG, goals, npg, assists,
xA, key_passes). We aggregate across leagues x seasons into per-90 club rates —
the rich signal the tournament-only model lacks. Append-only raw cache with a
fetch timestamp (project discipline); rates derived from the cache.

Coverage caveat (review H1): Understat is top-5 + Russian only. The coverage
census measures how many WC-prediction players this misses; that number gates
the re-backtest and extremizing. FBref (broader leagues) is the next round.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from jumpprob.normalize.players import fold_name

GETSTATS = "https://understat.com/main/getPlayersStats/"
HDR = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Serie_A", "Ligue_1", "RFPL"]
SEASONS = ["2024", "2023"]  # 2024 = 2024/25; last two seasons of signal
RAW = Path("data/raw/understat")
# raw count fields (Understat returns them as strings)
COUNTS = ["games", "time", "goals", "npg", "shots", "assists", "key_passes", "xG", "npxG", "xA"]
# per-90 rate fields we expose to the player model
RATE_FROM = ["shots", "npg", "goals", "assists", "key_passes", "xG", "npxG", "xA"]


def fetch_player_season(league: str, season: str, force: bool = False, raw_dir: Path = RAW) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{league}_{season}.json"
    if force or not dest.exists():
        r = requests.post(GETSTATS, data={"league": league, "season": season}, headers=HDR, timeout=30)
        r.raise_for_status()
        d = r.json()
        if not isinstance(d, dict) or not d.get("success") or "players" not in d:
            raise RuntimeError(f"understat getPlayersStats unexpected payload for {league} {season}")
        dest.write_text(json.dumps(d["players"]))
        (raw_dir / f"{league}_{season}.meta.json").write_text(
            json.dumps({"fetched_at": datetime.now(timezone.utc).isoformat()}))
    return json.loads(dest.read_text(encoding="utf-8"))


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def players_to_rates(players_by_src: list[list[dict]]) -> pd.DataFrame:
    """Pure aggregation (no network) — testable. players_by_src: list of player
    lists (one per league-season). Aggregates per folded name into per-90 rates."""
    rows = []
    for players in players_by_src:
        for p in players:
            rec = {"player_name": p.get("player_name", ""), "position": p.get("position", ""),
                   "team_title": p.get("team_title", "")}
            for k in COUNTS:
                rec[k] = _num(p.get(k))
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["folded"] = df["player_name"].map(fold_name)
    df = df[df["folded"] != ""]
    agg = df.groupby("folded").agg(
        player_name=("player_name", "last"), position=("position", "last"),
        minutes=("time", "sum"), games=("games", "sum"),
        **{k: (k, "sum") for k in RATE_FROM},
    ).reset_index()
    per90 = (agg["minutes"] / 90.0).clip(lower=1e-6)
    for k in RATE_FROM:
        agg[f"{k}_p90"] = agg[k] / per90
    return agg.sort_values("minutes", ascending=False).reset_index(drop=True)


def build_player_rates(leagues=LEAGUES, seasons=SEASONS, force: bool = False) -> pd.DataFrame:
    srcs = [fetch_player_season(lg, sn, force) for lg in leagues for sn in seasons]
    return players_to_rates(srcs)
