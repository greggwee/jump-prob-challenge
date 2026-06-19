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

import numpy as np
import requests

from jumpprob.models.devig import consensus, overround
from jumpprob.models.market_anchor import MatchAnchor
from jumpprob.normalize.teams import canonical
from jumpprob.submit.env import load_dotenv

BASE = "https://api.the-odds-api.com/v4"
SPORT = "soccer_fifa_world_cup"  # confirm exact key via list_sports() when live
RAW_DIR = Path("data/raw/odds")


def _key() -> str:
    load_dotenv()
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


def fetch_events(sport: str = SPORT, markets: str = "h2h,totals") -> list[dict]:
    """Live odds for all upcoming games (also caches a snapshot)."""
    resp = requests.get(
        f"{BASE}/sports/{sport}/odds",
        params={"apiKey": _key(), "regions": "eu,uk", "markets": markets, "oddsFormat": "decimal"},
        timeout=60,
    )
    resp.raise_for_status()
    events = resp.json()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RAW_DIR / f"odds_{sport}_{stamp}.json").write_text(json.dumps(events))
    return events


def load_latest(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Most recent cached snapshot (avoids spending a credit when fresh)."""
    snaps = sorted(raw_dir.glob(f"odds_{SPORT}_*.json"))
    if not snaps:
        raise FileNotFoundError("no cached odds snapshot; call fetch_events() first")
    return json.loads(snaps[-1].read_text())


def _books_1x2(event: dict) -> list[tuple[str, list[float]]]:
    home, away = event["home_team"], event["away_team"]
    out = []
    for bm in event.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m["key"] != "h2h":
                continue
            px = {o["name"]: o["price"] for o in m["outcomes"]}
            if home in px and away in px and "Draw" in px:
                out.append((bm["key"], [px[home], px["Draw"], px[away]]))
    return out


def _books_over25(event: dict) -> list[tuple[str, list[float]]]:
    out = []
    for bm in event.get("bookmakers", []):
        for m in bm.get("markets", []):
            if m["key"] != "totals":
                continue
            over = under = None
            for o in m["outcomes"]:
                if o.get("point") == 2.5:
                    if o["name"] == "Over":
                        over = o["price"]
                    elif o["name"] == "Under":
                        under = o["price"]
            if over and under:
                out.append((bm["key"], [over, under]))
    return out


def event_to_anchor(event: dict) -> MatchAnchor | None:
    """One Odds API event -> de-vigged book consensus (wisdom of crowds)."""
    books = _books_1x2(event)
    if not books:
        return None
    cons = consensus(books)  # [P(home), P(draw), P(away)], Shin + sharpness weights
    over_books = _books_over25(event)
    p_over = float(consensus(over_books)[0]) if over_books else None
    return MatchAnchor(
        home=canonical(event["home_team"]), away=canonical(event["away_team"]),
        p_home=float(cons[0]), p_draw=float(cons[1]), p_away=float(cons[2]),
        p_over25=p_over, n_books=len(books),
        avg_overround=float(np.mean([overround(o) for _, o in books])),
    )


def build_anchors(events: list[dict]) -> dict[tuple[str, str], MatchAnchor]:
    """{(canonical_home, canonical_away): MatchAnchor} for all priced games.

    Per-event guard: a single malformed/degenerate quote (e.g. a market locking near
    kickoff that returns decimal odds <= 1.0) must NOT crash the whole submission run —
    the live scheduler calls this path. Skip just that event; the match falls back to
    the pure model (or, if locked, isn't open for submission anyway)."""
    out = {}
    for ev in events:
        try:
            a = event_to_anchor(ev)
        except Exception as e:
            print(f"  ! anchor skipped: {ev.get('home_team', '?')} vs "
                  f"{ev.get('away_team', '?')} ({str(e)[:50]})")
            continue
        if a is not None:
            out[(a.home, a.away)] = a
    return out
