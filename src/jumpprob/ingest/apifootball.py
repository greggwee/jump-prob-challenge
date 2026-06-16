"""API-Football (api-sports.io) broad-league player rates.

Fills the coverage hole Understat misses: Saudi Pro League, Qatar Stars League,
Liga MX, MLS, Championship, Primeira Liga, Eredivisie (review H1). Two things make
this worth a second source despite the rate limit:
  * it returns shots.on (shots on target) DIRECTLY, which Understat does not — and
    that is exactly the signal the `player_sot_ge` prop needs;
  * it covers leagues outside Understat's top-5 + RFPL.

Free tier = 100 req/day AND ~10 req/min, and /players paginates ~20 players/page,
so crawling whole leagues is unaffordable. Instead, per hole player: resolve their
api-sports id via `/players/profiles?search=<surname>` (no league needed, loose
match — scan all hits for best name overlap), then pull season stats via
`/players?id=<id>&season=` (id+season needs no league). ~2 requests each, throttled
under the per-minute cap with 429 backoff. Append-only raw cache with a fetch
timestamp; rates derived from the cache (pure, testable). Key read from env or .env
as API_FOOTBALL_KEY (never logged).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from jumpprob.normalize.players import fold_name, tokens

BASE = "https://v3.football.api-sports.io"
RAW = Path("data/raw/apifootball")
THROTTLE = 6.5  # free tier is ~10 req/min — stay under it (per-minute, not just /day)
# the leagues that make up our coverage hole (api-sports league ids), for filtering
# a searched player's statistics[] down to the relevant domestic competition
HOLE_LEAGUE_IDS = {
    307: "Saudi Pro League", 305: "Qatar Stars League", 262: "Liga MX",
    253: "MLS", 40: "Championship", 94: "Primeira Liga", 88: "Eredivisie",
    203: "Super Lig", 144: "Jupiler Pro League", 197: "Super League Greece",
    179: "Premiership",  # Scotland
}
# per-90 rate fields exposed to the player model (union with Understat's schema)
RATE_FROM = ["shots", "sot", "goals", "npg", "assists", "key_passes"]


def _key() -> str:
    k = os.environ.get("API_FOOTBALL_KEY")
    if not k:
        envf = Path(".env")
        if envf.exists():
            for line in envf.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("API_FOOTBALL_KEY"):
                    k = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not k:
        raise RuntimeError("API_FOOTBALL_KEY not set (add it to .env)")
    return k


def _get(path: str, params: dict, retries: int = 4) -> dict:
    """Single GET with the api-sports key header. Retries on 429 (per-minute rate
    limit) with backoff. Raises on other HTTP errors and on a plan/quota error in
    the JSON body (so callers fail loudly, not silently empty)."""
    for attempt in range(retries):
        r = requests.get(f"{BASE}{path}", params=params,
                         headers={"x-apisports-key": _key()}, timeout=30)
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 0)) or (THROTTLE * (attempt + 2))
            print(f"  ... 429 rate-limited, waiting {wait:.0f}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        d = r.json()
        errs = d.get("errors")
        # api-sports returns errors as a dict (plan/quota/params) or [] when clean
        if isinstance(errs, dict) and errs:
            # a per-minute-cap message also arrives here on some plans -> back off once
            if any("rate" in str(v).lower() or "minute" in str(v).lower() for v in errs.values()):
                time.sleep(THROTTLE * (attempt + 2))
                continue
            raise RuntimeError(f"api-football error for {path} {params}: {errs}")
        return d
    raise RuntimeError(f"api-football still rate-limited after {retries} retries: {path}")


def _search_token(name: str) -> str:
    """api-sports `search` needs >=4 chars; use the longest name token (usually the
    surname), which is the most distinctive and accent-folded for the URL."""
    toks = sorted(tokens(name), key=len, reverse=True)
    for t in toks:
        if len(t) >= 4:
            return t
    return fold_name(name).replace(" ", "")[:20]


def _profile_full_name(pl: dict) -> str:
    """Best human name from a profile for both matching and storage. Uses
    firstname+lastname (api-sports stores 'R. Mahrez' as `name` but 'Riyad Karim' /
    'Mahrez' as first/last), then appends any distinct `name`-field token — the
    nickname surname markets use (e.g. Trézéguet, whose first/last are 'Mahmoud Ahmed
    Ibrahim'/'Hassan' but whose `name` is 'Trézéguet'). Single-letter abbreviations
    ('R.' -> 'r') are dropped so they don't pollute the clean name."""
    fn, ln = pl.get("firstname"), pl.get("lastname")
    base = f"{fn or ''} {ln or ''}".strip()
    nm = (pl.get("name") or "").strip()
    if not base:
        return nm
    # append the original `name` only if it carries a surname token first/last lack
    new = [t for t in tokens(nm) if t not in tokens(base) and len(t) >= 2]
    return f"{base} {nm}".strip() if new else base


def search_profiles(token: str, force: bool = False, raw_dir: Path = RAW) -> list[dict]:
    """`/players/profiles?search=<token>` — loose surname/firstname substring match,
    no league required. Returns the player dicts. Cached by token. One request."""
    raw_dir = raw_dir / "profiles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{token}.json"
    if force or not dest.exists():
        d = _get("/players/profiles", {"search": token})
        dest.write_text(json.dumps(d.get("response", [])))
        time.sleep(THROTTLE)
    resp = json.loads(dest.read_text(encoding="utf-8"))
    return [h.get("player", h) for h in resp]


def resolve_profile(name: str, force: bool = False, raw_dir: Path = RAW) -> tuple[int | None, str, int]:
    """Find a player's api-sports id by scanning ALL profile hits for the best
    name-token overlap (the target is often buried in a loose-match list). Tries
    query tokens longest-first and stops once a confident (>=2 token) match is found,
    to spend as few requests as possible. Returns (id|None, full_name, overlap)."""
    qtok = tokens(name)
    best, bo = None, 0
    for tk in sorted(qtok, key=len, reverse=True):
        if len(tk) < 4:
            continue
        for pl in search_profiles(tk, force, raw_dir):
            ov = len(qtok & tokens(_profile_full_name(pl)))
            if ov > bo:
                best, bo = pl, ov
        if bo >= 2:  # firstname AND surname matched — confident, stop searching
            break
    # accept >=2 always; accept ==1 only for single-token queries (no more to match)
    if best and (bo >= 2 or (bo >= 1 and len(qtok) <= 1)):
        return best.get("id"), _profile_full_name(best), bo
    return None, "", bo


def fetch_player_stats(player_id: int, season: int, force: bool = False,
                       raw_dir: Path = RAW) -> list[dict]:
    """`/players?id=<id>&season=<season>` — full per-competition statistics for one
    player (id+season needs no league). Cached by id+season. One request."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"id{player_id}_{season}.json"
    if force or not dest.exists():
        d = _get("/players", {"id": player_id, "season": season})
        resp = d.get("response", [])
        dest.write_text(json.dumps(resp[0].get("statistics", []) if resp else []))
        (raw_dir / f"id{player_id}_{season}.meta.json").write_text(json.dumps({
            "fetched_at": datetime.now(timezone.utc).isoformat(), "id": player_id, "season": season}))
        time.sleep(THROTTLE)
    return json.loads(dest.read_text(encoding="utf-8"))


def _num(v) -> float:
    try:
        return float(v) if v is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _primary_stat(stats: list[dict]) -> dict | None:
    """Pick the statistics entry to represent a player's club season: prefer a
    hole-league competition, else the entry with the most minutes (the player's
    main club competition), ignoring national-team cameos when possible."""
    if not stats:
        return None
    hole = [s for s in stats if (s.get("league") or {}).get("id") in HOLE_LEAGUE_IDS]
    pool = hole or stats
    return max(pool, key=lambda s: _num((s.get("games") or {}).get("minutes")))


def players_to_rates_af(payloads: list[dict]) -> pd.DataFrame:
    """Pure aggregation (no network). For each searched payload, resolve the queried
    player among the search hits (best token-set overlap), take their primary club
    competition, and emit per-90 rates. Schema aligns with Understat plus sot_p90."""
    rows = []
    for pl in payloads:
        query = pl.get("query", "")
        qtok = tokens(query)
        best, best_overlap = None, -1
        for hit in pl.get("response", []):
            nm = (hit.get("player") or {}).get("name", "")
            overlap = len(qtok & tokens(nm))
            if overlap > best_overlap:
                best, best_overlap = hit, overlap
        if best is None or best_overlap < 1:
            continue
        st = _primary_stat(best.get("statistics") or [])
        if not st:
            continue
        games, shots, goals = st.get("games") or {}, st.get("shots") or {}, st.get("goals") or {}
        passes, penalty = st.get("passes") or {}, st.get("penalty") or {}
        minutes = _num(games.get("minutes"))
        if minutes <= 0:
            continue
        g, pk = _num(goals.get("total")), _num(penalty.get("scored"))
        rows.append({
            "player_name": (best.get("player") or {}).get("name", query),
            "position": games.get("position", ""),
            "league": (st.get("league") or {}).get("name", ""),
            "minutes": minutes, "games": _num(games.get("appearences")),
            "shots": _num(shots.get("total")), "sot": _num(shots.get("on")),
            "goals": g, "npg": max(g - pk, 0.0),
            "assists": _num(goals.get("assists")), "key_passes": _num(passes.get("key")),
            "source": "apifootball",
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["folded"] = df["player_name"].map(fold_name)
    df = df[df["folded"] != ""]
    per90 = (df["minutes"] / 90.0).clip(lower=1e-6)
    for k in RATE_FROM:
        df[f"{k}_p90"] = df[k] / per90
    return df.sort_values("minutes", ascending=False).reset_index(drop=True)


def build_hole_rates(names: list[str], seasons=(2024, 2023), force: bool = False) -> pd.DataFrame:
    """Fetch club rates for a list of (hole) players via profile-id -> stats. Tries
    each season in order and keeps the first that yields minutes — so a stale
    free-tier season still helps. ~2 requests per resolvable player."""
    payloads = []
    for nm in names:
        try:
            pid, full, ov = resolve_profile(nm, force)
        except Exception as e:  # quota/plan/network — stop the round, keep what we have
            print(f"  ! {nm}: {str(e)[:90]}")
            break
        if not pid:
            print(f"  ? no profile match: {nm} (best overlap {ov})")
            continue
        got = False
        for sn in seasons:
            try:
                stats = fetch_player_stats(pid, sn, force)
            except Exception as e:
                print(f"  ! {nm} id{pid} ({sn}): {str(e)[:90]}")
                break
            if any(_num((s.get("games") or {}).get("minutes")) > 0 for s in stats):
                payloads.append({"query": full or nm,
                                 "response": [{"player": {"name": full or nm}, "statistics": stats}]})
                got = True
                break
        if not got:
            print(f"  ? no minutes any season: {nm} (id {pid})")
    return players_to_rates_af(payloads)
