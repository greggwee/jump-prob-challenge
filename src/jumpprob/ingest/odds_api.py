"""The Odds API client for live WC 2026 odds (h2h, totals, BTTS where offered).

Requires THE_ODDS_API_KEY in the environment (free tier: 500 credits/month).
Snapshots are stored append-only with fetch timestamps (point-in-time
discipline) under data/raw/odds/.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
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


# --- player props (verified live: player_shots_on_target Over 0.5 = "1+ SoT", -------
#     player_goal_scorer_anytime Yes = "scores", player_assists Over 0.5) ------------
PLAYER_MARKETS = "player_shots_on_target,player_goal_scorer_anytime,player_assists"


# Latin letters that do NOT decompose under NFKD, so a plain ascii-encode DROPS them
# ('Ødegaard'->'degaard') while the books transliterate them ('odegaard'). Map them the way
# the feeds do so the join survives — this was the Sørloth/Ødegaard miss class.
_TRANSLIT = str.maketrans({
    "ø": "o", "Ø": "o", "å": "a", "Å": "a", "æ": "ae", "Æ": "ae",
    "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ð": "d", "Ð": "d",
    "þ": "th", "Þ": "th", "ß": "ss", "ı": "i", "œ": "oe", "Œ": "oe",
})


def norm_player(name: str) -> str:
    """Transliterate + accent/punct-strip to lowercase, for joining a question subject to a
    prop name. Drops a trailing team qualifier ('Kai Havertz (Germany)'). The _TRANSLIT pass
    handles non-decomposing Latin letters (ø/æ/å/ł/đ...) that NFKD+ascii would otherwise drop."""
    s = re.sub(r"\(.*?\)", "", str(name))
    s = unicodedata.normalize("NFKD", s.translate(_TRANSLIT)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).strip()


def _first_match(a: str, b: str) -> bool:
    """First names align: equal, or one is an initial/prefix of the other ('e' ~ 'erling')."""
    return a == b or a.startswith(b) or b.startswith(a)


# Nickname/registered-name aliases the token matcher cannot bridge (platform name ->
# the book's normalized key). Populated from the submit-time name-join audit as new
# cases surface; keep entries normalized (lowercase, ascii).
_ALIASES = {
    "mahmoud trezeguet": "mahmoud hassan",   # Trezeguet = nickname; books list Mahmoud Hassan
    "trezeguet": "mahmoud hassan",
}


def resolve_player(subject: str, props: dict) -> dict:
    """Join a question subject ('Erling Haaland') to a book prop entry, robust to the book
    carrying middle names ('erling braut haaland') or initials ('e haaland') — the exact
    norm_player key often misses ('erling haaland' != 'erling braut haaland'), which silently
    drops the anchor and leaves the player on the shrinkage model. Exact key wins; then a
    NICKNAME alias (Trezeguet -> mahmoud hassan); otherwise require the LAST name to match
    and the FIRST names to align; 0 or AMBIGUOUS (>1) matches -> {} so we fall back to the
    model rather than guess the wrong player."""
    if not props:
        return {}
    key = norm_player(subject)
    if key in props:
        return props[key]
    alias = _ALIASES.get(key)
    if alias and alias in props:
        return props[alias]
    toks = key.split()
    if len(toks) < 2:
        return {}                       # single token (or empty) -> too risky to fuzzy-match
    first, last = toks[0], toks[-1]
    cands = [v for k, v in props.items()
             if (kt := k.split()) and len(kt) >= 2 and kt[-1] == last and _first_match(first, kt[0])]
    return cands[0] if len(cands) == 1 else {}


def _two_way(books: list[tuple[str, list[float]]]) -> float | None:
    """books [(name, [over, under]), ...] -> de-vigged P(over) consensus, or None."""
    books = [b for b in books if len(b[1]) == 2 and all(o > 1.0 for o in b[1])]
    if not books:
        return None
    try:
        return float(consensus(books)[0])
    except Exception:
        return None


def _over_prob(books: list[tuple[str, list[float]]]) -> float | None:
    """P(Over) from SoT/assist books: prefer a two-way (Over/Under) Shin de-vig; fall
    back to a vig-haircut on ONE-SIDED (Over-only) books — many books quote only the
    Over for the 0.5 line, which the two-way path silently dropped (the SoT-coverage bug)."""
    two = _two_way([b for b in books if len(b[1]) == 2])
    if two is not None:
        return two
    one = [b for b in books if len(b[1]) == 1 and b[1][0] > 1.0]
    if one:
        return float(np.clip(np.mean([1.0 / b[1][0] for b in one]) / 1.08, 0.01, 0.97))
    return None


def _parse_player_event(event: dict) -> dict:
    """One event's player markets -> {norm_player: {archetype: prob}} (de-vigged)."""
    sot, ast, scorer = {}, {}, {}
    for bm in event.get("bookmakers", []):
        bk = bm["key"]
        for m in bm.get("markets", []):
            k = m["key"]
            if k in ("player_shots_on_target", "player_assists"):
                byp: dict = {}
                for o in m["outcomes"]:
                    if o.get("point") == 0.5 and o["name"] in ("Over", "Under"):
                        byp.setdefault(norm_player(o.get("description", "")), {})[o["name"]] = o["price"]
                tgt = sot if k == "player_shots_on_target" else ast
                for pl, ou in byp.items():
                    if "Over" in ou and "Under" in ou:
                        tgt.setdefault(pl, []).append((bk, [ou["Over"], ou["Under"]]))
                    elif "Over" in ou:
                        tgt.setdefault(pl, []).append((bk, [ou["Over"]]))  # one-sided (Over-only)
            elif k == "player_goal_scorer_anytime":
                byp = {}
                for o in m["outcomes"]:
                    if o["name"] in ("Yes", "No"):
                        byp.setdefault(norm_player(o.get("description", "")), {})[o["name"]] = o["price"]
                for pl, yn in byp.items():
                    if "Yes" in yn and "No" in yn:
                        scorer.setdefault(pl, []).append((bk, [yn["Yes"], yn["No"]]))
                    elif "Yes" in yn:
                        scorer.setdefault(pl, []).append((bk, [yn["Yes"]]))   # one-sided
    out: dict = {}
    for pl in set(sot) | set(ast) | set(scorer):
        d: dict = {}
        p_sot = _over_prob(sot.get(pl, []))
        if p_sot is not None:
            d["player_sot_ge"] = p_sot
        sc = scorer.get(pl, [])
        two = [b for b in sc if len(b[1]) == 2]
        if two:
            p_sc = _two_way(two)
        elif sc:                       # only "Yes" quoted: 1/price with a small vig haircut
            p_sc = float(np.clip(np.mean([1.0 / b[1][0] for b in sc]) / 1.08, 0.01, 0.95))
        else:
            p_sc = None
        if p_sc is not None:
            d["player_scores"] = p_sc
        p_ast = _over_prob(ast.get(pl, []))
        if p_sc is not None or p_ast is not None:   # score OR assist (independence approx)
            d["player_ga"] = float(1.0 - (1.0 - (p_sc or 0.0)) * (1.0 - (p_ast or 0.0)))
        if d:
            out[pl] = d
    return out


def fetch_player_props(sport: str = SPORT, regions: str = "eu,uk", hours_ahead: float = 36.0) -> dict:
    """{(canonical_home, canonical_away): {norm_player: {archetype: prob}}}, cached.

    Per-event (player markets aren't on the bulk endpoint), so this costs ~markets x
    regions credits PER event -- callers should fetch on the full daily run and reuse
    load_latest_props() for the per-match scheduler ticks."""
    key = _key()
    evs = requests.get(f"{BASE}/sports/{sport}/events", params={"apiKey": key}, timeout=30).json()
    now = datetime.now(timezone.utc)
    out: dict = {}
    for ev in evs if isinstance(evs, list) else []:
        try:
            ko = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
            if (ko - now).total_seconds() / 3600.0 > hours_ahead:
                continue                       # too far out — props thin, credits scarce
            r = requests.get(
                f"{BASE}/sports/{sport}/events/{ev['id']}/odds",
                params={"apiKey": key, "regions": regions, "markets": PLAYER_MARKETS, "oddsFormat": "decimal"},
                timeout=60,
            )
            if r.status_code != 200:
                continue
            props = _parse_player_event(r.json())
        except Exception:
            continue
        if props:
            out[(canonical(ev["home_team"]), canonical(ev["away_team"]))] = props
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RAW_DIR / f"props_{sport}_{stamp}.json").write_text(
        json.dumps({f"{h}|{a}": v for (h, a), v in out.items()}))
    return out


def load_latest_props(sport: str = SPORT) -> dict:
    """Most recent cached player-prop snapshot (reuse to avoid re-spending credits)."""
    snaps = sorted(RAW_DIR.glob(f"props_{sport}_*.json"))
    if not snaps:
        return {}
    ser = json.loads(snaps[-1].read_text())
    return {tuple(k.split("|", 1)): v for k, v in ser.items()}


def refresh_props_targeted(pairs: list[tuple[str, str]], regions: str = "eu,uk",
                           sport: str = SPORT) -> dict:
    """Refresh player props for ONLY the given (canonical_home, canonical_away) pairs and
    MERGE into the latest snapshot. The near-kickoff, low-credit refresh: SoT/scorer lines
    publish close to KO, so the scheduler can top up just the imminent match(es) without
    re-fetching the whole board. Returns the merged snapshot; writes it so the cached path
    (load_latest_props) picks it up before the live submit."""
    key = _key()
    evs = requests.get(f"{BASE}/sports/{sport}/events", params={"apiKey": key}, timeout=30).json()
    want = {frozenset(p) for p in pairs}
    merged = load_latest_props()
    changed = False
    for ev in evs if isinstance(evs, list) else []:
        ch, ca = canonical(ev["home_team"]), canonical(ev["away_team"])
        if frozenset([ch, ca]) not in want:
            continue
        try:
            r = requests.get(
                f"{BASE}/sports/{sport}/events/{ev['id']}/odds",
                params={"apiKey": key, "regions": regions, "markets": PLAYER_MARKETS, "oddsFormat": "decimal"},
                timeout=60)
            if r.status_code != 200:
                continue
            props = _parse_player_event(r.json())
        except Exception:
            continue
        if props:
            merged[(ch, ca)] = props
            changed = True
    if changed:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (RAW_DIR / f"props_{sport}_{stamp}.json").write_text(
            json.dumps({f"{h}|{a}": v for (h, a), v in merged.items()}))
    return merged
