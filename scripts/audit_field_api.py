"""Phase A: audit the SportsPredict API for ANY field / crowd / leaderboard signal.

If per-question field data (field-average Brier, awarded RBP, crowd consensus,
leaderboard distribution) exists anywhere, it dominates every proxy and we pivot
to using it directly. This dumps the full field set of every known object and
probes candidate undocumented endpoints.
"""

import json
import os
import time

import requests

from jumpprob.submit.env import load_dotenv

load_dotenv()
BASE = "https://api.sportspredict.com/api/v1"
EVENT = "aa5572ec-5930-4d99-b06b-f8966333d172"
LOBBY = "8df8038c-fd2c-4a5f-be4e-0e11d5966c05"
HDR = {"Authorization": f"Bearer {os.environ['SPORTSPREDICT_KEY_A']}"}


def get(path, **params):
    try:
        r = requests.get(f"{BASE}{path}", params=params, headers=HDR, timeout=20)
        return r.status_code, (r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    except Exception as e:
        return "ERR", str(e)[:120]
    finally:
        time.sleep(1.1)  # stay well under 60/min


def keys_of(obj):
    if isinstance(obj, list) and obj:
        obj = obj[0]
    return sorted(obj.keys()) if isinstance(obj, dict) else f"<{type(obj).__name__}>"


FIELD_WORDS = ("field", "crowd", "consensus", "average", "avg", "rbp", "point",
               "rank", "leaderboard", "standing", "percentile", "count", "volume",
               "distribution", "edge", "score")


def scan(label, obj):
    """Print keys and flag any that smell like field/crowd data."""
    blob = json.dumps(obj).lower()
    hits = sorted({w for w in FIELD_WORDS if w in blob})
    print(f"\n[{label}] keys={keys_of(obj)}")
    if hits:
        print(f"    >>> field-ish tokens present: {hits}")


# 1. Full field sets of known objects
print("=== KNOWN OBJECTS (full key sets) ===")
sc, ev = get("/events", limit=1); scan("events", ev)
sc, lo = get("/lobbies", event_id=EVENT); scan("lobbies", lo)
sc, ma = get("/matches", event_id=EVENT, lobby_id=LOBBY); scan("matches", ma)
sc, mk = get("/markets", lobby_id=LOBBY); scan("markets[list]", mk)
sc, pr = get("/predictions", lobby_id=LOBBY); scan("predictions", pr)
sc, rs = get("/results", lobby_id=LOBBY); scan("results", rs)

# a single settled result/market/prediction — may carry extra fields the list omits
settled = [r for r in (rs if isinstance(rs, list) else []) if r.get("brier_score") is not None]
if settled:
    mid = settled[0]["market_id"]
    print(f"\n=== SINGLE settled market {mid[:8]} (probe for per-market crowd/field fields) ===")
    for path in (f"/markets/{mid}", f"/markets/{mid}/stats", f"/markets/{mid}/results",
                 f"/predictions/{settled[0].get('id','')}"):
        sc, ob = get(path)
        print(f"  GET {path[:48]:48} -> {sc}  keys={keys_of(ob) if sc==200 else ob if isinstance(ob,str) else ''}")
        if sc == 200:
            scan(path, ob)

# 2. Probe candidate field/leaderboard endpoints
print("\n=== PROBE candidate endpoints (200 = exists) ===")
candidates = [
    "/leaderboard", f"/leaderboard?lobby_id={LOBBY}", "/leaderboards",
    f"/lobbies/{LOBBY}/leaderboard", f"/lobbies/{LOBBY}/standings",
    f"/lobbies/{LOBBY}/rankings", f"/lobbies/{LOBBY}/members",
    f"/lobbies/{LOBBY}", "/rankings", "/standings", "/me", "/users/me",
    "/profile", "/stats", f"/stats?lobby_id={LOBBY}",
    f"/results?lobby_id={LOBBY}&include=field",
]
for path in candidates:
    base = path.split("?")[0]
    params = dict(p.split("=") for p in path.split("?")[1].split("&")) if "?" in path else {}
    sc, ob = get(base, **params)
    flag = ""
    if sc == 200:
        blob = json.dumps(ob).lower()
        flag = " <<< FIELD TOKENS: " + ",".join(w for w in FIELD_WORDS if w in blob) if any(w in blob for w in FIELD_WORDS) else " (200, no field tokens)"
    print(f"  {sc}  {path[:55]:55}{flag}")
