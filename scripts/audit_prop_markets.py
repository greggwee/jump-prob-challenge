"""Phase D.0 audit: does The Odds API expose bookmaker PROP markets for the WC
(player shots on target, anytime scorer, corners, cards)? #4 depends on this."""

import os

import requests

from jumpprob.submit.env import load_dotenv

load_dotenv()
KEY = os.environ["THE_ODDS_API_KEY"]
BASE = "https://api.the-odds-api.com/v4"
SPORT = "soccer_fifa_world_cup"

# candidate soccer prop market keys (The Odds API naming)
PROP_KEYS = [
    "player_shots_on_target", "player_shots", "player_goal_scorer_anytime",
    "player_goal_scorer_first", "player_assists", "player_to_receive_card",
    "alternate_totals", "team_totals", "totals_corners", "btts", "draw_no_bet",
]

# need an event id first
ev = requests.get(f"{BASE}/sports/{SPORT}/events", params={"apiKey": KEY}, timeout=30)
print(f"events: HTTP {ev.status_code}, quota used {ev.headers.get('x-requests-used')} "
      f"remaining {ev.headers.get('x-requests-remaining')}")
events = ev.json()
if not events:
    print("no upcoming events"); raise SystemExit
e = events[0]
eid = e["id"]
print(f"probing event: {e['home_team']} vs {e['away_team']} ({eid[:8]})\n")

# probe each prop market key on the event-odds endpoint
for mk in PROP_KEYS:
    r = requests.get(f"{BASE}/sports/{SPORT}/events/{eid}/odds",
                     params={"apiKey": KEY, "regions": "eu,uk,us", "markets": mk,
                             "oddsFormat": "decimal"}, timeout=30)
    if r.status_code != 200:
        try:
            msg = r.json().get("message", r.text)[:90]
        except Exception:
            msg = r.text[:90]
        print(f"  {mk:28} HTTP {r.status_code}  {msg}")
        continue
    data = r.json()
    books = data.get("bookmakers", [])
    n_books = len(books)
    outs = 0
    sample = ""
    for bm in books:
        for m in bm.get("markets", []):
            if m["key"] == mk:
                outs += len(m["outcomes"])
                if not sample and m["outcomes"]:
                    o = m["outcomes"][0]
                    sample = f"{o.get('description','')} {o.get('name','')} @ {o.get('price','')}"
    flag = " <== AVAILABLE" if n_books and outs else " (no books offer it)"
    print(f"  {mk:28} HTTP 200  {n_books} books, {outs} outcomes{flag}  {sample[:50]}")
