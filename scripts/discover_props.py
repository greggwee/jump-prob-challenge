"""Phase D / improvement #4 feasibility probe: does The Odds API expose prop
markets (player SoT, anytime scorer, cards, corners, btts) for the World Cup,
on our plan, and at what credit cost?

Player/alternate props use the per-EVENT endpoint and are billed per market x
region. We probe ONE event with eu region only to keep the cost minimal.
"""

import os

import requests

from jumpprob.submit.env import load_dotenv

load_dotenv()
BASE = "https://api.the-odds-api.com/v4"
SPORT = "soccer_fifa_world_cup"
KEY = os.environ["THE_ODDS_API_KEY"]

# 1. event list (free — no markets requested)
ev = requests.get(f"{BASE}/sports/{SPORT}/events", params={"apiKey": KEY}, timeout=30)
print(f"events: HTTP {ev.status_code}  remaining={ev.headers.get('x-requests-remaining')}")
events = ev.json()
if not events:
    print("no upcoming events"); raise SystemExit
e0 = events[0]
print(f"probing event: {e0['home_team']} vs {e0['away_team']}  ({e0['id']})\n")

# 2. probe candidate prop market keys via the per-event endpoint
candidates = [
    "btts", "alternate_totals", "double_chance", "draw_no_bet", "team_totals",
    "player_goal_scorer_anytime", "player_shots_on_target", "player_shots",
    "player_assists", "player_to_receive_card",
    "alternate_team_totals",
]
for mk in candidates:
    r = requests.get(f"{BASE}/sports/{SPORT}/events/{e0['id']}/odds",
                     params={"apiKey": KEY, "regions": "eu", "markets": mk,
                             "oddsFormat": "decimal"}, timeout=30)
    rem = r.headers.get("x-requests-remaining")
    if r.status_code == 200:
        data = r.json()
        bks = data.get("bookmakers", [])
        n_out = sum(len(m["outcomes"]) for b in bks for m in b.get("markets", []))
        books = sorted({b["key"] for b in bks for m in b.get("markets", []) if m["key"] == mk})
        print(f"  {mk:28} 200  books={len(books)} outcomes={n_out} rem={rem}  {books[:4]}")
    else:
        msg = ""
        try:
            msg = r.json().get("message", "")[:60]
        except Exception:
            msg = r.text[:60]
        print(f"  {mk:28} {r.status_code}  {msg}  rem={rem}")
