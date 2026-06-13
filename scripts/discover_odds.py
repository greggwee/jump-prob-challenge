"""Ground the odds-integration plan: what does the live Odds API actually
offer for the World Cup, and how many credits does a refresh cost?"""

import os
import requests

from jumpprob.submit.env import load_dotenv

load_dotenv()
KEY = os.environ.get("THE_ODDS_API_KEY")
print("key present:", bool(KEY))
BASE = "https://api.the-odds-api.com/v4"

# 1. list sports (free, no quota cost) — find soccer / WC keys
sports = requests.get(f"{BASE}/sports", params={"apiKey": KEY}, timeout=30).json()
soccer = [s for s in sports if "soccer" in s.get("key", "")]
print(f"\n{len(soccer)} soccer competitions available:")
for s in soccer:
    flag = "  <-- WORLD CUP" if "world_cup" in s["key"] or "fifa" in s["key"] else ""
    print(f"  {s['key']:42} active={s.get('active')}{flag}")

# 2. one odds snapshot for the WC key (costs credits) — inspect structure
wc_keys = [s["key"] for s in soccer if "world_cup" in s["key"] or "fifa" in s["key"]]
for wc in wc_keys:
    r = requests.get(f"{BASE}/sports/{wc}/odds",
                     params={"apiKey": KEY, "regions": "eu,uk", "markets": "h2h,totals",
                             "oddsFormat": "decimal"}, timeout=30)
    print(f"\n=== {wc}: HTTP {r.status_code} ===")
    print("quota remaining:", r.headers.get("x-requests-remaining"),
          "| used:", r.headers.get("x-requests-used"))
    if r.status_code != 200:
        print(r.text[:300]); continue
    games = r.json()
    print(f"{len(games)} games with odds")
    for g in games[:3]:
        print(f"\n  {g['home_team']} vs {g['away_team']}  @ {g['commence_time']}")
        for bm in g.get("bookmakers", [])[:2]:
            mk = ", ".join(f"{m['key']}({len(m['outcomes'])})" for m in bm["markets"])
            print(f"    {bm['key']:14} markets: {mk}")
        # show one h2h price set
        for bm in g.get("bookmakers", []):
            for m in bm["markets"]:
                if m["key"] == "h2h":
                    prices = {o['name']: o['price'] for o in m['outcomes']}
                    print(f"    h2h ({bm['key']}): {prices}")
                    break
            break
