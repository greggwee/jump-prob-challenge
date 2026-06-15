"""Evidence for the player-data-scarcity question: how thin is our per-player
history, and is FBref/Understat reachable for richer club rates?"""

import pandas as pd
import requests

pm = pd.read_parquet("data/processed/player_match.parquet")
tot = pm.groupby("player").agg(matches=("minutes", "size"), minutes=("minutes", "sum"),
                               sot=("sot", "sum"), goals=("goals", "sum"))
print(f"player_match table: {len(pm)} player-matches, {pm.player.nunique()} distinct players")
print(f"median career minutes per player in our table: {tot.minutes.median():.0f}")
print(f"players with < 180 min (under ~2 full matches) of history: "
      f"{(tot.minutes < 180).mean():.0%}\n")

# how much history do current WC starters have? sample some names from recent losses
for name in ["Takefusa Kubo", "Cody Gakpo", "Folarin Balogun", "Jonathan David",
             "Granit Xhaka", "Akram Afif"]:
    hit = tot[[name.lower() in p.lower() for p in tot.index]]
    if len(hit):
        r = hit.iloc[0]
        print(f"  {name:18} {int(r.minutes):>5} min, {int(r.matches)} matches, "
              f"{int(r.sot)} SoT, {int(r.goals)} goals  (our entire data on them)")
    else:
        print(f"  {name:18} NOT in our data (0 prior minutes -> pure population prior)")

# FBref / Understat reachability (club-season rates would fix the scarcity)
print("\nclub-data source reachability:")
for name, url in [("FBref", "https://fbref.com/en/comps/1/World-Cup-Stats"),
                  ("Understat", "https://understat.com/league/EPL")]:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        print(f"  {name:10} HTTP {r.status_code}  ({len(r.content)//1024} KB)")
    except Exception as e:
        print(f"  {name:10} ERR {str(e)[:70]}")
