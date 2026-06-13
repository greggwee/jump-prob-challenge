"""Demonstrate the wisdom-of-crowds consensus on LIVE odds vs our pure model.

For a few upcoming matches: build the multi-book Shin consensus, invert to
market lambdas, and compare the market's win probability with our pure-Elo
model's. This shows concretely what the odds anchor changes.
"""

import os

import numpy as np
import requests

from jumpprob.models.devig import consensus, overround
from jumpprob.models.market_anchor import implied_lambdas
from jumpprob.normalize.teams import canonical
from jumpprob.pipeline import Pipeline
from jumpprob.simulate import markets as mk
from jumpprob.submit.env import load_dotenv

load_dotenv()
BASE = "https://api.the-odds-api.com/v4"
ev = requests.get(f"{BASE}/sports/soccer_fifa_world_cup/odds",
                  params={"apiKey": os.environ["THE_ODDS_API_KEY"], "regions": "eu,uk",
                          "markets": "h2h,totals", "oddsFormat": "decimal"}, timeout=30).json()

pipe = Pipeline()
WATCH = {"United States", "Paraguay", "Qatar", "Switzerland", "Brazil", "Morocco"}

for g in ev:
    home, away = g["home_team"], g["away_team"]
    if canonical(home) not in WATCH and canonical(away) not in WATCH:
        continue
    # collect h2h books in [home, draw, away] order
    h2h_books = []
    for bm in g.get("bookmakers", []):
        for m in bm["markets"]:
            if m["key"] == "h2h":
                px = {o["name"]: o["price"] for o in m["outcomes"]}
                if home in px and away in px and "Draw" in px:
                    h2h_books.append((bm["key"], [px[home], px["Draw"], px[away]]))
    if not h2h_books:
        continue
    cons = consensus(h2h_books)  # [P(home), P(draw), P(away)]
    avg_margin = np.mean([overround(o) for _, o in h2h_books])

    # our pure model
    ra, rb = pipe.ratings.get(canonical(home)), pipe.ratings.get(canonical(away))
    if ra is None or rb is None:
        print(f"\n{home} vs {away}: team not in ratings ({home if ra is None else away}) — skip")
        continue
    grid = pipe.goal_model.scoreline_grid(*pipe.goal_model.intensities(ra, rb))
    model = [mk.p_win(grid, "a"), mk.p_draw(grid), mk.p_win(grid, "b")]

    la_m, lb_m = pipe.goal_model.intensities(ra, rb)
    la_k, lb_k = implied_lambdas(cons[0], cons[1], cons[2], rho=pipe.goal_model.rho_dc)

    print(f"\n{home} vs {away}  ({len(h2h_books)} books, avg margin {avg_margin:.1%})")
    print(f"  market consensus : home {cons[0]:.0%}  draw {cons[1]:.0%}  away {cons[2]:.0%}"
          f"   (lam {la_k:.2f}-{lb_k:.2f})")
    print(f"  our pure model   : home {model[0]:.0%}  draw {model[1]:.0%}  away {model[2]:.0%}"
          f"   (lam {la_m:.2f}-{lb_m:.2f})")
    # 0.85 market / 0.15 model logit blend on the home-win prob
    from jumpprob.models.market_anchor import blend
    blended_home = blend([cons[0], model[0]], [0.85, 0.15])
    print(f"  blended (0.85/0.15) home-win: {blended_home:.0%}")
