"""Fit ensemble blend weights on real tournament data (assumption A10).

Anchor = FiveThirtyEight's published pre-match forecasts for WC 2018 + WC 2022
(prob1/probtie/prob2 and projected scores, i.e. expected-goal intensities).
For each blend weight w (anchor share), we blend:
  - result archetypes (win/draw): logit-space blend of probabilities
  - goal-count archetypes: log-space blend of intensities, then DC grid
and report Brier per archetype per w. The optimal w is what Bot A ships with
until live odds data replaces 538 as the anchor.
"""

import numpy as np
import pandas as pd

from jumpprob.backtest.harness import ARCHETYPES, _settle, load_backtest_data
from jumpprob.ingest import fivethirtyeight as fte
from jumpprob.models.goal_model import fit_goal_model, dc_grid
from jumpprob.models.market_anchor import blend
from jumpprob.normalize.teams import canonical
from jumpprob.simulate import markets

print("fetching 538 SPI forecasts...")
fte.fetch()
spi = fte.load_world_cup_forecasts()
spi["team1_c"] = spi["team1"].map(canonical)
spi["team2_c"] = spi["team2"].map(canonical)

data = load_backtest_data()
fixtures = data.fixtures[data.fixtures["tournament"].isin(["wc2022", "euro2020"])]

W_GRID = [0.0, 0.25, 0.5, 0.65, 0.75, 0.85, 1.0]
RESULT_ARCHS = {"win_a", "win_b", "draw"}

rows = []
joined = 0
for tournament, fx in fixtures.groupby("tournament"):
    model = fit_goal_model(data.results_elo, fx["date"].min(), half_life_days=730.0)
    for m in fx.itertuples():
        cand = spi[
            (spi["date"].between(m.date - pd.Timedelta(days=1), m.date + pd.Timedelta(days=1)))
            & (
                ((spi["team1_c"] == m.home_team) & (spi["team2_c"] == m.away_team))
                | ((spi["team1_c"] == m.away_team) & (spi["team2_c"] == m.home_team))
            )
        ]
        if cand.empty:
            continue
        joined += 1
        s = cand.iloc[0]
        flipped = s["team1_c"] != m.home_team
        p538 = {
            "win_a": s["prob2"] if flipped else s["prob1"],
            "win_b": s["prob1"] if flipped else s["prob2"],
            "draw": s["probtie"],
        }
        lam538_a = s["proj_score2"] if flipped else s["proj_score1"]
        lam538_b = s["proj_score1"] if flipped else s["proj_score2"]

        la_m, lb_m = model.intensities(m.elo_home, m.elo_away, not m.neutral)
        probs_model = markets.answer_all(model, m.elo_home, m.elo_away, not m.neutral)
        outcomes = _settle(m, data.half_goals)

        for w in W_GRID:
            la = float(np.exp(w * np.log(lam538_a) + (1 - w) * np.log(la_m)))
            lb = float(np.exp(w * np.log(lam538_b) + (1 - w) * np.log(lb_m)))
            grid = dc_grid(la, lb, model.rho_dc)
            blended = {
                "win_a": blend([p538["win_a"], probs_model["win_a"]], [w, 1 - w]),
                "win_b": blend([p538["win_b"], probs_model["win_b"]], [w, 1 - w]),
                "draw": blend([p538["draw"], probs_model["draw"]], [w, 1 - w]),
                "total_ge_2": markets.p_total_ge(grid, 2),
                "total_ge_3": markets.p_total_ge(grid, 3),
                "btts": markets.p_btts(grid),
                "btts_and_3plus": markets.p_btts_and_total_ge(grid, 3),
                "a_scores_2h": markets.p_scores_in_half(la, 2),
                "b_scores_2h": markets.p_scores_in_half(lb, 2),
                "a_scores_1h": markets.p_scores_in_half(la, 1),
            }
            for arch in ARCHETYPES:
                if outcomes.get(arch) is None:
                    continue
                rows.append(
                    {"tournament": tournament, "w": w, "archetype": arch,
                     "p": blended[arch], "outcome": outcomes[arch]}
                )

print(f"joined {joined}/{len(fixtures)} fixtures to 538 forecasts")
bt = pd.DataFrame(rows)
bt.to_csv("data/processed/blend_fit.csv", index=False)

pivot = (
    bt.assign(sq=(bt["p"] - bt["outcome"]) ** 2)
    .groupby(["archetype", "w"])["sq"].mean().unstack("w").round(4)
)
pivot["best_w"] = pivot.idxmin(axis=1)
print(pivot.to_string())
overall = bt.assign(sq=(bt["p"] - bt["outcome"]) ** 2).groupby("w")["sq"].mean()
print("\noverall Brier by w (0 = pure model, 1 = pure 538):")
print(overall.round(4).to_string())
