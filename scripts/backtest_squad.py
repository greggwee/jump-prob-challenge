"""Squad-feature decision backtest (keep only if OOS Brier improves).

Per tournament t (point-in-time): compute squad quality for t's teams from
player history before t; fit the one-parameter squad adjustment b_sq on PRIOR
tournaments' group matches; apply to t's fixtures; compare match-archetype
Brier with vs without the adjustment. Decision rule: keep iff pooled OOS Brier
improves.
"""

import numpy as np
import pandas as pd

from jumpprob.backtest.harness import ARCHETYPES, _settle, load_backtest_data
from jumpprob.features.squad import fit_squad_coef, squad_quality
from jumpprob.models.goal_model import fit_goal_model
from jumpprob.normalize.teams import canonical
from jumpprob.simulate import markets as mk

ORDER = ["wc2018", "euro2020", "wc2022", "afcon2023", "copa2024", "euro2024"]

players = pd.read_parquet("data/processed/player_match.parquet")
players["date"] = pd.to_datetime(players["date"])
players["team_c"] = players["team"].map(canonical)
data = load_backtest_data()

# squad quality per tournament (point-in-time)
sq_by_t = {}
for t in ORDER:
    fx = data.fixtures[data.fixtures["tournament"] == t]
    if fx.empty:
        continue
    sq = squad_quality(players.assign(team=players["team_c"]), t, fx["date"].min())
    sq_by_t[t] = sq

rows = []
for i, t in enumerate(ORDER):
    if i == 0 or t not in sq_by_t:
        continue
    fx = data.fixtures[data.fixtures["tournament"] == t]
    as_of = fx["date"].min()
    gm = fit_goal_model(data.results_elo, as_of, 730.0)

    # training rows from prior tournaments: (lam_v1, goals, sq_diff) per side
    lam_tr, goals_tr, sqd_tr = [], [], []
    for tp in ORDER[:i]:
        if tp not in sq_by_t:
            continue
        for m in data.fixtures[data.fixtures["tournament"] == tp].itertuples():
            sq = sq_by_t[tp]
            if m.home_team not in sq.index or m.away_team not in sq.index:
                continue
            la, lb = gm.intensities(m.elo_home, m.elo_away, not m.neutral)
            lam_tr += [la, lb]
            goals_tr += [m.home_score, m.away_score]
            sqd_tr += [sq[m.home_team] - sq[m.away_team], sq[m.away_team] - sq[m.home_team]]
    if len(lam_tr) < 40:
        continue
    sqd_tr = np.asarray(sqd_tr)
    b_sq = fit_squad_coef(np.asarray(lam_tr), np.asarray(goals_tr), sqd_tr)
    z_mu, z_sd = sqd_tr.mean(), max(sqd_tr.std(), 1e-9)

    sq = sq_by_t[t]
    for m in fx.itertuples():
        if m.home_team not in sq.index or m.away_team not in sq.index:
            continue
        la, lb = gm.intensities(m.elo_home, m.elo_away, not m.neutral)
        d = sq[m.home_team] - sq[m.away_team]
        la2 = la * np.exp(b_sq * ((d - z_mu) / z_sd))
        lb2 = lb * np.exp(b_sq * ((-d - z_mu) / z_sd))
        outcomes = _settle(m, data.half_goals)
        grid0 = gm.scoreline_grid(la, lb)
        grid1 = gm.scoreline_grid(la2, lb2)
        for arch in ("win_a", "win_b", "draw", "total_ge_3", "btts"):
            y = outcomes.get(arch)
            if y is None:
                continue
            p0 = {"win_a": mk.p_win(grid0, "a"), "win_b": mk.p_win(grid0, "b"),
                  "draw": mk.p_draw(grid0), "total_ge_3": mk.p_total_ge(grid0, 3),
                  "btts": mk.p_btts(grid0)}[arch]
            p1 = {"win_a": mk.p_win(grid1, "a"), "win_b": mk.p_win(grid1, "b"),
                  "draw": mk.p_draw(grid1), "total_ge_3": mk.p_total_ge(grid1, 3),
                  "btts": mk.p_btts(grid1)}[arch]
            rows.append({"tournament": t, "b_sq": b_sq, "archetype": arch,
                         "p_plain": p0, "p_squad": p1, "outcome": y})

bt = pd.DataFrame(rows)
bt.to_csv("data/processed/backtest_squad.csv", index=False)
bt["sq0"] = (bt.p_plain - bt.outcome) ** 2
bt["sq1"] = (bt.p_squad - bt.outcome) ** 2
print("fitted b_sq per tournament:",
      bt.groupby("tournament")["b_sq"].first().round(4).to_dict())
print(bt.groupby("archetype")[["sq0", "sq1"]].mean().round(4)
      .rename(columns={"sq0": "brier_plain", "sq1": "brier_squad"}).to_string())
print(f"\npooled: plain {bt.sq0.mean():.4f} squad {bt.sq1.mean():.4f} "
      f"-> {'KEEP' if bt.sq1.mean() < bt.sq0.mean() else 'REJECT'} squad feature")
