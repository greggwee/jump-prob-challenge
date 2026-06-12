"""Point-in-time backtest of count models on the question archetypes.

Each tournament is predicted by a count model trained ONLY on tournaments that
finished before it (wc2018 is training-only). Baseline = same NegBin family
with the training global mean per half (no covariates, no referee).

Archetypes:
  corners_ht_a_gt_b   more corners than opponent at half time
  cards_2h_ge_2       2+ total cards in the second half
  offsides_a_ge_2     team A caught offside 2+ times (full match)
  sot_a_gt_b_2h       A more shots on target than B in the 2nd half
"""

import numpy as np
import pandas as pd

from jumpprob.backtest.harness import load_backtest_data
from jumpprob.models.count_model import FAMILIES, fit_count_model, p_a_gt_b, p_ge, p_sum_ge
from jumpprob.models.goal_model import fit_goal_model
from jumpprob.normalize.teams import canonical

ORDER = ["wc2018", "euro2020", "wc2022", "afcon2023", "copa2024", "euro2024"]

team_half = pd.read_parquet("data/processed/team_half_counts.parquet")
team_half["team_c"] = team_half["team"].map(canonical)
team_half["cards"] = team_half["yellows"] + team_half["reds"]
team_half["offsides"] = team_half["pass_offsides"] + team_half["offside_events"]

data = load_backtest_data()
fx = data.fixtures[["match_id", "tournament", "date", "home_team", "away_team", "neutral",
                    "elo_home", "elo_away", "referee"]].copy()

# attach lambdas per fixture using the goal model fit at each tournament start
lams = {}
for t in ORDER:
    sub = fx[fx["tournament"] == t]
    if sub.empty:
        continue
    gm = fit_goal_model(data.results_elo, sub["date"].min(), 730.0)
    for m in sub.itertuples():
        lams[m.match_id] = gm.intensities(m.elo_home, m.elo_away, not m.neutral)

rows = []
for r in team_half.itertuples():
    f = fx[fx["match_id"] == r.match_id]
    if f.empty or r.match_id not in lams:
        continue
    f = f.iloc[0]
    is_a = r.team_c == f["home_team"]
    la, lb = lams[r.match_id]
    rows.append({
        "match_id": r.match_id, "tournament": f["tournament"], "period": r.period,
        "side": "a" if is_a else "b", "referee": f["referee"],
        "lam_team": la if is_a else lb, "lam_opp": lb if is_a else la,
        "corners": r.corners, "cards": r.cards, "offsides": r.offsides,
        "shots_on_target": r.shots_on_target,
    })
th = pd.DataFrame(rows)
print(f"prepared {len(th)} team-half rows across {th.match_id.nunique()} matches")

out = []
for i, t in enumerate(ORDER):
    if i == 0:
        continue  # wc2018: training only
    train = th[th["tournament"].isin(ORDER[:i])]
    test = th[th["tournament"] == t]
    if test.empty or train["tournament"].nunique() < 1:
        continue
    model = fit_count_model(train)
    base_mu = {(fam, h): train[train.period == h][fam].mean() for fam in FAMILIES for h in (1, 2)}

    for mid, g in test.groupby("match_id"):
        a1 = g[(g.side == "a") & (g.period == 1)]; a2 = g[(g.side == "a") & (g.period == 2)]
        b1 = g[(g.side == "b") & (g.period == 1)]; b2 = g[(g.side == "b") & (g.period == 2)]
        if len(a1) * len(a2) * len(b1) * len(b2) == 0:
            continue
        a1, a2, b1, b2 = a1.iloc[0], a2.iloc[0], b1.iloc[0], b2.iloc[0]
        ref = a1["referee"]

        def pmf(fam, half, side_row, use_base):
            if use_base:
                mu = base_mu[(fam, half)]
                k = model.params[fam]["k"]
                from scipy.stats import nbinom, poisson
                xs = np.arange(31)
                pm = poisson.pmf(xs, mu) if np.isinf(k) else nbinom.pmf(xs, k, k / (k + mu))
                return pm / pm.sum()
            return model.pmf(fam, model.mu(fam, half, side_row["lam_team"], side_row["lam_opp"],
                                           ref if fam == "cards" else None))

        for use_base, tag in ((False, "model"), (True, "base")):
            preds = {
                "corners_ht_a_gt_b": p_a_gt_b(pmf("corners", 1, a1, use_base), pmf("corners", 1, b1, use_base)),
                "cards_2h_ge_2": p_sum_ge(pmf("cards", 2, a2, use_base), pmf("cards", 2, b2, use_base), 2),
                "offsides_a_ge_2": p_sum_ge(pmf("offsides", 1, a1, use_base), pmf("offsides", 2, a2, use_base), 2),
                "sot_a_gt_b_2h": p_a_gt_b(pmf("shots_on_target", 2, a2, use_base), pmf("shots_on_target", 2, b2, use_base)),
            }
            outcomes = {
                "corners_ht_a_gt_b": float(a1["corners"] > b1["corners"]),
                "cards_2h_ge_2": float(a2["cards"] + b2["cards"] >= 2),
                "offsides_a_ge_2": float(a1["offsides"] + a2["offsides"] >= 2),
                "sot_a_gt_b_2h": float(a2["shots_on_target"] > b2["shots_on_target"]),
            }
            for arch in preds:
                out.append({"tournament": t, "archetype": arch, "src": tag,
                            "p": preds[arch], "outcome": outcomes[arch]})

bt = pd.DataFrame(out)
bt.to_csv("data/processed/backtest_counts.csv", index=False)
summary = (
    bt.assign(sq=(bt.p - bt.outcome) ** 2)
    .pivot_table(index="archetype", columns="src", values="sq", aggfunc="mean")
    .assign(n=bt[bt.src == "model"].groupby("archetype").size())
)
summary["edge"] = summary["base"] - summary["model"]
print(summary.round(4).to_string())
print("\nbase rates:", bt[bt.src == "model"].groupby("archetype")["outcome"].mean().round(3).to_dict())
