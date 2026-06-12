"""Backtest player prop rates: P(1+ SoT), P(scores), conditioned on minutes.

Point-in-time: each player-match uses only earlier matches for the rate;
population priors come from tournaments strictly before the evaluated one.
Baseline: population prior rate alone (no player identity).
Eligibility: players with >= 30 minutes in the match (props are quoted for
expected starters; minutes models handle the rest in live use).
"""

import numpy as np
import pandas as pd

from jumpprob.models.player_model import expanding_rates, p_at_least_one

ORDER = ["wc2018", "euro2020", "wc2022", "afcon2023", "copa2024", "euro2024"]

players = pd.read_parquet("data/processed/player_match.parquet")
players["date"] = pd.to_datetime(players["date"])

rows = []
for i, t in enumerate(ORDER):
    if i == 0:
        continue
    train = players[players["tournament"].isin(ORDER[:i])]
    test_ids = players["tournament"] == t
    prior_sot = train["sot"].sum() / train["minutes"].sum()
    prior_goal = train["goals"].sum() / train["minutes"].sum()

    for m0 in (90.0, 270.0, 540.0):
        # expanding rates over all data up to and including tournament t,
        # priors fixed from training window
        upto = players[players["tournament"].isin(ORDER[: i + 1])]
        rated = expanding_rates(upto, m0, prior_sot, prior_goal)
        ev = rated[(rated["tournament"] == t) & (rated["minutes"] >= 30)]
        for kind, rate_col, out_col, prior in (
            ("sot_ge_1", "rate_sot", "sot", prior_sot),
            ("scores", "rate_goal", "goals", prior_goal),
        ):
            p_model = p_at_least_one(ev[rate_col], ev["minutes"])
            p_base = p_at_least_one(prior, ev["minutes"])
            y = (ev[out_col] >= 1).astype(float).to_numpy()
            rows.append({
                "tournament": t, "archetype": kind, "m0": m0, "n": len(ev),
                "brier_model": float(np.mean((p_model - y) ** 2)),
                "brier_base": float(np.mean((p_base - y) ** 2)),
                "base_rate": float(y.mean()),
            })

bt = pd.DataFrame(rows)
agg = bt.groupby(["archetype", "m0"]).apply(
    lambda g: pd.Series({
        "n": g["n"].sum(),
        "brier_model": np.average(g["brier_model"], weights=g["n"]),
        "brier_base": np.average(g["brier_base"], weights=g["n"]),
    }), include_groups=False,
)
agg["edge"] = agg["brier_base"] - agg["brier_model"]
print(agg.round(4).to_string())
bt.to_csv("data/processed/backtest_players.csv", index=False)
