"""GBM ensemble decision backtest: per tournament point-in-time, train the
hybrid GBM on prior internationals (2006+, competitive), evaluate on the
tournament's match-result archetypes, and fit the logit blend weight on
out-of-sample predictions pooled from PRIOR evaluation tournaments only."""

import numpy as np
import pandas as pd

from jumpprob.backtest.harness import load_backtest_data, _settle
from jumpprob.models.gbm_model import add_rest_days, build_features, predict_gbm, train_gbm
from jumpprob.models.goal_model import fit_goal_model
from jumpprob.models.market_anchor import blend

ORDER = ["wc2018", "euro2020", "wc2022", "afcon2023", "copa2024", "euro2024"]
W_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

data = load_backtest_data()
results = add_rest_days(data.results_elo)

rows = []
for i, t in enumerate(ORDER):
    fx = data.fixtures[data.fixtures["tournament"] == t]
    if fx.empty or i == 0:
        continue
    as_of = fx["date"].min()
    gm = fit_goal_model(data.results_elo, as_of, 730.0)
    train = results[(results["date"] < as_of)
                    & (results["date"] >= as_of - pd.Timedelta(days=365 * 20))
                    & (~results["tournament"].str.contains("Friendly"))]
    feat_tr = build_features(train, gm)
    model = train_gbm(feat_tr)

    fx_r = results.merge(fx[["match_id", "date", "home_team", "away_team"]],
                         on=["date", "home_team", "away_team"]).drop_duplicates("match_id")
    feat_te = build_features(fx_r, gm)
    probs = predict_gbm(model, feat_te)

    for j, m in enumerate(feat_te.itertuples()):
        fx_row = fx[fx["match_id"] == m.match_id].iloc[0]
        outcomes = _settle(fx_row, data.half_goals)
        for arch, cls, p1 in (("win_a", 0, m.p1_home), ("draw", 1, m.p1_draw), ("win_b", 2, m.p1_away)):
            rows.append({"tournament": t, "archetype": arch, "p_gbm": float(probs[j, cls]),
                         "p_v1": float(p1), "outcome": outcomes[arch]})

bt = pd.DataFrame(rows)
bt.to_csv("data/processed/backtest_gbm.csv", index=False)

bt["sq_gbm"] = (bt.p_gbm - bt.outcome) ** 2
bt["sq_v1"] = (bt.p_v1 - bt.outcome) ** 2
print(bt.groupby("archetype")[["sq_v1", "sq_gbm"]].mean().round(4).to_string())
print(f"pooled: v1 {bt.sq_v1.mean():.4f} gbm {bt.sq_gbm.mean():.4f}")

# blend weight fit on expanding OOS: weight chosen on tournaments < t, applied to t
print("\nblend (w = GBM share, logit space, expanding OOS):")
applied = []
for i, t in enumerate(ORDER):
    if t not in set(bt.tournament):
        continue
    prior = bt[bt.tournament.isin(ORDER[:i])]
    cur = bt[bt.tournament == t]
    if prior.empty:
        w_star = 0.0
    else:
        scores = {w: ((prior.apply(lambda r: blend([r.p_gbm, r.p_v1], [w, 1 - w]), axis=1)
                       - prior.outcome) ** 2).mean() for w in W_GRID}
        w_star = min(scores, key=scores.get)
    pb = cur.apply(lambda r: blend([r.p_gbm, r.p_v1], [w_star, 1 - w_star]), axis=1)
    applied.append(pd.DataFrame({"tournament": t, "w": w_star,
                                 "sq_blend": (pb - cur.outcome) ** 2,
                                 "sq_v1": cur.sq_v1}))
ap = pd.concat(applied)
print(ap.groupby("tournament")[["w"]].first().join(
    ap.groupby("tournament")[["sq_v1", "sq_blend"]].mean()).round(4).to_string())
print(f"\npooled blended {ap.sq_blend.mean():.4f} vs v1 {ap.sq_v1.mean():.4f} "
      f"-> {'KEEP' if ap.sq_blend.mean() < ap.sq_v1.mean() else 'REJECT'} GBM member")
