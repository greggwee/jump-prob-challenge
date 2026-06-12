"""Groll-style hybrid GBM: gradient boosting on engineered features WITH the
statistical model's probabilities as inputs, predicting {home win, draw, away win}.

Point-in-time: features for a match use only pre-match information (Elo columns
are pre-match by construction; rest days from prior fixtures; v1 probabilities
come from a goal model fit before the evaluation tournament — generating
features for PAST training matches with it leaks nothing into evaluation).

Guards against the small-evaluation-set overfit risk (V2-PLAN risk #5):
time-ordered validation split for early stopping, shallow trees, and the
ensemble weight is fit on out-of-sample predictions only (scripts/backtest_gbm.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jumpprob.features.elo import k_factor
from jumpprob.models.goal_model import GoalModel
from jumpprob.simulate import markets as mk

FEATURES = ["elo_diff", "home_ground", "k_importance", "rest_diff",
            "p1_home", "p1_draw", "p1_away"]


def add_rest_days(results: pd.DataFrame) -> pd.DataFrame:
    """Days since each team's previous match (capped at 60)."""
    long = pd.concat([
        results[["date", "home_team"]].rename(columns={"home_team": "team"}),
        results[["date", "away_team"]].rename(columns={"away_team": "team"}),
    ]).sort_values("date")
    long["prev"] = long.groupby("team")["date"].shift(1)
    long["rest"] = (long["date"] - long["prev"]).dt.days.clip(upper=60).fillna(60)
    rest = long.drop_duplicates(["date", "team"], keep="last").set_index(["date", "team"])["rest"]
    out = results.copy()
    out["rest_home"] = [rest.get((d, t), 60.0) for d, t in zip(out["date"], out["home_team"])]
    out["rest_away"] = [rest.get((d, t), 60.0) for d, t in zip(out["date"], out["away_team"])]
    return out


def build_features(df: pd.DataFrame, gm: GoalModel) -> pd.DataFrame:
    out = df.copy()
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out["home_ground"] = (~out["neutral"]).astype(int)
    out["k_importance"] = out["tournament"].map(k_factor)
    out["rest_diff"] = out["rest_home"] - out["rest_away"]
    p_h, p_d, p_a = [], [], []
    for r in out.itertuples():
        grid = gm.scoreline_grid(*gm.intensities(r.elo_home, r.elo_away, not r.neutral))
        p_h.append(mk.p_win(grid, "a"))
        p_d.append(mk.p_draw(grid))
        p_a.append(mk.p_win(grid, "b"))
    out["p1_home"], out["p1_draw"], out["p1_away"] = p_h, p_d, p_a
    out["target"] = np.select(
        [out["home_score"] > out["away_score"], out["home_score"] == out["away_score"]], [0, 1], 2)
    return out


def train_gbm(feat: pd.DataFrame):
    import lightgbm as lgb

    feat = feat.sort_values("date")
    split = int(len(feat) * 0.9)
    tr, va = feat.iloc[:split], feat.iloc[split:]
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=600, learning_rate=0.05,
        num_leaves=15, min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        verbose=-1,
    )
    model.fit(tr[FEATURES], tr["target"],
              eval_set=[(va[FEATURES], va["target"])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def predict_gbm(model, feat: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(feat[FEATURES])
